"""Fetch an image-caption corpus from a Hugging Face webdataset repo.

Written for the 1.2 data plan, where the lever is caption DENSITY rather than
image count. The corpus 1.1 trained on was 84% DiffusionDB — Stable Diffusion
1.x output paired with tag-salad prompts ("artstation, cinematic, 8k"), which
teach style and say almost nothing about which objects are present. That is the
measured cause of "a horse standing in a field" producing a field: RECAP froze
the model and the compute, changed only the captions, and object-presence
accuracy went 80.8 -> 86.2 while the control arm with the original captions did
not move at all (arXiv:2310.16656).

Three repos are wired in, all public, all shipping actual image bytes rather
than URLs to scrape (checked: LAION-5B, LAION-COCO and PixelProse are all
either gated or URL-only, so none of them are usable here):

  t2i2m   jackyhate/text-to-image-2M — FLUX images, dense VLM captions
  dalle3  ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions
  cc12m   pixparse/cc12m-wds — real photographs, thin alt-text, for diversity

Runs on a Kaggle CPU kernel on purpose: CPU notebooks have no weekly quota, so
fetching and transcoding costs nothing out of the 45 GPU-hours. Only the encode
pass afterwards spends them.
"""

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request

from PIL import Image

REPOS = {
    "t2i2m": {
        "repo": "jackyhate/text-to-image-2M",
        "path": "data_512_2M/data_{i:06d}.tar",
        "shards": 46,
        "caption": lambda d: d.get("prompt", ""),
    },
    "dalle3": {
        "repo": "ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions",
        "path": "data/data-{i:06d}.tar",
        "shards": 69,
        # Ten zbior ma opis dlugi i krotki. Bierzemy dlugi, ale w 10%
        # przypadkow krotki: DALL-E 3 trenowal na 95% podpisow syntetycznych i
        # 5% prawdziwych wlasnie dlatego, ze model uczony wylacznie na dlugich
        # opisach przestaje rozumiec krotkie polecenie od czlowieka.
        "caption": lambda d: (d.get("long_caption") or d.get("caption")
                              or d.get("prompt") or ""),
        "short": lambda d: (d.get("short_caption") or d.get("caption") or ""),
    },
    "cc12m": {
        "repo": "pixparse/cc12m-wds",
        "path": "cc12m-train-{i:04d}.tar",
        "shards": 2176,
        "caption": lambda d: d.get("caption", ""),
    },
}

BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def square_jpeg(data, res, quality=88):
    """Centre-crop to square, resize, hand back JPEG bytes.

    Square crop rather than letterbox, as in every other fetcher here: grey
    bars would teach the model that grey bars are part of what a picture looks
    like.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    s = min(img.size)
    l, t = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((res, res), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def shard_size(url, tries=5):
    """How many bytes the shard is supposed to be, from the server."""
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range", "")
                if "/" in cr:
                    return int(cr.rsplit("/", 1)[1])
                return int(r.headers.get("Content-Length", 0))
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))
    return 0


def stream_shard(url, dst, want, tries=10):
    """Shard to a file on disk, resumed until it is the size the server said.

    **The loop ends on a byte count, never on a closed connection.** The first
    version returned as soon as `read()` came back empty, which is exactly what
    a connection dropped mid-transfer looks like: the first run died on shard 5
    with `tarfile.ReadError: unexpected end of data` after writing 1.71 of
    3.5 GB, having already banked 240,000 images. That is the third time in
    this project that "the transfer finished" has been mistaken for "the file
    is complete" — a truncated Colab download and a zero-byte Kaggle checkpoint
    were the other two.

    Not into memory either: shards are gigabytes each against a 20 GB working
    disk, so one is fetched, used and deleted at a time.
    """
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    for t in range(tries):
        if have >= want:
            return have
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "ab") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
        except Exception as e:
            print(f"    blad przy {have/1e9:.2f} GB: {e}", flush=True)
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have < want:
            print(f"    mam {have/1e9:.2f} z {want/1e9:.2f} GB, ponawiam",
                  flush=True)
            time.sleep(min(30, 3 * (t + 1)))
    if have != want:
        raise SystemExit(f"{dst}: {have} z {want} B po {tries} probach")
    return have


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=sorted(REPOS))
    p.add_argument("--first-shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--out-prefix", default="./gweird")
    p.add_argument("--max-images", type=int, default=0,
                   help="zatrzymaj sie po tylu obrazach; 0 = bez limitu")
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=400,
                   help="400, nie 300: przy 64 tokenach BPE dluzszy opis wreszcie "
                        "ma gdzie sie zmiescic")
    p.add_argument("--short-every", type=int, default=10,
                   help="co ile obrazow uzyc krotkiego podpisu zamiast dlugiego "
                        "(tylko dalle3); 10 daje mieszanke 90/10")
    a = p.parse_args()

    spec = REPOS[a.source]
    images_path = f"{a.out_prefix}_images.jpgbin"
    fh = open(images_path, "wb")
    captions, offsets, kept, seen = [], [0], 0, 0

    last = min(a.first_shard + a.shards, spec["shards"])
    for i in range(a.first_shard, last):
        url = BASE.format(repo=spec["repo"], path=spec["path"].format(i=i))
        print(f"[{i}] {url.rsplit('/', 1)[-1]}", flush=True)
        tmp = f"{a.out_prefix}.shard.tar"
        if os.path.exists(tmp):
            os.remove(tmp)
        want = shard_size(url)
        nbytes = stream_shard(url, tmp, want)
        print(f"    {nbytes/1e9:.2f} GB (komplet)", flush=True)

        # ignore_zeros: shardy sklejane z kawalkow miewaja bloki zerowe w
        # srodku, na ktorych zwykly czytnik tar konczy w polowie po cichu.
        tf = tarfile.open(tmp, mode="r:", ignore_zeros=True)
        pending = {}
        # Jeden uszkodzony shard nie moze zabrac ze soba calego biegu. Poprzedni
        # bieg mial juz zebrane 240 000 obrazow, kiedy padl na szostym
        # shardzie, i stracil wszystko, bo wyjatek doszedl na sama gore.
        try:
            for m in tf:
                if not m.isfile():
                    continue
                stem, ext = os.path.splitext(m.name)
                ext = ext.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp", ".json", ".txt"):
                    continue
                slot = pending.setdefault(stem, {})
                slot["meta" if ext in (".json", ".txt") else "img"] = tf.extractfile(m).read()
                if "img" not in slot or "meta" not in slot:
                    continue
                blob = pending.pop(stem)
                seen += 1

                try:
                    meta = json.loads(blob["meta"])
                except Exception:
                    meta = {"caption": blob["meta"].decode("utf-8", "replace")}
                cap = spec["caption"](meta)
                if a.source == "dalle3" and kept % a.short_every == 0:
                    cap = spec["short"](meta) or cap
                cap = " ".join(str(cap).split())[:a.max_chars]
                if len(cap.split()) < a.min_words:
                    continue
                try:
                    jpg = square_jpeg(blob["img"], a.res)
                except Exception:
                    continue

                fh.write(jpg)
                offsets.append(offsets[-1] + len(jpg))
                captions.append(cap)
                kept += 1
                if kept % 20000 == 0:
                    fh.flush()
                    print(f"    {kept} zachowanych z {seen}", flush=True)
                if a.max_images and kept >= a.max_images:
                    break
        except (tarfile.ReadError, EOFError) as e:
            print(f"    shard uszkodzony ({e}) — pomijam, mam {kept}",
                  flush=True)
        tf.close()
        os.remove(tmp)          # 9 GB na dysku o pojemnosci 20 GB
        if a.max_images and kept >= a.max_images:
            break

    fh.close()

    # Sprawdzane PRZED zapisem meta, zeby bieg, ktory rozjechal pliki, nie
    # zostawil po sobie meta.json blogoslawiacego ten stan.
    size = os.path.getsize(images_path)
    if size != offsets[-1] or len(captions) != kept or len(offsets) != kept + 1:
        raise SystemExit(f"NIEZGODNOSC: plik {size} B, offsety {offsets[-1]}, "
                         f"podpisow {len(captions)}, obrazow {kept}")

    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_offsets.json", "w") as f:
        json.dump(offsets, f)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": kept, "res": a.res, "format": "jpeg",
                   "source": a.source, "shards": [a.first_shard, last]}, f)

    words = sum(len(c.split()) for c in captions) / max(kept, 1)
    print(f"gotowe: {kept} obrazow z {seen}, {size/1e9:.2f} GB, "
          f"srednio {words:.1f} slow na podpis", flush=True)


if __name__ == "__main__":
    main()
