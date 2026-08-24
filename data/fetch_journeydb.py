"""Midjourney images with clean captions — the coherence teacher.

DiffusionDB is Stable Diffusion 1.x output, and SD 1.x produces exactly the
failure this version exists to remove: limbs that merge, animals that are
assemblages. Learning from it alone means learning from a teacher with the same
problem. Midjourney images hold together far better, so this corpus raises the
ceiling on "a horse is a horse" rather than only on sharpness.

Two things make it awkward and both are handled here rather than paid for.

**The archives are 15.3 GB each**, 200 of them, about 3 TB in total. Taking whole
archives would spend the entire download budget on a handful of them. A .tgz can
be read as a stream, so each archive is opened, a slice of images is taken, and
the connection is dropped — the same transfer buys images from many archives
instead of all of one, which is better for diversity as well as cheaper.

**The captions are better than the prompts.** Each record carries the raw
Midjourney prompt ("cinematic, realistic, 1920s girls at college, ...") and, under
Task2, a written sentence: "A group of 1920s girls at college immersed in their
studies at a dark academia university." The sentence is what teaches a subject;
the comma salad mostly teaches style, and DiffusionDB already supplies plenty of
that. So the caption is preferred and the prompt is the fallback.
"""

import argparse
import io
import json
import os
import tarfile

import requests
from PIL import Image

REPO = "https://huggingface.co/datasets/JourneyDB/JourneyDB/resolve/main"
ANNO = "data/train/train_anno_realease_repath.jsonl.tgz"


def square_jpeg(data, res, quality=88):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    s = min(img.size)
    l, t = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((res, res), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def download(url, path, token, tries=8):
    """Download that survives a broken connection by resuming from what it has.

    The first attempt at the 859 MB annotation file died at 593 MB with
    IncompleteRead and took the whole kernel with it — the same failure already
    fixed once today for the 8 GB DiffusionDB shards, written fresh here without
    the protection. A Range request continues from the bytes already on disk
    instead of starting over.
    """
    for attempt in range(1, tries + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        headers = {"Authorization": f"Bearer {token}"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code == 200:
                    # Server ignored the range; start clean rather than append
                    # a second copy of the beginning onto the first.
                    have = 0
                r.raise_for_status()
                with open(path, "ab" if have else "wb") as out:
                    for chunk in r.iter_content(1 << 22):
                        out.write(chunk)
            return path
        except Exception as e:
            print(f"  proba {attempt}/{tries} przerwana ({type(e).__name__}), "
                  f"mam {os.path.getsize(path)/1e6:.0f} MB, wznawiam", flush=True)
    raise SystemExit(f"nie udalo sie pobrac {url}")


def load_captions(token, tmp):
    """img_path -> caption, read once from the annotation archive."""
    path = download(f"{REPO}/{ANNO}", os.path.join(tmp, "anno.tgz"), token)
    print(f"  podpisy: {os.path.getsize(path)/1e6:.0f} MB pobrane", flush=True)

    caps = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = ((rec.get("Task2") or {}).get("Caption")
                        or rec.get("prompt") or "").strip()
                if text:
                    # "./000/uuid.jpg" -> "000/uuid.jpg"
                    caps[rec["img_path"].lstrip("./")] = text
    os.remove(path)
    print(f"  {len(caps):,} podpisow wczytanych", flush=True)
    return caps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=0, help="pierwsze archiwum")
    p.add_argument("--archives", type=int, default=20)
    p.add_argument("--per-archive", type=int, default=7500,
                   help="ile obrazow wziac z kazdego, zanim przejdziemy dalej")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument("--out-prefix", default="./gweird")
    p.add_argument("--tmp", default="/kaggle/tmp")
    a = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("brak HF_TOKEN — JourneyDB jest bramkowane")
    os.makedirs(a.tmp, exist_ok=True)
    caps = load_captions(token, a.tmp)

    fh = open(f"{a.out_prefix}_images.jpgbin", "wb")
    captions, offsets, kept, seen = [], [0], 0, 0
    headers = {"Authorization": f"Bearer {token}"}

    for n in range(a.first, a.first + a.archives):
        name = f"data/train/imgs/{n:03d}.tgz"
        taken = 0
        try:
            with requests.get(f"{REPO}/{name}", headers=headers, stream=True,
                              timeout=120) as r:
                r.raise_for_status()
                r.raw.decode_content = False
                # Streaming mode: entries arrive in order and the connection is
                # dropped once the slice is full, so a 15.3 GB archive costs
                # only the few GB actually read.
                with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                    for member in tf:
                        if taken >= a.per_archive:
                            break
                        if not member.isfile() or not member.name.lower().endswith(".jpg"):
                            continue
                        seen += 1
                        key = member.name.lstrip("./")
                        text = caps.get(key)
                        if not text:
                            continue
                        blob_in = tf.extractfile(member)
                        if blob_in is None:
                            continue
                        try:
                            blob = square_jpeg(blob_in.read(), a.res)
                        except Exception:
                            continue
                        fh.write(blob)
                        offsets.append(offsets[-1] + len(blob))
                        captions.append(text[:a.max_chars])
                        kept += 1
                        taken += 1
        except Exception as e:
            print(f"  archiwum {n:03d}: przerwane ({type(e).__name__}: {e})", flush=True)
        fh.flush()
        print(f"  archiwum {n:03d}: +{taken}, razem {kept} par, "
              f"{os.path.getsize(f'{a.out_prefix}_images.jpgbin')/1e9:.2f} GB", flush=True)

    fh.close()
    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_offsets.json", "w") as f:
        json.dump(offsets, f)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": kept, "res": a.res, "format": "jpeg",
                   "source": "journeydb", "first": a.first,
                   "archives": a.archives, "seen": seen}, f)
    size = os.path.getsize(f"{a.out_prefix}_images.jpgbin")
    print(f"gotowe: {kept} par, {size/1e9:.2f} GB, {size/max(kept,1)/1024:.1f} kB na obraz",
          flush=True)


if __name__ == "__main__":
    main()
