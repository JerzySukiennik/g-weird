"""Build a G-Weird shard from DiffusionDB instead of Conceptual Captions.

Two things this corpus fixes, both measured rather than assumed:

**No watermarks.** The old corpus was largely stock photography, and the
Shutterstock and Alamy marks were visible in every training batch. A
discriminator's job is to match the training distribution, so decoder
fine-tuning learned to render those marks crisply — twice — and the marks are
baked into the code sequences the transformer learned, which is why cropping the
decoder's data could not remove them from generations. DiffusionDB is Stable
Diffusion output: no stock marks anywhere.

**No dead links.** Conceptual Captions ships URLs, 38% of which were already
gone, so a shard spent most of its time waiting on HTTP for images it would
never get. Here the pixels are in the dataset.

The trade, stated plainly: these are not photographs. They are SD 1.x outputs,
tilted hard toward fantasy, portraits and artstation. That buys sharpness and
consistency, not photographic realism — if photorealism were the goal the right
corpus would be unwatermarked photographs, not this.

Streaming by design: one part is downloaded, unpacked, re-encoded and deleted
before the next is fetched, so peak disk is one archive rather than hundreds of
gigabytes. Each ~500 MB part of 1000 PNGs becomes ~25 MB of 256px JPEG.
"""

import argparse
import io
import json
import os
import zipfile

from PIL import Image


def square_jpeg(data, res, quality=88):
    """Centre-crop to square, resize, hand back JPEG bytes.

    Square crop rather than letterbox for the same reason as the old fetcher:
    grey bars would teach the model that grey bars are part of what a picture
    looks like.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    s = min(img.size)
    l, t = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((res, res), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first-part", type=int, default=1)
    p.add_argument("--parts", type=int, default=200, help="ile archiwow przerobic")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--out-prefix", default="./gweird")
    p.add_argument("--min-words", type=int, default=2,
                   help="prompty krotsze niz to sa odrzucane")
    p.add_argument("--max-chars", type=int, default=300)
    a = p.parse_args()

    from huggingface_hub import hf_hub_download

    images_path = f"{a.out_prefix}_images.jpgbin"
    fh = open(images_path, "wb")
    captions, offsets, kept, seen = [], [0], 0, 0

    for n in range(a.first_part, a.first_part + a.parts):
        name = f"images/part-{n:06d}.zip"
        try:
            path = hf_hub_download("poloclub/diffusiondb", name,
                                   repo_type="dataset")
        except Exception as e:
            print(f"  {name}: pominiete ({type(e).__name__})", flush=True)
            continue

        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                meta_name = next((f for f in names if f.endswith(".json")), None)
                meta = json.loads(z.read(meta_name)) if meta_name else {}
                for f in names:
                    if not f.lower().endswith(".png"):
                        continue
                    seen += 1
                    # The official loader keys this JSON by bare image name.
                    # Falling back to the basename costs nothing and covers an
                    # archive that stores its files under a directory.
                    entry = meta.get(f) or meta.get(os.path.basename(f)) or {}
                    prompt = (entry.get("p") or "").strip()
                    # A picture with no prompt teaches the text side nothing, and
                    # a one-word prompt teaches it almost nothing.
                    if len(prompt.split()) < a.min_words:
                        continue
                    if len(prompt) > a.max_chars:
                        prompt = prompt[:a.max_chars]
                    try:
                        blob = square_jpeg(z.read(f), a.res)
                    except Exception:
                        continue
                    fh.write(blob)
                    offsets.append(offsets[-1] + len(blob))
                    captions.append(prompt)
                    kept += 1
        finally:
            # The cache would otherwise grow by 500 MB a part and fill the disk
            # long before the run finished.
            try:
                os.remove(path)
            except OSError:
                pass

        if (n - a.first_part + 1) % 10 == 0:
            fh.flush()
            print(f"  {n - a.first_part + 1} archiwow, {kept} par, "
                  f"{os.path.getsize(images_path)/1e9:.2f} GB", flush=True)

    fh.close()
    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_offsets.json", "w") as f:
        json.dump(offsets, f)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": kept, "res": a.res, "format": "jpeg",
                   "source": "diffusiondb", "first_part": a.first_part,
                   "parts": a.parts, "seen": seen}, f)

    size = os.path.getsize(images_path)
    print(f"gotowe: {kept} par z {seen} obrazow, {size/1e9:.2f} GB, "
          f"{size/max(kept,1)/1024:.1f} kB na obraz", flush=True)


if __name__ == "__main__":
    main()
