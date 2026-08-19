"""Build a caption corpus for G-Weird: 256px images plus the text that described them.

Conceptual Captions ships URLs rather than pixels, so this is a few hundred
thousand HTTP fetches per shard. That is the slow part, not the resizing, which
is why downloads run on a thread pool and the packing loop simply consumes
whatever arrives. Dead links are normal — roughly a fifth of the web rots
between a dataset's publication and today — so the target is a pair COUNT, not
a row range, and the shard keeps pulling until it has enough.

Why the whole thing is worth the trouble: prompt obedience at this scale is
bought with pairs, not parameters. DALL-E mini saw 15M+ of them at 400M
parameters. We will not reach that, but the difference between 300k and 3M is
the difference between "sushi z zieleniny" meaning something and meaning
nothing.

Images are stored as JPEG bytes, not raw uint8. Raw was the obvious first choice
and the wrong one: it costs 196 KB an image, which caps a shard at 90k pairs
against Kaggle's 20 GB output limit, so a million pairs would have wanted eleven
shards and ~195 GB. JPEG at quality 88 costs ~25 KB, which is roughly 700k pairs
in the same shard. The pictures arrive as JPEG in the first place, so this is
mostly a matter of not throwing the compression away.

The artefacts do not matter here — the model exists to make ugly pictures, and
the 2021 VQGANs whose look it is chasing were trained on web JPEGs too.

Even this is an intermediate: once the VQ-VAE exists, each image becomes 256
tokens — 512 bytes against 196 KB raw — and the pixels can be dropped.
"""

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 (compatible; g-weird-prep/1.0)"}


def fetch_one(url, res, quality=88, timeout=10):
    """Download, square-crop, resize, and hand back JPEG bytes."""
    try:
        r = requests.get(url, headers=UA, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        # Square centre crop, then resize: letterboxing would teach the model
        # that grey bars are part of what a picture looks like.
        s = min(img.size)
        l, t = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((l, t, l + s, t + s)).resize((res, res), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--want", type=int, default=200000, help="pairs to collect")
    p.add_argument("--skip", type=int, default=0, help="rows to skip before starting")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--out-prefix", default="./gweird")
    a = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/conceptual_captions",
                      split="train", streaming=True)
    if a.skip:
        ds = ds.skip(a.skip)

    # Variable-length records now, so the reader needs an offset table; raw
    # uint8 could be indexed by arithmetic alone.
    images_path = f"{a.out_prefix}_images.jpgbin"
    fh = open(images_path, "wb")
    captions, offsets, kept, seen = [], [0], 0, 0

    def rows():
        for row in ds:
            yield row.get("image_url"), row.get("caption")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending = {}
        for url, cap in rows():
            if kept >= a.want:
                break
            if not url or not cap:
                continue
            pending[pool.submit(fetch_one, url, a.res)] = cap
            seen += 1
            if len(pending) < a.workers * 4:
                continue
            done = [f for f in pending if f.done()]
            for f in done:
                cap_of = pending.pop(f)
                arr = f.result()
                if arr is None:
                    continue
                fh.write(arr)
                offsets.append(offsets[-1] + len(arr))
                captions.append(cap_of)
                kept += 1
                if kept % 5000 == 0:
                    print(f"{kept} par ({seen} sprawdzonych, "
                          f"{100*kept/max(seen,1):.0f}% zywych)", flush=True)
        for f, cap_of in pending.items():
            arr = f.result()
            if arr is not None and kept < a.want:
                fh.write(arr)
                offsets.append(offsets[-1] + len(arr))
                captions.append(cap_of)
                kept += 1
    fh.close()

    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_offsets.json", "w") as f:
        json.dump(offsets, f)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": kept, "res": a.res, "skipped": a.skip, "seen": seen,
                   "format": "jpeg"}, f)
    size = os.path.getsize(images_path)
    print(f"gotowe: {kept} par, {size/1e9:.2f} GB, "
          f"{size/max(kept,1)/1024:.1f} kB na obraz, "
          f"{100*kept/max(seen,1):.0f}% linkow zylo", flush=True)


if __name__ == "__main__":
    main()
