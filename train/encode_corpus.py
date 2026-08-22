"""Encode the image corpus into token ids, once and for all.

This is the step that makes the transformer affordable: an image stops being
196 KB of pixels and becomes 256 uint16 tokens — 512 bytes. The 31.6 GB corpus
becomes about 1 GB, and a million pairs stops being a storage problem.

**The encoder and codebook become a contract here.** The transformer will learn
that token 4711 means one particular vector; retraining or fine-tuning either of
them silently changes what every id means and invalidates the transformer
completely. The decoder is exempt — it only turns ids back into pixels, so it can
be improved later and the transformer keeps working. That asymmetry is the whole
reason it is safe to move on from a tokenizer that is not yet beautiful.

Written to a flat uint16 array plus the captions, in corpus order, so training
can memory-map it.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vqvae import VQVAE  # noqa: E402


def shard_reader(prefix, res):
    meta = json.load(open(f"{prefix}_meta.json"))
    n = meta["n"]
    if meta.get("format") == "jpeg":
        offs = json.load(open(f"{prefix}_offsets.json"))
        fh = open(f"{prefix}_images.jpgbin", "rb")
        from PIL import Image
        import io

        def get(i):
            fh.seek(offs[i])
            return np.asarray(Image.open(io.BytesIO(fh.read(offs[i + 1] - offs[i])))
                              .convert("RGB"))
    else:
        arr = np.memmap(f"{prefix}_images.bin", dtype=np.uint8, mode="r",
                        shape=(n, res, res, 3))

        def get(i):
            return np.array(arr[i])
    caps = json.load(open(f"{prefix}_captions.json"))
    return n, get, caps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-prefix", default="./tokens")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = VQVAE(**ck["arch"]).to(dev).eval()
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    print(f"tokenizer z kroku {ck['step']}, arch {ck['arch']}", flush=True)
    if missing:
        print(f"  brakujace klucze: {list(missing)}", flush=True)

    grid = a.res // (2 ** len(ck["arch"]["mults"]))
    per_image = grid * grid
    n_codes = ck["arch"]["n_codes"]
    assert n_codes <= 65535, "uint16 nie pomiesci tego codebooka"
    print(f"siatka {grid}x{grid} = {per_image} tokenow na obraz", flush=True)

    tok_path = f"{a.out_prefix}_tokens.u16"
    fh = open(tok_path, "wb")
    captions, total = [], 0

    for prefix in a.data:
        n, get, caps = shard_reader(prefix, a.res)
        print(f"{prefix}: {n} obrazow", flush=True)
        for start in range(0, n, a.batch):
            end = min(start + a.batch, n)
            batch = np.stack([get(i) for i in range(start, end)])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(dev)
            x = x / 127.5 - 1.0
            with torch.no_grad():
                idx = model.encode(x)
            fh.write(idx.to(torch.int32).cpu().numpy().astype(np.uint16).tobytes())
            captions.extend(caps[start:end])
            total += end - start
            if total % 20000 < a.batch:
                print(f"  {total} obrazow zakodowanych", flush=True)
    fh.close()

    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": total, "grid": grid, "per_image": per_image,
                   "n_codes": n_codes, "vqvae_step": ck["step"],
                   "arch": ck["arch"]}, f)

    size = os.path.getsize(tok_path)
    print(f"gotowe: {total} obrazow, {size/1e9:.2f} GB "
          f"({size/max(total,1):.0f} B na obraz), podpisow {len(captions)}", flush=True)


if __name__ == "__main__":
    main()
