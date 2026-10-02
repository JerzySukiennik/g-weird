"""Hand the model a real image with part of it hidden and see what it puts back.

This is the task the editing feature actually performs — paint over a region,
the model refills it — so it is the right thing to judge a masked model on.
It is also the only test that separates "has not learned the task" from "cannot
start from an empty canvas", which cost this project three wrong diagnoses.

Exact-token accuracy is reported but is close to worthless as a quality number:
with 8192 codes many different tokens decode to nearly the same patch, and a
sheet indistinguishable from the original once scored 2.3%. It is here to catch
a model that is outputting constants, not to rank two good ones. Look at the
picture.
"""

import argparse
import io
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.maskgit import MaskGITConfig, MaskGIT   # noqa: E402
from model.vqvae import VQVAE                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vqvae", required=True)
    p.add_argument("--data", required=True, help="prefiks sharda z obrazami JPEG")
    p.add_argument("--out", default="out/fill.png")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = MaskGITConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in MaskGITConfig.__dataclass_fields__})
    model = MaskGIT(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])

    vk = torch.load(a.vqvae, map_location="cpu", weights_only=False)
    vq = VQVAE(**vk["arch"]).to(dev).eval()
    vq.load_state_dict(vk["model"], strict=False)

    offs = json.load(open(f"{a.data}_offsets.json"))
    caps = json.load(open(f"{a.data}_captions.json"))
    fh = open(f"{a.data}_images.jpgbin", "rb")
    ims = []
    for i in range(a.n):
        fh.seek(offs[i])
        ims.append(np.asarray(Image.open(io.BytesIO(
            fh.read(offs[i + 1] - offs[i]))).convert("RGB")))
    res = int(round(math.sqrt(cfg.image_len))) * (2 ** len(vk["arch"]["mults"]))
    if ims[0].shape[0] != res:
        ims = [np.asarray(Image.fromarray(im).resize((res, res), Image.LANCZOS))
               for im in ims]
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev)
    x = x / 127.5 - 1.0
    with torch.no_grad():
        idx = vq.encode(x)
    grid = idx.shape[-1]

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(os.path.dirname(a.vqvae), "text.json")) \
        if os.path.exists(os.path.join(os.path.dirname(a.vqvae), "text.json")) else None
    rows = []
    for i in range(a.n):
        ids = tok.encode(caps[i]).ids[: cfg.text_len] if tok else []
        rows.append([cfg.text_token(t) for t in ids]
                    + [cfg.PAD] * (cfg.text_len - len(ids)))
    text_rows = torch.tensor(rows, dtype=torch.long, device=dev)

    lo, hi = cfg.image_token(0), cfg.image_token(cfg.n_image - 1)
    flat = idx.view(a.n, -1) + lo
    with torch.no_grad():
        sheets = [vq.decode(idx)]
    g = torch.Generator().manual_seed(a.seed)
    for frac in a.fracs:
        img = flat.clone()
        k = int(frac * cfg.image_len)
        for r in range(a.n):
            cut = torch.randperm(cfg.image_len, generator=g)[:k]
            img[r, cut.to(dev)] = cfg.MASK
        with torch.no_grad():
            logits = model(torch.cat([text_rows, img], dim=1))[:, cfg.text_len:]
            logits[..., :lo] = -float("inf")
            logits[..., hi + 1:] = -float("inf")
            filled = torch.where(img == cfg.MASK, logits.argmax(-1), img)
            sheets.append(vq.decode((filled - lo).view(-1, grid, grid)))
        m = img == cfg.MASK
        acc = (filled[m] == flat[m]).float().mean().item()
        print(f"zaslonione {int(frac * 100)}%: trafnosc tokenow {acc * 100:.1f}%",
              flush=True)

    arr = [((s.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
           for s in sheets]
    h, w = arr[0].shape[1:3]
    out = np.zeros((len(arr) * h, a.n * w, 3), dtype=np.uint8)
    for r, blk in enumerate(arr):
        for c, im in enumerate(blk):
            out[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(out).save(a.out)
    print(f"zapisane {a.out} — wiersze: oryginal, " +
          ", ".join(f"{int(f * 100)}%" for f in a.fracs), flush=True)


if __name__ == "__main__":
    main()
