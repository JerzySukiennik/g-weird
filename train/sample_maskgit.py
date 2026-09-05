"""Draw images from a trained MaskGIT: caption in, 576 tokens out in 12 rounds.

The MaskGIT twin of train/sample.py. Same fixed prompts, same sheet layout, so
the two architectures can be judged side by side on identical captions — the
only comparison that has ever settled anything in this project.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.maskgit import MaskGITConfig, MaskGIT, generate   # noqa: E402
from model.vqvae import VQVAE                                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vqvae", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--out", default="out/sample-maskgit.png")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--choice-temp", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("cfg")
    if saved is None:
        raise SystemExit("checkpoint bez konfiguracji — nie zgaduje architektury")
    cfg = MaskGITConfig(**{k: v for k, v in saved.items()
                           if k in MaskGITConfig.__dataclass_fields__})
    model = MaskGIT(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"maskgit z kroku {ck.get('step', '?')}", flush=True)

    vk = torch.load(a.vqvae, map_location="cpu", weights_only=False)
    vq = VQVAE(**vk["arch"]).to(dev).eval()
    vq.load_state_dict(vk["model"], strict=False)

    rows = []
    for pr in a.prompts:
        ids = tok.encode(pr).ids[: cfg.text_len]
        rows.append([cfg.text_token(i) for i in ids]
                    + [cfg.PAD] * (cfg.text_len - len(ids)))
    text_rows = torch.tensor(rows, dtype=torch.long, device=dev)

    with torch.no_grad():
        codes = generate(model, text_rows, cfg, steps=a.steps, scale=a.cfg_scale,
                         temp=a.temp, choice_temp=a.choice_temp)
        grid = int(round(math.sqrt(cfg.image_len)))
        imgs = vq.decode(codes.view(-1, grid, grid))
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()

    from PIL import Image
    cols = min(len(a.prompts), 4)
    rws = math.ceil(len(a.prompts) / cols)
    h, w = imgs.shape[1:3]
    sheet = np.zeros((rws * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(sheet).save(a.out)
    print(f"zapisane {a.out}", flush=True)


if __name__ == "__main__":
    main()
