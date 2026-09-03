"""Draw images from a trained G-Weird: caption in, 256 tokens out, VQ-VAE decodes.

Generation is plain autoregression over the same sequence training saw —
[32 text][BOS_IMG][256 image] — continued past BOS_IMG. The two things that are
not plain:

**Classifier-free guidance.** Each prompt is run twice, once with its caption and
once with the caption blanked to PAD, and the logits are extrapolated away from
the unconditional answer:

    logits = uncond + scale * (cond - uncond)

This works for an autoregressive model exactly as it does for a diffusion one —
the mechanism is the conditioning dropout train_ar.py applies from step zero, not
anything specific to denoising. Both branches ride in one batch, so the cost is a
doubled batch rather than a second pass.

**A vocabulary mask.** The model shares one 16388-entry vocabulary across text,
image and special tokens, so nothing in the architecture stops it from emitting a
word in the middle of a picture. Early in training it does. Sampling only from
the image range costs one masked_fill and removes a whole class of garbage
output that would otherwise be misread as the model being worse than it is.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import WeirdGPT, WeirdConfig      # noqa: E402
from model.vqvae import VQVAE                            # noqa: E402
from model.gpt_base import KVCache                       # noqa: E402


def build_prefix(prompts, tok, cfg, device):
    """[text tokens padded to text_len][BOS_IMG], conditional then unconditional."""
    rows = []
    for p in prompts:
        ids = tok.encode(p).ids[:cfg.text_len]
        row = [cfg.text_token(i) for i in ids] + [cfg.PAD] * (cfg.text_len - len(ids))
        rows.append(row + [cfg.BOS_IMG])
    uncond = [[cfg.PAD] * cfg.text_len + [cfg.BOS_IMG] for _ in prompts]
    return torch.tensor(rows + uncond, dtype=torch.long, device=device)


@torch.no_grad()
def generate(model, prefix, cfg, scale, temp, top_k):
    n = prefix.size(0) // 2
    lo = cfg.image_token(0)
    hi = cfg.image_token(cfg.n_image - 1)

    kv = KVCache(cfg.n_layer)
    logits = model(prefix, kv=kv, pos=0)[:, -1]
    out = []
    for step in range(cfg.image_len):
        # Prowadzenie liczone w fp32, nawet gdy model chodzi w fp16.
        # Ekstrapolacja mnozy roznice logitow przez skale, wiec przy scale 4
        # i logitach rzedu 10 wynik siega setek — a fp16 konczy sie na 65504.
        # Po przekroczeniu zakresu softmax dostaje inf i zwraca degeneracje:
        # ten sam checkpoint MaskGIT-a dawal plaskie plamy koloru na GPU w
        # autocascie i normalna teksture na CPU w fp32. Ten sam wzor byl tutaj.
        cond, uncond = logits[:n].float(), logits[n:].float()
        guided = uncond + scale * (cond - uncond)

        guided[:, :lo] = -math.inf          # never emit text or special tokens
        guided[:, hi + 1:] = -math.inf
        guided = guided / max(temp, 1e-5)
        if top_k:
            kth = guided.topk(min(top_k, hi - lo + 1), dim=-1).values[:, -1:]
            guided = guided.masked_fill(guided < kth, -math.inf)

        nxt = torch.multinomial(F.softmax(guided, dim=-1), 1)   # (n, 1)
        out.append(nxt)
        if step == cfg.image_len - 1:
            break
        # Both branches continue with the SAME sampled token. Letting them
        # diverge would make the unconditional branch describe a different image
        # than the one being drawn, and the guidance term would be nonsense.
        step_in = torch.cat([nxt, nxt], dim=0)
        logits = model(step_in, kv=kv, pos=cfg.text_len + 1 + step)[:, -1]

    return torch.cat(out, dim=1) - lo        # (n, 256) codebook ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vqvae", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--out", default="out/sample.png")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)

    ck = torch.load(a.ckpt, map_location="cpu")
    # train_ar.py stores the config under "cfg" (as cfg.__dict__). Falling back
    # to a default WeirdConfig would load a checkpoint whose shapes happen to
    # match into a model with different settings — silently, if the defaults
    # ever drift from what was trained.
    saved = ck.get("cfg") or ck.get("arch")
    if saved is None:
        raise SystemExit("checkpoint bez konfiguracji — nie zgaduje architektury")
    cfg = WeirdConfig(**{k: v for k, v in saved.items()
                         if k in WeirdConfig.__dataclass_fields__})
    model = WeirdGPT(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"transformer z kroku {ck.get('step', '?')}", flush=True)

    vk = torch.load(a.vqvae, map_location="cpu")
    vq = VQVAE(**vk["arch"]).to(dev).eval()
    vq.load_state_dict(vk["model"])

    prefix = build_prefix(a.prompts, tok, cfg, dev)
    codes = generate(model, prefix, cfg, a.cfg_scale, a.temp, a.top_k)

    grid = int(round(math.sqrt(cfg.image_len)))
    imgs = vq.decode(codes.view(-1, grid, grid))
    imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()

    from PIL import Image
    cols = min(len(a.prompts), 4)
    rows = math.ceil(len(a.prompts) / cols)
    h, w = imgs.shape[1:3]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(sheet).save(a.out)
    print(f"zapisano {a.out} — {len(a.prompts)} obrazow, CFG {a.cfg_scale}", flush=True)
    for i, pr in enumerate(a.prompts):
        print(f"  {i}: {pr}", flush=True)


if __name__ == "__main__":
    main()
