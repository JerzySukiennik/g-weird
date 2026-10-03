"""Paint over a region of a real picture, give a new caption, see what comes back.

The feature this model line is being built for: the user marks the part of an
image that should change, says what it should become, and only that part is
redrawn. The random-token fill test (fill_test.py) cannot answer whether that
works — a hole made of scattered tokens has neighbours everywhere to copy from,
a contiguous region does not.

Three rows per picture: the original through the tokenizer round trip with the
region greyed out (so the ceiling of the whole exercise is visible), the region
refilled under the ORIGINAL caption (a control: does it put back something
plausible and keep the context?), and refilled under the NEW caption (does it
change what it was told to change?).

Output is the raw decode with no pixel paste-back, on purpose: it exposes how much
the context drifts through the token round trip. A product would composite the
original pixels back outside the region.
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
    p.add_argument("--strip", required=True, help="PNG z obrazami obok siebie, jeden rzad")
    p.add_argument("--tile", type=int, default=192)
    p.add_argument("--old", nargs="+", required=True, help="podpisy oryginalow")
    p.add_argument("--new", nargs="+", required=True, help="nowe podpisy")
    p.add_argument("--box", type=int, nargs=4, default=[6, 6, 18, 18],
                   metavar=("R0", "C0", "R1", "C1"),
                   help="prostokat w WSPOLRZEDNYCH TOKENOW, [R0,R1) x [C0,C1)")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/edit.png")
    a = p.parse_args()

    from PIL import Image
    from tokenizers import Tokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = MaskGITConfig(**{k: v for k, v in ck["cfg"].items()
                           if k in MaskGITConfig.__dataclass_fields__})
    model = MaskGIT(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"maskgit z kroku {ck.get('step', '?')}", flush=True)

    vk = torch.load(a.vqvae, map_location="cpu", weights_only=False)
    vq = VQVAE(**vk["arch"]).to(dev).eval()
    vq.load_state_dict(vk["model"], strict=False)
    grid = int(round(math.sqrt(cfg.image_len)))
    res = grid * (2 ** len(vk["arch"]["mults"]))

    strip = np.asarray(Image.open(a.strip).convert("RGB"))
    n = strip.shape[1] // a.tile
    if len(a.old) != n or len(a.new) != n:
        raise SystemExit(f"{n} obrazow w pasku, a podpisow {len(a.old)} / {len(a.new)}")
    ims = []
    for i in range(n):
        t = Image.fromarray(strip[:a.tile, i * a.tile:(i + 1) * a.tile])
        if a.tile != res:
            t = t.resize((res, res), Image.LANCZOS)
        ims.append(np.asarray(t))
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1.0
    with torch.no_grad():
        idx = vq.encode(x)                                   # (n, g, g)
    lo = cfg.image_token(0)

    r0, c0, r1, c1 = a.box
    if not (0 <= r0 < r1 <= grid and 0 <= c0 < c1 <= grid):
        raise SystemExit(f"prostokat {a.box} poza siatka {grid}x{grid}")
    hole = torch.zeros(grid, grid, dtype=torch.bool)
    hole[r0:r1, c0:c1] = True
    print(f"obszar: {int(hole.sum())} z {grid * grid} tokenow "
          f"({100 * int(hole.sum()) / grid ** 2:.0f}%)", flush=True)
    init = (idx.view(n, -1) + lo).clone()
    init[:, hole.view(-1).to(dev)] = cfg.MASK

    tok = Tokenizer.from_file(a.tokenizer)

    def text_rows(caps):
        rows = []
        for c in caps:
            ids = tok.encode(c).ids[: cfg.text_len]
            rows.append([cfg.text_token(t) for t in ids]
                        + [cfg.PAD] * (cfg.text_len - len(ids)))
        return torch.tensor(rows, dtype=torch.long, device=dev)

    def draw(codes):
        with torch.no_grad():
            img = vq.decode(codes.view(-1, grid, grid).to(dev))
        return ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()

    base = draw(idx.view(n, -1))
    shaded = base.copy()
    f = res // grid
    for r in range(r0, r1):
        for c in range(c0, c1):
            shaded[:, r * f:(r + 1) * f, c * f:(c + 1) * f] = \
                (shaded[:, r * f:(r + 1) * f, c * f:(c + 1) * f] * 0.35 + 90 * 0.65)
    shaded = shaded.astype(np.uint8)

    rows = [shaded]
    for label, caps in (("oryginalny podpis", a.old), ("nowy podpis", a.new)):
        torch.manual_seed(a.seed)
        with torch.no_grad():
            codes = generate(model, text_rows(caps), cfg, steps=a.steps,
                             scale=a.cfg_scale, temp=a.temp, init=init)
        # Poza obszarem tokeny musza zostac dokladnie te same — to jest
        # gwarancja edycji, wiec sprawdzamy ja zamiast jej zakladac.
        keep = ~hole.view(-1).to(dev)
        same = bool((codes[:, keep] == idx.view(n, -1)[:, keep]).all())
        print(f"{label}: tokeny poza obszarem nietkniete = {same}", flush=True)
        if not same:
            raise SystemExit("sampler zmienil tokeny poza obszarem")
        rows.append(draw(codes))

    sheet = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    Image.fromarray(sheet).save(a.out)
    print(f"zapisane {a.out} — wiersze: oryginal z zaznaczonym obszarem, "
          f"obszar z oryginalnym podpisem, obszar z nowym podpisem", flush=True)


if __name__ == "__main__":
    main()
