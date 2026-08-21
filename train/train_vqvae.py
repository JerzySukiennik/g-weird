"""Train the VQ-VAE that decides what G-Weird's wrongness looks like.

Reconstruction loss alone will not tell you whether this worked. A codebook can
collapse to a few dozen live entries and still post a falling loss, because the
encoder simply learns to route everything through the survivors — the pictures
come back as uniform mush rather than interesting wrongness, and the curve looks
healthy the whole way. So the number to watch is `codes`: distinct entries used
in the batch, out of 8192. It should climb into the thousands. If it sits in the
dozens after a few thousand steps, the run is already dead and the hours after
that are wasted.

L1 rather than MSE for the reconstruction: MSE averages its way to blur, and blur
is the one failure mode this project cannot use. No VGG perceptual loss, for the
same reason as everywhere else in the family — it would import pretrained
ImageNet weights.

Sessions end voluntarily at --max-steps. Two G-Images runs died by being killed
mid-write, once losing 8h11m, so the checkpoint goes to a temp file and renames.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vqvae import VQVAE  # noqa: E402
from model.discriminator import (PatchDiscriminator, hinge_d_loss,  # noqa: E402
                                 g_adv_loss, adaptive_weight)


class Images(Dataset):
    """Frames from the prep shards, in either storage format.

    Shards 0 and 1 hold raw uint8 and are memory-mapped: an image is a fixed
    stride into the file. Everything from shard 2 holds JPEG, which is seven or
    eight times smaller but variable length, so those carry an offset table and
    are decoded per item. Both are concatenated logically — an item is
    (shard, row) and the training loop never learns which file it came from.
    """

    def __init__(self, prefixes, res=256):
        self.shards, self.index = [], []
        for p in prefixes:
            meta = json.load(open(f"{p}_meta.json"))
            n, r = meta["n"], meta["res"]
            if r != res:
                raise SystemExit(f"{p}: {r}px, oczekiwano {res}px")
            if meta.get("format") == "jpeg":
                offs = json.load(open(f"{p}_offsets.json"))
                self.shards.append(("jpeg", f"{p}_images.jpgbin", offs, None))
            else:
                arr = np.memmap(f"{p}_images.bin", dtype=np.uint8, mode="r",
                                shape=(n, res, res, 3))
                self.shards.append(("raw", None, None, arr))
            self.index += [(len(self.shards) - 1, i) for i in range(n)]
        self.res = res
        self._fh = {}
        kinds = ", ".join(sorted({k for k, _, _, _ in self.shards}))
        print(f"{len(self.shards)} shardow ({kinds}), {len(self.index)} obrazow",
              flush=True)

    def __len__(self):
        return len(self.index)

    def _pixels(self, s, r):
        kind, path, offs, arr = self.shards[s]
        if kind == "raw":
            return np.array(arr[r])
        # One handle per worker process, opened lazily: a memmap survives fork,
        # an open file position does not.
        fh = self._fh.get(s)
        if fh is None:
            fh = self._fh[s] = open(path, "rb")
        fh.seek(offs[r])
        from PIL import Image as _Image
        import io as _io
        return np.asarray(_Image.open(_io.BytesIO(fh.read(offs[r + 1] - offs[r])))
                          .convert("RGB"))

    def __getitem__(self, i):
        s, r = self.index[i]
        x = torch.from_numpy(self._pixels(s, r).copy()).permute(2, 0, 1).float()
        x = x / 127.5 - 1.0
        if torch.rand(()) < 0.5:                     # h-flip; free data, no downside
            x = torch.flip(x, dims=[2])
        return x


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default="run")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--max-steps", type=int, default=0, help="stop early, keep ckpt")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--commit", type=float, default=0.25)
    p.add_argument("--disc-start", type=int, default=0,
                   help="krok, od ktorego dziala dyskryminator; 0 = od razu")
    p.add_argument("--disc-lr", type=float, default=2e-4)
    p.add_argument("--restart-below", type=float, default=0.1,
                   help="prog przesiewania; 1.0 wybijalo kody uzywane rzadziej niz srednia")
    p.add_argument("--n-codes", type=int, default=8192)
    p.add_argument("--workers", type=int, default=2,
                   help="0 na macOS: forkowane workery potrafia zawisnac")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--res", type=int, default=256,
                   help="obrazy w shardach; mniejsze tylko do testow")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Images(a.data, res=a.res)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=(dev == "cuda"),
                    persistent_workers=(a.workers > 0))

    model = VQVAE(n_codes=a.n_codes).to(dev)
    model.quant.restart_below = a.restart_below
    print(f"VQ-VAE: {sum(q.numel() for q in model.parameters())/1e6:.1f}M parametrow, "
          f"codebook {a.n_codes}", flush=True)
    raw = model
    # Deliberately single-GPU. DataParallel broadcasts module buffers to each
    # replica as views, so the quantizer's in-place EMA update raises — and had
    # it not raised, the update would have happened on the replica and been
    # thrown away at the end of the step, leaving the codebook frozen at its
    # initialisation with nothing in the loss to show for it. A silent
    # correctness bug is worse than a crash, and this model's whole point is a
    # codebook that moves.
    if dev == "cuda" and torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPU widoczne, uzywam jednej: "
              f"EMA codebooka nie przezywa DataParallel", flush=True)

    opt = torch.optim.AdamW(raw.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    disc = PatchDiscriminator().to(dev)
    # betas (0.5, 0.9) rather than AdamW's usual: the standard GAN setting,
    # because a discriminator that remembers too much momentum overshoots and
    # drags the generator into oscillation.
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.disc_lr, betas=(0.5, 0.9))
    print(f"dyskryminator: {sum(q.numel() for q in disc.parameters())/1e6:.1f}M, "
          f"start od kroku {a.disc_start}", flush=True)

    step = 0
    ckpt_path = os.path.join(a.out, "vqvae.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Tolerant, because the quantizer gained a step counter after the first
        # session and a strict load would refuse a checkpoint that is otherwise
        # perfectly good. Missing keys are reported rather than swallowed.
        missing, unexpected = raw.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"wczytane z roznicami — brakujace: {list(missing)}, "
                  f"nieoczekiwane: {list(unexpected)}", flush=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            print("brak stanu optymalizatora — AdamW startuje w spoczynku", flush=True)
        step = ck["step"]
        if "disc" in ck:
            disc.load_state_dict(ck["disc"])
            opt_d.load_state_dict(ck["opt_d"])
        else:
            print("checkpoint sprzed dyskryminatora — startuje od zera", flush=True)
        print(f"wznowione z kroku {step}", flush=True)

    ceiling = min(a.steps, step + a.max_steps) if a.max_steps else a.steps
    print(f"sesja: krok {step} -> {ceiling} (cel {a.steps})", flush=True)

    t0 = time.time()
    model.train()
    it = iter(dl)
    while step < ceiling:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl); x = next(it)
        x = x.to(dev, non_blocking=True)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)

        # Generator. The adversarial term runs in fp32: the adaptive weight is a
        # ratio of gradient norms, and fp16 loses the small one to underflow.
        with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            out, commit, idx = model(x)
        out = out.float()
        rec = F.l1_loss(out, x)
        adv = torch.zeros((), device=dev)
        w = torch.zeros((), device=dev)
        if step >= a.disc_start:
            adv = g_adv_loss(disc(out))
            last = raw.decoder.out.weight
            w = adaptive_weight(rec, adv, last)
        loss = rec + a.commit * commit.float().mean() + w * adv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()

        # Discriminator, on the detached reconstruction so its gradient never
        # reaches the generator.
        d_loss = torch.zeros((), device=dev)
        if step >= a.disc_start:
            d_loss = hinge_d_loss(disc(x), disc(out.detach()))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        step += 1

        if step % a.log_every == 0:
            live = int(idx.unique().numel())
            print(f"step {step}/{ceiling}  rec {rec.item():.4f}  "
                  f"commit {commit.float().mean().item():.4f}  "
                  f"adv {adv.item():.3f}  d {d_loss.item():.3f}  w {float(w):.2f}  "
                  f"codes {live}/{a.n_codes}  {time.time()-t0:.0f}s", flush=True)

        if step % a.ckpt_every == 0 or step == ceiling:
            blob = {"model": raw.state_dict(), "opt": opt.state_dict(),
                    "disc": disc.state_dict(), "opt_d": opt_d.state_dict(),
                    "step": step, "arch": raw.arch}
            tmp = ckpt_path + ".tmp"
            torch.save(blob, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, ckpt_path)          # atomic; a kill mid-write costs one interval

    print(f"done — {step} krokow w {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
