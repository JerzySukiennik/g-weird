"""Sharpen the decoder without touching the codes the transformer already learned.

The images out of G-Weird look like oil paint. That is not the transformer: real
photographs pushed through this VQ-VAE come back equally soft (mean error
15.7/255, measured 2026-08-23). The decoder is the ceiling, so the decoder is
what this trains.

**Why not simply raise the adversarial weight.** That was tried during VQ-VAE
training and it is recorded in train_vqvae.py: uncapped, the adaptive weight
reached 0.61, sharpness rose from 0.050 to 0.113 — and everything acquired a
green cast while people disappeared from the pictures. The cause is named there
too: with no perceptual loss, L1 is the ONLY term holding content together, and
this project will not import a pretrained VGG to supply one.

So two things change here, and together they make this a different experiment
rather than a rerun of the one that failed:

**The encoder and codebook are frozen.** This is required anyway — the
transformer predicts code ids, and re-training the codebook would silently
invalidate 11 h of transformer training. But it also removes the freedom that
let the earlier run fail: with codes fixed, the latent still says "a person",
and the decoder can only change how that is drawn, not whether it survives.

**A perceptual loss built from our own encoder.** The output is pushed back
through the frozen encoder and compared with the target's latent:

    perc = | E(decode(q)) - E(x) |

This is cycle-consistency: whatever the decoder draws must re-encode to the same
place. It constrains content in a space that ignores exact pixel values, which
is precisely what L1 cannot do — and L1's pixel averaging is what produces the
soft look in the first place. No foreign weights: E is our own encoder, trained
from scratch, and frozen.

With content held by something other than L1, the adversarial cap can be
loosened, which is the whole point.

A small colour term guards the specific failure that was observed — per-channel
means of output and target are pulled together, so a global cast cannot creep in
unnoticed.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vqvae import VQVAE                                    # noqa: E402
from model.discriminator import (PatchDiscriminator, hinge_d_loss,  # noqa: E402
                                 g_adv_loss, adaptive_weight)
from train.train_vqvae import Images, lr_at                      # noqa: E402


def sharpness(img):
    """Mean absolute neighbour difference — high for crisp edges, low for mush.

    Reported next to the same measure on the real batch every log line, because
    the absolute number means nothing on its own: what matters is how close the
    reconstruction gets to the photographs it is copying.
    """
    dx = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    dy = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    return ((dx + dy) / 2).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--vqvae", required=True, help="zamrozony enkoder + codebook")
    p.add_argument("--out", default="run")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--disc-lr", type=float, default=5e-5)
    p.add_argument("--disc-every", type=int, default=2)
    p.add_argument("--adv-max", type=float, default=0.5,
                   help="sufit wagi adwersarialnej; 0.1 dawalo efekt olejny, "
                        "bez sufitu (0.61) znikala tresc — ale wtedy nie bylo "
                        "straty percepcyjnej ani zamrozonych kodow")
    p.add_argument("--perc", type=float, default=1.0, help="waga straty percepcyjnej")
    p.add_argument("--colour", type=float, default=0.5, help="waga kotwicy kolorow")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--sample-every", type=int, default=1000,
                   help="zrzut PNG oryginal/rekonstrukcja — liczby dzis kilka razy "
                        "wygladaly dobrze, gdy tresc sie psula")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    ds = Images(a.data, res=a.res)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)

    ck = torch.load(a.vqvae, map_location="cpu", weights_only=False)
    model = VQVAE(**ck["arch"]).to(dev)
    model.load_state_dict(ck["model"])
    base_step = ck.get("step", 0)
    print(f"vqvae z kroku {base_step}, arch {ck['arch']}", flush=True)

    # Frozen: everything the transformer depends on. Only the decoder learns.
    for q in model.encoder.parameters():
        q.requires_grad_(False)
    for q in model.quant.parameters():
        q.requires_grad_(False)
    model.encoder.eval()
    model.quant.eval()
    trainable = [q for q in model.decoder.parameters()]
    print(f"dekoder uczy sie: {sum(q.numel() for q in trainable)/1e6:.1f}M; "
          f"enkoder i codebook zamrozone", flush=True)

    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.5, 0.9), weight_decay=0.0)
    disc = PatchDiscriminator().to(dev)
    opt_d = torch.optim.Adam(disc.parameters(), lr=a.disc_lr, betas=(0.5, 0.9))
    if "disc" in ck:
        disc.load_state_dict(ck["disc"])
        print("dyskryminator wznowiony z checkpointu vqvae (bez zimnego startu)",
              flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    scaler_d = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    step = 0
    ckpt_path = os.path.join(a.out, "decoder.pt")
    if a.resume and os.path.exists(ckpt_path):
        r = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        disc.load_state_dict(r["disc"])
        opt_d.load_state_dict(r["opt_d"])
        step = r["step"]
        print(f"wznowione z kroku {step}", flush=True)

    raw = model
    if dev == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel na {torch.cuda.device_count()} GPU", flush=True)

    ceiling = min(a.steps, step + a.max_steps) if a.max_steps else a.steps
    print(f"sesja: krok {step} -> {ceiling} (cel {a.steps})", flush=True)

    t0 = time.time()
    it = iter(dl)
    while step < ceiling:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl); x = next(it)
        x = x.to(dev, non_blocking=True)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)

        with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            # No gradient through the encoder or the quantizer: they are frozen,
            # and the straight-through estimator would otherwise carry gradient
            # into a codebook that must not move.
            with torch.no_grad():
                z = raw.encoder(x)
                q, _, _ = raw.quant(z)
            out = raw.decoder(q)
        out = out.float()

        rec = F.l1_loss(out, x)
        # Cycle-consistency in latent space. E is frozen, so this asks the
        # decoder to draw something that still encodes to the same content —
        # a content constraint that does not average pixels.
        with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
            z_out = raw.encoder(out)
        perc = F.l1_loss(z_out.float(), z.float().detach())
        colour = F.l1_loss(out.mean(dim=(2, 3)), x.mean(dim=(2, 3)))
        content = rec + a.perc * perc + a.colour * colour

        adv = g_adv_loss(disc(out))
        w = adaptive_weight(content, adv, raw.decoder.out.weight).clamp(max=a.adv_max)
        loss = content + w * adv

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()

        # Keyed to the step number AFTER the increment below, so that a logged
        # step is always one the discriminator ran on. Keyed to the pre-increment
        # value, every logged hundred fell on a skipped step and printed the
        # freshly zeroed variable — "d 0.000" on every line of a 10000-step run,
        # which reads as a dead discriminator and hid the real one from view.
        d_loss = torch.zeros(())
        if (step + 1) % a.disc_every == 0:
            d_loss = hinge_d_loss(disc(x), disc(out.detach()))
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

        step += 1

        if step % a.log_every == 0:
            s_out, s_real = sharpness(out), sharpness(x)
            print(f"step {step}/{ceiling}  rec {rec.item():.4f}  perc {perc.item():.4f}  "
                  f"col {colour.item():.4f}  adv {adv.item():.3f}  d {d_loss.item():.3f}  "
                  f"w {float(w):.2f}  ostrosc {s_out:.3f}/{s_real:.3f} "
                  f"({100*s_out/max(s_real,1e-6):.0f}%)  {time.time()-t0:.0f}s", flush=True)

        if step % a.sample_every == 0 or step == ceiling:
            from PIL import Image
            k = min(4, x.size(0))
            pair = torch.cat([x[:k], out[:k].detach()], dim=0).clamp(-1, 1)
            grid = ((pair + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
            h, w = grid.shape[1:3]
            sheet = np.zeros((2 * h, k * w, 3), dtype=np.uint8)
            for j in range(k):
                sheet[0:h, j * w:(j + 1) * w] = grid[j]
                sheet[h:2 * h, j * w:(j + 1) * w] = grid[k + j]
            Image.fromarray(sheet).save(os.path.join(a.out, f"probka-{step:06d}.png"))

        if step % a.ckpt_every == 0 or step == ceiling:
            blob = {"model": raw.state_dict(), "opt": opt.state_dict(),
                    "disc": disc.state_dict(), "opt_d": opt_d.state_dict(),
                    "step": step, "arch": ck["arch"], "vqvae_step": base_step}
            tmp = ckpt_path + ".tmp"
            torch.save(blob, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, ckpt_path)

    print(f"gotowe — {step} krokow w {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
