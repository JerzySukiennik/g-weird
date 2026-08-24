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
                                 g_adv_loss, adaptive_weight,
                                 feature_match_loss)
from train.train_vqvae import Images, lr_at                      # noqa: E402


class Cropped(torch.utils.data.Dataset):
    """Random crops that always drop the bottom band and move everything else.

    The corpus is largely watermarked stock photography — Shutterstock,
    Dreamstime and Alamy marks are plainly visible in the training batches. A
    discriminator's whole job is to make outputs match that distribution, so the
    first run at cap 0.5 dutifully learned to render the watermark crisply: white
    "piano key" blocks turned up in every generation, sharper than the subjects.

    Two defences here. The caption bar lives in the last rows, so crops are taken
    from above it and it is never seen. The tiled marks cannot be removed that
    way, but a random offset means they no longer sit at a fixed place, which is
    what made them easy to memorise.

    Everything downstream is fully convolutional, so training at 224 and running
    at 256 is fine — the grid is 14x14 here and 16x16 there, and no layer cares.
    """

    def __init__(self, base, size=224, drop_bottom=16):
        self.base, self.size, self.drop = base, size, drop_bottom

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x = self.base[i]
        _, h, w = x.shape
        top_max = max(0, h - self.drop - self.size)
        left_max = max(0, w - self.size)
        top = int(torch.randint(0, top_max + 1, ()))
        left = int(torch.randint(0, left_max + 1, ()))
        return x[:, top:top + self.size, left:left + self.size]


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
    p.add_argument("--fm", type=float, default=1.0,
                   help="waga dopasowania cech dyskryminatora; 0 wylacza")
    p.add_argument("--dec-base", type=int, default=0,
                   help="szerszy dekoder; 0 = jak w checkpoincie")
    p.add_argument("--dec-res", type=int, default=2, help="bloki rezydualne na poziom")
    p.add_argument("--dec-attn", type=int, default=0,
                   help="na ilu poziomach dodac attention (poza waskim gardlem)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--sample-every", type=int, default=1000,
                   help="zrzut PNG oryginal/rekonstrukcja — liczby dzis kilka razy "
                        "wygladaly dobrze, gdy tresc sie psula")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--crop", type=int, default=0,
                   help="losowy kadr tej wielkosci; 0 = bez kadrowania")
    p.add_argument("--drop-bottom", type=int, default=16,
                   help="ile dolnych wierszy nigdy nie trafia do kadru (pasek stempla)")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    ds = Images(a.data, res=a.res)
    if a.crop:
        ds = Cropped(ds, size=a.crop, drop_bottom=a.drop_bottom)
        print(f"kadr {a.crop}px, dolne {a.drop_bottom} wierszy zawsze odciete",
              flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)

    ck = torch.load(a.vqvae, map_location="cpu", weights_only=False)
    arch = dict(ck["arch"])
    if a.dec_base:
        arch.update(dec_base=a.dec_base, dec_res=a.dec_res, dec_attn=a.dec_attn)
    model = VQVAE(**arch).to(dev)

    # A wider decoder has different shapes, so its weights cannot come from the
    # checkpoint and start fresh. The encoder and codebook MUST come from it:
    # they define what a token id means, and the transformer was trained against
    # those meanings. Loading them by name rather than with strict=False, so a
    # silently missing encoder tensor cannot pass as "just the decoder differs".
    frozen = {k: v for k, v in ck["model"].items()
              if k.startswith("encoder.") or k.startswith("quant.")}
    missing, unexpected = model.load_state_dict(frozen, strict=False)
    still_needed = [k for k in missing
                    if k.startswith("encoder.") or k.startswith("quant.")]
    if still_needed or unexpected:
        raise SystemExit(f"enkoder/codebook nie wczytal sie w calosci: "
                         f"brakuje {still_needed[:4]}, nieoczekiwane {unexpected[:4]}")
    if a.dec_base:
        n = sum(q.numel() for q in model.decoder.parameters())
        print(f"dekoder od zera: base {a.dec_base}, res {a.dec_res}, "
              f"attn {a.dec_attn}, {n/1e6:.1f}M", flush=True)
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

        # One pass with features, reused for both the verdict and the matching
        # term — running the discriminator twice would double its cost for
        # numbers we already have.
        if a.fm > 0:
            fake_logits, fake_feats = disc(out, features=True)
            with torch.no_grad():
                _, real_feats = disc(x, features=True)
            fm = feature_match_loss(real_feats, fake_feats)
            content = content + a.fm * fm
        else:
            fake_logits = disc(out)
            fm = torch.zeros((), device=dev)
        adv = g_adv_loss(fake_logits)
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
                  f"fm {float(fm):.4f}  "
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
