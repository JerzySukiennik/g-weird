"""Train the text-to-image transformer on the encoded corpus.

Sequence per example: [32 text tokens] [BOS_IMG] [256 image tokens]. Loss on the
whole thing — predicting the caption is a free auxiliary task that costs nothing
and gives the text half a reason to be well modelled — with padding ignored.

**Conditioning dropout runs from step zero and that is the point.** Classifier-
free guidance needs an unconditional branch trained alongside the conditional
one, and G-Images proved what happens otherwise: dropout added for the last
21600 steps of 70000 left the unconditional branch with 1.5% of the exposure,
cos(e_cond, e_null) measured 1.000 at every timestep, guidance had nothing to
amplify, and raising its scale only produced colour garbage. Weeks of GPU went
into discovering that. Here 10% of examples have their caption blanked from the
first step.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import WeirdConfig, WeirdGPT  # noqa: E402
from data.text_tokenizer import load as load_tok  # noqa: E402


class TokenPairs(Dataset):
    def __init__(self, prefix, tok_path, cfg):
        meta = json.load(open(f"{prefix}_meta.json"))
        self.per = meta["per_image"]
        if self.per != cfg.image_len:
            raise SystemExit(f"korpus ma {self.per} tokenow na obraz, "
                             f"model oczekuje {cfg.image_len}")

        # The count comes from the file, not from meta. When the corpus was
        # finished in a second kernel run, meta recorded that run's counter
        # (400015) instead of the whole corpus (1780125) — training would then
        # have quietly used 22% of the data with nothing in the logs to say so.
        # The bytes on disk cannot be wrong in that way, so they decide.
        size = os.path.getsize(f"{prefix}_tokens.u16")
        if size % (self.per * 2):
            raise SystemExit(f"plik tokenow ma {size} B, nie dzieli sie na obrazy "
                             f"po {self.per * 2} B — jest urwany")
        self.n = size // (self.per * 2)
        if self.n != meta["n"]:
            print(f"UWAGA: meta mowi {meta['n']:,} par, plik ma {self.n:,} — "
                  f"ufam plikowi", flush=True)

        self.toks = np.memmap(f"{prefix}_tokens.u16", dtype=np.uint16, mode="r",
                              shape=(self.n, self.per))
        self.caps = json.load(open(f"{prefix}_captions.json"))
        if len(self.caps) != self.n:
            raise SystemExit(f"{len(self.caps):,} podpisow na {self.n:,} obrazow — "
                             f"pary sie rozjechaly, nie trenuj na tym")
        self.tok = load_tok(tok_path)
        self.cfg = cfg
        print(f"{self.n:,} par, {self.per} tokenow na obraz", flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        c = self.cfg
        ids = self.tok.encode(self.caps[i]).ids[:c.text_len]
        text = [c.text_token(t) for t in ids] + [c.PAD] * (c.text_len - len(ids))
        img = [c.image_token(int(t)) for t in self.toks[i]]
        seq = text + [c.BOS_IMG] + img
        return torch.tensor(seq, dtype=torch.long), len(ids)


def collate(batch, cfg, drop_p):
    seqs = torch.stack([b[0] for b in batch])
    if drop_p > 0:
        # Blank the caption, not the whole prefix: BOS_IMG must stay so the model
        # always knows where the picture starts, conditioned or not.
        mask = torch.rand(seqs.size(0)) < drop_p
        seqs[mask, :cfg.text_len] = cfg.PAD
    return seqs


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="run")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--text-dropout", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WeirdConfig()
    ds = TokenPairs(a.data, a.tokenizer, cfg)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=(dev == "cuda"),
                    persistent_workers=(a.workers > 0),
                    collate_fn=lambda b: collate(b, cfg, a.text_dropout))

    model = WeirdGPT(cfg).to(dev)
    raw = model
    print(f"transformer: {sum(q.numel() for q in model.parameters())/1e6:.1f}M, "
          f"slownik {cfg.vocab_size}, sekwencja {cfg.block_size}", flush=True)
    if dev == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel na {torch.cuda.device_count()} GPU", flush=True)

    opt = torch.optim.AdamW(raw.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    step = 0
    ckpt_path = os.path.join(a.out, "gweird.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = raw.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"wczytane z roznicami: {list(missing)} / {list(unexpected)}",
                  flush=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"wznowione z kroku {step}", flush=True)

    ceiling = min(a.steps, step + a.max_steps) if a.max_steps else a.steps
    print(f"sesja: krok {step} -> {ceiling} (cel {a.steps})", flush=True)

    t0 = time.time()
    model.train()
    it = iter(dl)
    while step < ceiling:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(a.accum):
            try:
                seq = next(it)
            except StopIteration:
                it = iter(dl); seq = next(it)
            seq = seq.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                _, loss = model(seq[:, :-1], seq[:, 1:])
                loss = loss.mean() / a.accum
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % a.log_every == 0:
            print(f"step {step}/{ceiling}  loss {total:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s",
                  flush=True)

        if step % a.ckpt_every == 0 or step == ceiling:
            blob = {"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": cfg.__dict__}
            tmp = ckpt_path + ".tmp"
            torch.save(blob, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, ckpt_path)      # atomic; a kill mid-write costs one interval

    print(f"done — {step} krokow w {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
