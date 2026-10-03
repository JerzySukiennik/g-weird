"""Train the masked-token model, so it can be compared against autoregression.

Same corpus, same tokenizer, same 62.1M — the only thing that differs is how the
image is produced, which is the whole point of running both.

The training objective is not next-token prediction. A random share of the image
tokens is replaced with MASK and the model predicts those, seeing every other
position in both directions. The share is drawn from a cosine schedule rather
than uniformly: at generation time the first round faces an almost entirely
masked image, and a uniform draw would train mostly for the easy middle.

Conditioning dropout runs from step zero, exactly as in the autoregressive
trainer and for the same measured reason: guidance needs an unconditional branch
that has been trained alongside, and G-Images showed what adding it late gives —
1.5% of the exposure and guidance that does not work.
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
from model.maskgit import MaskGIT, MaskGITConfig, mask_ratio    # noqa: E402
from train.corpus import TokenPairs  # noqa: E402
from train.train_ar import lr_at                                # noqa: E402


def region_hidden(n, grid, lo=0.08, hi=0.6):
    """(n, grid*grid) bool: each row hides the union of 1-3 random rectangles
    covering roughly lo..hi of the picture, aspect ratio between 1:2 and 2:1.

    Random scattered tokens never teach what a region edit needs. A hole made of
    scattered tokens has neighbours everywhere to copy from; a rectangle does not,
    and its content has to come from the caption and the far context together.
    Tested on the 50000-step Live checkpoint: the context is kept perfectly but a
    new caption does not change the content, because training never showed it a
    contiguous hole.
    """
    out = torch.zeros(n, grid, grid, dtype=torch.bool)
    for i in range(n):
        frac = lo + (hi - lo) * float(torch.rand(()))
        boxes = 1 + int(torch.randint(0, 3, ()))
        for _ in range(boxes):
            aspect = math.exp((float(torch.rand(())) - 0.5) * 2 * math.log(2.0))
            area = frac / boxes * grid * grid
            h = max(2, min(grid, int(round(math.sqrt(area * aspect)))))
            w = max(2, min(grid, int(round(math.sqrt(area / aspect)))))
            r0 = int(torch.randint(0, grid - h + 1, ()))
            c0 = int(torch.randint(0, grid - w + 1, ()))
            out[i, r0:r0 + h, c0:c0 + w] = True
    return out.view(n, grid * grid)


def collate(batch, cfg, drop_p, region_p=0.0):
    # TokenPairs oddaje (sekwencja, dlugosc podpisu); tu potrzebna jest tylko
    # sekwencja. Pierwszy bieg Live padl na tym w pierwszej partii — test
    # dymny sprawdzal dataset, ale nie collate.
    seqs = torch.stack([b[0] for b in batch])
    target = seqs.clone()

    # One ratio per example, from the cosine schedule.
    u = torch.rand(seqs.size(0))
    k = (mask_ratio(u) * cfg.image_len).long().clamp(1, cfg.image_len)

    img = seqs[:, cfg.text_len:]
    scores = torch.rand_like(img, dtype=torch.float)
    # Mask the k lowest-scoring positions per row — a per-row random subset of
    # exactly the right size, without a Python loop.
    order = scores.argsort(dim=1)
    ranks = order.argsort(dim=1)
    hidden = ranks < k[:, None]
    if region_p > 0:
        # Czesc przykladow dostaje zamiast tego ciagle prostokaty — to jest
        # trening pod edycje zamalowanego obszaru.
        use = torch.rand(seqs.size(0)) < region_p
        if bool(use.any()):
            grid = int(round(math.sqrt(cfg.image_len)))
            hidden = hidden.clone()
            hidden[use] = region_hidden(int(use.sum()), grid)
    img = img.masked_fill(hidden, cfg.MASK)
    seqs = torch.cat([seqs[:, :cfg.text_len], img], dim=1)

    # Loss on hidden image positions only; the text half and the visible tokens
    # would otherwise reward copying the input.
    loss_mask = torch.zeros_like(seqs, dtype=torch.float)
    loss_mask[:, cfg.text_len:] = hidden.float()

    if drop_p > 0:
        blanked = torch.rand(seqs.size(0)) < drop_p
        seqs[blanked, :cfg.text_len] = cfg.PAD
    return seqs, target, loss_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True,
                   help="jeden lub wiele prefiksow shardow, jak w train_ar.py")
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
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--region-p", type=float, default=0.0,
                   help="czesc przykladow z ciaglymi prostokatami zamiast losowych "
                        "tokenow (trening pod edycje obszaru); 0 = jak dotad")
    p.add_argument("--init-from", default="",
                   help="wagi startowe z innego checkpointu (krok liczony od zera, "
                        "swiezy optymalizator). Dziala tylko w pierwszej sesji")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="zakoncz sesje po tylu godzinach, z koncowym checkpointem "
                        "(Kaggle ubija sesje o 12 h bez zapisu)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    # Dlugosc obrazu bierze sie z korpusu, nie z domyslnej wartosci w klasie.
    # Tokenizer 576-tokenowy zastapil 256-tokenowy, a kazde miejsce, w ktorym ta
    # liczba jest wpisana recznie, to okazja, zeby trenowac model o zlym
    # ksztalcie i dowiedziec sie o tym po godzinach.
    per = json.load(open(f"{a.data[0]}_meta.json"))["per_image"]
    cfg = MaskGITConfig(image_len=per, label_smoothing=a.label_smoothing)
    print(f"korpus: {per} tokenow na obraz, sekwencja {cfg.block_size}", flush=True)
    ds = TokenPairs(a.data, cfg, insert_bos=False)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=(dev == "cuda"),
                    collate_fn=lambda b: collate(b, cfg, a.text_dropout, a.region_p))

    model = MaskGIT(cfg).to(dev)
    raw = model
    print(f"maskgit: {sum(q.numel() for q in model.parameters())/1e6:.1f}M, "
          f"slownik {cfg.vocab_size}, sekwencja {cfg.block_size}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

    step = 0
    ckpt_path = os.path.join(a.out, "maskgit.pt")
    # Start z cudzych wag. Sensowne, bo siatka tokenow zmienia tylko dlugosc
    # sekwencji, a model nie ma absolutnych pozycji (RoPE), wiec te same wagi
    # laduja sie w modelu z inna liczba tokenow obrazu; slownik kodow jest ten
    # sam, jesli tokenizer jest ten sam. Ladowanie SCISLE: brakujacy lub
    # nadmiarowy klucz ma wywrocic trening, nie zostac pominiety. Tylko gdy nie
    # ma checkpointu wlasnej sesji, zeby kolejna sesja wznawiala, a nie
    # zaczynala od zrodla.
    if a.init_from and not (a.resume and os.path.exists(ckpt_path)):
        src = torch.load(a.init_from, map_location="cpu", weights_only=False)
        raw.load_state_dict(src["model"])
        sc = src.get("cfg", {})
        print(f"wagi startowe z {a.init_from} (krok zrodlowy {src.get('step', '?')}, "
              f"image_len zrodla {sc.get('image_len', '?')} -> {cfg.image_len}); "
              f"krok od zera, swiezy optymalizator", flush=True)
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"wznowione z kroku {step}", flush=True)

    if dev == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel na {torch.cuda.device_count()} GPU", flush=True)

    ceiling = min(a.steps, step + a.max_steps) if a.max_steps else a.steps
    print(f"sesja: krok {step} -> {ceiling} (cel {a.steps})", flush=True)

    t0 = time.time()
    it = iter(dl)
    while step < ceiling:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(a.accum):
            try:
                seq, tgt, lm = next(it)
            except StopIteration:
                it = iter(dl); seq, tgt, lm = next(it)
            seq, tgt, lm = (seq.to(dev, non_blocking=True),
                            tgt.to(dev, non_blocking=True),
                            lm.to(dev, non_blocking=True))
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                _, loss = model(seq, tgt, lm)
                loss = loss.mean() / a.accum
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        # Zegar, nie liczba krokow — te same powody co w train_ar.py: dwie sesje
        # liczone po szacunku uderzyly w sciane 12 h bez koncowego zapisu.
        if a.max_hours and step < ceiling and \
                (time.time() - t0) / 3600 >= a.max_hours:
            print(f"limit czasu {a.max_hours} h — koncze na kroku {step}", flush=True)
            ceiling = step

        if step % a.log_every == 0:
            print(f"step {step}/{ceiling}  loss {total:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s",
                  flush=True)
        if step % a.ckpt_every == 0 or step == ceiling:
            blob = {"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": cfg.__dict__}
            tmp = ckpt_path + ".tmp"
            torch.save(blob, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, ckpt_path)

    print(f"done — {step} krokow w {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
