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
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vqvae import VQVAE  # noqa: E402


def shard_reader(prefix, res):
    """Random-access reader for one shard, in whichever format it was written."""
    meta = json.load(open(f"{prefix}_meta.json"))
    n = meta["n"]
    if meta.get("format") == "jpeg":
        offs = json.load(open(f"{prefix}_offsets.json"))
        fh = open(f"{prefix}_images.jpgbin", "rb")

        def raw(i):
            # Seek+read under no concurrency; only the DECODE is threaded, since
            # a single file handle cannot be shared across threads safely.
            fh.seek(offs[i])
            return fh.read(offs[i + 1] - offs[i])

        def decode(buf):
            return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
    else:
        arr = np.memmap(f"{prefix}_images.bin", dtype=np.uint8, mode="r",
                        shape=(n, res, res, 3))

        def raw(i):
            return i

        def decode(i):
            return np.array(arr[i])

    caps = json.load(open(f"{prefix}_captions.json"))
    return n, raw, decode, caps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-prefix", default="./tokens")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=16,
                   help="watki dekodujace JPEG; bez nich karta czeka na procesor")
    p.add_argument("--skip", type=int, default=0,
                   help="pomin pierwsze N obrazow (wznowienie po przerwanym biegu)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = VQVAE(**ck["arch"]).to(dev).eval()
    missing, _ = model.load_state_dict(ck["model"], strict=False)
    raw_model = model
    print(f"tokenizer z kroku {ck['step']}, arch {ck['arch']}", flush=True)
    if missing:
        print(f"  brakujace klucze: {list(missing)}", flush=True)

    # Measured on the card rather than assumed, after two wrong guesses about
    # where the time was going: pure compute in fp32 on one T4 runs at 71.9
    # images/s against 50.2 in production, so the forward pass — not I/O, not
    # JPEG decode — is the dominant cost. fp16 buys 1.5x (this encoder is heavy
    # on GroupNorm and SiLU, which tensor cores do not accelerate) and the second
    # card another 1.27x, for 134 images/s total.
    use_amp = dev == "cuda"
    if dev == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel na {torch.cuda.device_count()} GPU", flush=True)

    grid = a.res // (2 ** len(ck["arch"]["mults"]))
    per_image = grid * grid
    n_codes = ck["arch"]["n_codes"]
    assert n_codes <= 65535, "uint16 nie pomiesci tego codebooka"
    print(f"siatka {grid}x{grid} = {per_image} tokenow na obraz", flush=True)

    tok_path = f"{a.out_prefix}_tokens.u16"
    fh = open(tok_path, "ab" if a.skip else "wb")
    # Captions stream to disk one per line instead of piling up in a list. The
    # first attempt held all of them in memory and was SIGKILLed at 1.47M images
    # after 4.65 h — no traceback, because an OOM kill leaves none. The tokens
    # survived only because they were being written as they went; the captions,
    # written at the end, did not exist at all.
    cap_fh = open(f"{a.out_prefix}_captions.jsonl", "a" if a.skip else "w")
    total = a.skip
    seen = 0

    # Decoding is what starves the GPU here. The first run managed 50 images a
    # second on a card that should do 300-500 — eight times below its ceiling,
    # with the encoder idle while one thread unpacked JPEGs. 9.8 h of GPU quota
    # for work a thread pool does in a fraction of that. The prep script already
    # used 64 threads for exactly this reason; not carrying that over was an
    # oversight that cost 2.5 h before it was caught.
    pool = ThreadPoolExecutor(max_workers=a.workers)
    for prefix in a.data:
        n, raw, decode, caps = shard_reader(prefix, a.res)
        # Captions are re-emitted even for the skipped range. The tokens for it
        # already exist in the file being appended to, but the captions do not —
        # they were lost with the killed process. Re-deriving them costs nothing
        # (no GPU, they are just strings from the shard) and is what keeps the
        # two files the same length, which the check at the end enforces.
        if seen + n <= a.skip:                    # whole shard already encoded
            for c in caps:
                cap_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            seen += n
            print(f"{prefix}: pomijam {n} juz zakodowanych "
                  f"(podpisy odtworzone)", flush=True)
            continue
        start0 = max(0, a.skip - seen)
        if start0:
            for c in caps[:start0]:
                cap_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"{prefix}: wznawiam od obrazu {start0}", flush=True)
        else:
            print(f"{prefix}: {n} obrazow", flush=True)
        for start in range(start0, n, a.batch):
            end = min(start + a.batch, n)
            bufs = [raw(i) for i in range(start, end)]
            batch = np.stack(list(pool.map(decode, bufs)))
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(dev)
            x = x / 127.5 - 1.0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                idx = model(x)[2]
            fh.write(idx.to(torch.int32).cpu().numpy().astype(np.uint16).tobytes())
            for c in caps[start:end]:
                cap_fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            total += end - start
            if total % 20000 < a.batch:
                fh.flush(); cap_fh.flush()
                print(f"  {total} obrazow zakodowanych", flush=True)
        seen += n
    pool.shutdown()
    fh.close()
    cap_fh.close()

    # One JSON array at the end, for consumers that want it in one piece.
    caps_all = [json.loads(l) for l in open(f"{a.out_prefix}_captions.jsonl")]
    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(caps_all, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": total, "grid": grid, "per_image": per_image,
                   "n_codes": n_codes, "vqvae_step": ck["step"],
                   "arch": ck["arch"]}, f)

    size = os.path.getsize(tok_path)
    print(f"gotowe: {total} obrazow, {size/1e9:.2f} GB "
          f"({size/max(total,1):.0f} B na obraz), podpisow {len(caps_all)}", flush=True)
    if size // 2 // per_image != len(caps_all):
        raise SystemExit(f"NIEZGODNOSC: {size//2//per_image} obrazow w tokenach "
                         f"vs {len(caps_all)} podpisow")


if __name__ == "__main__":
    main()
