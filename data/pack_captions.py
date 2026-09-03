"""Tokenise the captions once, to a fixed-width array beside the image tokens.

Two reasons, both measured rather than assumed.

**Memory.** The trainer used to hold every caption as a Python string. At
2.36M that was already a large list; the 1.2 corpus is heading for 3.8M with
captions averaging 47 words instead of DiffusionDB's tag salad, which is a
couple of gigabytes of strings before training allocates anything. Colab's
standard runtime has 12.7 GB. As a `text_len`-wide uint16 array the same
captions are 3.8M x 64 x 2 = 486 MB on disk and memory-mapped, so resident cost
is whatever the OS decides to cache.

**Speed.** BPE ran inside `__getitem__`, so every worker re-tokenised the same
caption on every epoch. The corpus is read two or three times over a full run;
tokenising it once and reading integers afterwards moves that work out of the
hot path entirely.

Ids are stored **plus one**, so a stored zero means "no token here". BPE id 0 is
a real token (`<unk>`), and using it as padding would have taught the model that
every short caption ends in a run of unknown words.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.text_tokenizer import load as load_tok  # noqa: E402


def pack(prefix, tok, text_len):
    caps = json.load(open(f"{prefix}_captions.json"))
    out = np.zeros((len(caps), text_len), dtype=np.uint16)
    lens = np.zeros(len(caps), dtype=np.uint16)
    over = 0
    for i, c in enumerate(caps):
        ids = tok.encode(c).ids
        if len(ids) > text_len:
            over += 1
        ids = ids[:text_len]
        if any(t + 1 > 65535 for t in ids):
            raise SystemExit("slownik nie miesci sie w uint16")
        out[i, :len(ids)] = np.asarray(ids, dtype=np.uint16) + 1
        lens[i] = len(ids)
    path = f"{prefix}_text{text_len}.u16"
    out.tofile(path)
    return len(caps), path, lens.mean(), over


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True,
                   help="prefiksy shardow, te same co dla treningu")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--text-len", type=int, default=64)
    a = p.parse_args()

    tok = load_tok(a.tokenizer)
    for prefix in a.data:
        n_img = None
        meta_path = f"{prefix}_meta.json"
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            per = meta.get("per_image")
            if per:
                # Liczba z bajtow, nie z meta: meta zapisala juz kiedys w tym
                # projekcie licznik sesji zamiast korpusu.
                n_img = os.path.getsize(f"{prefix}_tokens.u16") // 2 // per
        n, path, avg, over = pack(prefix, tok, a.text_len)
        if n_img is not None and n_img != n:
            raise SystemExit(f"{prefix}: {n_img} obrazow vs {n} podpisow")
        print(f"{prefix}: {n:,} podpisow -> {path} "
              f"({os.path.getsize(path)/1e6:.0f} MB), srednio {avg:.1f} tokenow, "
              f"obcietych {over} ({100*over/max(n,1):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
