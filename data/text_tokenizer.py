"""BPE over the captions — the difference between free text and a dropdown.

G-Doodle matched prompts against 345 fixed categories through a table of 766
aliases. That was honest for a model that genuinely knew 345 things, and it is
the wrong tool here: G-Weird has to accept a sentence it has never seen and make
something of it.

Trained on our own captions, not downloaded. The vocabulary that comes out is
shaped by Conceptual Captions — heavy on "person", "background", "illustration",
light on colour words — which is worth knowing when a prompt underperforms: the
model may simply never have had the words.

8192 merges, matching the image codebook so the two halves of the vocabulary are
the same size. Nothing forces that, it just keeps the arithmetic legible.
"""

import argparse
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


def train(captions, vocab_size=8192, out="text.json"):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # Lowercase and strip accents: captions are already lowercase web alt-text,
    # and a prompt typed with capitals should hit the same tokens as one without.
    tok.normalizer = normalizers.Sequence([normalizers.NFD(),
                                           normalizers.Lowercase(),
                                           normalizers.StripAccents()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                  special_tokens=["<unk>"],
                                  show_progress=False)
    tok.train_from_iterator(captions, trainer)
    tok.save(out)
    return tok


def load(path):
    return Tokenizer.from_file(path)


def encode(tok, text, length):
    """Ids, truncated and padded to a fixed length. Returns (ids, n_real)."""
    ids = tok.encode(text).ids[:length]
    return ids + [0] * (length - len(ids)), len(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", required=True)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--out", default="text.json")
    a = p.parse_args()

    caps = json.load(open(a.captions))
    print(f"{len(caps):,} podpisow", flush=True)
    tok = train(caps, a.vocab, a.out)
    print(f"slownik: {tok.get_vocab_size()}, zapisano {a.out} "
          f"({os.path.getsize(a.out)/1e6:.1f} MB)", flush=True)

    for s in ["a dog on a beach", "make me a man that eats a hotdog",
              "roblox character", "sushi made of leaves"]:
        ids = tok.encode(s).ids
        print(f"  {s!r} -> {len(ids)} tokenow: {tok.encode(s).tokens}", flush=True)


if __name__ == "__main__":
    main()
