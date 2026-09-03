"""The encoded corpus, read as one thing however many shards it arrives in.

Shared by both trainers because they differ in exactly one detail — whether a
BOS_IMG separator sits between the caption and the image. MaskGIT has none:
nothing there is being continued, the whole image is visible to attention at
once.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenPairs(Dataset):
    """Several encoded shards read as one corpus, with pre-tokenised captions.

    Multiple prefixes rather than one, because the corpus grows by adding
    kernel outputs and concatenating 6 GB of tokens before every run is pure
    waste — the token file is a flat array, so N of them read back to back are
    the same thing as one. Row i is located by a cumulative-count search.

    Captions come from `<prefix>_text<L>.u16` written by data/pack_captions.py,
    not from the JSON. Holding 3.8M dense captions as Python strings is a
    couple of gigabytes against Colab's 12.7 GB, and BPE inside __getitem__
    re-tokenised the same caption on every epoch.
    """

    def __init__(self, prefixes, cfg, insert_bos=True):
        self.insert_bos = insert_bos
        self.cfg = cfg
        self.parts, self.starts, total = [], [], 0
        for prefix in prefixes:
            meta = json.load(open(f"{prefix}_meta.json"))
            per = meta["per_image"]
            if per != cfg.image_len:
                raise SystemExit(f"{prefix}: {per} tokenow na obraz, model "
                                 f"oczekuje {cfg.image_len}")

            # The count comes from the file, not from meta. When the corpus was
            # finished in a second kernel run, meta recorded that run's counter
            # (400015) instead of the whole corpus (1780125) — training would
            # then have quietly used 22% of the data with nothing in the logs to
            # say so. The bytes on disk cannot be wrong in that way.
            size = os.path.getsize(f"{prefix}_tokens.u16")
            if size % (per * 2):
                raise SystemExit(f"{prefix}: {size} B nie dzieli sie na obrazy "
                                 f"po {per * 2} B — plik jest urwany")
            n = size // (per * 2)
            if n != meta.get("n"):
                print(f"UWAGA {prefix}: meta mowi {meta.get('n')}, plik ma "
                      f"{n:,} — ufam plikowi", flush=True)

            text_path = f"{prefix}_text{cfg.text_len}.u16"
            if not os.path.exists(text_path):
                raise SystemExit(
                    f"brak {text_path} — odpal najpierw data/pack_captions.py "
                    f"--data {prefix} --tokenizer <text.json> "
                    f"--text-len {cfg.text_len}")
            tsize = os.path.getsize(text_path)
            if tsize != n * cfg.text_len * 2:
                raise SystemExit(
                    f"{text_path}: {tsize} B na {n:,} obrazow po "
                    f"{cfg.text_len} tokenow — spakowane innym text_len albo "
                    f"innym korpusem, nie trenuj na tym")

            self.parts.append((
                np.memmap(f"{prefix}_tokens.u16", dtype=np.uint16, mode="r",
                          shape=(n, per)),
                np.memmap(text_path, dtype=np.uint16, mode="r",
                          shape=(n, cfg.text_len)),
            ))
            self.starts.append(total)
            total += n
            print(f"  {prefix}: {n:,} par", flush=True)

        self.n = total
        self.starts = np.asarray(self.starts)
        print(f"{self.n:,} par razem, {cfg.image_len} tokenow na obraz, "
              f"{cfg.text_len} na podpis", flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        c = self.cfg
        part = int(np.searchsorted(self.starts, i, side="right")) - 1
        toks, texts = self.parts[part]
        row = i - int(self.starts[part])

        # Zapisane jako id+1, wiec zero znaczy "brak tokenu". Id 0 to prawdziwy
        # token (<unk>) i uzycie go jako wypelnienia uczyloby model, ze kazdy
        # krotki podpis konczy sie seria nieznanych slow.
        raw = texts[row]
        ids = [int(t) - 1 for t in raw if t]
        text = ([c.text_token(t) for t in ids]
                + [c.PAD] * (c.text_len - len(ids)))
        img = [c.image_token(int(t)) for t in toks[row]]
        seq = text + ([c.BOS_IMG] if self.insert_bos else []) + img
        return torch.tensor(seq, dtype=torch.long), len(ids)
