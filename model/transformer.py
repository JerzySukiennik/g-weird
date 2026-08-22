"""Text to image tokens: the model that actually makes the pictures.

One decoder-only stack over a single concatenated sequence, exactly the way the
original DALL-E did it:

    [text tokens] [BOS_IMG] [256 image tokens]

No encoder-decoder split. A causal transformer reading that sequence learns to
continue it, and continuing it past BOS_IMG *is* generating an image. Text
attends to text, image tokens attend to everything before them, and nothing else
has to be arranged.

The blocks come from G-Doodle's transformer, which is already trained, tested,
and carries grouped-query attention — the same reason it was added there applies
here, since generation is 256 sequential steps and the KV cache is what makes
that bearable.

Two things are specific to this model:

**One vocabulary, not two.** Text ids and image ids live in the same embedding
table, offset apart. A separate image head would work equally well and would be
one more place for an off-by-one to hide.

**Conditioning dropout from step zero.** Classifier-free guidance is the only
knob this model has for making prompts actually matter, and it only works if an
unconditional branch was trained alongside. G-Images learned that the expensive
way: dropout added for the last 21600 steps of 70000 gave the unconditional
branch 1.5% of the conditional one's exposure, guidance had nothing to amplify,
and the whole run was wasted. Here it runs from the first step.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt_base import Block, RMSNorm, build_rope_cache  # noqa: E402


@dataclass
class WeirdConfig:
    n_text: int = 8192          # BPE vocabulary for captions
    n_image: int = 8192         # VQ codebook
    n_special: int = 4          # PAD, BOS_TEXT, BOS_IMG, EOS
    text_len: int = 32          # captions truncated to this
    image_len: int = 256        # 16x16 grid
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4          # grouped-query attention
    n_embd: int = 768
    ffn_hidden: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def vocab_size(self):
        return self.n_special + self.n_text + self.n_image

    @property
    def block_size(self):
        return self.text_len + 1 + self.image_len

    # Offsets, so a caller never hand-rolls the arithmetic.
    PAD = 0
    BOS_TEXT = 1
    BOS_IMG = 2
    EOS = 3

    def text_token(self, i):
        return self.n_special + i

    def image_token(self, i):
        return self.n_special + self.n_text + i


class WeirdGPT(nn.Module):
    def __init__(self, config: WeirdConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm_final = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Tied weights: the embedding and the output projection describe the same
        # relationship between ids and vectors, and tying them saves 12.6M
        # parameters that buy nothing when kept separate.
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(config.n_embd // config.n_head,
                                    config.block_size, config.rope_theta,
                                    device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, kv=None, pos=0):
        b, t = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[pos:pos + t].to(x.dtype)
        sin = self.rope_sin[pos:pos + t].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin, kv=kv)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        # PAD is ignored: captions are padded to a fixed length and predicting
        # padding is not a skill worth spending capacity on.
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1),
                               ignore_index=self.config.PAD)
        return logits, loss
