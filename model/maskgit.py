"""MaskGIT: the same tokens, filled in all at once instead of one at a time.

The autoregressive model draws an image by sampling 256 tokens in order, each
one a full pass through the network — measured at ~7 s per picture on this Mac,
and that cost is structural, not an implementation detail.

MaskGIT starts from an entirely masked image and fills it in over roughly ten
rounds. Each round predicts EVERY remaining position at once, keeps the ones it
is most confident about, and puts the rest back under the mask. Ten passes
instead of 256.

Two differences from the family's usual shape, both required rather than chosen:

**Attention is bidirectional.** A token in the middle of an image should see the
tokens after it — that is the whole point of filling in gaps rather than
extending a sequence. `causal=False` on the config flips it; the attention
itself is the same code every other G model uses.

**There is a MASK token.** It occupies its own id, so "not yet decided" is
something the model can read on its input rather than something inferred from
position.

Reported in the original paper as both faster and better than autoregression at
the same size — which is a claim about their setup, not ours. This exists so the
two can be trained briefly on the same data and compared on pictures.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gpt_base import Block, RMSNorm, build_rope_cache


@dataclass
class MaskGITConfig:
    n_layer: int = 12
    n_head: int = 10
    n_kv_head: int = 2
    n_embd: int = 640
    ffn_hidden: int = 1728
    rope_theta: float = 10000.0
    dropout: float = 0.0
    causal: bool = False          # the reason this class exists

    text_len: int = 32
    image_len: int = 256
    n_text: int = 8192
    n_image: int = 8192
    n_special: int = 5            # PAD, BOS_TEXT, BOS_IMG, EOS, MASK

    PAD: int = 0
    BOS_TEXT: int = 1
    BOS_IMG: int = 2
    EOS: int = 3
    MASK: int = 4

    @property
    def vocab_size(self):
        return self.n_special + self.n_text + self.n_image

    @property
    def block_size(self):
        return self.text_len + self.image_len

    def text_token(self, i):
        return self.n_special + i

    def image_token(self, i):
        return self.n_special + self.n_text + i


class MaskGIT(nn.Module):
    def __init__(self, config: MaskGITConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm_final = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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

    def forward(self, idx, targets=None, mask=None):
        """`mask` marks the positions that were hidden; loss is taken there only.

        Scoring unmasked positions would reward copying the input, which is free
        and teaches nothing.
        """
        b, t = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:t].to(x.dtype)
        sin = self.rope_sin[:t].to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_final(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            reduction="none").view(b, t)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return logits, loss


def mask_ratio(u):
    """Cosine schedule: mostly-visible early, almost-empty late.

    A uniform ratio would train the model mainly on the easy middle, and the
    hard case at generation time is the first round, where nearly everything is
    still hidden.
    """
    return torch.cos(u * torch.pi / 2)


@torch.no_grad()
def generate(model, text_rows, cfg, steps=10, scale=4.0, temp=1.0):
    """Fill a whole image in `steps` rounds instead of 256.

    Classifier-free guidance rides along the batch exactly as in the
    autoregressive sampler: the prompt and a blanked copy are predicted
    together and the logits extrapolated away from the unconditional answer.
    """
    device = text_rows.device
    n = text_rows.size(0)
    lo, hi = cfg.image_token(0), cfg.image_token(cfg.n_image - 1)

    img = torch.full((n, cfg.image_len), cfg.MASK, dtype=torch.long, device=device)
    blank = torch.full_like(text_rows, cfg.PAD)

    for step in range(steps):
        seq = torch.cat([torch.cat([text_rows, blank], 0),
                         img.repeat(2, 1)], dim=1)
        logits = model(seq)[:, cfg.text_len:]
        cond, uncond = logits[:n], logits[n:]
        g = uncond + scale * (cond - uncond)
        g[..., :lo] = -float("inf")
        g[..., hi + 1:] = -float("inf")

        probs = F.softmax(g / max(temp, 1e-5), dim=-1)
        flat = probs.view(-1, probs.size(-1))
        pick = torch.multinomial(flat, 1).view(n, cfg.image_len)
        conf = flat.gather(1, pick.view(-1, 1)).view(n, cfg.image_len)

        # Positions already decided keep their token and are never reconsidered;
        # giving them infinite confidence is how they stay put.
        decided = img != cfg.MASK
        pick = torch.where(decided, img, pick)
        conf = torch.where(decided, torch.full_like(conf, float("inf")), conf)

        # How many should still be hidden after this round.
        u = torch.tensor((step + 1) / steps, device=device)
        keep_masked = int(mask_ratio(u).item() * cfg.image_len)
        if step == steps - 1:
            keep_masked = 0

        img = pick
        if keep_masked > 0:
            # Re-mask the least confident: the model gets another look at them
            # once its neighbours are settled, which is the entire mechanism.
            cut = conf.topk(keep_masked, dim=1, largest=False).indices
            img = img.scatter(1, cut, cfg.MASK)

    return img - lo
