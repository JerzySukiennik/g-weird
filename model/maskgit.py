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
    # 0.1 jak w oryginalnym MaskGIT-cie. Przy 8192 kodach wiele roznych tokenow
    # dekoduje sie do prawie tego samego kawalka obrazu, wiec twarde "tylko ten
    # jeden jest dobry" karze model za odpowiedzi, ktore na obrazku sa poprawne.
    label_smoothing: float = 0.1

    text_len: int = 64          # jak w transformer.py — patrz komentarz tam
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
            reduction="none", label_smoothing=self.config.label_smoothing).view(b, t)
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
def generate(model, text_rows, cfg, steps=12, scale=4.0, temp=1.0,
             choice_temp=4.5):
    """Fill a whole image in `steps` rounds instead of 576.

    Classifier-free guidance rides along the batch exactly as in the
    autoregressive sampler: the prompt and a blanked copy are predicted
    together and the logits extrapolated away from the unconditional answer.

    **The Gumbel noise on the confidence is not optional and is not in the
    MaskGIT paper.** It lives only in Google's released code, and the paper's
    one sentence about it points at a section that does not exist. Without it
    the scheme is greedy: the first round keeps whatever the model is most sure
    about, which is flat background, and every later round is conditioned on
    that flatness until the whole image collapses to one colour. That is
    exactly what our first MaskGIT run produced. The PyTorch reproduction
    measured the cost with everything else held fixed: FID 66.7 without the
    noise against 7.7 with it (arXiv:2310.14400).

    The temperature decays linearly to zero, so the last round is pure greedy
    and the early rounds are nearly free choices — which is the point: the
    model must be allowed to commit to something other than the safest token
    while there is still context left to justify it.
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
        # fp32 dla prowadzenia — patrz komentarz w train/sample.py: przy
        # scale 4 ekstrapolacja wychodzi z zakresu fp16 i softmax degeneruje.
        cond, uncond = logits[:n].float(), logits[n:].float()
        g = uncond + scale * (cond - uncond)
        g[..., :lo] = -float("inf")
        g[..., hi + 1:] = -float("inf")

        probs = F.softmax(g / max(temp, 1e-5), dim=-1)
        flat = probs.view(-1, probs.size(-1))
        pick = torch.multinomial(flat, 1).view(n, cfg.image_len)
        conf = flat.gather(1, pick.view(-1, 1)).view(n, cfg.image_len)

        # log p + T*Gumbel, T decaying to zero. Ranking happens in log space
        # because that is where an additive Gumbel is the right perturbation;
        # adding noise to a raw probability would barely move a token whose
        # probability is 0.9 and would dominate one at 0.001.
        t_choice = choice_temp * (1.0 - (step + 1) / steps)
        gum = -torch.log(-torch.log(
            torch.rand_like(conf).clamp_min(1e-10)).clamp_min(1e-10))
        conf = conf.clamp_min(1e-10).log() + t_choice * gum

        # Positions already decided keep their token and are never reconsidered.
        # Infinity rather than 1.0: the noise above can push a fresh token's
        # score past any finite ceiling, and a decided token that loses its
        # place gets re-drawn, which is the one thing this loop must never do.
        decided = img != cfg.MASK
        pick = torch.where(decided, img, pick)
        conf = torch.where(decided, torch.full_like(conf, float("inf")), conf)

        # How many should still be hidden after this round.
        u = torch.tensor((step + 1) / steps, device=device)
        keep_masked = int(mask_ratio(u).item() * cfg.image_len)
        if step == steps - 1:
            keep_masked = 0
        # At least one token must be settled per round, or a schedule that
        # rounds badly can spin without ever finishing.
        keep_masked = min(keep_masked, int(decided.logical_not().sum(1).min()) - 1
                          if n else keep_masked)
        keep_masked = max(keep_masked, 0)

        img = pick
        if keep_masked > 0:
            # Re-mask the least confident: the model gets another look at them
            # once its neighbours are settled, which is the entire mechanism.
            cut = conf.topk(keep_masked, dim=1, largest=False).indices
            img = img.scatter(1, cut, cfg.MASK)

    return img - lo
