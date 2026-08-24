"""
G-Doodle — a ~18M parameter decoder-only transformer over pen strokes.

This is [[g-micro]]'s architecture shrunk to a twentieth of its size and pointed
at a different alphabet. Nothing here knows it is drawing: it predicts the next
token in a sequence, exactly as the text model does. Only the tokenizer differs,
which is the entire reason this project costs one night of GPU instead of a month.

Same modernised-GPT choices as G-Micro, kept deliberately identical so lessons
transfer in both directions:
  - RoPE instead of learned positional embeddings
  - RMSNorm instead of LayerNorm
  - SwiGLU feed-forward instead of a GELU MLP
  - no bias terms, pre-normalisation, tied embedding/output weights

The size target is not accuracy, it is the browser: this model ships as an ONNX
file a visitor downloads before they can doodle. At 18M parameters that is 73 MB
in fp32, 37 MB in fp16 and ~18 MB quantised to int8 — the last of which is the
number that actually has to be true, since nobody waits 73 MB for a doodle.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DoodleConfig:
    vocab_size: int = 1691     # 4 special + 345 categories + 2 x 671 delta bins
    block_size: int = 512      # ~250 pen points, longer than any QuickDraw sketch
    n_layer: int = 10
    n_head: int = 6
    n_kv_head: int = 2         # grouped-query attention: 3 query heads per KV head
    n_embd: int = 384          # head_dim = 64
    ffn_hidden: int = 1024     # ~8/3 * n_embd, rounded to a multiple of 64
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # Causal by default, which is what every model in the family has been. A
    # masked-token model needs every position to see every other one, so it
    # flips this rather than forking the attention.
    causal: bool = True


class KVCache:
    """Past keys and values, so each generated token costs one forward step.

    Without it, drawing a 400-token sketch re-reads the whole prefix 400 times.
    Keys and values of past tokens are a pure function of tokens that no longer
    change, so they are computed once and kept. This matters more here than in
    a chat model: the browser runs generation on the CPU.
    """

    def __init__(self, n_layer: int):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    def append(self, layer_idx: int, k, v):
        if self.k[layer_idx] is not None:
            k = torch.cat([self.k[layer_idx], k], dim=2)
            v = torch.cat([self.v[layer_idx], v], dim=2)
        self.k[layer_idx] = k
        self.v[layer_idx] = v
        return k, v

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].size(2)

    def trim(self, max_len: int):
        if self.length <= max_len:
            return
        cut = self.length - max_len
        for i in range(len(self.k)):
            self.k[i] = self.k[i][:, :, cut:, :]
            self.v[i] = self.v[i][:, :, cut:, :]


class RMSNorm(nn.Module):
    """Normalise by root-mean-square, then rescale. LayerNorm without the mean."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def build_rope_cache(head_dim: int, seq_len: int, theta: float, device, dtype):
    """cos/sin tables for rotary embeddings.

    Queries and keys are rotated by an angle proportional to position, so their
    dot product depends only on the difference of angles — attention sees
    relative distance for free. For strokes that is the right prior: what
    matters is how far back the previous point was, not where the sketch began.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):

    def __init__(self, config: DoodleConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.layer_idx = layer_idx
        self.dropout = config.dropout
        self.causal = getattr(config, "causal", True)

        # Grouped-query attention: full-width queries, narrow keys and values.
        # The browser, not the GPU, is what forces this. onnxruntime-web
        # materializes present_k/present_v as fresh buffers every step, so the
        # per-token cost grows with what is cached — measured at 65 ms/point
        # rising to 92 ms by point 40, which put a 400-token drawing at ~27 s
        # against a 2-5 s target. Two KV heads instead of six cut that traffic
        # threefold. Watching it draw IS the feature, so this is not a
        # micro-optimization; and after training the same change would cost the
        # entire training run, which is why it lands before the first step.
        kv_width = config.n_kv_head * self.head_dim
        self.qkv = nn.Linear(config.n_embd, config.n_embd + 2 * kv_width, bias=False)
        self.kv_width = kv_width
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, kv=None):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split([C, self.kv_width, self.kv_width], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv is not None:
            k, v = kv.append(self.layer_idx, k, v)

        T_k = k.size(2)
        # Expand only for the attention call, and only after caching: the cache
        # must hold the narrow tensors or the whole point is lost.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        # is_causal assumes queries and keys are the same sequence. During a
        # cached decode step there are T queries against T_k > T keys and every
        # query may legitimately see the whole cache, so the flag would mask the
        # wrong cells. Only pass it when the shapes actually match.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(self.causal and T == T_k and T > 1),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    """Gated feed-forward: one branch proposes values, the other gates them."""

    def __init__(self, config: DoodleConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, config.ffn_hidden, bias=False)
        self.w_down = nn.Linear(config.ffn_hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):

    def __init__(self, config: DoodleConfig, layer_idx: int):
        super().__init__()
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm_ffn = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def forward(self, x, cos, sin, kv=None):
        x = x + self.attn(self.norm_attn(x), cos, sin, kv)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class DoodleGPT(nn.Module):

    def __init__(self, config: DoodleConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm_final = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('w_down.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self._rope_cache = None

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope(self, start, T, device, dtype):
        need = start + T
        if (self._rope_cache is None
                or self._rope_cache[0].shape[0] < need
                or self._rope_cache[0].device != device
                or self._rope_cache[0].dtype != dtype):
            self._rope_cache = build_rope_cache(
                self.config.n_embd // self.config.n_head,
                max(need, self.config.block_size),
                self.config.rope_theta, device, dtype,
            )
        cos, sin = self._rope_cache
        return cos[start:need], sin[start:need]

    def forward(self, idx, targets=None, kv=None, return_logits: bool = True):
        B, T = idx.shape
        past = kv.length if kv is not None else 0
        assert past + T <= self.config.block_size, \
            f"sequence of {past + T} exceeds block_size {self.config.block_size}"

        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope(past, T, idx.device, x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin, kv)

        x = self.norm_final(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            # Training throws the logits away, and under DataParallel every
            # returned tensor is shipped back to GPU 0 and concatenated. Sending
            # them costs bandwidth and memory for nothing.
            return (logits if return_logits else None), loss

        return self.lm_head(x[:, [-1], :]), None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_p=0.9,
                 stop_token=None, use_cache=True):
        """Sample tokens autoregressively, yielding one id at a time.

        Nucleus sampling rather than top-k: the stroke vocabulary is ordinal and
        heavily peaked — after a long straight line the plausible next dx is a
        tight cluster of neighbouring bins, while at a corner it is wide. A
        fixed k either truncates the corner or lets noise into the straight.
        """
        self.eval()
        kv = KVCache(self.config.n_layer) if use_cache else None
        step_in = idx[:, -self.config.block_size:]

        for _ in range(max_new_tokens):
            logits, _ = self(step_in, kv=kv)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cutoff = probs.cumsum(dim=-1) - probs > top_p
                sorted_logits[cutoff] = float('-inf')
                logits = torch.full_like(logits, float('-inf')).scatter(
                    1, sorted_idx, sorted_logits)

            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            token = next_id.item()
            yield token

            if stop_token is not None and token == stop_token:
                return

            if kv is not None:
                kv.trim(self.config.block_size - 1)
                step_in = next_id
            else:
                step_in = idx[:, -self.config.block_size:]


if __name__ == '__main__':
    cfg = DoodleConfig()
    m = DoodleGPT(cfg)
    print(f"params: {m.num_params():,}")
    print(f"fp32 on disk: ~{m.num_params() * 4 / 1e6:.1f} MB")
    x = torch.randint(0, cfg.vocab_size, (2, 65))
    logits, loss = m(x[:, :-1], targets=x[:, 1:])
    print(f"forward ok — logits {tuple(logits.shape)}, loss {loss.item():.3f}")
    print(f"expected at random init: {math.log(cfg.vocab_size):.3f}")
