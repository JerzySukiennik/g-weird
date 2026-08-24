"""VQ-VAE: the part of G-Weird that decides what "wrong" looks like.

The encoder and decoder are lifted from G-Images' autoencoder, which was written
and tested for 2.2 — same ResBlock/AttnBlock stack, same shape. What is added
here is the quantizer, and it is the whole aesthetic argument of this project.

A continuous autoencoder degrades GENTLY: starve it and you get blur, which
reads as "low quality" and is boring. A quantizer degrades HARSHLY: every patch
must be replaced by the nearest of N codebook entries, so a face the codebook
has no entry for comes back as the closest thing it does have. That is where
melted features and wrong eye counts come from — Craiyon's signature, and the
thing Jurek actually asked for.

Two numbers control how uncanny the result is:

  codebook_size  — fewer entries, cruder substitutions. 8192 is roughly what
                   2021-era VQGANs used; dropping it makes things stranger.
  16x16 grid     — 256 tokens per image, from 256px. That is the DALL-E mini
                   layout, and four times cheaper for the transformer than the
                   32x32 grid 2.2's autoencoder produces. The extra downsample
                   is the reason this file does not simply import that class.

Commitment loss and the EMA codebook are the standard fixes for the standard
failure: without them most entries are never selected and the effective
vocabulary collapses to a few dozen, which produces uniform mush rather than
interesting wrongness.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoencoder_base import ResBlock, AttnBlock  # noqa: E402


class VectorQuantizer(nn.Module):
    """Nearest-neighbour lookup with an EMA-updated codebook.

    EMA rather than a learned embedding matrix because the gradient path to
    unused entries is zero — an entry that stops being selected can never come
    back on its own, and the codebook quietly shrinks to whatever it started
    near. EMA updates every selected entry toward the mean of what it matched
    this batch, and dead entries get restarted from live encoder output.
    """

    def __init__(self, n_codes=8192, dim=256, decay=0.99, eps=1e-5,
                 restart_below=1.0, restart_every=100):
        super().__init__()
        self.n_codes, self.dim, self.decay, self.eps = n_codes, dim, decay, eps
        # Restarting EVERY step looked reasonable and was measurably wrong.
        # cluster_size decays 1% per step, so an entry reseeded to 1.0 falls to
        # 0.99 on the very next step, trips the threshold again, and is reseeded
        # before it ever had a chance to be selected. After 20000 steps the
        # checkpoint showed 5670 of 8192 entries sitting at exactly 1.0 — the
        # value _restart_dead writes — i.e. churning, never used. Effective
        # vocabulary was 2522, not 8192, and reconstructions were correspondingly
        # smeared.
        # Every N steps instead gives a reseeded entry N chances to win a vector
        # and start accumulating real usage.
        self.restart_below = restart_below
        self.restart_every = restart_every
        self.register_buffer("steps", torch.zeros(1))
        embed = torch.randn(n_codes, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("inited", torch.zeros(1))

    def _seed_from(self, flat):
        """Start the codebook on real encoder output instead of noise.

        A Gaussian codebook and an encoder whose outputs live at a different
        scale collapse immediately: every vector has the same nearest entry, EMA
        drags that one entry around, and the other 8191 are never selected
        again. Measured on the first smoke test — codes 1/256 after 30 steps.
        Seeding from the first real batch puts the entries where the data is.
        """
        # Under autocast `flat` arrives as fp16 while the buffers are fp32, and an
        # indexed assignment will not convert — it raises. The CPU smoke test never
        # saw this because AMP is off there, which is exactly the gap that let it
        # reach Kaggle and die 36 seconds in.
        flat = flat.to(self.embed.dtype)
        n = flat.shape[0]
        idx = torch.randint(0, n, (self.n_codes,), device=flat.device)
        pick = flat[idx]
        if n < self.n_codes:                      # tiny batch: jitter the copies
            pick = pick + 0.01 * torch.randn_like(pick)
        self.embed.copy_(pick)
        self.embed_avg.copy_(pick)
        self.cluster_size.fill_(1.0)
        self.inited.fill_(1.0)

    def _restart_dead(self, flat):
        """Reseed entries nothing has selected lately from live encoder output.

        Without this the vocabulary only ever shrinks: an unselected entry gets
        no gradient and no EMA update, so it can never come back on its own. A
        codebook that decays to a few dozen live entries produces uniform mush —
        and the reconstruction loss keeps falling the whole time, so the curve
        will not tell you."""
        flat = flat.to(self.embed.dtype)
        dead = self.cluster_size < self.restart_below
        k = int(dead.sum())
        if k == 0:
            return 0
        idx = torch.randint(0, flat.shape[0], (k,), device=flat.device)
        self.embed[dead] = flat[idx]
        self.embed_avg[dead] = flat[idx]
        self.cluster_size[dead] = 1.0
        return k

    def forward(self, z):
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)

        if self.training and float(self.inited) == 0.0:
            self._seed_from(flat.detach())

        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embed.t()
             + self.embed.pow(2).sum(1))
        idx = d.argmin(1)
        q = self.embed[idx].view(b, h, w, c).permute(0, 3, 1, 2)

        if self.training:
            # Under no_grad because these buffers are statistics, not parameters:
            # autograd has no business tracking them, and without this the
            # in-place update trips "a view is being modified inplace" as soon as
            # anything upstream broadcasts them.
            with torch.no_grad():
                flat32 = flat.detach().to(self.embed.dtype)
                onehot = F.one_hot(idx, self.n_codes).type(flat32.dtype)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0),
                                                        alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat32,
                                                     alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster = ((self.cluster_size + self.eps)
                           / (n + self.n_codes * self.eps) * n)
                self.embed.copy_(self.embed_avg / cluster.unsqueeze(1))
                self.steps += 1
                if int(self.steps) % self.restart_every == 0:
                    self._restart_dead(flat32)

        # Commitment only: the codebook itself moves by EMA, so the encoder is
        # the only thing this gradient should be pulling on.
        loss = F.mse_loss(q.detach(), z)
        # Straight-through: the decoder sees quantized values, the encoder gets
        # the gradient as if nothing had been rounded.
        q = z + (q - z).detach()
        return q, loss, idx.view(b, h, w)

    def lookup(self, idx):
        """Token grid -> latent, for decoding what the transformer dreams up."""
        b, h, w = idx.shape
        return self.embed[idx.reshape(-1)].view(b, h, w, self.dim).permute(0, 3, 1, 2)


class _Down(nn.Module):
    """Encoder with one more downsample than 2.2's: 256px -> 16x16, not 32x32."""

    def __init__(self, base, mults, dim):
        super().__init__()
        self.conv_in = nn.Conv2d(3, base, 3, padding=1)
        blocks, ch = [], base
        for m in mults:
            out = base * m
            blocks += [ResBlock(ch, out), ResBlock(out, out),
                       nn.Conv2d(out, out, 3, stride=2, padding=1)]
            ch = out
        self.blocks = nn.Sequential(*blocks)
        self.mid = nn.Sequential(ResBlock(ch, ch), AttnBlock(ch), ResBlock(ch, ch))
        self.norm = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, dim, 1)

    def forward(self, x):
        h = self.mid(self.blocks(self.conv_in(x)))
        return self.out(F.silu(self.norm(h)))


class _Up(nn.Module):
    """Codes back to pixels.

    `n_res` and `attn_levels` exist because of a measurement, not a preference.
    A public VQGAN reconstructing the SAME 16x16 grid with a SMALLER codebook
    (1024 against our 8192) recovered fur and window frames where ours produced
    smears. Profiling both decoders side by side, the differences were: 42.4M
    parameters against 10.3M, 17 residual blocks against 10, and — the part that
    is not just "bigger" — **four attention blocks against one**.

    Ours attends only at the bottleneck. Theirs attends at several resolutions,
    so it can relate distant parts of the picture while it still has the detail
    to act on. On a corpus full of repeated texture and lettering that plausibly
    matters more than width alone, which is why it is adjustable here rather
    than fixed.
    """

    def __init__(self, base, mults, dim, n_res=2, attn_levels=0):
        super().__init__()
        ch = base * mults[-1]
        self.conv_in = nn.Conv2d(dim, ch, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(ch, ch), AttnBlock(ch), ResBlock(ch, ch))
        blocks = []
        # Levels run coarse to fine, and attention is spent on the coarse ones:
        # at 16x16 it costs 256 positions, at 256x256 it would cost 65536 and be
        # quadratic in that.
        for level, m in enumerate(reversed(mults)):
            out = base * m
            blocks.append(ResBlock(ch, out))
            for _ in range(n_res - 1):
                blocks.append(ResBlock(out, out))
            if level < attn_levels:
                blocks.append(AttnBlock(out))
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(out, out, 3, padding=1)]
            ch = out
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z):
        return self.out(F.silu(self.norm(self.blocks(self.mid(self.conv_in(z))))))


class VQVAE(nn.Module):
    """256x256x3 <-> 16x16 integer tokens."""

    def __init__(self, base=64, mults=(1, 2, 4, 4), dim=256, n_codes=8192,
                 dec_base=None, dec_res=2, dec_attn=0):
        """`dec_base` widens the DECODER alone.

        The encoder and the codebook define what a token id means, and the
        transformer is trained against those meanings — so they must not move
        once it exists. The decoder only renders those ids into pixels, which
        makes it the one part that can grow freely.

        That matters because capacity is where the measured ceiling is. At the
        same 16x16 grid and with a SMALLER codebook (1024 against our 8192), a
        public VQGAN reconstructed fur and window frames where ours produced
        smears — and the difference between them is width: 72M against 19.5M.

        Defaults to `base`, so every checkpoint written before this argument
        existed loads unchanged.
        """
        super().__init__()
        dec_base = base if dec_base is None else dec_base
        self.encoder = _Down(base, mults, dim)
        self.quant = VectorQuantizer(n_codes, dim)
        self.decoder = _Up(dec_base, mults, dim, n_res=dec_res,
                           attn_levels=dec_attn)
        self.arch = dict(base=base, mults=tuple(mults), dim=dim, n_codes=n_codes)
        if dec_base != base:
            self.arch["dec_base"] = dec_base
        if dec_res != 2:
            self.arch["dec_res"] = dec_res
        if dec_attn:
            self.arch["dec_attn"] = dec_attn

    def encode(self, x):
        _, _, idx = self.quant(self.encoder(x))
        return idx

    def decode(self, idx):
        return self.decoder(self.quant.lookup(idx))

    def forward(self, x):
        z = self.encoder(x)
        q, commit, idx = self.quant(z)
        return self.decoder(q), commit, idx
