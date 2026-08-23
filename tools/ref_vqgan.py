"""A yardstick, NOT part of G-Weird — a public VQGAN used only to measure.

Nothing here ever touches the model. The question it answers is narrow: our
decoder turns 256 codes back into soft, oil-painted pictures, and before
spending GPU hours sharpening it we want to know whether 256 codes can BE sharp.
A public VQGAN at the same compression (f16: 256x256 -> 16x16 codes, the family
DALL-E mini used) settles that. If its reconstructions are crisp there is room
to gain; if they are soft too, no amount of adversarial training will help and
the hours are better spent elsewhere.

Its codebook is 1024 entries against our 8192, so it carries LESS information per
image than we do — which makes it a conservative yardstick: whatever it manages,
we should be able to match.

This is the standard taming-transformers architecture, reimplemented here only so
the published weights load. Correctness is checked by load_state_dict(strict=True):
a single wrong name or shape and it refuses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(ch):
    return nn.GroupNorm(32, ch, eps=1e-6, affine=True)


def swish(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = Normalize(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.use_shortcut = in_ch != out_ch
        if self.use_shortcut:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.use_shortcut:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = Normalize(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)
        k = k.reshape(b, c, hh * ww)
        w = torch.bmm(q, k) * (c ** -0.5)
        w = torch.softmax(w, dim=2).permute(0, 2, 1)
        v = v.reshape(b, c, hh * ww)
        h = torch.bmm(v, w).reshape(b, c, hh, ww)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, attn_res, z_ch, res):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, 1, 1)
        self.num_res = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.down = nn.ModuleList()
        cur, in_ch = res, ch
        for i in range(self.num_res):
            out_ch = ch * ch_mult[i]
            block, attn = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
                if cur in attn_res:
                    attn.append(AttnBlock(in_ch))
            stage = nn.Module()
            stage.block, stage.attn = block, attn
            if i != self.num_res - 1:
                stage.downsample = Downsample(in_ch)
                cur //= 2
            self.down.append(stage)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch)
        self.mid.attn_1 = AttnBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch)
        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, z_ch, 3, 1, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for i, stage in enumerate(self.down):
            for j, blk in enumerate(stage.block):
                h = blk(h)
                if len(stage.attn) > j:
                    h = stage.attn[j](h)
            if hasattr(stage, "downsample"):
                h = stage.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(swish(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, attn_res, z_ch, res):
        super().__init__()
        self.num_res = len(ch_mult)
        in_ch = ch * ch_mult[-1]
        cur = res // 2 ** (self.num_res - 1)
        self.conv_in = nn.Conv2d(z_ch, in_ch, 3, 1, 1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_ch, in_ch)
        self.mid.attn_1 = AttnBlock(in_ch)
        self.mid.block_2 = ResnetBlock(in_ch, in_ch)
        # Stored lowest-resolution-last so indices match the published names:
        # taming builds this list in reverse and indexes it with the same i.
        self.up = nn.ModuleList([None] * self.num_res)
        for i in reversed(range(self.num_res)):
            out_ch = ch * ch_mult[i]
            block, attn = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
                if cur in attn_res:
                    attn.append(AttnBlock(in_ch))
            stage = nn.Module()
            stage.block, stage.attn = block, attn
            if i != 0:
                stage.upsample = Upsample(in_ch)
                cur *= 2
            self.up[i] = stage
        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv2d(in_ch, 3, 3, 1, 1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for i in reversed(range(self.num_res)):
            stage = self.up[i]
            for j, blk in enumerate(stage.block):
                h = blk(h)
                if len(stage.attn) > j:
                    h = stage.attn[j](h)
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)
        return self.conv_out(swish(self.norm_out(h)))


class Quantize(nn.Module):
    def __init__(self, n_e, e_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)

    def forward(self, z):
        b, c, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(1)
        q = self.embedding(idx).view(b, h, w, c).permute(0, 3, 1, 2)
        return q, idx.view(b, h, w)


class RefVQGAN(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,
                 attn_res=(16,), z_ch=256, embed_dim=256, n_embed=1024, res=256):
        super().__init__()
        self.encoder = Encoder(ch, ch_mult, num_res_blocks, attn_res, z_ch, res)
        self.decoder = Decoder(ch, ch_mult, num_res_blocks, attn_res, z_ch, res)
        self.quantize = Quantize(n_embed, embed_dim)
        self.quant_conv = nn.Conv2d(z_ch, embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_ch, 1)

    @torch.no_grad()
    def reconstruct(self, x):
        q, idx = self.quantize(self.quant_conv(self.encoder(x)))
        return self.decoder(self.post_quant_conv(q)), idx
