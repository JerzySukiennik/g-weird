"""PatchGAN discriminator — the piece that stops reconstructions being blurry.

Why this exists, measured rather than assumed: training the VQ-VAE on L1 alone
from step 20000 to 45000 cost 5.08 h of GPU and moved the reconstruction error
from 0.1574 to 0.1527. Three percent. The pictures were visually identical.

That is not slow convergence, it is the loss doing exactly what it is asked. L1
is minimised by the AVERAGE of every plausible pixel value, and the average of
many sharp possibilities is a blur. No amount of further training escapes it,
because the model is already at the optimum of the wrong objective.

A discriminator changes the objective. It looks at patches and judges whether
they look like real photographs, so the decoder is rewarded for COMMITTING to
one sharp answer instead of hedging across all of them. This is the "GAN" in
VQGAN, and it is why every serious image tokenizer has one.

Patches, not whole images, on purpose: a 70x70 receptive field judges local
texture, which is precisely what is missing. A whole-image discriminator would
also judge composition — which the encoder already preserves fine — and would be
far easier for the generator to fool.

Trained from scratch like everything else in the G family.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    """Convolutional real/fake map. Each output cell judges one patch."""

    def __init__(self, in_ch=3, base=64, layers=3):
        super().__init__()
        seq = [nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
               nn.LeakyReLU(0.2, inplace=True)]
        ch = base
        for i in range(1, layers + 1):
            out = base * min(2 ** i, 8)
            stride = 2 if i < layers else 1
            seq += [nn.Conv2d(ch, out, 4, stride=stride, padding=1, bias=False),
                    # GroupNorm, not BatchNorm: batch statistics make the
                    # discriminator's verdict depend on what else happens to be
                    # in the batch, which is a nasty source of instability when
                    # the batch is small.
                    nn.GroupNorm(8, out),
                    nn.LeakyReLU(0.2, inplace=True)]
            ch = out
        seq += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*seq)

    def forward(self, x, features=False):
        """`features=True` also returns the activation after each LeakyReLU.

        Those intermediate maps are what feature matching compares. Asking the
        discriminator only "real or fake" gives the generator a single scalar to
        chase, and chasing it is what produced our two failures: at cap 0.5 the
        decoder found texture that satisfied the verdict while faces melted.
        Matching what the discriminator SEES on the way to its verdict is a much
        denser signal, and it cannot be satisfied by inventing plausible grain
        in the wrong places.
        """
        if not features:
            return self.net(x)
        feats = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return x, feats


def feature_match_loss(real_feats, fake_feats):
    """Mean L1 between the discriminator's own activations.

    The real side is detached: this is a target for the generator, not another
    thing for the discriminator to learn.
    """
    return sum(F.l1_loss(f, r.detach())
               for f, r in zip(fake_feats, real_feats)) / max(len(real_feats), 1)


def hinge_d_loss(real_logits, fake_logits):
    """Standard VQGAN discriminator loss.

    Hinge rather than cross-entropy because it stops pushing once a patch is
    confidently classified, which keeps the discriminator from racing ahead of
    the generator and turning the whole thing into noise.
    """
    return 0.5 * (torch.relu(1.0 - real_logits).mean()
                  + torch.relu(1.0 + fake_logits).mean())


def g_adv_loss(fake_logits):
    return -fake_logits.mean()


def adaptive_weight(rec_loss, adv_loss, last_layer, max_weight=1e4):
    """Balance the two objectives by their gradient magnitudes at the output.

    A fixed adversarial weight is a guess that goes stale: early on the
    reconstruction gradient dwarfs the adversarial one, later the reverse. VQGAN
    computes the ratio against the decoder's final layer every step, which keeps
    the discriminator from overwhelming the thing it is meant to sharpen.
    """
    rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    adv_grad = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]
    w = rec_grad.norm() / (adv_grad.norm() + 1e-4)
    return w.clamp(0.0, max_weight).detach()
