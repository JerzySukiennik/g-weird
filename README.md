# G-Weird

Text-to-image, trained from scratch, aiming squarely at the DALL-E mini / Craiyon
look of 2021: melted faces, wrong numbers of eyes, objects that almost cohere.

The uncanny part is the deliverable, not a limitation to apologise for. Every
constraint this project runs under — a small model, 256px, a coarse codebook — is
what produced that aesthetic in the first place.

**Architecture:** own VQ-VAE (no pretrained VQGAN — the G family trains
everything from scratch) plus an autoregressive transformer over image tokens,
with classifier-free guidance so prompts are actually followed. CFG works for
autoregressive models too, via conditioning dropout at training time and a
logit extrapolation at sampling time.

Full plan and measured costs: `ClaudeMemory/projects/g-weird.md`.
