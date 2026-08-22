"""Kaggle GPU cell: finish the corpus encode killed at 83%. T4, Internet ON.

The first attempt ran 4.65 h and died with no traceback — the signature of an OOM
kill, not an exception. The cause was in this pipeline: every caption was held in
a list for the whole run and written only at the end. Tokens were written as they
went, so 1470798 of 1780125 images survived; the captions did not exist at all.

This kernel therefore does not redo the 4.65 h. Two facts make that cheap:

  * the survivors are byte-exact — 753048576 B / 512 B, no partial image;
  * everything still missing lives in ONE shard. Cumulative counts are
    90070 / 90022 / 400001 / 400015 / 400002 / 400015 = 1780125, so prep-5 starts
    at 1380110 and the resume point sits 90688 images into it.

So only prep-5 is needed, and it is fetched over HTTP instead of mounted. That is
not a stylistic choice: this token cannot attach kernel outputs as sources (the
API denies kernels.get, which is what source validation needs), while it can read
their output URLs perfectly well. Downloading one 8 GB shard sidesteps the whole
problem.

Captions for prep-0..4 are pre-seeded into the .jsonl before encoding starts, so
that when encode_corpus.py appends prep-5's the file ends up the same length as
the token stream. Its closing check enforces exactly that, and is the only thing
standing between a silent caption/token misalignment and a model trained on
images paired with the wrong words.
"""

import json
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
TMP = "/kaggle/tmp/prep5"

DONE = 1470798          # images already in the partial token file
SKIP_IN_SHARD = 90688   # DONE - 1380110, the offset inside prep-5
URLS = json.loads(os.environ.get("GW_URLS") or "{}")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
os.makedirs(TMP, exist_ok=True)


def grab(key, dest):
    if key not in URLS:
        raise SystemExit(f"brak linku dla {key}")
    subprocess.run(["curl", "-sSL", "--retry", "3", "-o", dest, URLS[key]], check=True)
    mb = os.path.getsize(dest) / 1e6
    print(f"  {key}: {mb:.1f} MB", flush=True)
    return dest


print("pobieram czesciowe tokeny...", flush=True)
tok = grab("tokens", f"{WORK}/gwtok_tokens.u16")
size, want = os.path.getsize(tok), DONE * 256 * 2
if size != want:
    raise SystemExit(f"czesciowy plik ma {size} B, oczekiwano {want} B — "
                     f"link wygasl albo wskazuje co innego")

print("skladam podpisy shardow 0-4...", flush=True)
seeded = 0
with open(f"{WORK}/gwtok_captions.jsonl", "w") as out:
    for i in range(5):
        p = grab(f"caps{i}", f"{TMP}/caps{i}.json")
        for c in json.load(open(p)):
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
            seeded += 1
        os.remove(p)
print(f"  {seeded} podpisow (oczekiwano 1380110)", flush=True)
if seeded != 1380110:
    raise SystemExit(f"shardy 0-4 daja {seeded} podpisow, nie 1380110")

print("pobieram shard prep-5...", flush=True)
for key, name in [("meta5", "gweird_meta.json"), ("offs5", "gweird_offsets.json"),
                  ("caps5", "gweird_captions.json"), ("img5", "gweird_images.jpgbin")]:
    grab(key, f"{TMP}/{name}")
ckpt = grab("vqvae", f"{WORK}/vqvae.pt")

subprocess.run([sys.executable, "train/encode_corpus.py",
                "--data", f"{TMP}/gweird", "--ckpt", ckpt,
                "--out-prefix", f"{WORK}/gwtok", "--batch", "64",
                "--skip", str(SKIP_IN_SHARD)], check=True)

os.remove(ckpt)
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
