"""Kaggle CPU cell: fetch text-to-image-2M shardy 11-21 into shard format for the 1.2 corpus.

CPU on purpose. Kaggle CPU notebooks have no weekly quota — only GPU hours are
capped, and all 45 of those are needed for encoding and training. Fetching and
transcoding therefore costs nothing, which is what makes the whole data plan
affordable.

Shards stream to disk one at a time and are deleted after use: they are ~3.5 GB
each against a 20 GB working disk, and the output is growing in the same place.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

subprocess.run([sys.executable, "data/fetch_webdataset.py",
                "--source", "t2i2m",
                "--first-shard", "11", "--shards", "11",
                "--res", "256", "--max-images", "480000",
                "--out-prefix", f"{WORK}/gweird"], check=True)

# Rozmiar w logu, zeby dalo sie zobaczyc, czy kernel zmiescil sie w limicie
# dysku, zanim ktos oprze na tym kodowanie.
for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"{f}: {os.path.getsize(p)/1e9:.2f} GB", flush=True)
