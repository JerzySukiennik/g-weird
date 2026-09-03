"""Kaggle CPU cell: text-to-image-2M, gęste opisy VLM.

CPU on purpose: Kaggle caps GPU hours weekly but not CPU notebooks, so fetching
and transcoding costs nothing out of the 45 hours that encoding and training
need.

**Sized to the disk, not to ambition.** The shards turned out to be 7-11 GB
each, not the 3.5 GB the first attempt assumed — that number came from a
download that had itself been truncated. One shard plus the growing output has
to fit in 20 GB, so each kernel takes 250000 images and no more.
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
                "--first-shard", "14", "--shards", "7",
                "--res", "256", "--max-images", "250000",
                "--out-prefix", f"{WORK}/gweird"], check=True)

for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"{f}: {os.path.getsize(p)/1e9:.2f} GB", flush=True)
