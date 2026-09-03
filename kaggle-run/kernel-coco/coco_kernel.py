"""Kaggle CPU cell: COCO photographs into a G-Weird shard. GPU OFF, Internet ON.

CPU only, so it costs nothing from the GPU quota. About 19.5 GB comes down and
~2.4 GB of 256px JPEG goes back out.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow", "requests"],
               check=True)
subprocess.run([sys.executable, "data/fetch_coco.py", "--res", "256",
                "--per-image", "2", "--out-prefix", f"{WORK}/gweird"], check=True)
for f in sorted(os.listdir(WORK)):
    p = os.path.join(WORK, f)
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB", flush=True)
