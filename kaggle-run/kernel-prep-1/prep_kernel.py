"""Bootstrap for the G-Weird caption corpus, shard 1. CPU only — no GPU quota."""

import os
import subprocess
import sys

os.environ["GWEIRD_SHARD"] = "1"
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/JerzySukiennik/g-weird.git", "/tmp/gw"], check=True)
subprocess.run([sys.executable, "/tmp/gw/kaggle/01-prep.py"], check=True)
