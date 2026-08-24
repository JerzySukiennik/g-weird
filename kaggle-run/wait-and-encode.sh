#!/bin/bash
# Wait for the prep shards, then launch the encoding — unattended.
#
# Kaggle has no scheduler, so the hand-off between "corpus finished downloading"
# and "corpus starts encoding" has to happen from here. Seven and a half hours of
# encoding are headless once started; only the starting needs a machine awake.
#
# The encoding MUST see every shard: prefixes come from a sorted glob, so a shard
# that arrives late would change the order and pair images with other images'
# captions from that point on.
cd "$(dirname "$0")/.." || exit 1
PY=/Users/jurek/Downloads/Claude/Projects/AIe/G-Images/.venv/bin/python
export KAGGLE_CONFIG_DIR=$HOME/.kaggle
LOG=kaggle-run/wait-and-encode.log
WATCH="gweird-ddb-6 gweird-ddb-7 gweird-jdb-0"

say() { echo "[$(date '+%H:%M')] $*" >> "$LOG"; }
say "czekam na: $WATCH"

while true; do
  left=0
  for k in $WATCH; do
    st=$($PY -m kaggle kernels status "jerzysukiennik/$k" 2>&1 | tail -1 \
         | sed 's/.*KernelWorkerStatus\.//;s/"//')
    case "$st" in
      COMPLETE) ;;
      ERROR|CANCEL*) say "$k zakonczyl sie jako $st — ide dalej bez niego" ;;
      *) left=$((left+1)) ;;
    esac
  done
  [ "$left" -eq 0 ] && break
  sleep 300
done

say "wszystkie prepy zakonczone, skladam kernel kodujacy"
SRC=$(ls -d kaggle-run/kernel-ddb-* kaggle-run/kernel-coco kaggle-run/kernel-jdb-0 2>/dev/null)
D=kaggle-run/kernel-encmix
rm -rf "$D" && mkdir -p "$D"
cp kaggle/10-encode-mixed.py "$D/encmix_kernel.py"

# Every finished prep output plus the frozen tokenizer.
$PY - "$D" <<'PY'
import json, subprocess, sys, os
d = sys.argv[1]
PY_BIN = sys.executable
srcs = []
for s in list(range(8)):
    srcs.append(f"jerzysukiennik/gweird-ddb-{s}")
srcs += ["jerzysukiennik/gweird-coco", "jerzysukiennik/gweird-jdb-0",
         "jerzysukiennik/gweird-vqvae"]
json.dump({"id": "jerzysukiennik/gweird-encmix", "title": "gweird encmix",
           "code_file": "encmix_kernel.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True,
           "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [], "competition_sources": [],
           "kernel_sources": srcs, "model_sources": []},
          open(os.path.join(d, "kernel-metadata.json"), "w"), indent=2)
print("zrodel:", len(srcs))
PY

OUT=$($PY -m kaggle kernels push -p "$D" 2>&1 | tail -2)
say "push: $OUT"
say "gotowe — kodowanie ruszylo, dalej leci bez tego Maca"
