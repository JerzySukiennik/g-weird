#!/bin/bash
# Retry the VQ-VAE push until the weekly GPU quota resets, then stop.
#
# Written so Jurek can sleep through the reset. Kaggle rejects a push outright
# while the quota is spent ("Maximum weekly GPU quota of 30.00 hours reached"),
# which makes the rejection itself a reliable probe — no need to know the exact
# reset time, just keep asking.
#
# Guards for things this project has already been bitten by:
#  - stops on the FIRST success, so a reset does not turn into a push loop;
#  - checks the output text rather than the exit code, because the CLI reports
#    success and failure the same way;
#  - logs every attempt, since nobody will be watching it happen.

set -u
PROJ=/Users/jurek/Downloads/Claude/Projects/AIe/G-Weird
LOG="$PROJ/kaggle-run/wait-and-launch.log"
EVERY=600          # 10 min
MAX=48             # ~8 h of trying, well past a 2h40m reset

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$PROJ" || exit 1
say "czekam na reset limitu; probuje co $((EVERY/60)) min, maks $MAX razy"

for i in $(seq 1 $MAX); do
  out=$("$PROJ/kaggle-run/kg" kernels push -p "$PROJ/kaggle-run/kernel-vqvae" 2>&1 | tail -1)
  case "$out" in
    *"successfully pushed"*)
      say "PUSZCZONE (proba $i): $out"
      sleep 240
      st=$("$PROJ/kaggle-run/kg" kernels status jerzysukiennik/gweird-vqvae 2>/dev/null \
             | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1)
      say "status po starcie: $st"
      exit 0
      ;;
    *"quota"*)
      say "proba $i: limit wciaz wyczerpany"
      ;;
    *)
      say "proba $i: nieoczekiwana odpowiedz: $out"
      ;;
  esac
  sleep "$EVERY"
done
say "poddaje sie po $MAX probach — limit nie wrocil"
