"""Kaggle GPU cell: continue G-Weird 1.2 from the latest checkpoint on Kaggle.

The A100 session did the first 70000 steps; this is the workhorse for the
remaining 230000 on T4x2. Same code path as the Colab cell, one difference —
nothing is downloaded here. The four token shards are attached as kernel
outputs and the checkpoint, tokenizer and VQ-VAE as datasets, so a session
starts training within minutes instead of after 4 GB of transfer.

Resumes from whatever version of the checkpoint dataset is attached. Two
training kernels must never run at once: both would resume from the same step
and the second upload would silently overwrite the first.
"""

import glob
import json
import os
import subprocess
import sys

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
WORK = "/kaggle/working"
STEPS_TOTAL = 300000
MAX_STEPS = 30000          # zmierzone 1.21 s/krok na T4x2 -> ~10 h. 36000 wyszlo 11.9 h i sesja zostala ubita o 12 h bez zapisu koncowego i bez probki
DATASET = "jerzysukiennik/gweird-12-ar"
PROMPTS = ["a horse standing in a field", "a red double decker bus on a street",
           "a cat wearing sunglasses", "portrait of an old man with a beard",
           "a bowl of soup on a wooden table", "a castle on a mountain at sunset",
           "two people riding bicycles", "a robot playing a piano"]

if not torch.cuda.is_available():
    raise SystemExit("brak GPU")
cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}, "
      f"compute {cap[0]}.{cap[1]}", flush=True)
if cap[0] < 7:
    raise SystemExit("bez rdzeni tensor — nie palmy na to kwoty")

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob("/kaggle/input/**/gwtok*_meta.json", recursive=True))
prefixes = [m[: -len("_meta.json")] for m in metas]
if len(prefixes) != 4:
    raise SystemExit(f"oczekiwalem 4 shardow, widze {len(prefixes)}: {prefixes}")
n = sum(os.path.getsize(f"{p}_tokens.u16") // 1152 for p in prefixes)
print(f"korpus: {n:,} par w {len(prefixes)} shardach", flush=True)
if n != 4017878:
    raise SystemExit(f"korpus ma {n} par, oczekiwalem 4017878")

txt = glob.glob("/kaggle/input/**/text.json", recursive=True)
vqs = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
cks = glob.glob("/kaggle/input/**/gweird.pt", recursive=True)
if len(txt) != 1 or len(vqs) != 1 or not cks:
    raise SystemExit(f"wejscia: text {txt}, vqvae {vqs}, ckpt {cks}")
# Kilka checkpointow naraz jest normalne: dataset z Colaba plus wyjscie
# poprzedniego kernela. Wygrywa najwyzszy krok, wiec kolejne sesje lancuchuja
# sie przez wyjscia kerneli bez sekretow i bez recznego wersjonowania datasetu.
steps = {c: torch.load(c, map_location="cpu", weights_only=False)["step"] for c in cks}
for c, st in sorted(steps.items(), key=lambda kv: kv[1]):
    print(f"  checkpoint krok {st}: {c}", flush=True)
cks = [max(steps, key=steps.get)]

# Wejscia sa tylko do odczytu, a pack_captions pisze obok tokenow — wiec
# shardy dostaja dowiazania w katalogu roboczym, a tablice podpisow laduja tam.
os.makedirs(f"{WORK}/data", exist_ok=True)
local = []
for p in prefixes:
    tag = os.path.basename(p)
    for suf in ("tokens.u16", "captions.json", "meta.json"):
        dst = f"{WORK}/data/{tag}_{suf}"
        if not os.path.exists(dst):
            os.symlink(f"{p}_{suf}", dst)
    local.append(f"{WORK}/data/{tag}")
subprocess.run([sys.executable, "data/pack_captions.py", "--data", *local,
                "--tokenizer", txt[0], "--text-len", "64"], check=True)

os.makedirs(f"{WORK}/run", exist_ok=True)
subprocess.run(["cp", cks[0], f"{WORK}/run/gweird.pt"], check=True)
before = torch.load(f"{WORK}/run/gweird.pt", map_location="cpu", weights_only=False)["step"]
print(f"wznawiam z kroku {before}", flush=True)

subprocess.run([sys.executable, "train/train_ar.py", "--data", *local,
                "--out", f"{WORK}/run", "--steps", str(STEPS_TOTAL),
                "--max-steps", str(MAX_STEPS), "--batch", "32", "--accum", "2",
                "--lr", "3e-4", "--warmup", "2000", "--workers", "4",
                "--log-every", "100", "--ckpt-every", "1000", "--resume"], check=True)

after = torch.load(f"{WORK}/run/gweird.pt", map_location="cpu", weights_only=False)["step"]
if after <= before:
    raise SystemExit(f"krok nie ruszyl: {before} -> {after}")
print(f"krok {before} -> {after}", flush=True)

subprocess.run([sys.executable, "train/sample.py", "--ckpt", f"{WORK}/run/gweird.pt",
                "--vqvae", vqs[0], "--tokenizer", txt[0],
                "--out", f"{WORK}/proba-{after}.png", "--prompts", *PROMPTS], check=False)

# Nowa wersja datasetu z checkpointem. Token z sekretu kernela, nie z kodu:
# repo jest publiczne i juz raz wyciekl przez nie token.
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("KAGGLE_ACCESS_TOKEN")
except Exception as e:
    print("brak sekretu KAGGLE_ACCESS_TOKEN:", e, flush=True)
if tok:
    os.makedirs("/root/.kaggle", exist_ok=True)
    open("/root/.kaggle/access_token", "w").write(tok)
    os.chmod("/root/.kaggle/access_token", 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = "/root/.kaggle"
    subprocess.run(["pip", "install", "-q", "-U", "kaggle"], check=False)
    os.makedirs(f"{WORK}/up", exist_ok=True)
    subprocess.run(["cp", f"{WORK}/run/gweird.pt", f"{WORK}/up/"], check=True)
    for p in glob.glob(f"{WORK}/proba-*.png"):
        subprocess.run(["cp", p, f"{WORK}/up/"], check=False)
    json.dump({"id": DATASET, "title": "gweird 12 ar",
               "licenses": [{"name": "CC0-1.0"}]},
              open(f"{WORK}/up/dataset-metadata.json", "w"))
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version",
                        "-p", f"{WORK}/up", "-m", f"krok {after}", "-q"],
                       capture_output=True, text=True)
    print("kaggle:", r.returncode, r.stdout[-300:], r.stderr[-300:], flush=True)
    subprocess.run(["rm", "-rf", f"{WORK}/up"], check=False)
else:
    print("checkpoint zostaje w wyjsciu kernela; wersje datasetu trzeba "
          "dodac recznie", flush=True)
