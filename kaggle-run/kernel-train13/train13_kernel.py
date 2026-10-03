"""Kaggle GPU cell: G-Weird 1.3, autoregressive, 256px / 1024 tokens.

The drawing half of the hybrid. Autoregression is the architecture that has
already shown objects here (1.2 had a horse at 70000 steps; masked-from-scratch
had none at 50000), so it draws; a short-trained fill-in model does the editing.

Corpus: three groups of kernel outputs from enc1024 — DiffusionDB 0-3 (~1.0M),
coco + jdb + FLUX 0-1 + DALL-E 3 (~1.04M), FLUX 2-3 + CC12M (~1.0M) — 3,026,852
pairs. DiffusionDB 4-7 is deliberately left out: those are Stable Diffusion 1.x
images, the source of the melted look this model is meant to get rid of, and
leaving them out takes DiffusionDB from 49% to 33% of the mix.

Tokenizer: the untouched 40000-step checkpoint run at 256px (see vq256).

**Sessions end by the clock.** --max-hours 9 stops the trainer at a step
boundary with a final save, whatever the speed turns out to be. Time per step at
1089 tokens is not measured yet (expected ~2.4 s on T4x2); two earlier sessions
sized by estimate hit Kaggle's 12 h wall and lost the final save. The cost of
that was ~20 GPU hours.

First batch tried is 32 x accum 2 (effective 64, as in 1.2). If that fails
before the first checkpoint — memory at 1089 tokens is estimated, not measured —
it retries once at 16 x accum 4, the same effective batch.

Checkpoint versions are published from the Mac after each session, so the next
one resumes from the dataset and never from a cancelled run's output.
"""

import glob
import json
import os
import subprocess
import sys

import torch

REPO = "https://github.com/JerzySukiennik/g-weird.git"
INPUT = os.environ.get("GW_INPUT", "/kaggle/input")
WORK = os.environ.get("GW_WORK", "/kaggle/working")
EXPECT_SHARDS = int(os.environ.get("GW_EXPECT_SHARDS", "3"))
EXPECT_N = int(os.environ.get("GW_EXPECT_N", "3026852"))
MAX_HOURS = float(os.environ.get("GW_MAX_HOURS", "9"))
STEPS_TOTAL = 100000
MAX_STEPS = 60000          # tylko bezpiecznik; o koncu sesji decyduje zegar
DATASET = "jerzysukiennik/gweird-13-ar"
PER = 1024
PROMPTS = ["a horse standing in a field", "a red double decker bus on a street",
           "a cat wearing sunglasses", "portrait of an old man with a beard",
           "a bowl of soup on a wooden table", "a castle on a mountain at sunset",
           "two people riding bicycles", "a robot playing a piano"]
HARD = ["a horse standing in a field", "a cat wearing sunglasses"]

if not torch.cuda.is_available() and not os.environ.get("GW_ALLOW_CPU"):
    raise SystemExit("brak GPU")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}, "
          f"compute {cap[0]}.{cap[1]}", flush=True)
    if cap[0] < 7:
        raise SystemExit("bez rdzeni tensor — nie palmy na to kwoty")

if os.environ.get("GW_REPO_DIR"):
    os.chdir(os.environ["GW_REPO_DIR"])
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-weird"], check=True)
    os.chdir(f"{WORK}/g-weird")

metas = sorted(glob.glob(f"{INPUT}/**/gwtok1024*_meta.json", recursive=True)) \
    or sorted(glob.glob(f"{INPUT}/**/*_meta.json", recursive=True))
prefixes = [m[: -len("_meta.json")] for m in metas]
if len(prefixes) != EXPECT_SHARDS:
    raise SystemExit(f"oczekiwalem {EXPECT_SHARDS} shardow, widze {len(prefixes)}: {prefixes}")
n = 0
for p in prefixes:
    meta = json.load(open(f"{p}_meta.json"))
    if meta.get("per_image") != PER:
        raise SystemExit(f"{p}: {meta.get('per_image')} tokenow na obraz, oczekiwalem {PER}")
    size = os.path.getsize(f"{p}_tokens.u16")
    if size % (PER * 2):
        raise SystemExit(f"{p}: {size} B nie dzieli sie na obrazy po {PER * 2} B")
    n += size // (PER * 2)
print(f"korpus: {n:,} par w {len(prefixes)} shardach", flush=True)
if n != EXPECT_N:
    raise SystemExit(f"korpus ma {n} par, oczekiwalem {EXPECT_N}")

txt = glob.glob(f"{INPUT}/**/text.json", recursive=True)
vqs = glob.glob(f"{INPUT}/**/vqvae.pt", recursive=True)
if len(txt) != 1 or len(vqs) != 1:
    raise SystemExit(f"wejscia: text {txt}, vqvae {vqs}")
vstep = torch.load(vqs[0], map_location="cpu", weights_only=False)["step"]
if vstep != 40000:
    raise SystemExit(f"vqvae z kroku {vstep}, oczekiwalem zamrozonego 40000")

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

# Wznowienie: najwyzszy krok sposrod podpietych checkpointow; brak = od zera.
os.makedirs(f"{WORK}/run", exist_ok=True)
cks = glob.glob(f"{INPUT}/**/gweird.pt", recursive=True)
before = 0
if cks:
    steps = {c: torch.load(c, map_location="cpu", weights_only=False)["step"] for c in cks}
    for c, st in sorted(steps.items(), key=lambda kv: kv[1]):
        print(f"  checkpoint krok {st}: {c}", flush=True)
    best = max(steps, key=steps.get)
    before = steps[best]
    subprocess.run(["cp", best, f"{WORK}/run/gweird.pt"], check=True)
    print(f"wznawiam z kroku {before}", flush=True)
else:
    print("pierwsza sesja — od zera", flush=True)


def current_step():
    p = f"{WORK}/run/gweird.pt"
    return torch.load(p, map_location="cpu", weights_only=False)["step"] \
        if os.path.exists(p) else 0


def train(batch, accum, extra=()):
    return subprocess.run(
        [sys.executable, "train/train_ar.py", "--data", *local, "--out", f"{WORK}/run",
         "--steps", str(STEPS_TOTAL), "--max-steps", str(MAX_STEPS),
         "--batch", str(batch), "--accum", str(accum), "--lr", "3e-4",
         "--warmup", "2000", "--workers", os.environ.get("GW_WORKERS", "4"),
         "--log-every", "100",
         "--ckpt-every", "1000", "--max-hours", str(MAX_HOURS), "--resume",
         *extra]).returncode


# GW_FORCE_FIRST_FAIL: tylko do testu lokalnego. Ponowienie jako 16 x 4 to jedyna
# siatka bezpieczenstwa na pamiec przy 1089 tokenach (nie zmierzona), wiec jest
# uruchamiana przed pierwszym prawdziwym biegiem, nie dopiero w nim.
rc = train(32, 2, ["--wymuszony-blad"] if os.environ.get("GW_FORCE_FIRST_FAIL") else [])
if rc != 0 and current_step() == before:
    print(f"pierwsza proba (32 x 2) padla kodem {rc} bez zadnego kroku — "
          f"ponawiam jako 16 x 4 (ta sama efektywna partia)", flush=True)
    rc = train(16, 4)
if rc != 0:
    raise SystemExit(f"trening padl kodem {rc} — nie wysylam checkpointu")

after = current_step()
if after <= before:
    raise SystemExit(f"krok nie ruszyl: {before} -> {after}")
print(f"krok {before} -> {after}", flush=True)

if not os.environ.get("GW_SKIP_SAMPLES"):
    # Osiem podpisow na jednym ziarnie, plus dwa najtrudniejsze na trzech
    # kolejnych: jeden obraz na podpis to anegdota, nie ocena.
    subprocess.run([sys.executable, "train/sample.py", "--ckpt", f"{WORK}/run/gweird.pt",
                    "--vqvae", vqs[0], "--tokenizer", txt[0], "--seed", "0",
                    "--out", f"{WORK}/proba-{after}.png", "--prompts", *PROMPTS], check=False)
    for seed in (1, 2, 3):
        subprocess.run([sys.executable, "train/sample.py", "--ckpt", f"{WORK}/run/gweird.pt",
                        "--vqvae", vqs[0], "--tokenizer", txt[0], "--seed", str(seed),
                        "--out", f"{WORK}/proba-{after}-ziarno{seed}.png",
                        "--prompts", *HARD], check=False)

# Wersja datasetu z checkpointem, jesli kernel ma sekret; inaczej wersjonuje
# ja Mac po zakonczeniu sesji. Token z sekretu, nie z kodu: repo jest publiczne
# i juz raz wyciekl przez nie token.
tok = None
if not os.environ.get("GW_ALLOW_CPU"):
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
    for pth in glob.glob(f"{WORK}/proba-*.png"):
        subprocess.run(["cp", pth, f"{WORK}/up/"], check=False)
    json.dump({"id": DATASET, "title": "gweird 13 ar",
               "licenses": [{"name": "CC0-1.0"}]},
              open(f"{WORK}/up/dataset-metadata.json", "w"))
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets",
                        "version" if before else "create", "-p", f"{WORK}/up",
                        *(["-m", f"krok {after}"] if before else []), "-q"],
                       capture_output=True, text=True)
    print("kaggle:", r.returncode, r.stdout[-300:], r.stderr[-300:], flush=True)
    subprocess.run(["rm", "-rf", f"{WORK}/up"], check=False)
else:
    print("checkpoint zostaje w wyjsciu kernela; wersje datasetu robi Mac po "
          "zakonczeniu sesji", flush=True)
subprocess.run(["rm", "-rf", f"{WORK}/data"], check=False)   # dowiazania, nie dane
print("gotowe", flush=True)
