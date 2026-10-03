"""Kaggle GPU cell: the masked model on the 256px / 1024-token corpus, warm-started.

One model for drawing, for region editing and for speed (12 parallel rounds
instead of 1024 sequential passes). An earlier version of this plan split the job
in two — autoregression to draw, a masked model to edit — because masked-from-
scratch looked unable to draw. That evidence came from a sampler whose Gumbel
noise was NaN for every input (2026-09-03 to 2026-10-03), so the ranking of
"least confident" tokens never ran. With the sampler fixed, the same 12000-step
duel checkpoint is on par with autoregression at equal steps, and the 50000-step
Live checkpoint draws a red vehicle on a road, a person, a castle silhouette.

Warm start: the tokenizer is the untouched 40000-step checkpoint, so the codebook
means the same thing at 256px as at 192px, and the model has RoPE instead of
absolute positions, so a 576-token grid and a 1024-token grid differ only in
sequence length. The Live weights (~35 GPU hours) become the starting point.
Two checkpoints share the name maskgit.pt and are told apart by the dataset in
their path: gweird-live-mg is the source, gweird-live1024-mg is this run's own.

Corpus: three enc1024 outputs, 3,026,852 pairs, DiffusionDB 4-7 left out (Stable
Diffusion 1.x images, the source of the melted look this is meant to lose).

Sessions end by the clock (--max-hours 9). 32 x 2 first; if that fails before any
step, 16 x 4 — memory at 1088 tokens is estimated, not measured.
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
DATASET = "jerzysukiennik/gweird-live1024-mg"
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

# Dwa rozne checkpointy o tej samej nazwie, rozroznione po datasecie w sciezce.
own = sorted(glob.glob(f"{INPUT}/**/gweird-live1024-mg/maskgit.pt", recursive=True))
src = sorted(glob.glob(f"{INPUT}/**/gweird-live-mg/maskgit.pt", recursive=True))
os.makedirs(f"{WORK}/run", exist_ok=True)
before = 0
extra = []
if own:
    if len(own) != 1:
        raise SystemExit(f"wiele checkpointow tej sesji: {own}")
    subprocess.run(["cp", own[0], f"{WORK}/run/maskgit.pt"], check=True)
    before = torch.load(f"{WORK}/run/maskgit.pt", map_location="cpu",
                        weights_only=False)["step"]
    print(f"wznawiam wlasny checkpoint z kroku {before}", flush=True)
elif src:
    if len(src) != 1:
        raise SystemExit(f"wiele checkpointow zrodlowych: {src}")
    sck = torch.load(src[0], map_location="cpu", weights_only=False)
    print(f"start z wag Live, krok zrodlowy {sck['step']}, "
          f"image_len {sck['cfg']['image_len']}", flush=True)
    if sck["cfg"]["image_len"] != 576 or sck["cfg"]["text_len"] != 64:
        raise SystemExit("checkpoint zrodlowy to nie Live 576 / 64")
    extra = ["--init-from", src[0]]
else:
    print("brak checkpointu — od zera", flush=True)


def current_step():
    p = f"{WORK}/run/maskgit.pt"
    return torch.load(p, map_location="cpu", weights_only=False)["step"] \
        if os.path.exists(p) else 0


def train(batch, accum, more=()):
    return subprocess.run(
        [sys.executable, "train/train_maskgit.py", "--data", *local,
         "--out", f"{WORK}/run", "--steps", str(STEPS_TOTAL),
         "--max-steps", str(MAX_STEPS), "--batch", str(batch), "--accum", str(accum),
         "--lr", "3e-4", "--warmup", "1000", "--label-smoothing", "0.1",
         # 40% przykladow z ciaglymi prostokatami: bez tego nowy opis nie zmienia
         # tresci zamalowanego obszaru (sprawdzone na Live 50k: kontekst
         # zachowany, tekst ignorowany).
         "--region-p", "0.4",
         "--workers", os.environ.get("GW_WORKERS", "4"), "--log-every", "100",
         "--ckpt-every", "1000", "--max-hours", str(MAX_HOURS), "--resume",
         *extra, *more]).returncode


# GW_FORCE_FIRST_FAIL: tylko test lokalny. Ponowienie jako 16 x 4 to jedyna siatka
# bezpieczenstwa na pamiec przy 1088 tokenach, wiec ma byc przetestowana przed
# pierwszym prawdziwym biegiem.
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
    # Domyslne ustawienia samplera (12 rund, prowadzenie 4, temperatura 1,0).
    # Osiem podpisow na jednym ziarnie plus dwa najtrudniejsze na trzech
    # kolejnych: jeden obraz na podpis to anegdota, nie ocena.
    subprocess.run([sys.executable, "train/sample_maskgit.py",
                    "--ckpt", f"{WORK}/run/maskgit.pt", "--vqvae", vqs[0],
                    "--tokenizer", txt[0], "--seed", "0",
                    "--out", f"{WORK}/proba-live1024-{after}.png",
                    "--prompts", *PROMPTS], check=False)
    for seed in (1, 2, 3):
        subprocess.run([sys.executable, "train/sample_maskgit.py",
                        "--ckpt", f"{WORK}/run/maskgit.pt", "--vqvae", vqs[0],
                        "--tokenizer", txt[0], "--seed", str(seed),
                        "--out", f"{WORK}/proba-live1024-{after}-ziarno{seed}.png",
                        "--prompts", *HARD], check=False)

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
    subprocess.run(["cp", f"{WORK}/run/maskgit.pt", f"{WORK}/up/"], check=True)
    for pth in glob.glob(f"{WORK}/proba-*.png"):
        subprocess.run(["cp", pth, f"{WORK}/up/"], check=False)
    json.dump({"id": DATASET, "title": "gweird live1024 mg",
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
subprocess.run(["rm", "-rf", f"{WORK}/data"], check=False)
print("gotowe", flush=True)
