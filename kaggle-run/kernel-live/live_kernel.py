"""Kaggle GPU cell: G-Weird Live — MaskGIT on the 1.2 corpus, mirrored to 1.2.

Same tokenizer, same 4.02M pairs, same BPE, same 62M, same effective batch 64,
same schedule. The only differences are the architecture (bidirectional,
12 parallel rounds instead of 576 sequential passes) and label smoothing 0.1
from the original MaskGIT recipe. That is what makes the side-by-side at equal
steps a fair one.

Sessions chain through kernel outputs: the checkpoint with the highest step
among all attached inputs wins.
"""

STEPS_TOTAL = 60000      # horyzont harmonogramu (jak 1.1, ta sama efektywna partia 64)
MAX_STEPS   = 25000      # zmierzone 1.21 s/krok -> ~8.4 h; 36000 uderzylo w sciane 12 h na 35400 bez zapisu koncowego
PROMPTS = ["a horse standing in a field", "a red double decker bus on a street",
           "a cat wearing sunglasses", "portrait of an old man with a beard",
           "a bowl of soup on a wooden table", "a castle on a mountain at sunset",
           "two people riding bicycles", "a robot playing a piano"]

import glob, json, os, subprocess, sys, torch
WORK = "/kaggle/working"
assert torch.cuda.is_available(), "brak GPU — ustaw Accelerator na GPU T4 x2"
print(f"GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}", flush=True)

subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/JerzySukiennik/g-weird.git", f"{WORK}/g-weird"], check=True)
os.chdir(f"{WORK}/g-weird")

# --- korpus: cztery shardy jako jeden, bez sklejania -------------------------
metas = sorted(glob.glob("/kaggle/input/**/gwtok*_meta.json", recursive=True))
prefixes = [m[:-len("_meta.json")] for m in metas]
assert len(prefixes) == 4, f"oczekiwalem 4 shardow (enc576-a..d), widze {len(prefixes)}: {prefixes}"
n = sum(os.path.getsize(f"{p}_tokens.u16") // 1152 for p in prefixes)
assert n == 4017878, f"korpus ma {n} par, oczekiwalem 4017878"
print(f"korpus: {n:,} par", flush=True)

txt = glob.glob("/kaggle/input/**/text.json", recursive=True)
vqs = glob.glob("/kaggle/input/**/vqvae.pt", recursive=True)
assert len(txt) == 1 and len(vqs) == 1, f"text {txt} / vqvae {vqs} — podepnij gweird-text-12 i gweird-vqvae-576"

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

# --- wznowienie: najwyzszy krok sposrod podpietych checkpointow ---------------
os.makedirs(f"{WORK}/run", exist_ok=True)
cks = glob.glob("/kaggle/input/**/maskgit.pt", recursive=True)
before = 0
if cks:
    steps = {c: torch.load(c, map_location="cpu", weights_only=False)["step"] for c in cks}
    best = max(steps, key=steps.get); before = steps[best]
    subprocess.run(["cp", best, f"{WORK}/run/maskgit.pt"], check=True)
    print(f"wznawiam z kroku {before} ({best})", flush=True)
else:
    print("pierwsza sesja — od zera", flush=True)

# --- trening ------------------------------------------------------------------
subprocess.run([sys.executable, "train/train_maskgit.py", "--data", *local,
                "--out", f"{WORK}/run", "--steps", str(STEPS_TOTAL),
                "--max-steps", str(MAX_STEPS), "--batch", "32", "--accum", "2",
                "--lr", "3e-4", "--warmup", "2000", "--label-smoothing", "0.1",
                "--workers", "4", "--log-every", "100", "--ckpt-every", "1000",
                "--resume"], check=True)

after = torch.load(f"{WORK}/run/maskgit.pt", map_location="cpu", weights_only=False)["step"]
assert after > before, f"krok nie ruszyl: {before} -> {after}"
print(f"krok {before} -> {after}", flush=True)

# --- dowod na obrazkach: te same osiem podpisow co 1.1 i 1.2 -----------------
subprocess.run([sys.executable, "train/sample_maskgit.py",
                "--ckpt", f"{WORK}/run/maskgit.pt", "--vqvae", vqs[0],
                "--tokenizer", txt[0], "--steps", "24", "--cfg-scale", "2.0", "--temp", "0.7",
                "--out", f"{WORK}/proba-live-{after}.png", "--prompts", *PROMPTS], check=False)
subprocess.run(["rm", "-rf", f"{WORK}/data"], check=False)   # dowiazania, nie dane
print("gotowe — checkpoint w /kaggle/working/run/maskgit.pt, probka w proba-live-*.png", flush=True)
