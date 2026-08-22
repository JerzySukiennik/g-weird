"""Two-minute measurement: how much faster is the encoder in fp16, and on two GPUs.

Guessing has cost 3.8 h of quota across two failed fixes. This settles it before
a third restart.
"""
import glob, subprocess, sys, time, torch
subprocess.run(["git","clone","--depth","1",
                "https://github.com/JerzySukiennik/g-weird.git","/tmp/gw"], check=True)
sys.path.insert(0,"/tmp/gw")
from model.vqvae import VQVAE

ck_path = sorted(glob.glob("/kaggle/input/**/vqvae.pt", recursive=True))[0]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
m = VQVAE(**ck["arch"]).cuda().eval()
m.load_state_dict(ck["model"], strict=False)
print(f"kart: {torch.cuda.device_count()}", flush=True)

x = torch.randn(64,3,256,256).cuda()

def bench(fn, warm=2, iters=6):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return 64*iters/(time.time()-t0)

with torch.no_grad():
    r32 = bench(lambda: m.encode(x))
    def f16():
        with torch.cuda.amp.autocast():
            return m.encode(x)
    r16 = bench(f16)
    md = torch.nn.DataParallel(m)
    xd = torch.randn(128,3,256,256).cuda()
    def f16dp():
        with torch.cuda.amp.autocast():
            return md.module.encode(xd) if False else md(xd)[2]
    try:
        r16dp = bench(lambda: f16dp(), iters=4) * 2
    except Exception as e:
        r16dp = float('nan'); print("DataParallel:", type(e).__name__, str(e)[:80], flush=True)

print(f"fp32, 1 GPU:        {r32:6.1f} obr./s", flush=True)
print(f"fp16, 1 GPU:        {r16:6.1f} obr./s   ({r16/r32:.1f}x)", flush=True)
print(f"fp16, 2 GPU:        {r16dp:6.1f} obr./s   ({r16dp/r32:.1f}x)", flush=True)
print(f"obecne w produkcji:   50.2 obr./s", flush=True)
