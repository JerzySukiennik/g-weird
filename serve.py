"""Local G-Weird: type a prompt in the browser, get an image. No Kaggle, no internet.

Generation runs on this Mac's CPU in about 3 seconds per image, so nothing here
needs a GPU. The models load once at startup — that is the slow part (a few
seconds), and after it every prompt is cheap.

    python3 serve.py --ckpt run/gweird.pt --vqvae run/vqvae.pt --tokenizer run/text.json

Then open http://localhost:8000.
"""

import argparse
import base64
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.transformer import WeirdGPT, WeirdConfig      # noqa: E402
from model.vqvae import VQVAE                            # noqa: E402
from train.sample import build_prefix, generate          # noqa: E402

PAGE = """<!doctype html><meta charset=utf-8><title>G-Weird</title>
<style>
 :root { color-scheme: dark }
 body { margin:0; background:#0d0d0f; color:#e8e8ea; font:15px/1.5 ui-sans-serif,system-ui,sans-serif;
        display:flex; flex-direction:column; align-items:center; padding:32px 16px }
 h1 { font-size:20px; font-weight:600; letter-spacing:.02em; margin:0 0 4px }
 .sub { color:#8a8a92; font-size:13px; margin-bottom:24px }
 form { width:100%; max-width:560px; display:flex; gap:8px; margin-bottom:14px }
 input[type=text] { flex:1; background:#17171b; border:1px solid #2a2a31; color:inherit;
        border-radius:8px; padding:11px 13px; font:inherit; min-width:0 }
 input[type=text]:focus { outline:none; border-color:#4a4a57 }
 button { background:#e8e8ea; color:#0d0d0f; border:0; border-radius:8px; padding:11px 18px;
        font:inherit; font-weight:600; cursor:pointer }
 button:disabled { opacity:.45; cursor:default }
 .knobs { width:100%; max-width:560px; display:flex; gap:18px; flex-wrap:wrap;
        color:#8a8a92; font-size:13px; margin-bottom:22px; align-items:center }
 .knobs label { display:flex; gap:7px; align-items:center }
 .knobs input[type=range] { width:96px; accent-color:#8a8a92 }
 #out { display:grid; grid-template-columns:repeat(auto-fill,minmax(256px,1fr));
        gap:12px; width:100%; max-width:560px }
 figure { margin:0 }
 img { width:100%; border-radius:10px; display:block; background:#17171b; aspect-ratio:1 }
 figcaption { color:#8a8a92; font-size:12px; margin-top:6px; word-break:break-word }
 .err { color:#ff8f8f; font-size:13px }
</style>
<h1>G-Weird</h1>
<div class=sub id=meta>ładowanie…</div>
<form id=f>
  <input type=text id=p placeholder="a horse made of meat in a living room" autofocus>
  <button id=go>Rysuj</button>
</form>
<div class=knobs>
  <label>CFG <input type=range id=cfg min=1 max=10 step=.5 value=4><span id=cfgv>4</span></label>
  <label>temperatura <input type=range id=t min=.5 max=1.5 step=.05 value=1><span id=tv>1.00</span></label>
  <label><input type=checkbox id=rand checked> losowe ziarno</label>
</div>
<div id=out></div>
<script>
const $ = s => document.querySelector(s)
cfg.oninput = () => cfgv.textContent = cfg.value
t.oninput = () => tv.textContent = (+t.value).toFixed(2)
fetch('/info').then(r=>r.json()).then(d=>{
  meta.textContent = `${d.params} parametrów · krok ${d.step} · ${d.device}`
})
let n = 0
f.onsubmit = async e => {
  e.preventDefault()
  const prompt = p.value.trim(); if (!prompt) return
  go.disabled = true; const label = go.textContent; go.textContent = 'rysuję…'
  const fig = document.createElement('figure')
  fig.innerHTML = '<img><figcaption></figcaption>'
  fig.querySelector('figcaption').textContent = prompt
  out.prepend(fig)
  try {
    const r = await fetch('/gen', {method:'POST', headers:{'content-type':'application/json'},
      body: JSON.stringify({prompt, cfg:+cfg.value, temp:+t.value,
                            seed: rand.checked ? -1 : ++n})})
    const d = await r.json()
    if (d.error) throw new Error(d.error)
    fig.querySelector('img').src = 'data:image/png;base64,' + d.png
    fig.querySelector('figcaption').textContent = `${prompt} · ${d.secs}s`
  } catch (err) {
    fig.querySelector('figcaption').innerHTML =
      `${prompt} <span class=err>— ${err.message}</span>`
  }
  go.disabled = false; go.textContent = label
}
</script>"""


def load_tokenizer(path):
    """Tokenizer files written by newer `tokenizers` store merges as pairs; older
    installs expect "a b" strings and fail with an opaque ModelWrapper error.
    Convert in memory so the same checkpoint works on either version."""
    from tokenizers import Tokenizer
    try:
        return Tokenizer.from_file(path)
    except Exception:
        d = json.load(open(path))
        m = d.get("model", {}).get("merges")
        if not m or isinstance(m[0], str):
            raise
        d["model"]["merges"] = [" ".join(p) for p in m]
        tmp = path + ".compat.json"
        json.dump(d, open(tmp, "w"))
        print("  tokenizer przekonwertowany do starszego formatu", flush=True)
        return Tokenizer.from_file(tmp)


class G:
    lock = threading.Lock()


def build(a):
    dev = "cpu"
    ck = torch.load(a.ckpt, map_location="cpu")
    saved = ck.get("cfg") or ck.get("arch") or {}
    cfg = WeirdConfig(**{k: v for k, v in saved.items()
                         if k in WeirdConfig.__dataclass_fields__})
    model = WeirdGPT(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])

    vk = torch.load(a.vqvae, map_location="cpu")
    vq = VQVAE(**vk["arch"]).to(dev).eval()
    vq.load_state_dict(vk["model"])

    G.model, G.vq, G.cfg, G.dev = model, vq, cfg, dev
    G.tok = load_tokenizer(a.tokenizer)
    G.step = ck.get("step", "?")
    G.params = sum(p.numel() for p in model.parameters())
    print(f"  transformer krok {G.step}, {G.params/1e6:.1f}M; vqvae krok {vk.get('step','?')}",
          flush=True)


def draw(prompt, scale, temp, seed):
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "big")
    torch.manual_seed(seed)
    prefix = build_prefix([prompt], G.tok, G.cfg, G.dev)
    codes = generate(G.model, prefix, G.cfg, scale, temp, 100)
    grid = int(round(math.sqrt(G.cfg.image_len)))
    with torch.no_grad():
        img = G.vq.decode(codes.view(-1, grid, grid))
    arr = ((img.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send(200, json.dumps({"step": G.step, "device": G.dev,
                                        "params": f"{G.params/1e6:.1f}M"}))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path != "/gen":
            return self._send(404, "{}")
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or "{}")
        t0 = time.time()
        try:
            # One prompt at a time: two concurrent generations on the same CPU
            # make both slower and neither is waiting on anything else.
            with G.lock:
                png = draw(req.get("prompt", ""), float(req.get("cfg", 4.0)),
                           float(req.get("temp", 1.0)), int(req.get("seed", -1)))
            self._send(200, json.dumps({"png": png, "secs": round(time.time() - t0, 1)}))
        except Exception as e:
            self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="run/gweird.pt")
    p.add_argument("--vqvae", default="run/vqvae.pt")
    p.add_argument("--tokenizer", default="run/text.json")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print("laduje modele...", flush=True)
    build(a)
    print(f"gotowe — otworz http://localhost:{a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()
