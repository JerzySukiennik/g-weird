"""Real photographs with human captions, to teach the model what things look like.

DiffusionDB gives rich prompts but its images are Stable Diffusion 1.x output —
and SD 1.x is itself prone to the exact failure we are trying to remove: limbs
that merge, animals that are assemblages rather than animals. Learning coherence
from a teacher that also produces incoherent pictures is a weak plan.

COCO is 118k photographs of ordinary things, each described five times by a
person. Nothing in it is generated, nothing is watermarked, and the captions name
plain subjects rather than styles.

There is also a measured reason to prefer photographs here. The frozen VQ-VAE was
trained on photographs, and it reconstructs them at 15.7/255 against 22.5 for
DiffusionDB images — so this corpus passes through our compressor noticeably
better than the one it is being mixed into.

Two captions per image rather than all five: five would repeat each photograph
five times in a corpus of ~2.6M, and at four epochs that is twenty passes over
the same picture, which invites memorising it instead of learning from it.
"""

import argparse
import io
import json
import os
import zipfile

import requests
from PIL import Image

IMAGES = "http://images.cocodataset.org/zips/train2017.zip"
ANNOTS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def stream(url, dest):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in r.iter_content(1 << 22):
                out.write(chunk)
    print(f"  {os.path.basename(dest)}: {os.path.getsize(dest)/1e9:.1f} GB", flush=True)
    return dest


def square_jpeg(data, res, quality=88):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    s = min(img.size)
    l, t = (img.width - s) // 2, (img.height - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((res, res), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--per-image", type=int, default=2, help="ile podpisow na obraz")
    p.add_argument("--out-prefix", default="./gweird")
    p.add_argument("--tmp", default="/kaggle/tmp")
    a = p.parse_args()

    os.makedirs(a.tmp, exist_ok=True)
    ann_zip = stream(ANNOTS, f"{a.tmp}/ann.zip")
    with zipfile.ZipFile(ann_zip) as z:
        caps = json.loads(z.read("annotations/captions_train2017.json"))
    os.remove(ann_zip)

    by_image = {}
    for c in caps["annotations"]:
        by_image.setdefault(c["image_id"], []).append(c["caption"].strip())
    name_of = {im["id"]: im["file_name"] for im in caps["images"]}
    print(f"  podpisy dla {len(by_image)} zdjec", flush=True)

    img_zip = stream(IMAGES, f"{a.tmp}/train2017.zip")

    fh = open(f"{a.out_prefix}_images.jpgbin", "wb")
    captions, offsets, kept = [], [0], 0
    with zipfile.ZipFile(img_zip) as z:
        for i, (img_id, texts) in enumerate(sorted(by_image.items())):
            name = name_of.get(img_id)
            if not name:
                continue
            try:
                blob = square_jpeg(z.read(f"train2017/{name}"), a.res)
            except Exception:
                continue
            # The same picture under each of its captions: the image tokens
            # repeat, which is the cost of teaching one subject several
            # phrasings.
            for text in texts[:a.per_image]:
                fh.write(blob)
                offsets.append(offsets[-1] + len(blob))
                captions.append(text)
                kept += 1
            if (i + 1) % 10000 == 0:
                fh.flush()
                print(f"  {i+1} zdjec, {kept} par", flush=True)
    fh.close()
    os.remove(img_zip)

    with open(f"{a.out_prefix}_captions.json", "w") as f:
        json.dump(captions, f, ensure_ascii=False)
    with open(f"{a.out_prefix}_offsets.json", "w") as f:
        json.dump(offsets, f)
    with open(f"{a.out_prefix}_meta.json", "w") as f:
        json.dump({"n": kept, "res": a.res, "format": "jpeg", "source": "coco"}, f)
    size = os.path.getsize(f"{a.out_prefix}_images.jpgbin")
    print(f"gotowe: {kept} par, {size/1e9:.2f} GB, {size/max(kept,1)/1024:.1f} kB na obraz",
          flush=True)


if __name__ == "__main__":
    main()
