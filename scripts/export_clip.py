"""Export a CLIP-style model to an image-search pack (two ONNX encoders).

    pip install torch transformers onnx onnxruntime pillow          # build machine
    python scripts/export_clip.py OFA-Sys/chinese-clip-vit-base-patch16 out/clip-zh
    # zip out/clip-zh -> publish as a release asset -> add its url/sha256/size to
    # localsearch/dlc.py (pack "clip-zh") -> users install it under Add-ons.

Writes text.onnx, vision.onnx, tokenizer.json and preprocess.json (resize /
crop / normalisation exactly as the model's own processor does them), then
checks the ONNX pipeline against transformers on sample images and captions.

Chinese-CLIP ViT-B/16 (188M params) is the default recommendation for Chinese
+ English captions; any transformers CLIPModel / ChineseCLIPModel works.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(src, out, check=True, fp16=True):
    import torch
    from transformers import AutoModel, AutoProcessor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(src, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(src)
    tok, ip = proc.tokenizer, proc.image_processor
    max_len = int(min(getattr(tok, "model_max_length", 77) or 77, 77))
    if "chinese" in type(model).__name__.lower():
        max_len = 52                      # Chinese-CLIP's context length

    class Text(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            kw = dict(input_ids=input_ids, attention_mask=attention_mask)
            if "token_type_ids" in tok.model_input_names:
                kw["token_type_ids"] = token_type_ids
            f = self.m.get_text_features(**kw)
            return getattr(f, "pooler_output", f)

    class Vision(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            f = self.m.get_image_features(pixel_values=pixel_values)
            return getattr(f, "pooler_output", f)

    enc = tok(["一只猫", "a photo of a dog"], padding="max_length", truncation=True,
              max_length=max_len, return_tensors="pt")
    tti = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(Text(model), (enc["input_ids"], enc["attention_mask"], tti),
                      str(out / "text.onnx"), input_names=names, output_names=["embeds"],
                      dynamic_axes={**{k: {0: "batch", 1: "seq"} for k in names},
                                    "embeds": {0: "batch"}}, opset_version=17, dynamo=False)
    size = ip.crop_size if getattr(ip, "do_center_crop", False) else ip.size
    side = int(size.get("height") or size.get("shortest_edge"))
    pv = torch.zeros(1, 3, side, side)
    torch.onnx.export(Vision(model), (pv,), str(out / "vision.onnx"),
                      input_names=["pixel_values"], output_names=["embeds"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "embeds": {0: "batch"}},
                      opset_version=17, dynamo=False)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    resize = ({"shortest_edge": int(ip.size["shortest_edge"])} if "shortest_edge" in ip.size
              else {"height": int(ip.size["height"]), "width": int(ip.size["width"])})
    pre = dict(resize=resize, crop=int(side) if getattr(ip, "do_center_crop", False) else None,
               mean=[float(x) for x in ip.image_mean], std=[float(x) for x in ip.image_std],
               max_length=max_len, pad_id=int(tok.pad_token_id or 0),
               uses_token_type_ids="token_type_ids" in tok.model_input_names,
               source=str(src))
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1), encoding="utf-8")
    if fp16:                  # store weights as fp16, compute in fp32 (halves the pack)
        sys.path.insert(0, str(ROOT / "scripts"))
        from export_onnx import _fp16_storage
        for f in ("text.onnx", "vision.onnx"):
            _fp16_storage(out / f)
    if check:
        return parity(src, out)
    return None


def parity(src, out):
    """Min cosine between transformers and the ONNX pack (texts and images).
    Uses a FRESHLY loaded model: an object that went through torch.onnx.export
    (tracing) is not a trustworthy reference afterwards."""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    from localsearch.clip import ClipBackend
    model = AutoModel.from_pretrained(src, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(src)
    texts = ["一只在草地上奔跑的狗", "sunset over a bridge", "发票扫描件"]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, color in enumerate([(200, 30, 30), (30, 160, 60), (240, 240, 240)]):
            img = Image.new("RGB", (320 + 40 * i, 240), color)
            for x in range(0, img.width, 16):        # some structure, odd aspect ratios
                for y in range(i, img.height, 23):
                    img.putpixel((x, y), (y % 255, x % 255, 90))
            p = Path(tmp) / f"s{i}.png"
            img.save(p)
            paths.append(str(p))
        pack = ClipBackend(out)
        a_img, _ = pack.embed_images(paths)
        a_txt = pack.embed_texts(texts)
        with torch.no_grad():
            ims = [Image.open(p).convert("RGB") for p in paths]
            pv = proc(images=ims, return_tensors="pt")["pixel_values"]
            fi = model.get_image_features(pixel_values=pv)
            fi = getattr(fi, "pooler_output", fi)
            enc = proc.tokenizer(texts, padding="max_length", truncation=True,
                                 max_length=pack.pre["max_length"], return_tensors="pt")
            ft = model.get_text_features(**enc)
            ft = getattr(ft, "pooler_output", ft)
        n = lambda x: (x / x.norm(dim=-1, keepdim=True)).numpy()
        return (float(np.min(np.sum(a_img * n(fi), axis=1))),
                float(np.min(np.sum(a_txt * n(ft), axis=1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="HF id or local dir of a CLIP / Chinese-CLIP model")
    ap.add_argument("out")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="keep fp32 weights (2x larger)")
    a = ap.parse_args()
    res = export(a.model, a.out, check=not a.no_check, fp16=not a.fp32)
    if res:
        print("min cosine vs transformers -- images %.6f, texts %.6f" % res)
    print("pack written to", a.out)


if __name__ == "__main__":
    main()
