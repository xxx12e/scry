"""Export the embedding model to ONNX so the app can run WITHOUT torch.

    pip install torch transformers onnx onnxruntime        # build machine only
    python scripts/export_onnx.py models/embed             # writes models/embed/model.onnx

What it produces (measured on the bundled multilingual-e5-small, 4-core CPU):
  * identical embeddings to the sentence-transformers path (min cosine
    0.99999994), so existing indexes stay valid -- no rebuild;
  * weights STORED as fp16 (like the shipped safetensors) but computed in fp32
    (Cast nodes that ONNX Runtime constant-folds at load): same 235 MB on disk;
  * BERT attention/GELU/LayerNorm fused by ONNX Runtime's optimizer (export uses
    eager attention, the pattern the fuser recognizes): indexing throughput on
    par with torch, query embed 21 -> ~5 ms, model load ~9 s -> ~1 s.

With `model.onnx` present and onnxruntime installed, localsearch picks the ONNX
backend automatically; torch / transformers / sentence-transformers / scipy /
scikit-learn can then be left out of the bundle (~500 MB unpacked).

--int8 additionally writes a dynamically quantized variant to OUT_DIR (smaller
and faster, but embeddings drift -- min cosine ~0.97 -- so it is a different
model: indexes are rebuilt, and retrieval quality should be evaluated first).
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _export(src, out):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src)
    base = AutoModel.from_pretrained(src, dtype=torch.float32,
                                     attn_implementation="eager").eval()

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    enc = tok(["passage: hello world", "query: 你好 世界"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(Wrapped(base).eval(),
                      (enc["input_ids"], enc["attention_mask"], torch.zeros_like(enc["input_ids"])),
                      str(out), input_names=names, output_names=["last_hidden_state"],
                      dynamic_axes={k: {0: "batch", 1: "seq"} for k in names + ["last_hidden_state"]},
                      opset_version=17, dynamo=False)
    return base.config.num_attention_heads, base.config.hidden_size


def _fuse(path, heads, hidden):
    from onnxruntime.transformers import optimizer
    opt = optimizer.optimize_model(str(path), model_type="bert", num_heads=heads,
                                   hidden_size=hidden, opt_level=1)
    fused = {k: v for k, v in opt.get_fused_operator_statistics().items() if v}
    if not fused.get("Attention"):
        print("warning: attention was not fused; the model works but indexes slower")
    opt.save_model_to_file(str(path))
    return fused


def _fp16_storage(path, min_elems=1024):
    """Store big fp32 weights as fp16 + a Cast back to fp32 (folded at load)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(path))
    g, casts = m.graph, []
    for init in list(g.initializer):
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16),
                                           init.name + "__fp16")
            g.initializer.remove(init)
            g.initializer.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
    for c in reversed(casts):
        g.node.insert(0, c)
    onnx.save(m, str(path))
    return len(casts)


def _check(model_dir, onnx_path):
    """Compare against sentence-transformers on multilingual samples."""
    import torch
    from sentence_transformers import SentenceTransformer
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from localsearch.embedder import OnnxBackend
    texts = ["passage: The reservoir dropped to a record low during the drought.",
             "query: 上个季度的发票报销流程是怎么改的", "passage: Привет мир, документ о бюджете.",
             "passage: 학교 숙제 제출 마감", "passage: " + "lorem ipsum dolor sit amet " * 80]
    ref = SentenceTransformer(str(model_dir), device="cpu", model_kwargs={"dtype": torch.float32})
    a = ref.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    b = OnnxBackend(model_dir, onnx_path).encode(texts, batch_size=8)
    return float(np.min(np.sum(a * b, axis=1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", help="sentence-transformers model dir (config.json, tokenizer.json, ...)")
    ap.add_argument("--int8", metavar="OUT_DIR", help="also write an int8 variant to OUT_DIR")
    ap.add_argument("--no-check", action="store_true", help="skip the parity check")
    args = ap.parse_args()
    src = Path(args.model_dir)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "model.onnx"
        heads, hidden = _export(src, raw)
        print("fused:", _fuse(raw, heads, hidden))
        if args.int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            out = Path(args.int8)
            out.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file() and f.suffix != ".safetensors" and f.name != "model.onnx":
                    shutil.copy2(f, out / f.name)
            if (src / "1_Pooling").is_dir():
                shutil.copytree(src / "1_Pooling", out / "1_Pooling", dirs_exist_ok=True)
            quantize_dynamic(str(raw), str(out / "model.onnx"), weight_type=QuantType.QInt8,
                             op_types_to_quantize=["MatMul", "Gather"])
            # a different model as far as the index is concerned (see resources.model_tag)
            (out / "quantization.json").write_text(json.dumps({"onnx": "dynamic-int8"}))
            print("int8 variant ->", out / "model.onnx")
        n = _fp16_storage(raw)
        shutil.copy2(raw, src / "model.onnx")
    print(f"wrote {src / 'model.onnx'} ({(src / 'model.onnx').stat().st_size / 1e6:.0f} MB, "
          f"{n} weights stored as fp16)")
    if not args.no_check:
        print(f"min cosine vs sentence-transformers: {_check(src, src / 'model.onnx'):.6f}")


if __name__ == "__main__":
    main()
