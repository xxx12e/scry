import os

import numpy as np
import pytest

from localsearch import dlc, embedder
from localsearch.config import Config


def test_can_load_needs_some_backend(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(embedder, "_has", lambda m: False)
    assert not embedder.can_load(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"x")
    monkeypatch.setattr(embedder, "_has", lambda m: m == "onnxruntime")
    assert embedder.can_load(tmp_path)


def test_unloadable_dlc_pack_is_ignored(tmp_path, monkeypatch):
    pack = dlc.pack_dir(tmp_path, "model-large")
    pack.mkdir(parents=True)
    (pack / "config.json").write_text("{}")
    monkeypatch.setattr(embedder, "_has", lambda m: False)      # torch-free, no model.onnx
    assert dlc.model_override(tmp_path) is None
    monkeypatch.setattr(embedder, "_has", lambda m: True)
    assert dlc.model_override(tmp_path) == str(pack)


def test_quantized_variant_is_a_different_model(tmp_path):
    from localsearch import resources
    (tmp_path / "config.json").write_text('{"h": 384}')
    plain = resources.model_tag(tmp_path)
    (tmp_path / "quantization.json").write_text('{"onnx": "dynamic-int8"}')
    assert resources.model_tag(tmp_path) != plain


MODEL = os.environ.get("SCRY_TEST_MODEL")       # a model dir exported by scripts/export_onnx.py


@pytest.mark.skipif(not MODEL, reason="set SCRY_TEST_MODEL to a model dir with model.onnx")
def test_onnx_backend_matches_sentence_transformers():
    st = pytest.importorskip("sentence_transformers")
    torch = pytest.importorskip("torch")
    texts = ["passage: The reservoir dropped to a record low.", "query: 发票报销流程",
             "passage: Привет мир", "passage: " + "lorem ipsum " * 300]
    ox = embedder.Embedder(Config(model_name=MODEL, device="cpu"))
    assert ox.backend == "onnx"
    ref = st.SentenceTransformer(MODEL, device="cpu", model_kwargs={"dtype": torch.float32})
    a = ref.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    b = ox._ox.encode(texts, 4)
    assert float(np.min(np.sum(a * b, axis=1))) > 0.9999
