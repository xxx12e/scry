import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import localsearch.engine as E
from localsearch import Config, clip as C, dlc

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

COLORS = {"red": (220, 20, 20), "green": (20, 180, 40), "blue": (20, 40, 220)}
WORDS = {"红": "red", "red": "red", "绿": "green", "green": "green", "蓝": "blue", "blue": "blue"}


class FakeClip:
    """Images -> their mean colour; captions -> the colour they name."""
    tag = "clip:fake:1"

    def __init__(self, d=None):
        pass

    def embed_images(self, paths, batch=8):
        out, ok = [], []
        for p in paths:
            try:
                with Image.open(p) as im:
                    v = np.asarray(im.convert("RGB"), np.float32).reshape(-1, 3).mean(0)
            except Exception:
                continue
            out.append(v / np.linalg.norm(v))
            ok.append(p)
        return np.array(out, np.float32), ok

    def embed_texts(self, texts):
        vs = []
        for t in texts:
            name = next((c for w, c in WORDS.items() if w in t.lower()), "red")
            v = np.array(COLORS[name], np.float32)
            vs.append(v / np.linalg.norm(v))
        return np.array(vs)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "pics"
    d.mkdir()
    for name, rgb in COLORS.items():
        Image.new("RGB", (40, 30), rgb).save(d / f"{name}.png")
    (d / "note.txt").write_text("a note about colours", encoding="utf-8")
    return d


@pytest.fixture
def eng(fake_embedder, tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    monkeypatch.setattr(dlc, "clip_dir", lambda index_dir: str(pack))
    monkeypatch.setattr(C, "ClipBackend", FakeClip)
    e = E.SearchEngine(tmp_path / "idx", Config(personalize=False))
    yield e
    e.close()


def test_build_and_search_by_description(eng, folder):
    st = eng.build([folder])
    assert st["images_added"] == 3 and st["added"] == 1          # text indexing unaffected
    assert "png" not in str(st["unsupported"])                    # images are searchable now
    assert os.path.basename(eng.search_images("一张红色的照片", 1)[0].path) == "red.png"
    assert os.path.basename(eng.search_images("something blue", 1)[0].path) == "blue.png"
    assert eng.build([folder])["images_added"] == 0               # diff: unchanged skipped


def test_image_changes_follow_the_folder(eng, folder):
    eng.build([folder])
    Image.new("RGB", (40, 30), COLORS["green"]).save(folder / "red.png")   # red becomes green
    os.utime(folder / "red.png", (1e9, 1e9))
    eng.update_file(str(folder / "red.png"))
    top2 = {os.path.basename(h.path) for h in eng.search_images("green", 2)}
    assert top2 == {"green.png", "red.png"}
    (folder / "blue.png").unlink()
    eng.remove_file(str(folder / "blue.png"))
    assert all(not h.path.endswith("blue.png") for h in eng.search_images("blue", 3))


def test_index_persists_and_is_pack_specific(eng, folder, tmp_path):
    eng.build([folder])
    ix = C.ImageIndex.load(tmp_path / "idx", FakeClip.tag)
    assert len(ix) == 3
    assert len(C.ImageIndex.load(tmp_path / "idx", "clip:other:2")) == 0   # other model


def test_api_image_mode_and_thumbnails(eng, folder, tmp_path, monkeypatch):
    from conftest import settle
    from localsearch import app as A
    app, engine, state = A.create_app(tmp_path / "idx")
    settle(state)
    engine.build([folder])
    c = app.test_client()
    red = str((folder / "red.png").resolve())
    assert c.get("/api/thumb", query_string={"path": red}).status_code == 404   # not shown yet
    r = c.post("/api/search", json={"query": "红色", "mode": "image"}).get_json()
    assert r["results"][0]["name"] == "red.png" and r["results"][0]["thumb"]
    t = c.get("/api/thumb", query_string={"path": r["results"][0]["path"]})
    assert t.status_code == 200 and t.mimetype == "image/jpeg"
    assert c.get("/api/status").get_json()["images"]["available"] is True
    engine.close()


@pytest.mark.skipif(not all(__import__("importlib").util.find_spec(m)
                            for m in ("torch", "transformers", "onnx", "onnxruntime")),
                    reason="needs torch + transformers + onnx to build a real pack")
def test_real_onnx_pack_end_to_end(fake_embedder, folder, tmp_path, monkeypatch):
    """Random-weight tiny Chinese-CLIP -> scripts/export_clip.py -> the real
    ClipBackend inside the engine. Proves the plumbing, not the quality."""
    src, pack = tmp_path / "tiny", tmp_path / "pack"
    code = r'''
import sys, torch
from pathlib import Path
from transformers import (ChineseCLIPConfig, ChineseCLIPModel, ChineseCLIPProcessor,
                          ChineseCLIPImageProcessorPil, BertTokenizerFast)
out = Path(sys.argv[1]); out.mkdir()
torch.manual_seed(0)
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("红色的照片绿蓝") + ["red", "photo", "blue"]
(out / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"))
cfg = ChineseCLIPConfig(
    text_config=dict(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64),
    vision_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                       image_size=32, patch_size=8), projection_dim=16)
ChineseCLIPModel(cfg).eval().save_pretrained(out)
ChineseCLIPProcessor(image_processor=ChineseCLIPImageProcessorPil(size={"shortest_edge": 32},
                     crop_size={"height": 32, "width": 32}), tokenizer=tok).save_pretrained(out)
'''
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code, str(src)], check=True, capture_output=True)
    out = subprocess.run([sys.executable, str(root / "scripts" / "export_clip.py"), str(src), str(pack)],
                         check=True, capture_output=True, text=True).stdout
    assert "images 1.000000, texts 1.000000" in out
    monkeypatch.setattr(dlc, "clip_dir", lambda index_dir: str(pack))
    e = E.SearchEngine(tmp_path / "idx2", Config(personalize=False))
    assert e.build([folder])["images_added"] == 3
    assert len(e.search_images("红色的照片", 3)) == 3
    e.close()
