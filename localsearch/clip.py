"""Image search by description ("bridge at sunset", "发票的照片") -- optional pack.

A CLIP-style pack (scripts/export_clip.py; recommended: Chinese-CLIP ViT-B/16)
holds two ONNX encoders that map images and captions into one vector space.
Images in controlled folders are embedded once (incrementally, like text) into
a small exact index; a text query is embedded and compared against them.
Everything stays local; the pack is downloaded only if the user installs it.
"""
import json
import os
from pathlib import Path

import numpy as np


class ClipBackend:
    def __init__(self, pack_dir, threads=0):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = Path(pack_dir)
        self.pre = json.loads((d / "preprocess.json").read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        n = int(self.pre["max_length"])
        self.tok.enable_truncation(max_length=n)
        self.tok.enable_padding(length=n, pad_id=int(self.pre.get("pad_id", 0)))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.text = ort.InferenceSession(str(d / "text.onnx"), so, providers=prov)
        self.vision = ort.InferenceSession(str(d / "vision.onnx"), so, providers=prov)
        self.text_inputs = {i.name for i in self.text.get_inputs()}
        self.tag = "clip:%s:%d" % (d.name, (d / "vision.onnx").stat().st_size)

    @staticmethod
    def _norm(v):
        v = np.asarray(v, np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def embed_texts(self, texts):
        encs = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], np.int64)
        mask = np.array([e.attention_mask for e in encs], np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.text_inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        return self._norm(self.text.run(None, feed)[0])

    def _pixels(self, path):
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            r = self.pre["resize"]
            if "shortest_edge" in r:
                s = int(r["shortest_edge"])
                w, h = im.size
                if w <= h:
                    nw, nh = s, int(s * h / w)
                else:
                    nw, nh = int(s * w / h), s
            else:
                nw, nh = int(r["width"]), int(r["height"])
            im = im.resize((nw, nh), Image.BICUBIC)
            c = self.pre.get("crop")
            if c:
                left, top = int((nw - c) / 2.0), int((nh - c) / 2.0)
                im = im.crop((left, top, left + c, top + c))
            a = np.asarray(im, np.float32) / 255.0
        a = (a - np.array(self.pre["mean"], np.float32)) / np.array(self.pre["std"], np.float32)
        return a.transpose(2, 0, 1)

    def embed_images(self, paths, batch=8):
        """-> (vectors for the readable images, list of their paths)."""
        vecs, ok = [], []
        for a in range(0, len(paths), batch):
            px, good = [], []
            for p in paths[a:a + batch]:
                try:
                    px.append(self._pixels(p))
                    good.append(p)
                except Exception:
                    continue              # unreadable / truncated image: skip it
            if px:
                vecs.append(self.vision.run(None, {"pixel_values": np.stack(px)})[0])
                ok += good
        return (self._norm(np.concatenate(vecs)) if vecs else np.zeros((0, 1), np.float32)), ok


class ImageIndex:
    """path -> (mtime, size, vector); exact cosine search; fp16 on disk."""

    FILE = "images.npz"

    def __init__(self, tag=""):
        self.tag = tag
        self.meta = {}                    # path -> (mtime, size)
        self.vecs = {}                    # path -> np.float32[dim]
        self._mat = None                  # cached (paths, matrix) for search

    def __len__(self):
        return len(self.vecs)

    def unchanged(self, path, mtime, size):
        return self.meta.get(path) == (mtime, size)

    def add(self, path, mtime, size, vec):
        self.meta[path] = (mtime, size)
        self.vecs[path] = np.asarray(vec, np.float32)
        self._mat = None

    def remove(self, path):
        if self.vecs.pop(path, None) is not None:
            self.meta.pop(path, None)
            self._mat = None
            return True
        return False

    def search(self, qvec, k):
        if not self.vecs:
            return []
        if self._mat is None:
            paths = list(self.vecs)
            self._mat = (paths, np.vstack([self.vecs[p] for p in paths]))
        paths, mat = self._mat
        sims = mat @ np.asarray(qvec, np.float32).ravel()
        n = min(k, len(paths))
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        return [(paths[i], float(sims[i])) for i in top]

    def save(self, index_dir):
        paths = list(self.vecs)
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (self.FILE + ".tmp.npz")
        np.savez(tmp, paths=np.array(paths, dtype=str),
                 mtimes=np.array([self.meta[p][0] for p in paths], np.float64),
                 sizes=np.array([self.meta[p][1] for p in paths], np.int64),
                 vecs=(np.vstack([self.vecs[p] for p in paths]).astype(np.float16)
                       if paths else np.zeros((0, 1), np.float16)),
                 tag=np.array(self.tag))
        os.replace(tmp, d / self.FILE)

    @classmethod
    def load(cls, index_dir, tag):
        """The saved index, or an empty one if absent / built by another pack."""
        f = Path(index_dir) / cls.FILE
        ix = cls(tag)
        try:
            z = np.load(f, allow_pickle=False)
            if str(z["tag"]) != tag:
                return ix                 # different model: vectors incompatible
            for p, m, s, v in zip(z["paths"], z["mtimes"], z["sizes"], z["vecs"]):
                ix.meta[str(p)] = (float(m), int(s))
                ix.vecs[str(p)] = v.astype(np.float32)
        except (OSError, KeyError, ValueError):
            pass
        return ix
