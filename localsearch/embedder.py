"""Embedding model wrapper (multilingual-e5-small by default). Loaded once and
kept resident. Two interchangeable backends with identical outputs:

  * ONNX Runtime -- used when the model dir holds `model.onnx` (see
    scripts/export_onnx.py) and onnxruntime is installed, on CPU. No torch at
    all: ~1 s model load instead of ~9 s, and the app can ship without torch.
  * sentence-transformers (torch) -- everything else, including GPU.
"""
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("localsearch.embedder")


def _onnx_file(model_name):
    p = Path(str(model_name)) / "model.onnx"
    return p if p.is_file() else None


def _has(module):
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def can_load(model_dir):
    """Could some installed backend load this model dir? (a DLC model pack
    without model.onnx is unusable in a torch-free build)"""
    return bool((_onnx_file(model_dir) and _has("onnxruntime"))
                or _has("sentence_transformers"))


class OnnxBackend:
    """Tokenize (HF `tokenizers`) -> ONNX encoder -> mean pool -> L2 normalize:
    the same pipeline as the sentence-transformers Transformer/Pooling/Normalize
    modules of the e5 models."""

    def __init__(self, model_dir, onnx_path=None, threads=0):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = self._json(model_dir / "tokenizer_config.json")
        st_cfg = self._json(model_dir / "sentence_bert_config.json")
        max_len = int(st_cfg.get("max_seq_length") or cfg.get("model_max_length") or 512)
        self.tok.enable_truncation(max_length=min(max_len, 512))
        self.tok.no_padding()
        pad = self.tok.token_to_id(str(cfg.get("pad_token") or "<pad>"))
        self.pad = 0 if pad is None else pad
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path or model_dir / "model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        hidden = self._json(model_dir / "config.json").get("hidden_size")
        out_dim = self.sess.get_outputs()[0].shape[-1]
        self.dim = int(out_dim) if isinstance(out_dim, int) else int(hidden)

    @staticmethod
    def _json(p):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def encode(self, texts, batch_size=16):
        out = np.zeros((len(texts), self.dim), np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # less padding
        for a in range(0, len(order), max(1, batch_size)):
            idx = order[a:a + batch_size]
            encs = self.tok.encode_batch([texts[i] for i in idx])
            width = max(1, max(len(e.ids) for e in encs))
            ids = np.full((len(idx), width), self.pad, np.int64)
            mask = np.zeros((len(idx), width), np.int64)
            for r, e in enumerate(encs):
                ids[r, :len(e.ids)] = e.ids
                mask[r, :len(e.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
            out[idx] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return out


class Embedder:
    def __init__(self, config):
        self.cfg = config
        self.model = None
        self._ox = None
        onnx = _onnx_file(config.model_name)
        want = getattr(config, "backend", "auto")
        device = "cpu" if want == "onnx" else config.resolved_device()
        if want == "onnx" and onnx is None:
            raise FileNotFoundError("backend=onnx but no model.onnx in %s" % config.model_name)
        if (onnx is not None and device == "cpu" and want != "torch"
                and _has("onnxruntime")):
            self.backend, self.device = "onnx", "cpu"
            log.info("loading embedder %s (onnxruntime) ...", onnx)
            self._ox = OnnxBackend(config.model_name, onnx)
            self.dim = self._ox.dim
        else:
            self.backend, self.device = "torch", device
            self._load_torch(config)
        # Measured on CPU (e5-small, 150-word chunks): batch 16 runs ~35% faster
        # than 128 (attention working sets stay in cache); GPUs want big batches.
        self.batch = int(getattr(config, "encode_batch_size", 0) or
                         (128 if self.device == "cuda" else 16))
        log.info("embedder ready (%s, dim=%d, batch=%d)", self.backend, self.dim, self.batch)

    def _load_torch(self, config):
        from sentence_transformers import SentenceTransformer
        import torch
        self._torch = torch
        log.info("loading embedder %s on %s ...", config.model_name, self.device)
        for kw in ({"dtype": torch.float32}, {"torch_dtype": torch.float32}, None):
            try:
                self.model = (SentenceTransformer(config.model_name, device=self.device,
                                                  model_kwargs=kw) if kw else
                              SentenceTransformer(config.model_name,
                                                  device=self.device).float())
                break
            except (TypeError, ValueError) as e:
                log.debug("model_kwargs %s rejected (%s); trying next", kw, e)
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts, prefix):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if prefix:
            texts = [prefix + t for t in texts]
        if self._ox is not None:
            return self._ox.encode(texts, self.batch)
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        v = self.model.encode(texts, batch_size=self.batch,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        return v.astype(np.float32)

    def embed_docs(self, texts):
        return self._encode(texts, self.cfg.doc_prefix)

    def embed_query(self, text):
        return self._encode([text], self.cfg.query_prefix)
