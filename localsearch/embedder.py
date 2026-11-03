"""Embedding model wrapper (multilingual-e5-small by default). Loaded once and kept resident."""
import logging
import numpy as np

log = logging.getLogger("localsearch.embedder")


class Embedder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        import torch
        self.cfg = config
        self.device = config.resolved_device()
        self._torch = torch
        log.info("loading embedder %s on %s ...", config.model_name, self.device)


        self.model = None
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
        log.info("embedder ready (dim=%d)", self.dim)

    def _encode(self, texts, prefix):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if prefix:
            texts = [prefix + t for t in texts]
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        v = self.model.encode(texts, batch_size=self.cfg.embed_batch_size,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        return v.astype(np.float32)

    def embed_docs(self, texts):
        return self._encode(texts, self.cfg.doc_prefix)

    def embed_query(self, text):
        return self._encode([text], self.cfg.query_prefix)
