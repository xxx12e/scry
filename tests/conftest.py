"""Shared fixtures. The engine is exercised with a deterministic hashing
embedder, so the suite needs no torch / model download and runs anywhere."""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localsearch.chunker import tokenize  # noqa: E402


class FakeEmbedder:
    """Bag-of-words hashed into `dim` buckets (stable across processes)."""
    dim = 64
    device = "cpu"

    def __init__(self, cfg=None):
        self.cfg = cfg

    def _vec(self, text):
        v = np.zeros(self.dim, np.float32)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else np.full(self.dim, 1 / np.sqrt(self.dim), np.float32)

    def embed_docs(self, texts):
        return (np.vstack([self._vec(t) for t in texts]) if texts
                else np.zeros((0, self.dim), np.float32))

    def embed_query(self, text):
        return self._vec(text)[None, :]


@pytest.fixture
def fake_embedder(monkeypatch):
    import localsearch.engine as E
    monkeypatch.setattr(E, "Embedder", FakeEmbedder)
    monkeypatch.setenv("LOCALSEARCH_NO_LAUNCH", "1")
    return FakeEmbedder


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "reservoir.txt").write_text(
        "The Larkfield reservoir dropped to a record low water level during the drought.",
        encoding="utf-8")
    (d / "latency.md").write_text(
        "# Bench\nThe radix index cut tail query latency to nine milliseconds.", encoding="utf-8")
    (d / "invoice.txt").write_text("今年第三季度的发票报销流程已经改为线上审批。", encoding="utf-8")
    return d


def settle(state, timeout=10):
    """Wait for create_app's background warm-up/catch-up so no thread outlives
    the test (and the monkeypatched embedder)."""
    import time
    state.warm_done.wait(timeout)
    deadline = time.time() + timeout
    while state.building and time.time() < deadline:
        time.sleep(0.01)
