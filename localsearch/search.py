"""Filename (Everything) + hybrid dense/BM25 semantic search."""
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunker import tokenize

log = logging.getLogger("localsearch.search")

_SENT_SPLIT = re.compile(r"(?<=[。！？；])|(?<=[.!?])\s+|\n+")


@dataclass
class Hit:
    path: str
    score: float
    chunk_text: str
    char_start: int
    char_end: int
    snippet: str = ""
    terms: list = field(default_factory=list)
    cid: int = -1                 # winning chunk id (deep mode) -> vec reconstruct
    pscore: float = None          # personalized score (set by Personalizer.rerank)
    personal: bool = False        # True if history boosted this hit


def _minmax(pairs, floor=None):
    """Min-max normalize; `floor` lowers the minimum (BM25: chunks that were not
    returned score 0, and rank_bm25 used to return them as zero-score padding)."""
    if not pairs:
        return {}
    vs = [s for _, s in pairs]
    lo, hi = min(vs), max(vs)
    if floor is not None:
        lo = min(lo, floor)
    rng = (hi - lo) or 1.0
    return {cid: (s - lo) / rng for cid, s in pairs}


def _split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


class VecCache:
    """Small thread-safe LRU of text -> embedding (sentences, queries): refining
    a query re-ranks mostly the same files, so their sentences are reused."""

    def __init__(self, cap=20000):
        import threading
        from collections import OrderedDict
        self.cap, self._d, self._lock = cap, OrderedDict(), threading.Lock()

    def get_many(self, keys):
        with self._lock:
            out = []
            for k in keys:
                v = self._d.get(k)
                if v is not None:
                    self._d.move_to_end(k)
                out.append(v)
            return out

    def put_many(self, keys, vecs):
        with self._lock:
            for k, v in zip(keys, vecs):
                self._d[k] = v
                self._d.move_to_end(k)
            while len(self._d) > self.cap:
                self._d.popitem(last=False)


def _cut(best, local, max_chars):
    snip = best[:max_chars] + ("…" if len(best) > max_chars else "")
    return ("…" + snip) if local > 0 else snip


def lexical_snippet(chunk_text, qtokens, max_chars=260, max_sents=16):
    """Instant, model-free snippet: the sentence sharing the most query terms
    (the chunk's first sentence when nothing overlaps). Shown while the
    semantic snippet is computed."""
    sents = _split_sentences(chunk_text)[:max_sents]
    if not sents:
        return _cut(chunk_text, 0, max_chars)
    q = set(qtokens or ())
    best_i, best_n = 0, 0
    if q:
        for i, s in enumerate(sents):
            n = len(q.intersection(tokenize(s)))
            if n > best_n:
                best_i, best_n = i, n
    return _cut(sents[best_i], best_i, max_chars)


def make_snippets(hits, qvec, embedder, max_chars=260,
                  max_sent_chars=300, max_sents=16, cache=None):
    """Set each hit's snippet to its most query-relevant sentence (bounded cost).
    `cache` (VecCache) skips re-embedding sentences seen in earlier searches."""
    sent_lists = []
    for h in hits:
        sents = _split_sentences(h.chunk_text)[:max_sents]
        if len(sents) <= 1:
            best = sents[0] if sents else h.chunk_text
            h.snippet = _cut(best, 0, max_chars)
            sent_lists.append(None)
        else:
            sent_lists.append(sents)
    flat, owner = [], []
    for i, sents in enumerate(sent_lists):
        if sents is None:
            continue
        for s in sents:
            flat.append(s[:max_sent_chars])
            owner.append(i)
    if not flat:
        return
    if cache is not None:
        got = cache.get_many(flat)
        miss = [j for j, v in enumerate(got) if v is None]
        if miss:
            new = embedder.embed_docs([flat[j] for j in miss])
            cache.put_many([flat[j] for j in miss], list(new))
            for k, j in enumerate(miss):
                got[j] = new[k]
        embs = np.vstack(got)
    else:
        embs = embedder.embed_docs(flat)
    sims = embs @ qvec[0]
    best_idx = {}
    best_sim = {}
    for j, i in enumerate(owner):
        if sims[j] > best_sim.get(i, -1e9):
            best_sim[i], best_idx[i] = sims[j], j
    first = {}
    for j, i in enumerate(owner):
        first.setdefault(i, j)
    for i, h in enumerate(hits):
        sents = sent_lists[i]
        if sents is None:
            continue
        local = best_idx[i] - first[i]
        best = sents[local] if local < len(sents) else sents[0]
        h.snippet = _cut(best, local, max_chars)


def retrieve(index, qvec, qtokens, top_k, alpha, fetch, allowed=None):
    """Hybrid retrieval -> top_k files with best chunk; call under the lock.
    `allowed`: optional int64 array of chunk ids (query filters)."""
    dense = index.search_dense(qvec, fetch, allowed)
    bm25 = index.search_bm25(qtokens, fetch, allowed)
    # fewer matches than `fetch`: the unreturned chunks score 0, so normalize
    # against 0 -- exactly what the zero-padded rank_bm25 top-k used to give
    dn = _minmax(dense)
    bn = _minmax(bm25, floor=0.0 if len(bm25) < fetch else None)
    fused = {}
    for cid in set(dn) | set(bn):
        fused[cid] = alpha * dn.get(cid, 0.0) + (1 - alpha) * bn.get(cid, 0.0)

    best_per_file = {}
    for cid, sc in sorted(fused.items(), key=lambda x: -x[1]):
        meta = index.chunk_meta.get(cid)
        if meta is None:
            continue
        p = meta["path"]
        if p not in best_per_file:
            best_per_file[p] = Hit(p, sc, meta["text"], meta["start"], meta["end"],
                                   terms=qtokens, cid=cid)
        if len(best_per_file) >= top_k:
            break
    return list(best_per_file.values())


def semantic_search(index, embedder, query, top_k, alpha, fetch):
    """Convenience wrapper (demos): embed + retrieve + snippets in one call."""
    qvec = embedder.embed_query(query)
    qtokens = tokenize(query)
    index.ensure_bm25()
    hits = retrieve(index, qvec, qtokens, top_k, alpha, fetch)
    make_snippets(hits, qvec, embedder)
    return hits


def filename_search(es_exe, query, limit, timeout):
    """Everything filename search via es.exe -> list of paths. None if unavailable."""
    if not Path(es_exe).exists():
        log.warning("es.exe not found at %s -> filename search unavailable", es_exe)
        return None
    try:
        out = subprocess.run([str(es_exe), "-n", str(limit), query],
                             capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
        return [ln for ln in out.stdout.splitlines() if ln.strip()]
    except Exception as e:
        log.warning("es.exe call failed: %s: %s", type(e).__name__, e)
        return None
