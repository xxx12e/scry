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


def make_snippets(hits, qvec, embedder, max_chars=260,
                  max_sent_chars=300, max_sents=16):
    """Set each hit's snippet to its most query-relevant sentence (bounded cost)."""
    sent_lists = []
    for h in hits:
        sents = _split_sentences(h.chunk_text)[:max_sents]
        if len(sents) <= 1:
            best = sents[0] if sents else h.chunk_text
            h.snippet = best[:max_chars] + ("…" if len(best) > max_chars else "")
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
    embs = embedder.embed_docs(flat)
    sims = embs @ qvec[0]
    best_idx = {}
    best_sim = {}
    for j, i in enumerate(owner):
        if sims[j] > best_sim.get(i, -1e9):
            best_sim[i], best_idx[i] = sims[j], j
    for i, h in enumerate(hits):
        sents = sent_lists[i]
        if sents is None:
            continue
        local = sum(1 for k in range(best_idx[i]) if owner[k] == i)
        best = sents[local] if local < len(sents) else sents[0]
        snip = best[:max_chars] + ("…" if len(best) > max_chars else "")
        if local > 0:
            snip = "…" + snip
        h.snippet = snip


def retrieve(index, qvec, qtokens, top_k, alpha, fetch):
    """Hybrid retrieval -> top_k files with best chunk; call under the lock."""
    dense = index.search_dense(qvec, fetch)
    bm25 = index.search_bm25(qtokens, fetch)
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
