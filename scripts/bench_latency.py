"""End-to-end deep-search latency at scale (CPU, real query model).

    python scripts/bench_latency.py --model models/embed --chunks 100000

Builds a synthetic deep index of N chunks (random unit vectors + random
words: retrieval cost depends on N, not on what the text says), then times
the full SearchEngine.search_semantic() call -- query embedding with the real
model, exact FAISS search, BM25, fusion, language balance, personalization --
as the UI's first response ("results"), and separately the semantic snippets
that stream in after it.
"""
import argparse
import random
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = ["when did the reservoir reach a record low", "上个季度的发票报销流程是怎么改的",
           "laptop warranty broken screen", "出差住宿标准", "quarterly revenue review notes",
           "合同续签违约赔偿", "how long is maternity leave", "服务器维保合同到期"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", type=int, default=100_000)
    ap.add_argument("--files", type=int, default=0, help="default: chunks / 4")
    ap.add_argument("--top-k", type=int, default=20)
    a = ap.parse_args()
    import logging
    logging.disable(logging.WARNING)
    from localsearch import Config, SearchEngine
    from localsearch.chunker import tokenize
    from localsearch.index import DualIndex

    rng = np.random.default_rng(0)
    rnd = random.Random(0)
    vocab = [f"w{i}" for i in range(50_000)] + list("的一是在不了有和人这中大为上个国我以要他时来用们")
    n, files = a.chunks, a.files or max(1, a.chunks // 4)
    with tempfile.TemporaryDirectory() as tmp:
        eng = SearchEngine(Path(tmp) / "idx", Config(model_name=a.model))
        t0 = time.perf_counter()
        eng.warmup()
        load_s = time.perf_counter() - t0
        dim = eng.embedder.dim
        ix = DualIndex(dim)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        ix.dense.add_with_ids(vecs, np.arange(n, dtype=np.int64))
        for c in range(n):
            # ~10 sentences per chunk, like real prose (snippets embed each sentence)
            text = " ".join(" ".join(rnd.choice(vocab) for _ in range(12)) + "." for _ in range(10))
            path = f"/bench/f{c % files}.txt"
            ix.chunk_meta[c] = dict(path=path, idx=0, text=text, start=0, end=len(text))
            ix.file_chunks.setdefault(path, []).append(c)
            ix._tokens[c] = tokenize(text)
        for p in ix.file_chunks:
            ix.file_meta[p] = dict(mtime=0.0, size=0)
        ix.next_id = n
        ix._bm25 = None
        ix.rebuild_bm25()
        eng.index = ix
        eng.search_semantic("warm up", top_k=a.top_k, snippets="lexical")
        res, snip = [], []
        for q in QUERIES * 3:
            eng._q_cache = type(eng._q_cache)()         # cold: every query is embedded
            eng._sent_cache = type(eng._sent_cache)()
            t = time.perf_counter()
            hits = eng.search_semantic(q, top_k=a.top_k, snippets="lexical")
            res.append((time.perf_counter() - t) * 1000)
            sid = eng.queue_snippets("semantic", q, hits)
            t = time.perf_counter()
            eng.resolve_snippets(sid)
            snip.append((time.perf_counter() - t) * 1000)
        eng.close()
    print(f"backend={eng.embedder.backend}  model ready {load_s:.1f} s  "
          f"index {n:,} chunks / {files:,} files  top_k={a.top_k}")
    print(f"results (query -> ranked files): median {st.median(res):.0f} ms, "
          f"p90 {sorted(res)[int(0.9 * len(res))]:.0f} ms")
    print(f"semantic snippets after that:    median {st.median(snip):.0f} ms")


if __name__ == "__main__":
    main()
