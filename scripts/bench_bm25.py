"""BM25 query/update cost: rank_bm25 (what v1.4 used) vs localsearch.bm25.

    pip install rank-bm25
    python scripts/bench_bm25.py --chunks 100000

Synthetic chunks of 120 tokens over a 50k-word vocabulary; one 5-term query.
Scores are identical (tests/test_bm25.py); only the cost differs.
"""
import argparse
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def ms(f, n=5):
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        xs.append((time.perf_counter() - t) * 1000)
    return st.median(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=100_000)
    a = ap.parse_args()
    from localsearch.bm25 import BM25Index
    rnd = random.Random(0)
    vocab = [f"w{i}" for i in range(50_000)]
    docs = [[rnd.choice(vocab) for _ in range(120)] for _ in range(a.chunks)]
    q = ["w12", "w345", "w6789", "w100", "w2222"]
    t = time.perf_counter()
    ix = BM25Index.build(range(len(docs)), docs)
    print(f"{a.chunks:,} chunks | localsearch.bm25: build {time.perf_counter() - t:.1f} s, "
          f"query {ms(lambda: ix.top(q, 100), 20):.2f} ms, "
          f"file change {ms(lambda: (ix.remove(0, docs[0]), ix.add(0, docs[0])), 50) * 1000:.0f} us/chunk")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("(pip install rank-bm25 for the v1.4 comparison)")
        return
    t = time.perf_counter()
    ref = BM25Okapi(docs)
    build = time.perf_counter() - t
    print(f"{a.chunks:,} chunks | rank_bm25 (v1.4):  build {build:.1f} s (= the refit after every "
          f"file change), query {ms(lambda: ref.get_scores(q), 3):.0f} ms")


if __name__ == "__main__":
    main()
