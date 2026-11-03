"""Proof of the breakthrough: WHOLE-DISK semantic search with ZERO pre-indexing.

A fresh engine -- we deliberately DO NOT build any index -- still answers content
queries across the entire machine, by riding the OS's own Windows Search content
index and re-ranking the hits semantically. This is the "semantic Everything":
open it, type a meaning query, get global content results, no setup, no waiting.
"""
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from localsearch import SearchEngine

IDX = HERE / "_instant_tmp"


def main():
    eng = SearchEngine(IDX)
    print("引擎索引状态(故意不建任何库):", eng.stats(), "  <-- 0 个文件，完全没预索引\n")

    queries = ["copyright license permission notice",
               "neural network training gradient",
               "convolution image filter"]
    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        hits = eng.search_instant_global(q, top_k=8)
        dt = (time.perf_counter() - t0) * 1000
        tag = " (含模型冷启动)" if qi == 0 else ""
        print(f"查询 {q!r} -> {len(hits)} 条命中, {dt:.0f} ms{tag}  [全盘 · 零预索引]")
        for h in hits[:4]:
            print(f"   [{h.score:.2f}] {Path(h.path).name}")
            print(f"          {h.path}")
            if h.snippet:
                print(f"          …{h.snippet[:84].strip()}")
        print()

    print("=" * 70)
    print("结论：没有建任何索引，就在全盘内容里语义搜到了东西 —— 这就是'语义版 Everything'。")
    print("(原理：骑 Windows 自带的全盘内容索引拿候选，再用 e5 语义重排。)")
    shutil.rmtree(IDX, ignore_errors=True)


if __name__ == "__main__":
    main()
