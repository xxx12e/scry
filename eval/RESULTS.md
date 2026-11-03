# Scry Bilingual Retrieval Eval (v1) — results

> **中文摘要：** 在自带的中英双语评测集（60 篇文档、102 条查询）上，用内置的 235 MB 模型：跨语言查询的 **#1 从 13% → 77%**、**Top 5 从 43% → 100%**，总体 MRR **0.746 → 0.948**；逐条比对，**30 条查询排名上升、0 条下降**；关键词与精确短语查询保持 100%，纯中文语料同样没有任何查询变差。普通 4 核 CPU 上，10 万文本块的深度搜索从查询到出结果 **23 ms**（端到端，真实模型）。评测集小、由人工编写——请把它当成"明显更好"的证据，而不是排行榜成绩；所有原始结果都在 `eval/results/`，一条命令即可复现。

Everything below was produced by the scripts in this repo; the raw outputs are in
[`results/`](results/). Numbers are for the current code (the next release) — the v1.4
download predates them. "v1.4 settings" = dense weight α = 0.7, no language balance;
"now" = α = 0.8 + language balance 1.0 (the new defaults).

## The eval set

[`scry_eval_v1.json`](scry_eval_v1.json) — **60 short documents** (30 Chinese, 30 English;
30 topics, each target paired with a *same‑language look‑alike decoy*, e.g. travel
reimbursement vs. office‑supply claims) and **102 queries**:

| kind | n | what it tests |
|---|---|---|
| paraphrase | 30 | same language, few shared words ("出去办事住酒店最多能报多少钱") |
| cross‑language | 30 | the other language ("hotel cost limit when travelling for work") |
| keywords | 30 | a few content words |
| exact | 12 | literal numbers / phrases ("600元 450元", "GB/T 7714") |

**Limits, stated up front:** it is small and hand‑written by us, so it is biased toward how we
phrase things, and settings were chosen on it (we checked robustness on held‑back corpus
variants below, but there is no separate test set). Use it to compare configurations; check
on your own folders before trusting any absolute number.

## Retrieval quality — bundled model (multilingual‑e5‑small, 235 MB)

`python scripts/eval.py --model models/embed` (add `--alphas 0.7 --balance 0` for v1.4 settings)

| queries (n) | R@1 v1.4 → now | R@5 v1.4 → now | MRR v1.4 → now |
|---|---|---|---|
| cross‑language (30) | **0.133 → 0.767** | **0.433 → 1.000** | 0.282 → 0.878 |
| paraphrase (30) | 0.767 → 0.900 | 1.000 → 1.000 | 0.856 → 0.944 |
| keywords (30) | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| exact phrases (12) | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| **all (102)** | **0.676 → 0.902** | **0.833 → 1.000** | **0.746 → 0.948** |

R@1 = share of queries with the right file at **#1**; R@5 = in the **Top 5**; MRR = mean of 1/rank (1.0 = always #1).

Raw: [`e5small_v1.4.json`](results/e5small_v1.4.json), [`e5small_now.json`](results/e5small_now.json).
The ONNX backend gives identical results to torch (same vectors, min cosine 0.99999994).

**Why cross‑language was weak:** multilingual embedders score same‑language text higher
across the board, so a Chinese query ranked Chinese look‑alikes above the English file it was
about. The fix (`search.lang_balance`) lifts the other‑language group of dense candidates by the
gap between the two groups' top‑10 mean similarity before fusion; it does nothing when a folder
is in one language.

### Robustness: language‑imbalanced and monolingual corpora

| corpus | queries | MRR v1.4 → now |
|---|---|---|
| Chinese only (30 docs) | all 51 that target them | 0.977 → **0.990** |
| mostly Chinese (30 zh + 4 en docs) | the 37 Chinese queries | 0.914 → **0.973** |

`--keep-en 0` / `--keep-en 2 --qlang zh`; raw: `results/stress_*.json`.

**Per query, nothing got worse.** Comparing each query's rank of the right file, v1.4
settings vs. now: full set — 30 of 102 ranked higher, **0 lower**; Chinese‑only — 2 of 51
higher, 0 lower; mostly‑Chinese — 4 of 37 higher, 0 lower.

### Other models

| model | size | MRR v1.4 → now | notes |
|---|---|---|---|
| e5‑small (bundled) | 235 MB | 0.746 → 0.948 | |
| e5‑base (add‑on pack) | +520 MB | 0.787 → 0.954 | better paraphrase R@1 (0.967), same cross‑language, exact 0.917 |
| e5‑small int8 (experimental) | 118 MB | — → 0.948 | needs an index rebuild (different vectors) |

So the free fix (+0.20 MRR) matters far more than the larger model (+0.006 on top of it).

### Relevance floor (dimmed "less related" tail)

With balanced cosine, no relevant hit in the set sat more than **0.012** below the best hit;
the UI dims hits more than **0.05** below it — **54%** of irrelevant hits, **0** relevant ones.

## Speed — plain 4‑core CPU (Intel Xeon 2.1 GHz), no GPU

| | before | now | how |
|---|---|---|---|
| query → ranked files, **100,000 chunks / 25,000 files** | list waited for the snippets: 32 ms + 1.5 s (same run, torch) | **23 ms** median (p90 27) ONNX · 32 ms torch | `scripts/bench_latency.py`, end‑to‑end with the real query model |
| best‑sentence snippets for 20 results | blocked the list | stream in after: 0.4–1.5 s (text‑dependent) | same; 300‑file real‑text run: 0.4–0.55 s |
| model ready | 7.6–8.9 s (torch) | **1.5–2.9 s** (ONNX) | incl. one warm‑up query |
| BM25 query, 100k chunks | 121 ms (rank_bm25) | **0.30 ms** | `scripts/bench_bm25.py`, identical scores |
| after a file changes | full BM25 refit 3.7 s | 33 µs per chunk | same (our one‑time startup build is slower: 6.8 s vs 3.7 s, in the background) |
| indexing throughput | batch 128 | batch 16: **~35% faster** on CPU | sentence‑transformers, same model |

Raw: [`results/bench_100k_onnx.txt`](results/bench_100k_onnx.txt),
[`bench_100k_torch.txt`](results/bench_100k_torch.txt), [`bench_bm25_100k.txt`](results/bench_bm25_100k.txt).

## Size

The v1.4 Windows bundle unpacks to 1,026 MB, of which 523 MB are packages the torch‑free build
does not need (torch, transformers, scipy, scikit‑learn, …). The torch‑free spec
(`packaging/scry.spec`) built on Linux is **516 MB** with the 235 MB model inside; a Windows
build is still to be measured.

## Reproduce

```bash
pip install -r requirements.txt pytest rank-bm25 onnxruntime
python scripts/export_onnx.py models/embed          # optional: ONNX backend
python scripts/eval.py --model models/embed --json eval/results/mine.json
python scripts/bench_latency.py --model models/embed --chunks 100000
python scripts/bench_bm25.py --chunks 100000
python scripts/make_eval_chart.py                    # README charts from eval/results/
```
