# `localsearch` — the Scry engine

The backend engine + local web UI behind [Scry](../README.md). Fully local/offline.
Factory config: **multilingual‑e5‑small embeddings (100+ languages, 384‑dim) + dense‑weighted
hybrid (dense + BM25, α = 0.7), no reranker.**

## Run

```bash
python -m localsearch.app          # http://127.0.0.1:8731 (localhost only)
```

Three modes in one box — **Whole‑disk Semantic** (rides the Windows content index, zero
pre‑indexing), **Filename** (Everything via `es.exe`), **Deep Semantic / Control mode**
(our own index over folders you choose; full scan once, diff‑only after, watched in the
background). UI is a single `web/index.html`, bilingual (中/EN toggle), no build step.

## Python API

```python
from localsearch import SearchEngine, Config

eng = SearchEngine("my_index")               # index dir (created/loaded automatically)
eng.build(["C:/Users/you/Documents"])        # scan → parse → chunk → embed → index → persist

for hit in eng.search_semantic("when did the reservoir overflow", top_k=10):
    print(hit.path, round(hit.score, 3), hit.snippet)

print(eng.search_filename("*.pdf"))          # Everything (instant)
eng.search_instant_global("neural network training")   # whole‑disk, no pre‑index

eng.start_watching(["C:/Users/you/Documents"])  # live background incremental
eng.close()                                      # flush + persist on exit
```

`build()` is incremental: unchanged files (mtime+size, or md5 with
`Config(change_detection="hash")`) are skipped, changed ones re‑indexed, deleted ones
dropped. A fresh `SearchEngine(same_dir)` loads from disk — no rebuild.

## Modules

| module | responsibility |
|---|---|
| `config.py` | factory defaults + all tunables (model, α, chunk size, caps, exclude dirs) |
| `parser.py` | robust text extraction, 10 formats: txt / md / csv (BOM → UTF‑8 → GBK sniff → system ANSI code page) · pdf (PyMuPDF) · docx / xlsx / pptx (stdlib OOXML: tables, headers, text boxes, one line per paragraph) · html (meta charset, block structure) · rtf (`\ansicpg` / `\fcharset` / `\uN` decoding) · images (opt‑in Windows OCR) |
| `chunker.py` | CJK‑aware sentence chunking with a hard size cap (no chunk the model would truncate) + BM25 tokenizer (Unicode words, case/diacritic‑folded, + CJK bigrams) |
| `embedder.py` | e5 wrapper (resident; query/passage prefixing; fp16‑checkpoint safe) |
| `index.py` | `DualIndex`: fp16 FAISS dense (exact) + BM25 + file→chunk map + crash‑durable persistence |
| `bm25.py` | incremental Okapi BM25 (scores identical to `rank_bm25`): numpy CSR base + live delta + tombstones — queries touch only their terms' postings, edits never refit |
| `search.py` | hybrid fusion → files + best‑sentence snippets; `es.exe` filename search |
| `winsearch.py` | Windows Search content index (`Search.CollatorDSO`) for instant global mode |
| `watcher.py` | single debounced scheduler thread → incremental updates |
| `personalize.py` | local, training-free re-rank: decayed open-counts + folder + interest vector |
| `engine.py` | `SearchEngine`: orchestrates build/load/search/watch; lock guards mutation only |

## Robustness & performance

- **Lock guards mutation, not compute.** Parse/chunk/embed run outside the engine lock, so
  search and the live progress bar stay responsive *during* a whole‑disk scan.
- **Crash‑durable index.** Saves are fsync'd, tmp‑written, renamed back‑to‑back, and stamped
  with a generation counter; a torn pair (power loss mid‑save) is detected on load and
  quarantined instead of silently corrupting IDs.
- **Decoupled persistence.** Correctness lives in memory; disk saves coalesce on a long
  cadence plus save‑on‑exit — an unsaved change self‑heals via the next diff rescan.
- **Bounded work under bursts.** Cross‑file embed batching, one batched FAISS removal per
  diff, a single watcher scheduler thread (survives bulk copies), and an LRU cache for
  instant‑mode candidate embeddings.
- Every per‑file failure (corrupt PDF, locked handle, bad encoding, oversized) is logged and
  skipped — never crashes a build or the watcher. Logs → `<index_dir>/engine.log`.

## Personalization (local, training-free)

Opening a result (`/api/open` → `engine.record_open`) updates a per-user profile in
`<index_dir>/personal.json`: decayed open-counts per file and folder, plus a decayed
interest centroid of the embeddings you open. Searches re-rank by blending the base score
with those priors (`Config.pers_*`), so **files and folders you actually open float up** —
recency-weighted. The blend keeps base relevance dominant by design (a fresh query is never
hijacked); the model is **never trained**. Toggle via `/api/personalize`, wipe via
`/api/forget`. See [`../demo_personalize.py`](../demo_personalize.py).

## Dependencies

See [`../requirements.txt`](../requirements.txt). Filename mode also needs
[Everything](https://www.voidtools.com/) installed + running and its `es.exe` at
`../tools/es.exe` (semantic modes work without it). First source run downloads the
`multilingual-e5-small` model (~470 MB) once, then runs fully offline.

## Packaging a portable, offline `.exe`

Bundle everything into a single self‑contained folder (no Python needed on the target):

- **PyInstaller onedir**, CPU‑only torch.
- Bundle as data: `web/`, `tools/es.exe`, and the model dir (`models/embed`) — resolved
  at runtime via `resources.py` (`sys._MEIPASS` when frozen).
- Ship the model as **fp16** on disk and upcast to fp32 at load (halves the asset; CPU
  compute stays fp32). `torch` and `transformers` must keep their loose source (both read
  their own source at runtime).
- `collect_all(..., include_py_files=False)` for the ML packages, exclude `hf_xet`, and set
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` in the entry point so nothing ever hits the
  network.

## Scope

Small local models cap semantic quality below datacenter LLMs — that's the trade for
"instant, private, offline." Flat FAISS is exact and great to ~10⁴–10⁵ chunks (`DualIndex` is
the single seam to swap dense → ANN at larger scale); the BM25 arm is an incremental inverted
index (`bm25.py`) whose query cost depends on the query's terms, not the corpus size. Bundled
model is multilingual (`multilingual-e5-small`, 100+ languages); cross‑language retrieval is
strong but has a mild same‑language bias. 10 formats + opt‑in image OCR (Windows built‑in engine).

## Local API safety

The server binds `127.0.0.1`, and additionally rejects any request whose `Host` or `Origin` is
not a loopback name — so a web page open in your browser can neither drive the API cross‑site
nor reach it via DNS rebinding. `/api/open` only launches files that a search actually returned.

## Settings

`<index_dir>/settings.json`: `ocr` (bool) and `startup_rescan` (bool, default `true` — a diff
scan of the controlled folders right after the model loads, so edits made while the app was
closed are picked up; unchanged files are skipped and nothing is rewritten).

## Tests

```bash
pip install -r requirements.txt pytest rank-bm25   # rank-bm25 only as the BM25 reference
python -m pytest
```

The suite runs on any OS without the model (a hashing stand‑in embedder); Windows‑only pieces
(Windows Search, OCR, `os.startfile`) are covered by the `demo_*.py` / `test_ocr.py` scripts.
