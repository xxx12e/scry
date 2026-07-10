<div align="center">

🌐 **English** · [简体中文](README.zh.md)

# 🏮 Fathom

### Everything finds filenames. **Fathom finds meaning.**

Instant, whole‑disk **semantic** file search that runs **100% offline**.
Type what you *mean* — not the exact words — and find the file. Your files never leave your machine.

`全盘语义搜索 · 秒懂你要找什么 · 纯本地、不联网、隐私零外泄`

**Windows · CPU or GPU · no cloud, no account, no telemetry**

</div>

---

## Why Fathom

You remember *what a file was about*. You don't remember its name, or which of five drives it's on, or whether you wrote "invoice" or "发票" or "billing". Keyword search fails you exactly there.

Fathom searches by **meaning**. It rides the index Windows already built, adds a neural re‑ranking layer on top, and — when you want the best quality — builds its own deep semantic index over the folders you choose. All of it on your own machine, air‑gap‑friendly, nothing phoned home.

> It's the search **Everything** should have grown into: same instant, whole‑disk reflex — but it understands you.

## Three ways to search, one box

| Mode | What it does | Setup |
|---|---|---|
| 🌐 **Whole‑disk Semantic** | Search the *content* of every file on the machine by meaning. Rides the Windows Search content index, then re‑ranks the hits with a neural model. | **None.** Open and go. |
| ⚡ **Filename** | Instant whole‑disk filename search, the way you already know it. | None. |
| 🎯 **Deep Semantic (Control Mode)** | Point it at folders (or a whole drive) and it builds its *own* deep index — the highest quality, catches paraphrases with **zero word overlap**. First scan is full; after that it only touches what changed, and keeps itself up to date in the background. | One click per folder. |

Plus: **bilingual UI (中文 / English, one‑click toggle)**, keyboard‑driven (`Enter` to search, `↑/↓` to move, `Enter` to open), live‑highlighted snippets that show *why* a file matched, and a portable build that runs on any Windows PC with **no Python install**.

## The privacy stance (the whole point)

- **Fully local.** The embedding model is bundled. Unplug the network and everything still works.
- **Binds `127.0.0.1` only.** Nothing is served to your LAN, let alone the internet.
- **No accounts, no telemetry, no "anonymous usage stats."** There is no server to call.
- Your index and settings live in `~/.localsearch_index` on your own disk.

## Quick start

**Run from source**

```bash
pip install -r requirements.txt
python -m localsearch.app          # opens http://127.0.0.1:8731 in your browser
```

or just double‑click **`run.bat`** (Windows).

**Portable build (no Python needed)** — package it into a single self‑contained folder with a bundled CPU model, then double‑click the `.exe`. Any Windows machine, fully offline. See [`localsearch/README.md`](localsearch/README.md) for the packaging recipe.

## How it works

```
        your query ("the reservoir that hit a record low")
                              │
        ┌─────────────────────┼───────────────────────────┐
   Whole‑disk              Filename                    Deep Semantic
   (Windows Search   →     (Everything‑style,          (own FAISS index of
    content index)          MFT filename)               your chosen folders)
        │                                                    │
        └──────────────►  neural re‑rank (bge embeddings)  ◄─┘
                              │
                     ranked files + the exact matching sentence, highlighted
```

- **Embeddings:** off‑the‑shelf `BAAI/bge-base-en-v1.5` (768‑dim). No training, ever.
- **Retrieval:** exact FAISS (fp16 scalar‑quantized — half the RAM/disk, still exhaustive) fused with a BM25 lexical arm; CJK‑aware chunking + bigram tokenizer so Chinese documents are fully searchable.
- **Instant mode:** queries the OS content index via the `Search.CollatorDSO` provider, then embeds and re‑ranks the candidates — global content search with **zero pre‑indexing**.
- **Robust by design:** crash‑durable index writes (fsync + generation‑stamped, torn saves detected and quarantined), save‑on‑exit, live progress and search *during* indexing, and a single‑thread watcher that survives bulk file copies.

Honest scope: Fathom's semantic quality is "genuinely useful," not magic — small local models top out well below a datacenter. The bundled model is English‑first; Chinese search works (structure + lexical + bigram BM25), and you can drop in `bge-base-zh` if Chinese is your main language. The point isn't to beat the cloud on a benchmark — it's to give you cloud‑grade *convenience* with **zero** privacy cost.

## Project layout

```
localsearch/          the engine + local web UI (this is the product)
  ├─ engine.py        orchestrator: scan → parse → chunk → embed → index → search
  ├─ index.py         dual index: fp16 FAISS dense + BM25, crash‑durable persistence
  ├─ winsearch.py     the "instant whole‑disk" breakthrough (Windows content index)
  ├─ search.py        hybrid retrieval + best‑sentence snippets
  ├─ watcher.py       debounced background re‑indexing
  ├─ app.py           Flask backend (127.0.0.1 only) + JSON API
  └─ web/index.html   single‑page UI, bilingual, zero build step
demo_*.py             runnable end‑to‑end examples
run.bat               double‑click launcher
```

## License

**[PolyForm Noncommercial 1.0.0](LICENSE.md)** — free to use, study, modify, and share for any **noncommercial** purpose (personal use, research, education, nonprofits). Commercial use is not granted by this license; reach out if you want to talk.

Bundled/adjacent third parties are **not** included in this repo and remain under their own terms: the embedding model (BAAI, MIT), FAISS (MIT), and Everything's `es.exe` (voidtools — download it yourself for filename mode).

---

<div align="center">
<sub>Built for people who remember <i>what</i> they wrote, not <i>where</i> they put it.</sub>
</div>
