<div align="center">

🌐 **English** · [简体中文](README.zh.md)

# 🔮 Scry

### Everything finds filenames. **Scry finds meaning.**

Instant, whole‑disk **semantic** file search that runs **100% offline**.
Type what you *mean* — not the exact words — and find the file. Your files never leave your machine.

`全盘语义搜索 · 秒懂你要找什么 · 纯本地、不联网、隐私零外泄`

**Windows · 100+ languages · CPU or GPU · no cloud, no account, no telemetry**

### Search in 中文, find the English PDF — no filename to remember, nothing uploaded.

**⬇ [Download Scry for Windows](https://github.com/xxx12e/scry/releases/latest)** — unzip, double‑click the `.exe`. No Python, no install, 100% offline (~477 MB).

<sub>Unsigned build, so Windows may ask you to confirm — verify the <a href="#is-it-safe-to-run">SHA‑256</a> if you like.</sub>

</div>

---

## Search your files by what they *mean*

You remember *what a file was about*. You don't remember its name, or which of five drives it's on, or whether you wrote "invoice" or "发票" or "billing". Keyword search fails you exactly there.

**Scry searches by meaning.** At its core is a **purpose‑built multilingual semantic model** — a compact **118‑million‑parameter embedding network** that understands *what text means* across **100+ languages**, **not** a bloated multi‑gigabyte local LLM bolted onto a search box. It **re‑ranks whole‑disk candidates with that model** in milliseconds, and — when you want the very best quality — it **builds a complete deep semantic index of the folders you choose**, embedding every passage so it can match your *idea* against files that share **zero words** with your query — **even in a different language** (search 中文, find English files, and vice versa).

A real semantic engine, not a keyword matcher with a marketing label — and its model is **235 MB, lightweight, and 100% on your machine.**

> It's the search **Everything** should have grown into: same instant, whole‑disk reflex — but it understands you.

## By the numbers

| | |
|---|---|
| ⚡ **~15–35 ms** to results over **100,000 chunks — on a plain 4‑core CPU** | query embed 5–21 ms + exact FAISS 6 ms + BM25 0.3 ms; the best‑sentence snippets stream in right after |
| 🌐 **~0.1 s** whole‑disk semantic results | **zero pre‑indexing** — rides the OS content index |
| 🪶 **118M params · 235 MB** multilingual model | vs the **4–8 GB** local LLMs other "AI search" tools ship |
| 🌍 **100+ languages · cross‑language** | search 中文 → find English files: measured **1/10 → 10/10** vs an English‑only model |
| 🔁 **~35 s** to index 100k files on a GPU, **~0.1 s** to re‑scan | incremental diff measured **235× faster** than a full scan; ~20 chunks/s on a 4‑core CPU |
| 🎯 matches paraphrases with **zero shared words** | exact FAISS + BM25 hybrid, fp16 — half the RAM/disk |
| 🔒 **0 bytes** leave your machine | fully offline · binds `127.0.0.1` · no account, no telemetry |
| 📄 **10 formats** (txt·md·pdf·docx·xlsx·pptx·csv·html·rtf) · no Docker | single `.exe` · optional add‑on packs (DLC) |
| 🖼️ **Reads images too** — opt‑in OCR | screenshots / scans / photos searchable **by meaning** · Windows built‑in engine · **no model download** |

<sub>CPU figures measured with the shipped model on a 4‑core 2.1 GHz CPU, no GPU: results in ~15–35 ms at 100k chunks; the semantic snippets for 20 results follow in ~0.4–1.2 s. Indexing speed is the one place a GPU matters.</sub>

## Three ways to search, one box

| Mode | What it does | Setup |
|---|---|---|
| 🌐 **Whole‑disk Semantic** | Search the *content* of every file on the machine by meaning. Rides the Windows Search content index, then re‑ranks the hits with a neural model. | **None.** Open and go. |
| ⚡ **Filename** | Instant whole‑disk filename search, the way you already know it. | None. |
| 🎯 **Deep Semantic (Control Mode)** | Point it at folders (or a whole drive) and it builds its *own* deep index — the highest quality, catches paraphrases with **zero word overlap**. First scan is full; after that it only touches what changed, and keeps itself up to date in the background. | One click per folder. |

> **New here? Pick the right mode for your first try.** *Whole‑disk Semantic* is instant and needs zero setup, but it draws its candidates from the Windows index — so a **pure paraphrase or cross‑language query with no shared words** (exactly the "wow" demo) can come up empty there. For that, use **Deep Semantic**: add the folder once, let it index, and it matches *meaning* even when nothing overlaps. Rule of thumb — **Whole‑disk = broad & instant; Deep = deepest meaning.**

**Filters, in plain words.** Add `type:pdf`, `time:上周` / `time:lastweek`, `in:合同`, `after:2024-03` to any search — or *only* filters, e.g. `time:上周 type:pdf`, to list last week's PDFs newest‑first. Filters are applied *inside* retrieval, so a narrow filter never starves the results. A time word in your sentence ("去年的工作总结") is **not** silently applied — that summary was probably written this year — Scry offers it as a one‑click filter instead.

**Find similar.** Every result has a *Find similar* button: Scry uses that file's own meaning as the query and lists the closest other files in your deep index — even ones that share no words with it.

Plus: **bilingual UI (中文 / English, one‑click toggle)**, keyboard‑driven (`Enter` to search, `↑/↓` to move, `Enter` to open), live‑highlighted snippets that show *why* a file matched, and a portable build that runs on any Windows PC with **no Python install**.

**Optional add‑on packs (DLC).** The base stays tiny and 100% offline; heavier extras (starting with a larger multilingual model) download **only when you click**, from this repo's releases, SHA‑256‑verified — see **Add‑ons** in the app.

**Read text inside images (opt‑in OCR).** Flip on OCR in Control mode and Scry reads the text in your **screenshots, scans, and photos** with the **Windows built‑in OCR engine** — so a screenshotted receipt or a scanned contract becomes searchable **by meaning**, in whatever languages your Windows has OCR packs for (English + 中文 out of the box on most PCs). It downloads **no model** (Windows provides the engine and language data), stays **100% offline**, adds only ~4 MB to the app, and is **off by default** — turn it on and rescan. It's for *finding*, not perfect transcription: stylized or low‑res text can come out noisy, but semantic search shrugs that off.

## It learns *you* — on your machine

The more you use Scry, the better it fits you: the **files and folders you actually open float toward the top** for similar searches, weighted toward your recent habits. It's a gentle re‑rank with a hard rule: a familiar file can rise from far down the list, but it **never displaces the best match for your query** unless the two are practically tied — and a fresh query is **never** hijacked. And it's personalization with nothing to give up: it learns **only from your own clicks, on your own disk, and never trains the model or sends anything anywhere.** Toggle it off or hit **Forget me** whenever you like.

## The privacy stance (the whole point)

- **Read‑only.** Scry only ever *reads* your files to index them — it never modifies, moves, renames, or deletes them. Your originals are untouched.
- **Zero outbound connections.** The base app never talks to the network: it binds `127.0.0.1` only (not even your LAN), bundles the model, and has no server to call — no accounts, no telemetry, no "anonymous usage stats." Unplug the network and everything still works. The **only** time Scry reaches the internet is if *you* click to download an optional add‑on pack, from this repo's GitHub Releases, SHA‑256‑verified. (Image OCR uses a local Windows API — it doesn't go online either.)
- **Fully local storage.** Your index and settings live in `~/.localsearch_index` on your own disk; delete that folder to wipe everything.
- **Light footprint.** CPU‑only (no GPU required); the bundled model is ~235 MB, and a 100k‑file deep index is ~154 MB on disk. Whole‑disk Semantic and Filename modes keep no index at all.

## Get it (no Python needed)

**[⬇ Download the latest release](https://github.com/xxx12e/scry/releases/latest)**, unzip anywhere, and double‑click **`本地语义搜索器.exe`** (or **`Run 运行.bat`**). A console window opens — keep it open; your browser opens the search page in a few seconds. That's it: any Windows 10/11 PC, fully offline, nothing to install.

<a id="is-it-safe-to-run"></a>
**Is it safe to run?** Yes, and here's how to check rather than take my word for it:

- It's an **unsigned** build, so Windows SmartScreen may say *"Windows protected your PC."* Click **More info → Run anyway**.
- **Antivirus false positives are expected.** This is an *unsigned, PyInstaller‑packaged* app, and the PyInstaller bootloader is a byte pattern that some heuristic/ML antivirus engines flag on sight — regardless of what the code does. A VirusTotal scan will likely show **a handful of ~70 engines** flagging it (usually minor ones, rarely Microsoft Defender). That's normal for legit Python apps built this way and is **not** evidence of malware. Your real assurances: the whole thing is **open source — read it, or build the `.exe` yourself**; the **SHA‑256 above**; and its **read‑only, offline** behavior. Want more certainty? Run it in **Windows Sandbox**.
- Verify the download is exactly the file I published — the SHA‑256 of `Scry-v1.4-win64-portable.zip` is:

```
967e132346c158eaeea8e07871184a39795e73c3d4e7e083a9521c5b9ee21162
```
```powershell
Get-FileHash Scry-v1.4-win64-portable.zip -Algorithm SHA256   # should match the line above
```

- It **only reads** your files and **never phones home** — see [The privacy stance](#the-privacy-stance-the-whole-point) below.

**Run from source (developers)**

```bash
pip install -r requirements.txt
python -m localsearch.app          # opens http://127.0.0.1:8731 in your browser
```

or double‑click **`run.bat`** (Windows). Packaging recipe: [`localsearch/README.md`](localsearch/README.md).

## How it works

```
        your query ("the reservoir that hit a record low")
                              │
        ┌─────────────────────┼───────────────────────────┐
   Whole‑disk              Filename                    Deep Semantic
   (Windows Search   →     (Everything‑style,          (own FAISS index of
    content index)          MFT filename)               your chosen folders)
        │                                                    │
        └──────────────►  neural re‑rank · Scry semantic model  ◄─┘
                              │
                     ranked files + the exact matching sentence, highlighted
```

- **Scry's semantic model:** a compact **118M‑param multilingual embedding network** (`multilingual-e5-small`, 384‑dim, **100+ languages**), purpose‑built to represent *meaning* — not a text‑generating LLM. Runs on CPU, no GPU required, and **never trains on your files**.
- **Retrieval:** exact FAISS (fp16 scalar‑quantized — half the RAM/disk, still exhaustive) fused with a BM25 lexical arm; CJK‑aware chunking + bigram tokenizer so Chinese documents are fully searchable.
- **Instant mode:** queries the OS content index via the `Search.CollatorDSO` provider, then embeds and re‑ranks the candidates — global content search with **zero pre‑indexing**.
- **Robust by design:** crash‑durable index writes (fsync + generation‑stamped, torn saves detected and quarantined), save‑on‑exit, live progress and search *during* indexing, and a single‑thread watcher that survives bulk file copies.

Scry is **multilingual** — one model covering 100+ languages, so you can search in one language and find files written in another (Chinese and English are both first‑class). Cross‑language retrieval is strong but not magic: a query can occasionally lose to a topically‑adjacent same‑language file. The promise isn't to beat a datacenter on a benchmark — it's cloud‑grade *convenience* with **zero** privacy cost.

## Project layout

```
localsearch/          the engine + local web UI (this is the product)
  ├─ engine.py        orchestrator: scan → parse → chunk → embed → index → search
  ├─ index.py         dual index: fp16 FAISS dense + BM25, crash‑durable persistence
  ├─ bm25.py          incremental BM25 (inverted index; edits never refit)
  ├─ winsearch.py     the "instant whole‑disk" breakthrough (Windows content index)
  ├─ search.py        hybrid retrieval + best‑sentence snippets
  ├─ watcher.py       debounced background re‑indexing
  ├─ personalize.py   local, training‑free re‑rank from what you open
  ├─ app.py           Flask backend (127.0.0.1 only) + JSON API
  └─ web/index.html   single‑page UI, bilingual, zero build step
demo_*.py             runnable end‑to‑end examples (incl. demo_personalize.py)
tests/                pytest suite (any OS, no model download): python -m pytest
run.bat               double‑click launcher
```

## License & credits

**[PolyForm Noncommercial 1.0.0](LICENSE.md)** — free to use, study, modify, and share for any **noncommercial** purpose (personal use, research, education, nonprofits). Commercial use is not granted by this license; reach out if you want to talk.

<sub>Scry stands on excellent open source: its embedding weights come from the open **multilingual‑e5** model (intfloat, MIT), retrieval uses **FAISS** (MIT), and filename mode talks to **Everything** (`es.exe`, voidtools — download it yourself). These remain under their own terms.</sub>

---

<div align="center">
<sub>Built for people who remember <i>what</i> they wrote, not <i>where</i> they put it.</sub>
</div>
