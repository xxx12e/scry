# Packaging Scry

Two outputs from one PyInstaller spec (`packaging/scry.spec`), both **torch‑free**:

| | how | trust / updates |
|---|---|---|
| Portable zip (as today) | `pyinstaller packaging/scry.spec` → zip `dist/Scry` | unsigned → SmartScreen warning, AV false positives |
| **Microsoft Store (MSIX)** | `packaging/build_msix.ps1` → upload `dist/Scry.msix` | **signed by the Store for free, no SmartScreen warning, Store auto‑updates** |

## 1. Build the torch‑free bundle

```bash
pip install -r requirements-slim.txt pyinstaller
pip install torch transformers onnx onnxruntime        # export step only
python scripts/export_onnx.py models/embed              # writes models/embed/model.onnx (once)
pyinstaller packaging/scry.spec                          # -> dist/Scry/Scry(.exe)
```

Verified on Linux (the same spec): the frozen app starts, loads `models/embed/model.onnx`
through onnxruntime (~1.5 s), indexes and searches; the bundle is ~516 MB with the 235 MB
model inside — vs 1026 MB for the v1.4 torch bundle. Excluded on purpose: torch, transformers,
sentence‑transformers, scipy, scikit‑learn, onnx, pandas (see the spec).

Windows specifics the spec adds as hidden imports: pywin32 (Windows Search), the winrt OCR
modules, pywebview + pystray (desktop shell). `SCRY_CONSOLE=0` builds a windowed exe (use it with
`--window`). Put Everything's `es.exe` in `tools/` to bundle it (ES is MIT‑licensed; ship its
license. It talks to the Everything app, which users still install themselves — without it the
filename mode falls back to the Windows Search index).

## 2. Microsoft Store

Facts this plan relies on (Sept 2025 announcements, Microsoft docs):
* Individual developer registration on the Store is **free** (ID + selfie verification);
  company accounts still pay a one‑time $99.
* An **MSIX** submitted through the Store is **signed by Microsoft at no cost**, and Microsoft's
  signing comparison lists Store‑distributed MSIX as *no SmartScreen warnings*. (A plain EXE/MSI
  submitted to the Store must already be signed by you — so the MSIX route is the free one.)

Steps:
1. Partner Center → register as an individual → **reserve the name** "Scry" (or a variant).
2. Product identity page → copy *Package/Identity/Name* and *Publisher* (`CN=...`).
3. On Windows with the Windows SDK (`makeappx`):
   ```powershell
   .\packaging\build_msix.ps1 -Name "<Identity Name>" -Publisher "<CN=...>" -PublisherDisplay "<your name>" -Version 1.5.0.0
   ```
   Add `-TestCert` to sign with a throw‑away certificate and install locally first.
4. Upload `dist\Scry.msix` unsigned. Certification asks why the app needs `runFullTrust`
   ("classic desktop app: local search engine with a local web UI; no network access").
5. Store listing: privacy policy URL is required — the README's privacy section works.

## 3. Things to check on a real Windows machine (not verifiable in CI)

- [ ] MSIX install + launch: the native window opens (`--window` via `uap10:Parameters`),
      hotkey Ctrl+Alt+Space raises it, tray icon works, closing the window quits.
- [ ] Index dir `%USERPROFILE%\.localsearch_index` is written directly (MSIX only virtualizes
      AppData), so data survives reinstall and is shared with the portable build.
- [ ] Windows Search mode, OCR (WinRT), opening files and "open at page" from inside the package.
- [ ] Add‑on downloads (DLC) land in the index dir and load.
- [ ] Store certification feedback on `runFullTrust` and the loopback web server (full‑trust
      desktop apps may bind 127.0.0.1; nothing listens on the network).
