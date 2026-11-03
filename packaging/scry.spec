# -*- mode: python -*-
# Torch-free portable build (see packaging/README.md):
#     python scripts/export_onnx.py models/embed      # once: writes models/embed/model.onnx
#     pyinstaller packaging/scry.spec                  # -> dist/Scry/Scry.exe
# Env: SCRY_MODEL_DIR (default models/embed), SCRY_CONSOLE=0 for a windowed exe
# (then run it with --window, or set "desktop": true in settings.json).
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
MODEL = Path(os.environ.get("SCRY_MODEL_DIR", ROOT / "models" / "embed"))
if not (MODEL / "model.onnx").exists():
    sys.exit(f"{MODEL}/model.onnx missing -- run scripts/export_onnx.py first")

datas = [(str(ROOT / "localsearch" / "web"), "web")]
for f in MODEL.rglob("*"):
    if f.is_file() and f.name != "model.safetensors" and f.suffix != ".md":   # ONNX replaces it
        datas.append((str(f), str(Path("models/embed") / f.relative_to(MODEL).parent)))
es = ROOT / "tools" / "es.exe"
if es.exists():
    datas.append((str(es), "tools"))

hidden = ["onnxruntime", "tokenizers", "olefile", "xlrd", "faiss", "pymupdf", "fitz", "PIL"]
if sys.platform == "win32":
    hidden += ["win32com.client", "pythoncom", "pywintypes", "webview", "pystray",
               "winrt.windows.media.ocr", "winrt.windows.graphics.imaging",
               "winrt.windows.storage", "winrt.windows.storage.streams",
               "winrt.windows.globalization", "winrt.windows.foundation",
               "winrt.windows.foundation.collections"]

# the whole point of this build: none of the torch stack
excludes = ["torch", "torchvision", "torchaudio", "transformers", "sentence_transformers",
            "scipy", "sklearn", "onnx", "onnxscript", "tensorflow", "jax", "matplotlib",
            "IPython", "notebook", "sympy", "networkx", "huggingface_hub", "safetensors",
            "triton", "nvidia", "tkinter.test", "pandas", "pyarrow"]

a = Analysis([str(ROOT / "packaging" / "scry_main.py")], pathex=[str(ROOT)], datas=datas,
             hiddenimports=hidden, excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Scry",
          console=os.environ.get("SCRY_CONSOLE", "1") == "1", upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Scry", upx=False)
