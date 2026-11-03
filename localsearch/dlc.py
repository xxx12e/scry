"""Optional downloadable add-on packs (DLC).

The base app is 100% offline and NEVER touches the network on its own. A pack is
fetched ONLY when the user explicitly clicks Install — over HTTPS from this
project's own GitHub releases, and verified by SHA-256 before it is used.

Installed packs live under `<index_dir>/dlc/<id>/` (writable, outside the frozen
bundle) and are auto-detected at startup. Packs are plain data/binary drop-ins
(a model dir, an OCR engine) — never Python packages — so they slot into the
frozen exe cleanly.
"""
import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("localsearch.dlc")

# Registry of available packs. `url`/`sha256`/`size` point at a GitHub release
# asset; empty url => "coming soon" (listed but not installable yet).
REGISTRY = [
    {
        "id": "model-large",
        "kind": "model",
        "name_zh": "大模型包 · 更高质量",
        "name_en": "Larger model · higher quality",
        "desc_zh": "把语义模型升级为 multilingual-e5-base（768 维，跨语言/中文质量更强，约 520 MB）。安装后深度索引会自动重建一次。",
        "desc_en": "Upgrade the semantic model to multilingual-e5-base (768-dim, stronger cross-language / Chinese quality, ~520 MB). The deep index rebuilds once after install.",
        "url": "https://github.com/xxx12e/scry/releases/download/v1.3.0/model-large-e5base.zip",
        "size": 519983424,
        "sha256": "e36625c6e8eba10a2876d31f63b36417025b8e34677c3c2ed1ac7ea94c1bf681",
    },
    {
        # Build with scripts/export_clip.py (Chinese-CLIP ViT-B/16), zip the output
        # folder, publish it as a release asset, then fill url/size/sha256 here.
        "id": "clip-zh",
        "kind": "image-model",
        "name_zh": "以文搜图包 · 中英",
        "name_en": "Search images by description",
        "desc_zh": "用一句话找照片和图片（「海边日落」「发票的照片」），不需要图里有文字。约 380 MB；装好后重新扫描一次，图片会按内容建索引。",
        "desc_en": "Find photos and images by describing them (\"sunset at the beach\", \"photo of a receipt\") -- no text in the image needed. ~380 MB; rescan once after installing.",
        "url": "",
        "size": 0,
        "sha256": "",
    },
]


def _by_id(pid):
    return next((p for p in REGISTRY if p["id"] == pid), None)


def dlc_root(index_dir):
    return Path(index_dir) / "dlc"


def pack_dir(index_dir, pid):
    return dlc_root(index_dir) / pid


def is_installed(index_dir, pid):
    d = pack_dir(index_dir, pid)
    p = _by_id(pid)
    if p and p["kind"] == "model":
        return (d / "config.json").exists()
    return d.exists() and any(d.iterdir()) if d.exists() else False


def clip_dir(index_dir):
    """Installed image-search pack dir (text.onnx + vision.onnx), or None."""
    d = pack_dir(index_dir, "clip-zh")
    return str(d) if (d / "text.onnx").exists() and (d / "vision.onnx").exists() else None


def model_override(index_dir):
    """Path to an installed model pack, or None -> use the bundled model. A pack
    no installed backend can load (e.g. no model.onnx in a torch-free build) is
    ignored rather than leaving the app without a model."""
    d = pack_dir(index_dir, "model-large")
    if not (d / "config.json").exists():
        return None
    from .embedder import can_load
    if not can_load(d):
        log.warning("model pack %s cannot be loaded by any installed backend; "
                    "using the bundled model", d)
        return None
    return str(d)


def status(index_dir):
    return [
        {"id": p["id"], "kind": p["kind"], "name_zh": p["name_zh"], "name_en": p["name_en"],
         "desc_zh": p["desc_zh"], "desc_en": p["desc_en"],
         "size": p["size"], "available": bool(p["url"]),
         "installed": is_installed(index_dir, p["id"])}
        for p in REGISTRY
    ]


def install(index_dir, pid, progress=None):
    """Download (HTTPS, GitHub), verify SHA-256, extract to dlc/<id>/. Returns the
    pack dict on success. Raises on unknown id / no url / checksum mismatch."""
    p = _by_id(pid)
    if p is None:
        raise ValueError("unknown pack")
    if not p["url"]:
        raise ValueError("pack not available yet")
    root = dlc_root(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(root), suffix=".part")
    os.close(fd)
    h = hashlib.sha256()
    try:
        req = urllib.request.Request(p["url"], headers={"User-Agent": "localsearch-dlc"})
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or p["size"] or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if p["sha256"] and h.hexdigest() != p["sha256"]:
            raise ValueError("checksum mismatch — download rejected")
        dest = pack_dir(index_dir, pid)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(dest)
        _flatten(dest)
        log.info("installed DLC %s -> %s", pid, dest)
        return p
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _flatten(dest):
    """If the zip wrapped everything in one top folder, lift it up one level."""
    entries = [e for e in dest.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for c in list(inner.iterdir()):
            shutil.move(str(c), str(dest / c.name))
        inner.rmdir()


def uninstall(index_dir, pid):
    d = pack_dir(index_dir, pid)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False
