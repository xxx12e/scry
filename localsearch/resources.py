"""Locate bundled assets (web UI, es.exe, model) in source and frozen modes."""
import hashlib
import os
import re
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
_PKG = Path(__file__).resolve().parent
_MEI = Path(getattr(sys, "_MEIPASS", _PKG))

if FROZEN:
    # The release bundles its model: make "never touches the network" hold in
    # code, not only in the packaging script. Set before the HF libs are
    # imported (the embedder imports them lazily); setdefault keeps overrides.
    for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ.setdefault(_k, "1")


def web_dir():
    return (_MEI / "web") if FROZEN else (_PKG / "web")


def es_exe():
    return (_MEI / "tools" / "es.exe") if FROZEN else (_PKG.parent / "tools" / "es.exe")


def model_path():
    """Bundled model dir when frozen (fully offline); else the HF hub id.
    multilingual-e5-small: 100+ languages, strong cross-lingual retrieval, light
    (~118M params, 384-dim). Needs 'query: ' / 'passage: ' prefixes (see Config)."""
    if FROZEN:
        m = _MEI / "models" / "embed"
        if (m / "config.json").exists():
            return str(m)
    return "intfloat/multilingual-e5-small"


def model_tag(model_name):
    """Location-independent identity of an embedding model, stamped on the index.

    A local model dir is identified by its folder name + a hash of its
    config.json -- NOT its absolute path, which changes whenever the portable app
    folder is moved or a new release is unzipped next to the old one (that used
    to discard the whole deep index). A hub id is its own identity."""
    p = Path(str(model_name))
    try:
        cfg = p / "config.json"
        if cfg.is_file():
            h = hashlib.sha1(cfg.read_bytes())
            q = p / "quantization.json"          # a quantized variant embeds differently
            if q.is_file():
                h.update(q.read_bytes())
            return "local:%s:%s" % (p.name, h.hexdigest()[:12])
    except OSError:
        pass
    return str(model_name)


def model_tag_matches(stored, model_name):
    """Was an index stamped `stored` built by the model at `model_name`?"""
    if not stored:
        return True
    cur = model_tag(model_name)
    if stored in (cur, str(model_name)):
        return True
    # legacy (<= v1.4) stamps were the model dir's absolute path: accept the same
    # model dir name ("embed" = bundled, "model-large" = DLC) wherever it now lives
    if cur.startswith("local:") and not stored.startswith("local:"):
        return re.split(r"[\\/]", stored.rstrip("\\/"))[-1] == cur.split(":")[1]
    return False
