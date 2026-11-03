"""Local, training-free personalization.

Learns from what you OPEN (never from a model gradient — the embedding model is
never trained; this is pure counting + vector arithmetic on frozen embeddings),
so it stays consistent with the privacy promise: everything lives in the index
dir, nothing leaves the machine.

Two signals, blended as a gentle re-rank on top of the base relevance score:
  * behavioral prior  — files/folders you actually open rank higher (decayed
    open-counts, so recent habits win);
  * interest vector   — a decayed centroid of the embeddings of files you open;
    results closer to "what you care about" get nudged up.

Base semantics still dominate: personalization only reorders within the
candidate set (weight `lambda_`), it never invents hits.

Thread-safety: /api/open (fire-and-forget) and /api/search run on different
Flask threads, so all mutable state is guarded by `self._lock`; saves serialize
a COPY taken under the lock (never the live dicts) to a unique temp file.
"""
import json
import math
import os
import threading
from pathlib import Path

import numpy as np


def _norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def _minmax(vals):
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return [(v - lo) / rng for v in vals]


class Personalizer:
    def __init__(self, index_dir, cfg):
        self.dir = Path(index_dir)
        self.cfg = cfg
        self._lock = threading.Lock()
        self.enabled = cfg.personalize
        self.opens = {}          # normcased path   -> {"n": weighted count, "t": event #}
        self.folders = {}        # normcased folder -> {"n": weighted count, "t": event #}
        self._interest = None    # decayed accumulator (unnormalized), np.float32[dim]
        self.events = 0          # monotonic event counter (recency clock)
        self._load()

    # ---- persistence (atomic; snapshot serialized OUTSIDE the state lock) ----
    @property
    def _file(self):
        return self.dir / "personal.json"

    def _load(self):
        try:
            d = json.loads(self._file.read_text(encoding="utf-8"))
            self.enabled = bool(d.get("enabled", self.enabled))
            self.opens = d.get("opens", {})
            self.folders = self._migrate_folders(d.get("folders", {}))
            self.events = d.get("events", 0)
            iv = d.get("interest")
            self._interest = np.asarray(iv, dtype=np.float32) if iv else None
        except Exception:
            pass

    @staticmethod
    def _migrate_folders(f):
        # tolerate the old {folder: float} shape
        out = {}
        for k, v in (f or {}).items():
            out[k] = v if isinstance(v, dict) else {"n": float(v), "t": 0}
        return out

    def _snapshot(self):
        return dict(enabled=self.enabled, opens=dict(self.opens),
                    folders=dict(self.folders), events=self.events,
                    interest=None if self._interest is None else self._interest.tolist())

    def _write(self, snap):
        # unique temp per write (mkstemp) so concurrent saves from different
        # threads never share a temp file; os.replace onto the target is atomic
        # (last writer wins with a COMPLETE snapshot).
        import tempfile
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix="personal.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(snap, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self._file)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass

    def reset(self):
        with self._lock:
            self.opens, self.folders, self._interest, self.events = {}, {}, None, 0
            snap = self._snapshot()
        self._write(snap)

    def set_enabled(self, on):
        with self._lock:
            self.enabled = bool(on)
            snap = self._snapshot()
        self._write(snap)

    def stats(self):
        with self._lock:
            return dict(enabled=self.enabled, learned_opens=len(self.opens),
                        has_interest=self._interest is not None)

    # ---- learning ----
    def record_open(self, path, vec=None):
        """Register that the user opened `path` (optionally with its embedding).
        Older signal decays so the profile tracks recent habits."""
        with self._lock:
            self.events += 1
            g = self.cfg.pers_decay
            key = _norm(path)
            rec = self.opens.get(key)
            n = (rec["n"] * (g ** (self.events - rec["t"])) if rec else 0.0) + 1.0
            self.opens[key] = {"n": n, "t": self.events}
            fkey = _norm(os.path.dirname(path))
            frec = self.folders.get(fkey)
            fn = (frec["n"] * (g ** (self.events - frec["t"])) if frec else 0.0) + 1.0
            self.folders[fkey] = {"n": fn, "t": self.events}
            if vec is not None:
                v = np.asarray(vec, dtype=np.float32).ravel()
                nrm = np.linalg.norm(v)
                if nrm > 0:
                    v = v / nrm
                    if self._interest is None or self._interest.shape != v.shape:
                        self._interest = v.copy()
                    else:
                        self._interest = g * self._interest + v
            self._prune()
            snap = self._snapshot()
        self._write(snap)

    def _prune(self, cap=5000):
        """Bound the profile: past `cap`, keep the strongest-weight entries."""
        if len(self.opens) > cap:
            keep = sorted(self.opens, key=lambda k: self._w(self.opens[k]), reverse=True)[:cap * 3 // 4]
            self.opens = {k: self.opens[k] for k in keep}
        if len(self.folders) > cap:
            keep = sorted(self.folders, key=lambda k: self._w(self.folders[k]), reverse=True)[:cap * 3 // 4]
            self.folders = {k: self.folders[k] for k in keep}

    def _w(self, rec):
        """Lazily-decayed weight of an {n,t} record at the current event clock."""
        return rec["n"] * (self.cfg.pers_decay ** (self.events - rec["t"])) if rec else 0.0

    # ---- applying (rerank runs under the lock: reads are cheap, top_k*3 hits) ----
    def rerank(self, hits, vecs=None):
        """Return hits re-ordered by blending the base score with behavioral +
        interest priors. `vecs` maps hit.path -> unit embedding (optional)."""
        if not hits:
            return hits
        with self._lock:
            if not self.enabled:
                for h in hits:
                    h.personal = False
                return hits
            base = _minmax([h.score if h.score is not None else 0.0 for h in hits])
            click = _minmax([math.log1p(self._w(self.opens.get(_norm(h.path)))) for h in hits])
            fold = _minmax([math.log1p(self._w(self.folders.get(_norm(os.path.dirname(h.path)))))
                            for h in hits])
            inter = [0.0] * len(hits)
            iv = self._interest
            if vecs is not None and iv is not None:
                dim = next((v.shape[-1] for v in vecs.values() if v is not None), None)
                n = float(np.linalg.norm(iv))
                if dim is not None and iv.shape[-1] == dim and n > 0:   # dim guard
                    iu = iv / n
                    raw = []
                    for h in hits:
                        v = vecs.get(h.path)
                        raw.append(0.0 if v is None else max(0.0, float(np.dot(v, iu))))
                    inter = _minmax(raw)
            c = self.cfg
            lam = c.pers_lambda
            for i, h in enumerate(hits):
                personal = (c.pers_w_click * click[i] + c.pers_w_folder * fold[i]
                            + c.pers_w_interest * inter[i])
                h.personal = self._w(self.opens.get(_norm(h.path))) > 0
                h.pscore = (1 - lam) * base[i] + lam * personal
        return self._protect_best(hits, base, c.pers_top_margin)

    @staticmethod
    def _protect_best(hits, base, margin):
        """Sort by personalized score, but keep the guarantee the README makes:
        a familiar file may rise from far down the list, yet it never displaces
        the BEST match for the query unless the two are practically tied
        (base relevance within `margin` of the candidates' score range)."""
        order = sorted(range(len(hits)), key=lambda i: -hits[i].pscore)
        best = max(range(len(hits)), key=lambda i: base[i])
        pos = order.index(best)
        if pos:
            ahead = order[:pos]
            tied = [i for i in ahead if base[i] >= base[best] - margin]
            demoted = [i for i in ahead if base[i] < base[best] - margin]
            order = tied + [best] + demoted + order[pos + 1:]
        return [hits[i] for i in order]
