"""Minimal LOCAL web UI for the localsearch engine (brick 6)."""
import argparse
import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import webbrowser
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, request, jsonify, Response

from .engine import SearchEngine
from .config import Config
from . import resources
from . import query as query_mod

log = logging.getLogger("localsearch.app")
HERE = Path(__file__).resolve().parent
WEB = resources.web_dir()

# The API is bound to loopback, but a web page the user visits can still reach it
# from inside their browser (cross-site POSTs; DNS rebinding, where an attacker's
# hostname re-resolves to 127.0.0.1 and becomes "same-origin"). Rejecting any
# request whose Host or Origin is not a loopback name closes both holes.
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_TOP_K_MAX = 200
_SHOWN_MAX = 5000


def _host_name(host):
    """'127.0.0.1:8731' -> '127.0.0.1'; '[::1]:8731' -> '::1' (lowercased)."""
    h = (host or "").strip().lower()
    if h.startswith("["):
        return h[1:h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def _is_local_request(host, origin):
    if _host_name(host) not in _LOOPBACK:
        return False
    if origin is not None:          # browsers send Origin on every POST
        try:
            return (urlparse(origin).hostname or "") in _LOOPBACK
        except ValueError:
            return False
    return True


def _body():
    """JSON request body as a dict (anything else -> {}), never raises."""
    d = request.get_json(silent=True)
    return d if isinstance(d, dict) else {}


def _norm_path(p):
    return os.path.normcase(os.path.abspath(str(p)))


class AppState:
    def __init__(self, engine, index_dir):
        self.engine = engine
        self.index_dir = Path(index_dir)
        self.folders_file = self.index_dir / "folders.json"
        self.folders = self._load_folders()
        self.settings_file = self.index_dir / "settings.json"
        self.settings = self._load_settings()
        self.building = False
        self.catchup = False          # the running build is the startup catch-up scan
        self.warm_done = threading.Event()   # model warm-up (and catch-up kick-off) finished
        self.show_window = None              # set by the desktop shell (raise + focus)
        self.build_lock = threading.Lock()
        self.cancel = False
        self.last_build = None
        self.last_error = ""
        self.last_error_detail = ""
        self.model_ready = False
        self.progress = {"done": 0, "total": 0, "file": ""}
        self.dlc = {"id": None, "done": 0, "total": 0, "status": "idle", "error": ""}
        self.dlc_lock = threading.Lock()
        self.restart_required = False
        # paths recently returned by /api/search: /api/open only launches these,
        # so the API can never be used to run an arbitrary file on the machine.
        self.shown = OrderedDict()
        self.shown_lock = threading.Lock()

    def remember_shown(self, paths):
        with self.shown_lock:
            for p in paths:
                k = _norm_path(p)
                self.shown[k] = True
                self.shown.move_to_end(k)
            while len(self.shown) > _SHOWN_MAX:
                self.shown.popitem(last=False)

    def was_shown(self, path):
        with self.shown_lock:
            return _norm_path(path) in self.shown

    def _load_folders(self):
        try:
            return json.loads(self.folders_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_folders(self):
        """Atomically persist the folder list (tmp + fsync + replace)."""
        self._atomic_json(self.folders_file, self.folders)

    def _load_settings(self):
        try:
            return json.loads(self.settings_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self):
        self._atomic_json(self.settings_file, self.settings)

    def _atomic_json(self, target, data):
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(target)
        except Exception as e:
            log.warning("could not persist %s: %s", target.name, e)

    def resume_watch(self):
        existing = [f for f in self.folders if os.path.isdir(f)]
        if existing:
            try:
                self.engine.start_watching(existing)
            except Exception as e:
                log.warning("resume watch failed: %s", e)


def _snippet(text, n=240):
    t = " ".join(text.split())
    return t[:n] + ("…" if len(t) > n else "")


def _human_size(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024


def _result(path, snippet, score, terms, personal=False, page=None):
    """Build a result row with current metadata; None if the file is gone."""
    import time as _t
    try:
        st = os.stat(path)
    except OSError:
        return None
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "?"
    size = _human_size(st.st_size)
    mtime = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(st.st_mtime))
    return dict(name=os.path.basename(path), path=path, snippet=snippet, score=score,
                terms=terms, ext=ext, size=size, mtime=mtime, personal=personal, page=page)


def create_app(index_dir, config=None):
    from . import winsearch
    engine = SearchEngine(index_dir, config or Config())
    state = AppState(engine, index_dir)
    if state.settings.get("ocr"):
        engine.set_ocr(True)             # restore the persisted image-OCR choice
    if state.settings.get("index_cloud_files"):
        engine.cfg.index_cloud_files = True   # advanced: allow downloading placeholders
    state.resume_watch()

    def _warm():
        try:
            _warm_inner()
        finally:
            state.warm_done.set()

    def _warm_inner():
        # BM25 needs no model: build it while the model loads, not after
        prep = threading.Thread(target=engine.prepare_index, daemon=True)
        prep.start()
        ok = engine.warmup()
        if ok:
            state.model_ready = True
            # Catch-up scan: the watcher only sees changes while the app runs, so
            # files added/edited/deleted while it was closed (and an index that was
            # moved aside after a model change) are reconciled now. It is a diff --
            # unchanged files are skipped by mtime+size and nothing is rewritten.
            if state.settings.get("startup_rescan", True):
                paths = [f for f in state.folders if os.path.isdir(f)]
                if paths:
                    _start_build(paths, register=False, catchup=True)
        else:
            state.last_error = "AI 模型加载失败，语义搜索不可用（详见控制台/engine.log）。"

    ws_available = winsearch.available()
    app = Flask(__name__)

    @app.before_request
    def _local_only():
        if not _is_local_request(request.host, request.headers.get("Origin")):
            log.warning("rejected non-local request: host=%r origin=%r path=%s",
                        request.host, request.headers.get("Origin"), request.path)
            return jsonify(ok=False, code="forbidden", error="Local access only."), 403

    @app.get("/")
    def home():
        return Response((WEB / "index.html").read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/api/status")
    def status():
        s = engine.stats()
        return jsonify(files=s["files"], chunks=s["chunks"], model=engine.cfg.model_name,
                       building=state.building, catchup=state.catchup,
                       progress=state.progress, folders=state.folders,
                       last_build=state.last_build, ws_available=ws_available,
                       model_ready=state.model_ready, last_error=state.last_error,
                       last_error_detail=state.last_error_detail, index_error=engine.load_error,
                       personalize=engine.personalizer.stats(),
                       ocr=engine.ocr_status(), images=engine.image_status(),
                       qa=engine.qa_status(),
                       dlc=state.dlc, restart_required=state.restart_required)

    @app.post("/api/cancel_build")
    def cancel_build():
        state.cancel = True
        return jsonify(ok=True)

    @app.post("/api/personalize")
    def personalize():
        # toggle local, training-free personalization on/off
        on = bool(_body().get("enabled", True))
        engine.personalizer.set_enabled(on)
        return jsonify(ok=True, enabled=engine.personalizer.enabled)

    @app.post("/api/forget")
    def forget():
        # wipe the local behavioral profile (opens + interest vector)
        engine.personalizer.reset()
        return jsonify(ok=True)

    @app.get("/api/ocr")
    def ocr_get():
        return jsonify(engine.ocr_status())

    @app.post("/api/ocr")
    def ocr_set():
        # toggle image OCR (Windows built-in engine); images index on next rescan.
        # Refuse mid-build: flipping the supported-set while a scan is parsing would
        # record in-flight images as empty and (unchanged mtime) never retry them.
        on = bool(_body().get("enabled", False))
        with state.build_lock:
            if state.building:
                return jsonify(ok=False, code="busy",
                               error="Busy indexing; try again after it finishes."), 409
            st = engine.set_ocr(on)
        state.settings["ocr"] = st["enabled"]
        state._save_settings()
        return jsonify(ok=True, **st)

    @app.get("/api/dlc")
    def dlc_list():
        from . import dlc as _dlc
        return jsonify(packs=engine.dlc_status(), progress=state.dlc,
                       restart_required=state.restart_required)

    @app.post("/api/dlc/install")
    def dlc_install():
        from . import dlc as _dlc
        pid = _body().get("id", "")
        with state.dlc_lock:
            if state.dlc["status"] == "downloading":
                return jsonify(ok=False, error="A download is already in progress."), 409
            state.dlc = {"id": pid, "done": 0, "total": 0, "status": "downloading", "error": ""}

        def worker():
            try:
                pack = _dlc.install(str(state.index_dir), pid,
                                    progress=lambda d, t: state.dlc.update(done=d, total=t))
                if pack["kind"] == "model":
                    engine.quarantine_index()      # dim changed -> rebuild on restart
                state.restart_required = True
                state.dlc.update(status="done")
            except Exception as e:
                log.error("DLC install failed: %s", e)
                state.dlc.update(status="error", error=str(e))

        threading.Thread(target=worker, daemon=True).start()
        return jsonify(ok=True, started=True)

    @app.post("/api/dlc/uninstall")
    def dlc_uninstall():
        from . import dlc as _dlc
        pid = _body().get("id", "")
        removed = _dlc.uninstall(str(state.index_dir), pid)
        if removed:
            p = _dlc._by_id(pid)
            if p and p["kind"] == "model":
                engine.quarantine_index()
            state.restart_required = True
        return jsonify(ok=True, removed=removed)

    @app.post("/api/pick_folder")
    def pick_folder():


        try:
            import tkinter as tk
            from tkinter import filedialog
            r = tk.Tk()
            r.withdraw()
            r.attributes("-topmost", True)
            p = filedialog.askdirectory()
            r.destroy()
            return jsonify(path=p or "")
        except Exception as e:
            log.warning("folder dialog failed (type the path instead): %s", e)
            return jsonify(path="", error=str(e))

    def _start_build(paths, register, catchup=False):
        """Background full-or-diff build (the engine skips unchanged files itself)."""
        with state.build_lock:
            if state.building:
                return False
            state.building = True
            state.catchup = catchup
        state.cancel = False
        state.last_error = ""
        state.last_error_detail = ""
        state.progress = {"done": 0, "total": 0, "file": ""}

        def worker():
            def cb(done, total, f):
                state.progress = {"done": done, "total": total, "file": os.path.basename(f)}
            try:
                state.last_build = engine.build(paths, progress_callback=cb,
                                                cancel_check=lambda: state.cancel)
                engine.load_error = ""
                if register:
                    known = {_norm_path(f) for f in state.folders}
                    for p in paths:
                        if _norm_path(p) not in known:
                            state.folders.append(p)
                    state._save_folders()
                engine.start_watching([f for f in state.folders if os.path.isdir(f)])
            except Exception as e:
                log.error("build failed: %s", e)
                state.last_error = "build_failed"
                state.last_error_detail = str(e)
            finally:
                state.catchup = False
                state.building = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @app.post("/api/add_folder")
    def add_folder():
        path = str(_body().get("path") or "").strip()
        if not path or not os.path.isdir(path):
            return jsonify(ok=False, code="bad_folder", error="Not a valid folder."), 400
        path = os.path.abspath(path)        # one spelling per folder (no dup chips)
        if not _start_build([path], register=True):
            return jsonify(ok=False, code="busy", error="Already indexing, please wait."), 409
        return jsonify(ok=True, started=True)

    @app.post("/api/remove_folder")
    def remove_folder():
        """Un-control a folder: drop it and purge its indexed entries."""
        path = str(_body().get("path") or "").strip()
        with state.build_lock:
            if state.building:
                return jsonify(ok=False, code="busy", error="Busy indexing."), 409
            match = next((f for f in state.folders if f == path
                          or (path and _norm_path(f) == _norm_path(path))), None)
            if match is None:
                return jsonify(ok=False, code="not_controlled", error="Not in the controlled list."), 400
            path = match
            state.folders.remove(path)
            state._save_folders()
            n = engine.remove_tree(path, persist=True)
            engine.start_watching([f for f in state.folders if os.path.isdir(f)])
        return jsonify(ok=True, removed_chunks=n)

    @app.get("/api/drives")
    def drives():
        return jsonify(drives=engine.list_drives())

    @app.post("/api/rescan")
    def rescan():

        paths = [f for f in state.folders if os.path.isdir(f)]
        if not paths:
            return jsonify(ok=False, code="no_folders", error="No controlled folders yet."), 400
        if not _start_build(paths, register=False):
            return jsonify(ok=False, code="busy", error="Busy indexing."), 409
        return jsonify(ok=True, started=True)

    @app.post("/api/search")
    def search():
        d = _body()
        raw = str(d.get("query") or "").strip()
        mode = d.get("mode", "semantic")
        try:
            k = min(max(int(d.get("top_k", 20)), 1), _TOP_K_MAX)
        except (TypeError, ValueError):
            k = 20
        scope = (d.get("scope") or "").strip() or None
        if not raw:
            return jsonify(results=[])
        qf = query_mod.parse(raw)          # operators -> filters; words -> q
        q = qf.text
        meta = dict(filters=qf.filters, suggest=qf.suggest, unknown=qf.unknown)
        if not q and not qf.filtered:
            return jsonify(results=[], **meta)
        sid = None
        used = mode
        if not q:
            # only filters ("time:上周 type:pdf"): list files instead of ranking.
            # Whole-disk modes use Everything (instant + covers the disk) when present.
            paths = (engine.search_filename("", limit=k, scope=scope, filters=qf)
                     if mode != "semantic" else None)
            if paths is not None:
                used = "filename"
                res = [_result(p, snippet="", score=None, terms=[]) for p in paths]
                res.sort(key=lambda r: (r or {}).get("mtime") or "", reverse=True)
            else:
                used = "semantic"
                res = [_result(h.path, snippet=h.snippet, score=None, terms=[])
                       for h in engine.browse(qf, top_k=k)]
        elif mode == "image":
            hits = engine.search_images(q, top_k=k, filters=qf)
            if not hits:
                return jsonify(results=[], code="no_images",
                               error="No images indexed for search by description.", **meta)
            res = [_result(h.path, snippet="", score=round(h.score, 3), terms=[]) for h in hits]
            for r in res:
                if r is not None:
                    r["thumb"] = True
        elif mode == "filename":
            paths = engine.search_filename(q, limit=k, scope=scope, filters=qf)
            if paths is None:
                return jsonify(results=[], code="no_everything",
                               error="Everything (es.exe) not available.", **meta)
            res = [_result(p, snippet="", score=None, terms=[]) for p in paths]
            if engine.filename_backend == "windows":
                meta["note"] = "fn_windows"     # fallback: indexed locations only
        elif mode == "instant":
            if not ws_available:
                return jsonify(results=[], code="ws_down",
                               error="Windows Search service is off.", **meta)
            hits = engine.search_instant_global(q, top_k=k, scope=scope, deepen=False,
                                                filters=qf)
            if not hits:
                return jsonify(results=[], code="no_hits", error="No content matches.", **meta)
            sid = engine.queue_snippets("instant", q, hits) if engine.cfg.instant_deep_snippet else None
            res = [_result(h.path, snippet=h.snippet, score=round(h.score, 3),
                           terms=h.terms, personal=h.personal) for h in hits]
        else:
            # results first (~tens of ms); the semantic snippets follow via
            # /api/snippets -- they were ~95% of a CPU query's time
            hits = engine.search_semantic(q, top_k=k, snippets="lexical", filters=qf)
            sid = engine.queue_snippets("semantic", q, hits) if hits else None
            res = [_result(h.path, snippet=h.snippet or _snippet(h.chunk_text),
                           score=round(h.score, 3), terms=h.terms, personal=h.personal,
                           page=h.page) for h in hits]
            sims = [h.sim for h in hits if h.sim is not None]
            if sims:           # calibrated relevance floor: dim (never hide) the weak tail
                top = max(sims)
                for r, h in zip(res, hits):
                    if r is not None and h.sim is not None:
                        r["low"] = top - h.sim > engine.cfg.relevance_gap
        res = [r for r in res if r is not None]
        state.remember_shown(r["path"] for r in res)
        out = dict(results=res, mode_used=used, **meta)
        if sid:
            out["snip_id"] = sid
        return jsonify(**out)

    @app.post("/api/similar")
    def similar():
        """More like this: files whose content is closest to a shown result."""
        d = _body()
        path = str(d.get("path") or "")
        try:
            k = min(max(int(d.get("top_k", 20)), 1), _TOP_K_MAX)
        except (TypeError, ValueError):
            k = 20
        if not path or not os.path.exists(path):
            return jsonify(ok=False, code="no_file", error="File no longer exists."), 404
        if not state.was_shown(path):
            return jsonify(ok=False, code="not_result",
                           error="Only files from search results can be used."), 403
        hits, qvec = engine.similar(path, top_k=k)
        name = os.path.basename(path)
        if not hits:
            return jsonify(results=[], code="no_similar", similar_to=name,
                           error="No similar files in the deep index.")
        sid = engine.queue_snippets("semantic", qvec, hits)
        res = [r for r in (_result(h.path, snippet=h.snippet, score=round(h.score, 3),
                                   terms=[], page=h.page) for h in hits) if r is not None]
        state.remember_shown(r["path"] for r in res)
        return jsonify(results=res, snip_id=sid, similar_to=name)

    @app.post("/api/ask")
    def ask():
        """Answer a question from the deep index with a local model, streamed as
        NDJSON: {"type":"sources"} first, then "token"s, then "done" with the
        source numbers the answer actually cited."""
        from . import qa
        d = _body()
        question = str(d.get("question") or "").strip()
        if not question:
            return jsonify(ok=False, code="no_question"), 400
        try:
            k = min(max(int(d.get("top_k", 5)), 1), 8)
        except (TypeError, ValueError):
            k = 5
        llm = engine.qa_backend()
        qf = query_mod.parse(question)
        sources = qa.gather_sources(engine, qf.text or question, k=k, filters=qf)
        state.remember_shown(s["path"] for s in sources)       # citations are openable
        public = [{key: s[key] for key in ("n", "name", "path", "page")} for s in sources]
        if llm is None:
            return jsonify(ok=False, code="no_llm", sources=public), 409
        if not sources:
            return jsonify(ok=False, code="no_sources", sources=[]), 404
        messages = qa.build_messages(question, sources)

        def gen():
            yield json.dumps({"type": "sources", "sources": public}, ensure_ascii=False) + "\n"
            answer = ""
            with engine.llm_lock:
                try:
                    for piece in llm.stream(messages):
                        answer += piece
                        yield json.dumps({"type": "token", "text": piece}, ensure_ascii=False) + "\n"
                except Exception as e:
                    log.warning("answer failed: %s", e)
                    yield json.dumps({"type": "error", "error": str(e)}) + "\n"
            yield json.dumps({"type": "done", "cited": qa.cited(answer, len(sources))}) + "\n"

        return Response(gen(), mimetype="application/x-ndjson")

    @app.get("/api/thumb")
    def thumb():
        """Small JPEG preview of an image result (same rule as opening: only files
        a search returned)."""
        path = request.args.get("path", "")
        if not path or not os.path.isfile(path) or not state.was_shown(path):
            return Response(status=404)
        try:
            import io
            from PIL import Image
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((320, 240))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=80)
            return Response(buf.getvalue(), mimetype="image/jpeg",
                            headers={"Cache-Control": "private, max-age=300"})
        except Exception:
            return Response(status=415)

    @app.post("/api/show")
    def show():
        """A second launch asks the running desktop window to come forward."""
        if state.show_window is None:
            return jsonify(ok=False)
        state.show_window()
        return jsonify(ok=True)

    @app.post("/api/snippets")
    def snippets():
        """Second half of a search: the semantic best-sentence snippets."""
        sid = str(_body().get("id") or "")
        try:
            return jsonify(snippets=engine.resolve_snippets(sid) if sid else {})
        except Exception as e:
            log.warning("snippets failed: %s", e)
            return jsonify(snippets={})

    @app.post("/api/open")
    def open_path():
        d = _body()
        path = str(d.get("path") or "")
        reveal = bool(d.get("reveal", False))
        try:
            page = int(d.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if not path or not os.path.exists(path):
            return jsonify(ok=False, code="no_file", error="File no longer exists."), 404
        if not state.was_shown(path):
            return jsonify(ok=False, code="not_result",
                           error="Only files from search results can be opened."), 403
        engine.record_open(path)        # local behavioral profile (training-free)
        if os.environ.get("LOCALSEARCH_NO_LAUNCH"):
            return jsonify(ok=True, launched=False)
        try:
            if reveal:
                subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
            elif page > 0 and path.lower().endswith(".pdf"):
                # every browser's PDF viewer honours #page=N (Edge ships with Windows)
                webbrowser.open(Path(path).resolve().as_uri() + f"#page={page}")
            else:
                os.startfile(path)
            return jsonify(ok=True)
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 500

    # started last: _warm() may kick off the catch-up scan via _start_build
    threading.Thread(target=_warm, daemon=True).start()
    return app, engine, state


def _install_shutdown_hooks(engine):
    """Persist pending changes on every exit path (finally/atexit + console-close)."""
    closed = threading.Event()

    def close_once():
        if not closed.is_set():
            closed.set()
            try:
                engine.close()
            except Exception:
                pass

    atexit.register(close_once)
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            HANDLER = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def _ctrl(evt):

                if evt in (2, 5, 6):
                    close_once()
                return False

            _install_shutdown_hooks._keep = HANDLER(_ctrl)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_install_shutdown_hooks._keep, True)
        except Exception as e:
            log.warning("console close hook not installed: %s", e)
    return close_once


def main():
    ap = argparse.ArgumentParser(description="LocalSearch UI (fully local)")
    ap.add_argument("--port", type=int, default=8731)
    ap.add_argument("--index-dir", default=str(Path.home() / ".localsearch_index"))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--window", action="store_true",
                    help="native window + global hotkey + tray (Windows; needs pywebview)")
    ap.add_argument("--hotkey", default=None, help='global hotkey, default "ctrl+alt+space"')
    args = ap.parse_args()

    from .instance import InstanceLock, free_port
    lock = InstanceLock(args.index_dir)
    if not lock.acquire():
        # already running on this index: bring that one up instead of a second
        # copy that would fight it over the index files
        url = lock.running_url() or f"http://127.0.0.1:{args.port}"
        print(f"Scry is already running -> {url}")
        print("已经在运行了，正在打开它的页面 / opening the running instance.")
        if not args.no_browser and not _ask_to_show(url):
            webbrowser.open(url)
        return
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"

    app, engine, state = create_app(args.index_dir)
    close_once = _install_shutdown_hooks(engine)
    lock.publish(url)
    print("=" * 64)
    print(f"  LocalSearch UI  ->  {url}")
    print(f"  index dir: {args.index_dir}")
    print("  Fully local. Nothing is sent over the network.")
    print("  Press Ctrl+C (or close this window) to stop.")
    print("=" * 64)
    serve = lambda: app.run(host="127.0.0.1", port=port, threaded=True)
    want_window = args.window or bool(state.settings.get("desktop"))
    try:
        if want_window:
            from . import desktop
            hotkey = args.hotkey or state.settings.get("hotkey") or "ctrl+alt+space"
            ready = lambda show: setattr(state, "show_window", show)
            if desktop.run(url, serve, close_once, hotkey=hotkey, on_ready=ready):
                return
            print("  (native window unavailable -- pip install pywebview; using the browser)")
        if not args.no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        serve()
    finally:
        close_once()
        lock.release()


def _ask_to_show(url):
    """Ask a running desktop-mode instance to raise its window."""
    import urllib.request
    try:
        req = urllib.request.Request(url + "/api/show", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read().decode("utf-8")).get("ok") is True
    except Exception:
        return False


if __name__ == "__main__":
    main()
