"""SearchEngine: the unified backend entry point."""
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import Config
from . import parser
from . import resources
from . import ocr
from .parser import parse_file, is_supported
from .chunker import chunk_text, tokenize
from .embedder import Embedder
from .index import DualIndex, IndexVersionError, IndexCorruptError
from .search import retrieve, make_snippets, filename_search, Hit
from .personalize import Personalizer, _norm
from . import dlc

log = logging.getLogger("localsearch")


def _setup_logging(index_dir):
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        return
    log.setLevel(logging.INFO)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(index_dir) / "engine.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(ch)


class SearchEngine:
    def __init__(self, index_dir, config: Config = None):
        self.index_dir = str(Path(index_dir).resolve())
        self.cfg = config or Config()
        _setup_logging(self.index_dir)
        self._embedder = None
        self._watcher = None
        self._lock = threading.RLock()
        self._emb_lock = threading.Lock()
        self._dirty = False
        self._flush_timer = None
        self._save_pending = False
        self._save_timer = None
        self._io_lock = threading.Lock()
        self._last_written_gen = 0
        self._closing = False
        self._instant_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self.personalizer = Personalizer(self.index_dir, self.cfg)
        self._last_shown = OrderedDict()  # normcased path -> vec (recent shown, for record_open)
        self._shown_lock = threading.Lock()
        self.load_error = ""
        # optional DLC model pack overrides the bundled model (auto-scanned)
        ov = dlc.model_override(self.index_dir)
        if ov:
            self.cfg.model_name = ov
            log.info("DLC model pack active: %s", ov)
        parser.set_ocr(self.cfg.ocr)     # image OCR (opt-in; Windows built-in)
        self._model_tag = resources.model_tag(self.cfg.model_name)
        self.index = None
        if DualIndex.exists(self.index_dir):
            try:
                self.index = DualIndex.load(self.index_dir)
                if not resources.model_tag_matches(self.index.model_tag, self.cfg.model_name):
                    raise IndexVersionError("embedding model changed")
            except IndexVersionError as e:
                log.warning("index version changed (%s); rebuild needed", e)
                self.index = None
                DualIndex.quarantine(self.index_dir)
                self.load_error = "idx_version"
            except IndexCorruptError as e:
                log.error("index damaged (%s); quarantined", e)
                self.index = None
                DualIndex.quarantine(self.index_dir)
                self.load_error = "idx_corrupt"
            except Exception as e:
                log.error("failed to load index (%s); starting empty", e)
                self.index = None
                self.load_error = "idx_load"


    @property
    def embedder(self):
        if self._embedder is None:
            mismatch = False
            with self._emb_lock:
                if self._embedder is None:
                    emb = Embedder(self.cfg)
                    idx = self.index
                    mismatch = idx is not None and idx.dim != emb.dim
                    self._embedder = emb
            if mismatch:
                # vectors of another width would crash every FAISS search: move the
                # index aside so the next scan rebuilds it (outside _emb_lock to
                # keep the _lock -> _emb_lock order used by _ensure_index)
                with self._lock:
                    if self.index is not None and self.index.dim != self._embedder.dim:
                        log.error("model dim %d != index dim %d -- index quarantined, "
                                  "rescan to rebuild", self._embedder.dim, self.index.dim)
                        self.index = None
                        DualIndex.quarantine(self.index_dir)
                        self.load_error = "idx_version"
        return self._embedder

    def warmup(self):
        """Load the model + one dummy embed for a fast first query; True on success."""
        try:
            self.embedder.embed_query("warmup")
            log.info("embedder warmed up")
            return True
        except Exception as e:
            log.warning("warmup failed: %s", e)
            return False

    def prepare_index(self):
        """Build BM25 off the startup critical path (loads are lazy now)."""
        try:
            # derived state only: building it must not schedule a full index save
            # (that rewrote the whole index ~2 min after every launch)
            self._rebuild_bm25_coalesced(force=True, mark_save=False)
        except Exception as e:
            log.warning("bm25 prepare failed: %s", e)

    def _ensure_index(self):
        if self.index is None:
            self.index = DualIndex(self.embedder.dim)
        self.index.model_tag = self._model_tag    # stamp for the DLC/model guard


    def _hash(self, path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    def _fmeta(self, path, mtime, size):
        return dict(mtime=mtime, size=size,
                    hash=self._hash(path) if self.cfg.change_detection == "hash" else None)

    def _unchanged(self, old, path, mtime, size):
        if self.cfg.change_detection == "hash":
            try:
                return old.get("hash") == self._hash(path)
            except Exception:
                return False
        return old.get("mtime") == mtime and old.get("size") == size

    def _excluded_dir(self, name):
        low = name.lower()
        if low in self.cfg.exclude_dirs:
            return True
        if self.cfg.exclude_hidden and (name.startswith(".") or name.startswith("$")):
            return True
        return False

    def in_excluded_dir(self, path, roots):
        """True if `path` lies under a dir that _walk() would prune (system/junk/
        hidden), judged relative to the most specific of `roots` -- so the live
        watcher indexes exactly what a rescan would, and a folder the user added
        explicitly is never excluded just because of where it lives."""
        full = os.path.abspath(str(path))
        key = os.path.normcase(full)
        best = None
        for r in roots:
            rn = os.path.normcase(os.path.abspath(str(r))).rstrip("\\/")
            if key.startswith(rn + os.sep) and (best is None or len(rn) > len(best)):
                best = rn
        if best is None:
            return False
        parts = full[len(best) + 1:].split(os.sep)[:-1]     # dirs below the root
        return any(self._excluded_dir(d) for d in parts if d)

    def _walk(self, root):
        """scandir walk yielding (path, mtime, size); prunes system/junk dirs."""
        exts = tuple(e.lower() for e in parser.current_exts())
        max_bytes = self.cfg.max_file_mb * 1e6
        stack = [str(root)]
        while stack:
            d = stack.pop()
            try:
                it = os.scandir(d)
            except OSError as e:
                log.warning("scandir: %s", e)
                continue
            with it:
                for entry in it:
                    try:
                        name = entry.name
                        if entry.is_dir(follow_symlinks=False):
                            if not self._excluded_dir(name):
                                stack.append(entry.path)
                            continue
                        dot = name.rfind(".")
                        if dot < 0 or name[dot:].lower() not in exts:
                            continue


                        st = entry.stat()
                        if st.st_size > max_bytes:
                            log.info("skip oversized file: %s", entry.path)
                            continue
                        yield entry.path, st.st_mtime, st.st_size
                    except OSError as e:
                        log.warning("scan entry failed: %s", e)

    @staticmethod
    def list_drives():
        """Fixed drives available to scan in 掌控模式 (e.g. ['C:\\\\', 'D:\\\\'])."""
        import string
        import ctypes
        drives = []
        try:
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for i, letter in enumerate(string.ascii_uppercase):
                if bitmask & (1 << i):
                    root = f"{letter}:\\"

                    if ctypes.windll.kernel32.GetDriveTypeW(root) == 3:
                        drives.append(root)
        except Exception:
            drives = [f"{d}:\\" for d in "CD" if os.path.isdir(f"{d}:\\")]
        return drives


    def _parse_one(self, path, mtime, size):
        """Lock-free compute: parse + chunk one file."""
        fmeta = self._fmeta(path, mtime, size)
        text = parse_file(path)
        if not text:
            return [], fmeta, True
        try:
            chunks = chunk_text(text, self.cfg.chunk_max_words,
                                self.cfg.chunk_overlap_sentences)
            return chunks, fmeta, False
        except Exception as e:
            log.warning("chunk failed, skipping %s: %s: %s", path, type(e).__name__, e)
            return [], fmeta, True

    def _apply_batch(self, batch):
        """Embed a cross-file batch in one call, then apply per file under the lock."""
        texts, spans = [], []
        for path, chunks, fmeta in batch:
            spans.append((len(texts), len(texts) + len(chunks)))
            texts.extend(c[0] for c in chunks)
        try:
            vecs = (self.embedder.embed_docs(texts) if texts
                    else np.zeros((0, self.embedder.dim), np.float32))
        except Exception as e:
            log.warning("batch embed failed (%s); falling back per-file", e)
            return [path for path, chunks, fmeta in batch
                    if self._apply_one(path, chunks, fmeta)]
        with self._lock:
            self._ensure_index()
            self.index.remove_files([p for p, _, _ in batch
                                     if p in self.index.file_chunks])
            for (path, chunks, fmeta), (a, b) in zip(batch, spans):
                self.index.add_file(path, chunks, vecs[a:b], fmeta)
        return []

    def _apply_one(self, path, chunks, fmeta):
        """Embed + apply one file; on failure record fmeta with no chunks."""
        try:
            vecs = (self.embedder.embed_docs([c[0] for c in chunks]) if chunks
                    else np.zeros((0, self.embedder.dim), np.float32))
        except Exception as e:
            log.warning("index failed, skipping %s: %s: %s", path, type(e).__name__, e)
            chunks, vecs, failed = [], np.zeros((0, self.embedder.dim), np.float32), 1
        else:
            failed = 0
        with self._lock:
            self._ensure_index()
            self.index.add_file(path, chunks, vecs, fmeta)
        return failed

    def build(self, dirs, persist=True, progress_callback=None, cancel_check=None):
        """Incremental scan/parse/embed/index; cancellable, searches stay live."""
        dirs = [str(Path(d).resolve()) for d in dirs]
        _ = self.embedder
        current = {}
        for d in dirs:
            if not os.path.isdir(d):
                log.warning("not a directory, skipping: %s", d)
                continue
            for p, mtime, size in self._walk(d):
                current[p] = (mtime, size)


        with self._lock:
            self._ensure_index()
            removed = 0
            deleted = []
            for p in [p for p in list(self.index.file_meta) if _under_any(p, dirs)]:
                if p not in current:
                    deleted.append(p)
                    removed += 1
            todo, skipped = [], 0
            for path, (mtime, size) in current.items():
                old = self.index.file_meta.get(path)
                if old is not None and self._unchanged(old, path, mtime, size):
                    skipped += 1
                else:
                    todo.append((path, mtime, size, old is not None))
            self.index.remove_files(deleted)

        added = updated = failed = cancelled = 0
        total = len(current)
        done = skipped
        batch, batch_chunks = [], 0
        existed_of = {}
        emb_batch = max(1, self.cfg.embed_batch_size)

        def flush_batch():
            nonlocal added, updated, failed, batch, batch_chunks
            if not batch:
                return
            for p in self._apply_batch(batch):
                failed += 1
                if existed_of.get(p):
                    updated -= 1
                else:
                    added -= 1
            batch, batch_chunks = [], 0

        for path, mtime, size, existed in tqdm(sorted(todo), desc="indexing", ncols=88):
            if cancel_check is not None and cancel_check():
                cancelled = 1
                log.info("build cancelled by user at %d/%d", done, total)
                break
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(done, total, path)
                except Exception:
                    pass
            chunks, fmeta, fail = self._parse_one(path, mtime, size)
            if fail:
                failed += 1
                with self._lock:
                    self._ensure_index()
                    self.index.add_file(path, [], np.zeros((0, self.embedder.dim),
                                                           np.float32), fmeta)
            else:
                batch.append((path, chunks, fmeta))
                batch_chunks += len(chunks)
                existed_of[path] = existed
                if existed:
                    updated += 1
                else:
                    added += 1
                if batch_chunks >= emb_batch:
                    flush_batch()
        flush_batch()

        if added or updated or removed or failed:
            self._rebuild_bm25_coalesced(force=True)
            if persist:
                try:
                    self.save()
                except Exception as e:
                    log.error("build save failed (periodic save will retry): %s", e)
                    self._schedule_save()
        # else: a no-op diff (e.g. the startup catch-up scan) touches nothing on disk
        stats = dict(added=added, updated=updated, removed=removed, skipped=skipped,
                     failed=failed, cancelled=bool(cancelled))
        with self._lock:
            n_files = len(self.index.file_chunks) if self.index else 0
            n_chunks = len(self.index.chunk_meta) if self.index else 0
        log.info("build done: %s | total %d files / %d chunks", stats, n_files, n_chunks)
        return stats

    def update_file(self, path, persist=False, defer=False):
        path = str(Path(path).resolve())
        if not os.path.exists(path) or not is_supported(path):
            return 0
        try:
            st = os.stat(path)
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            return 0
        if size > self.cfg.max_file_mb * 1e6:
            return 0
        with self._lock:
            self._ensure_index()
            old = self.index.file_meta.get(path)
            if old is not None and self._unchanged(old, path, mtime, size):
                return 0
        chunks, fmeta, fail = self._parse_one(path, mtime, size)
        if fail:
            with self._lock:
                self.index.add_file(path, [], np.zeros((0, self.embedder.dim),
                                                       np.float32), fmeta)
            n = 0
        else:
            self._apply_one(path, chunks, fmeta)
            n = len(chunks)
        if defer:
            self._mark_dirty()
        else:
            self._rebuild_bm25_coalesced(force=True)
            if persist:
                self.save()
        log.info("updated %s (%d chunks)", path, n)
        return n

    def remove_file(self, path, persist=False, defer=False):
        path = str(Path(path).resolve())
        with self._lock:
            if self.index is None:
                return 0
            n = self.index.remove_file(path)
        if n:
            if defer:
                self._mark_dirty()
            else:
                self._rebuild_bm25_coalesced(force=True)
                if persist:
                    self.save()
            log.info("removed %s (%d chunks)", path, n)
        return n

    def remove_tree(self, prefix, persist=False, defer=False):
        """Drop every indexed file under a directory prefix; ~O(1) no-op if none."""
        prefix = os.path.normcase(str(Path(prefix)))
        with self._lock:
            if self.index is None or not self.index.dir_known(prefix):
                return 0
            victims = [p for p in self.index.file_meta
                       if os.path.normcase(p).startswith(prefix.rstrip("\\/") + os.sep)
                       or os.path.normcase(p) == prefix]
            n = self.index.remove_files(victims) if victims else 0
        if n:
            if defer:
                self._mark_dirty()
            else:
                self._rebuild_bm25_coalesced(force=True)
                if persist:
                    self.save()
            log.info("removed tree %s (%d files, %d chunks)", prefix, len(victims), n)
        return n


    def _mark_dirty(self):
        """Schedule one coalesced BM25 rebuild after a quiet period."""
        self._dirty = True
        self._save_pending = True
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(self.cfg.flush_debounce_s, self._flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _flush(self):
        if self._dirty:
            self._dirty = False
            self._rebuild_bm25_coalesced(force=True)

    def _rebuild_bm25_coalesced(self, force=False, mark_save=True):
        """Make sure BM25 exists, then (if `mark_save`) schedule a save.

        BM25 is maintained incrementally by the index, so this only does real
        work once per load: snapshot tokens under the lock, build outside it,
        install under it (mutations made meanwhile are replayed). `mark_save` says
        the index content changed (BM25 itself is never saved); re-tokenized
        chunks (tokenizer upgrade) are worth persisting either way."""
        regenerated = 0
        with self._lock:
            ix = self.index
            if ix is None:
                return
            if not ix.bm25_built():
                bid, cids, toks, missing = ix.bm25_inputs()
            else:
                bid = None
        if bid is not None:
            bm25 = DualIndex.build_bm25(cids, toks, missing)   # outside the lock
            with self._lock:
                if self.index is ix and ix.set_bm25(bid, bm25):
                    regenerated = len(bm25.regenerated)
        if mark_save or regenerated:
            self._save_pending = True
            self._schedule_save()

    def _schedule_save(self):
        if self._closing or self._save_timer is not None:
            return
        self._save_timer = threading.Timer(self.cfg.save_interval_s, self._periodic_save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _periodic_save(self):
        self._save_timer = None
        if self._save_pending:
            try:
                self.save()
            except Exception as e:
                log.error("periodic save failed: %s", e)
                self._schedule_save()

    def save(self):
        """Serialize under the lock (fast memcpy), write to disk OUTSIDE it."""
        with self._lock:
            if self.index is None:
                return
            gen, dense_arr, store = self.index.snapshot()
            self._save_pending = False
        try:
            with self._io_lock:
                if gen > self._last_written_gen:
                    DualIndex.write_snapshot(self.index_dir, gen, dense_arr, store)
                    self._last_written_gen = gen
        except Exception:
            self._save_pending = True
            raise


    def search_semantic(self, query, top_k=10):
        if self.index is None or self.index.dense.ntotal == 0:
            return []
        qvec = self.embedder.embed_query(query)
        qtokens = tokenize(query)
        # pull a wider pool when personalizing, so the personal prior can lift a
        # genuinely-relevant file the base score ranked just below the cut.
        want = top_k * 3 if self.personalizer.enabled else top_k
        with self._lock:
            if self.index is None or self.index.dense.ntotal == 0:
                return []
            if not self.index.bm25_building():
                self.index.ensure_bm25()       # library use; the app pre-builds it
            # else: still building in the background -> dense-only for now rather
            # than stalling every search (and the status poll) on the lock
            hits = retrieve(self.index, qvec, qtokens, want,
                            self.cfg.alpha, self.cfg.fusion_fetch)
            vecs = ({h.path: self._reconstruct(h.cid) for h in hits}
                    if self.personalizer.enabled else {})
        hits = self._personalize(hits, vecs, top_k)
        make_snippets(hits, qvec, self.embedder,
                      max_sent_chars=self.cfg.snippet_sentence_chars,
                      max_sents=self.cfg.snippet_sents_per_hit)
        return hits

    def search_filename(self, query, limit=50, scope=None):
        q = query
        if scope:
            q = f'"{scope}" {query}'
        return filename_search(self.cfg.es_exe, q, limit, self.cfg.es_timeout_s)

    # ---- personalization (local, training-free) ----
    def _reconstruct(self, cid):
        """Unit embedding of a chunk id from the FAISS index (fp16-dequantized)."""
        if cid is None or cid < 0:
            return None
        try:
            v = np.asarray(self.index.dense.reconstruct(int(cid)), dtype=np.float32)
            n = np.linalg.norm(v)
            return v / n if n > 0 else None
        except Exception:
            return None

    def _personalize(self, hits, vecs, top_k):
        if self.personalizer.enabled:
            hits = self.personalizer.rerank(hits, vecs)
        hits = hits[:top_k]
        # accumulate recent shown vecs (bounded) so a click still finds its vec
        # even after another search ran (a single overwritten field would lose it)
        with self._shown_lock:
            for h in hits:
                v = vecs.get(h.path)
                if v is not None:
                    k = _norm(h.path)
                    self._last_shown[k] = v
                    self._last_shown.move_to_end(k)
            while len(self._last_shown) > 256:
                self._last_shown.popitem(last=False)
        return hits

    def record_open(self, path):
        """The user opened a result -> update the local behavioral profile."""
        if not self.personalizer.enabled:
            return
        with self._shown_lock:
            vec = self._last_shown.get(_norm(path))
        try:
            self.personalizer.record_open(path, vec)
        except Exception as e:
            log.debug("record_open failed: %s", e)

    def _embed_candidates(self, cands):
        """Embed candidate snippets through an LRU cache (locked; embed runs outside)."""
        texts = [(snip if snip else os.path.basename(path)) for path, snip in cands]
        keys = [(path, hash(t)) for (path, _), t in zip(cands, texts)]
        cache = self._instant_cache
        rows = [None] * len(keys)
        with self._cache_lock:
            for i, k in enumerate(keys):
                v = cache.get(k)
                if v is not None:
                    cache.move_to_end(k)
                    rows[i] = v
        miss_idx = [i for i, r in enumerate(rows) if r is None]
        if miss_idx:
            new = self.embedder.embed_docs([texts[i] for i in miss_idx])
            with self._cache_lock:
                for j, i in enumerate(miss_idx):
                    rows[i] = new[j]
                    cache[keys[i]] = new[j]
                while len(cache) > self.cfg.instant_cache_entries:
                    cache.popitem(last=False)
        return texts, np.vstack(rows)

    def search_instant_global(self, query, top_k=20, candidates=None, scope=None):
        """Whole-disk semantic search, no pre-indexing (Windows Search + e5 rerank)."""
        from . import winsearch
        cands = winsearch.query(query, limit=candidates or self.cfg.instant_candidates,
                                scope=scope, min_results=top_k)
        if not cands:
            return []
        qvec = self.embedder.embed_query(query)
        texts, cvecs = self._embed_candidates(cands)
        sims = cvecs @ qvec[0]
        want = top_k * 3 if self.personalizer.enabled else top_k
        order = np.argsort(-sims)[:max(top_k, want)]
        qtokens = tokenize(query)
        hits, vecs = [], {}
        for i in order:
            path, snip = cands[int(i)]
            text = texts[int(i)]
            hits.append(Hit(path=path, score=float(sims[int(i)]), chunk_text=text,
                            char_start=0, char_end=0, snippet=text[:260], terms=qtokens))
            vecs[path] = cvecs[int(i)]
        hits = self._personalize(hits, vecs, top_k)
        if self.cfg.instant_deep_snippet:
            self._deepen_instant_snippets(hits, qvec)
        return hits

    def _deepen_instant_snippets(self, hits, qvec):
        """Upgrade weak instant snippets from small .txt/.md by best-sentence pick."""
        on_gpu = getattr(self.embedder, "device", "cpu") == "cuda"
        max_files = 8 if on_gpu else 4
        max_sents = 60 if on_gpu else 20
        max_bytes = self.cfg.instant_snippet_max_kb * 1024
        picked = []
        for h in hits:
            if len(picked) >= max_files:
                break
            if len(h.snippet) >= 60:
                continue
            low = h.path.lower()
            if not (low.endswith(".txt") or low.endswith(".md")):
                continue
            try:
                if os.path.getsize(h.path) > max_bytes:
                    continue
                text = Path(h.path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if text.strip():
                picked.append((h, text))
        if not picked:
            return
        try:
            from .search import _split_sentences
            flat, owner = [], []
            for i, (h, text) in enumerate(picked):
                for s in _split_sentences(text)[:max_sents]:
                    flat.append(s[:self.cfg.snippet_sentence_chars])
                    owner.append(i)
            if not flat:
                return
            embs = self.embedder.embed_docs(flat)
            sims = embs @ qvec[0]
            best = {}
            for j, i in enumerate(owner):
                if i not in best or sims[j] > sims[best[i]]:
                    best[i] = j
            for i, (h, _) in enumerate(picked):
                if i in best:
                    h.snippet = flat[best[i]][:260]
        except Exception as e:
            log.debug("instant snippet deepen failed: %s", e)


    def start_watching(self, dirs):
        self.stop_watching()
        from .watcher import Watcher
        self._watcher = Watcher(self, self.cfg.watch_debounce_s)
        self._watcher.start([str(Path(d).resolve()) for d in dirs])

    def stop_watching(self):
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None


    # ---- image OCR (optional; Windows built-in engine) ----
    def set_ocr(self, enabled):
        """Enable/disable image OCR. Images are only indexed on the next scan."""
        self.cfg.ocr = bool(enabled)
        parser.set_ocr(self.cfg.ocr)
        return self.ocr_status()

    def ocr_status(self):
        st = ocr.status()
        st["enabled"] = bool(self.cfg.ocr and st["available"])
        return st

    # ---- DLC (optional downloadable packs) ----
    def dlc_status(self):
        return dlc.status(self.index_dir)

    def quarantine_index(self):
        """Drop the deep index (e.g. after a model pack changes its dim); it
        rebuilds on the next scan. Instant/filename modes are unaffected."""
        with self._lock:
            self.index = None
        DualIndex.quarantine(self.index_dir)

    def stats(self):
        with self._lock:
            if self.index is None:
                return dict(files=0, chunks=0, vectors=0)
            return dict(files=len(self.index.file_chunks), chunks=len(self.index.chunk_meta),
                        vectors=self.index.dense.ntotal, dim=self.index.dim)

    def close(self):
        """Persist and stop; no BM25 rebuild here (rebuilt lazily next launch)."""
        self._closing = True
        self.stop_watching()
        if self._flush_timer is not None:
            self._flush_timer.cancel()
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._dirty = False
        if self._save_pending:
            try:
                self.save()
            except Exception as e:
                log.error("final save failed: %s", e)


def _under_any(path, dirs):
    p = os.path.normcase(os.path.abspath(path))
    for d in dirs:
        d = os.path.normcase(os.path.abspath(d))
        # a drive root ("c:\\", "/") already ends in a separator: strip before
        # appending one, or nothing would ever count as "under" a whole drive
        if p == d or p.startswith(d.rstrip("\\/") + os.sep):
            return True
    return False
