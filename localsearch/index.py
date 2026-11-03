"""Dual index: fp16 FAISS dense (exact) + BM25 lexical, crash-durable on disk."""
import logging
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import faiss

from .bm25 import BM25Index
from .chunker import tokenize, TOKENIZER_VERSION

log = logging.getLogger("localsearch.index")
faiss.omp_set_num_threads(1)

INDEX_VERSION = 3                 # bumped: embedding model swapped (dim 768->384)
_MAGIC = b"LSDX0003"


class IndexVersionError(Exception):
    """Index on disk was built by an incompatible (older) version."""


class IndexCorruptError(Exception):
    """Index files on disk are torn/inconsistent (e.g. power loss mid-save)."""


def _new_dense(dim):
    inner = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16,
                                       faiss.METRIC_INNER_PRODUCT)
    return faiss.IndexIDMap2(inner)


class DualIndex:
    def __init__(self, dim):
        self.dim = dim
        self.dense = _new_dense(dim)
        self.chunk_meta = {}
        self.file_chunks = {}
        self.file_meta = {}
        self.next_id = 0
        self.generation = 0
        self.model_tag = ""       # which embedding model built this index (dim guard)
        self._bm25 = BM25Index()  # live from the start; load() resets it to None
        self._bm25_log = None     # mutations made while an out-of-lock build runs
        self._bm25_build = 0      # id of the build whose result may be installed
        self._tokens = {}
        self._dir_refs = None


    @staticmethod
    def _ancestors(path):
        p = os.path.normcase(str(path))
        while True:
            parent = os.path.dirname(p)
            if not parent or parent == p:
                return
            yield parent
            p = parent

    def _dir_ref(self, path, delta):
        if self._dir_refs is None:
            return
        for d in self._ancestors(path):
            n = self._dir_refs.get(d, 0) + delta
            if n > 0:
                self._dir_refs[d] = n
            else:
                self._dir_refs.pop(d, None)

    def dir_known(self, prefix):
        """True if any indexed file lives under `prefix` (O(1) after lazy build)."""
        if self._dir_refs is None:
            self._dir_refs = {}
            for p in self.file_meta:
                self._dir_ref(p, +1)
        # normpath (not rstrip): keeps a drive root as "c:\\" / "/", which is
        # exactly how _ancestors() spells it
        return os.path.normcase(os.path.normpath(str(prefix))) in self._dir_refs


    def add_file(self, path, chunks, vecs, fmeta):
        """chunks: [(text,start,end)]; vecs: [n,dim] aligned to chunks."""
        path = str(path)
        if path in self.file_chunks:
            self.remove_file(path)
        self._dir_ref(path, +1)
        if len(chunks) == 0:
            self.file_meta[path] = fmeta
            self.file_chunks[path] = []
            return 0
        cids = list(range(self.next_id, self.next_id + len(chunks)))
        self.next_id += len(chunks)
        self.dense.add_with_ids(vecs, np.array(cids, dtype=np.int64))
        for cid, (text, st, en), i in zip(cids, chunks, range(len(chunks))):
            self.chunk_meta[cid] = dict(path=path, idx=i, text=text, start=st, end=en)
            toks = self._tokens[cid] = tokenize(text)
            self._bm25_apply(True, cid, toks)
        self.file_chunks[path] = cids
        self.file_meta[path] = fmeta
        return len(chunks)

    def _pop_file(self, path):
        """Drop a file's dict entries; return its cids (dense NOT touched)."""
        if path in self.file_chunks:
            self._dir_ref(path, -1)
        cids = self.file_chunks.pop(path, [])
        self.file_meta.pop(path, None)
        tracking = self._bm25 is not None or self._bm25_log is not None
        for cid in cids:
            meta = self.chunk_meta.pop(cid, None)
            toks = self._tokens.pop(cid, None)
            if tracking:
                if toks is None and meta is not None:
                    toks = tokenize(meta["text"])
                self._bm25_apply(False, cid, toks or [])
        return cids

    def remove_file(self, path):
        cids = self._pop_file(str(path))
        if cids:
            self.dense.remove_ids(faiss.IDSelectorBatch(np.array(cids, dtype=np.int64)))
        return len(cids)

    def remove_files(self, paths):
        """Batched removal: one remove_ids pass for many files."""
        all_cids = []
        for p in paths:
            all_cids.extend(self._pop_file(str(p)))
        if all_cids:
            self.dense.remove_ids(faiss.IDSelectorBatch(np.array(all_cids, dtype=np.int64)))
        return len(all_cids)


    def _bm25_apply(self, add, cid, toks):
        if self._bm25 is not None:
            (self._bm25.add if add else self._bm25.remove)(cid, toks)
        if self._bm25_log is not None:
            self._bm25_log.append((add, cid, toks))

    def bm25_built(self):
        return self._bm25 is not None

    def bm25_building(self):
        """An out-of-lock build is in flight (bm25_inputs() taken, not installed)."""
        return self._bm25 is None and self._bm25_log is not None

    def bm25_inputs(self):
        """Snapshot (build_id, cids, token_lists, missing) for an out-of-lock
        BM25 build (call under the lock; O(chunks) pointer copies only).
        `missing` maps position -> chunk text for chunks with no cached tokens
        (e.g. after a tokenizer upgrade): build_bm25() tokenizes them OUTSIDE the
        lock. Mutations from here on are logged and replayed by set_bm25()."""
        self._bm25_build += 1
        self._bm25_log = []
        cids = list(self.chunk_meta.keys())
        get = self._tokens.get
        toks = [get(c) for c in cids]
        missing = {i: self.chunk_meta[cids[i]]["text"]
                   for i, t in enumerate(toks) if t is None}
        return self._bm25_build, cids, toks, missing

    def set_bm25(self, build_id, bm25):
        """Install a finished build (under the lock): replay the mutations logged
        since its snapshot. A superseded build (another one started, or one was
        already installed) is dropped. Returns True if installed."""
        if build_id != self._bm25_build or self._bm25 is not None or self._bm25_log is None:
            return False
        for add, cid, toks in self._bm25_log:
            (bm25.add if add else bm25.remove)(cid, toks)
        for cid, toks in getattr(bm25, "regenerated", {}).items():
            if cid in self.chunk_meta and cid not in self._tokens:
                self._tokens[cid] = toks               # cache re-derived tokens
        self._bm25, self._bm25_log = bm25, None
        return True

    @staticmethod
    def build_bm25(cids, toks, missing=None):
        """Pure compute -- safe to run outside the engine lock on a snapshot.
        Tokenizes the `missing` chunks first; their tokens ride along on the
        result (`.regenerated`) so set_bm25() can cache them."""
        regenerated = {}
        if missing:
            toks = list(toks)
            for i, text in missing.items():
                toks[i] = regenerated[cids[i]] = tokenize(text)
        bm25 = BM25Index.build(cids, toks)
        bm25.regenerated = regenerated
        return bm25

    def rebuild_bm25(self):
        self._bm25 = None
        bid, cids, toks, missing = self.bm25_inputs()
        self.set_bm25(bid, self.build_bm25(cids, toks, missing))

    def ensure_bm25(self):
        """Build BM25 on first need (loads no longer build it -> fast startup).
        After that it is updated in place on every add/remove -- no refits."""
        if self._bm25 is None:
            self.rebuild_bm25()


    def search_dense(self, qvec, topk):
        if self.dense.ntotal == 0:
            return []
        D, I = self.dense.search(qvec, min(topk, self.dense.ntotal))
        return [(int(c), float(s)) for c, s in zip(I[0], D[0]) if c >= 0]

    def search_bm25(self, qtokens, topk):
        """Best chunks containing a query term (non-matching chunks are not
        returned; search.retrieve() normalizes against a zero floor)."""
        if self._bm25 is None:
            return []
        return self._bm25.top(qtokens, topk)


    def snapshot(self):
        """Serialize under the engine lock; write_snapshot() writes it outside."""
        self.generation += 1
        dense_arr = faiss.serialize_index(self.dense)
        store = dict(version=INDEX_VERSION, generation=self.generation, dim=self.dim,
                     chunk_meta=dict(self.chunk_meta), file_chunks=dict(self.file_chunks),
                     file_meta=dict(self.file_meta), next_id=self.next_id,
                     tokens=dict(self._tokens), tok_ver=TOKENIZER_VERSION,
                     model_tag=self.model_tag)
        return self.generation, dense_arr, store

    @staticmethod
    def write_snapshot(index_dir, generation, dense_arr, store):
        """Write tmp-first with fsync, then rename back-to-back (torn-save safe)."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp_d = index_dir / "dense.faiss.tmp"
        with open(tmp_d, "wb") as f:
            f.write(_MAGIC + struct.pack("<Q", generation))
            dense_arr.tofile(f)
            f.flush()
            os.fsync(f.fileno())
        tmp_s = index_dir / "store.pkl.tmp"
        with open(tmp_s, "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        tmp_d.replace(index_dir / "dense.faiss")
        tmp_s.replace(index_dir / "store.pkl")
        log.info("saved index gen %d: %d files, %d chunks -> %s",
                 generation, len(store["file_chunks"]), len(store["chunk_meta"]), index_dir)

    def save(self, index_dir):
        gen, dense_arr, store = self.snapshot()
        self.write_snapshot(index_dir, gen, dense_arr, store)

    @classmethod
    def load(cls, index_dir):
        index_dir = Path(index_dir)
        with open(index_dir / "store.pkl", "rb") as f:
            store = pickle.load(f)
        if store.get("version") != INDEX_VERSION:
            raise IndexVersionError(
                f"index version {store.get('version')} != {INDEX_VERSION} (rebuild needed)")
        with open(index_dir / "dense.faiss", "rb") as f:
            raw = f.read()
        if len(raw) < 16 or raw[:8] != _MAGIC:
            raise IndexCorruptError("dense.faiss: bad header")
        (gen,) = struct.unpack("<Q", raw[8:16])
        if gen != store.get("generation"):
            raise IndexCorruptError(
                f"dense.faiss gen {gen} != store gen {store.get('generation')} (torn save)")
        obj = cls(store["dim"])
        obj.dense = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8, offset=16))
        obj.chunk_meta = store["chunk_meta"]
        obj.file_chunks = store["file_chunks"]
        obj.file_meta = store["file_meta"]
        obj.next_id = store["next_id"]
        obj.generation = store["generation"]
        obj.model_tag = store.get("model_tag", "")
        obj._bm25 = None          # postings are not persisted: built after load
        # tokens from an older tokenizer are dropped and re-derived from the chunk
        # text by bm25_inputs() (cheap; embeddings are untouched)
        obj._tokens = ((store.get("tokens") or {})
                       if store.get("tok_ver", 1) == TOKENIZER_VERSION else {})


        if obj.dense.ntotal != len(obj.chunk_meta):
            raise IndexCorruptError(
                f"dense has {obj.dense.ntotal} vectors but store has "
                f"{len(obj.chunk_meta)} chunks")
        if obj.chunk_meta and obj.next_id <= max(obj.chunk_meta):
            raise IndexCorruptError("next_id <= max chunk id (id reuse would corrupt)")

        log.info("loaded index gen %d: %d files, %d chunks",
                 obj.generation, len(obj.file_chunks), len(obj.chunk_meta))
        return obj

    @staticmethod
    def exists(index_dir):
        index_dir = Path(index_dir)
        return (index_dir / "store.pkl").exists() and (index_dir / "dense.faiss").exists()

    @staticmethod
    def quarantine(index_dir):
        """Move damaged index files aside (kept for post-mortem, out of the way)."""
        index_dir = Path(index_dir)
        for name in ("dense.faiss", "store.pkl"):
            p = index_dir / name
            try:
                if p.exists():
                    p.replace(index_dir / (name + ".corrupt"))
            except OSError as e:
                log.warning("could not quarantine %s: %s", p, e)
