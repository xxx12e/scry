"""Incremental Okapi BM25 over an inverted index.

Same formula and constants as rank_bm25.BM25Okapi (k1=1.5, b=0.75, negative
IDFs floored to epsilon * mean IDF), so scores are identical -- but:

  * a query touches only the postings of its own terms, instead of a Python
    loop over every chunk per term (rank_bm25: ~140 ms per query at 100k
    chunks, ~480 ms at 300k, all of it while holding the engine lock);
  * adding/removing a chunk is an in-place update, instead of refitting the
    whole model after every watched file change (5-18 s of CPU at that scale).

Layout: an immutable term-major CSR "base" (numpy; bulk-built once per load,
~5x less RAM than per-term dicts), a small dict "delta" for chunks added since,
and tombstones for removed base chunks. Not thread-safe by itself: the engine
mutates and queries it under its lock.
"""
import math
from collections import Counter

import numpy as np


class BM25Index:
    k1 = 1.5
    b = 0.75
    epsilon = 0.25

    def __init__(self):
        # base (bulk-built): postings of term j are rows[indptr[j]:indptr[j+1]]
        self._tid = {}
        self._indptr = np.zeros(1, np.int64)
        self._rows = np.zeros(0, np.int32)
        self._tf = np.zeros(0, np.float32)
        self._row_cid = np.zeros(0, np.int64)
        self._row_dl = np.zeros(0, np.float64)
        self._alive = np.zeros(0, bool)
        self._row_of = {}                 # live base cid -> row
        # delta (added after the bulk build)
        self.post = {}                    # term -> {cid: tf}
        self.dl = {}                      # cid -> document length
        self.n = 0                        # live documents
        self.total = 0                    # sum of live document lengths

    def __len__(self):
        return self.n

    def __contains__(self, cid):
        return cid in self._row_of or cid in self.dl

    @classmethod
    def build(cls, cids, toks):
        ix = cls()
        cids = list(cids)
        terms, tfs, uniq, dls = [], [], [], []
        for t in toks:
            c = Counter(t)
            terms.extend(c)
            tfs.extend(c.values())
            uniq.append(len(c))
            dls.append(len(t))
        names = list(dict.fromkeys(terms))
        ix._tid = {t: j for j, t in enumerate(names)}
        ids = np.fromiter(map(ix._tid.__getitem__, terms), np.int32, len(terms))
        order = np.argsort(ids, kind="stable")
        ix._indptr = np.zeros(len(names) + 1, np.int64)
        np.cumsum(np.bincount(ids, minlength=len(names)), out=ix._indptr[1:])
        ix._rows = np.repeat(np.arange(len(cids), dtype=np.int32), uniq)[order]
        ix._tf = np.asarray(tfs, np.float32)[order]
        ix._row_cid = np.asarray(cids, np.int64)
        ix._row_dl = np.asarray(dls, np.float64)
        ix._alive = np.ones(len(cids), bool)
        ix._row_of = dict(zip(cids, range(len(cids))))
        ix.n, ix.total = len(cids), int(sum(dls))
        return ix

    def add(self, cid, toks):
        if cid in self:
            self.remove(cid, toks)
        self.dl[cid] = len(toks)
        self.n += 1
        self.total += len(toks)
        post = self.post
        for t, tf in Counter(toks).items():
            p = post.get(t)
            if p is None:
                post[t] = {cid: tf}
            else:
                p[cid] = tf

    def remove(self, cid, toks):
        r = self._row_of.pop(cid, None)
        if r is not None:                          # base chunk: tombstone it
            self._alive[r] = False
            self.n -= 1
            self.total -= int(self._row_dl[r])
            return
        n = self.dl.pop(cid, None)
        if n is None:
            return
        self.n -= 1
        self.total -= n
        post = self.post
        for t in set(toks):
            p = post.get(t)
            if p is not None:
                p.pop(cid, None)
                if not p:
                    del post[t]

    def _mean_idf(self, n_docs):
        """Mean raw IDF over the live vocabulary (rank_bm25's epsilon floor)."""
        live = np.concatenate(([0], np.cumsum(self._alive[self._rows], dtype=np.int64)))
        df = (live[self._indptr[1:]] - live[self._indptr[:-1]]).astype(np.float64)
        extra = []
        for t, p in self.post.items():
            j = self._tid.get(t)
            if j is None:
                extra.append(len(p))
            else:
                df[j] += len(p)
        df = np.concatenate((df, np.asarray(extra, np.float64)))
        df = df[df > 0]
        if not len(df):
            return 0.0
        return float(np.mean(np.log(n_docs - df + 0.5) - np.log(df + 0.5)))

    def top(self, qtokens, k, allowed=None):
        """[(cid, score)] of the best `k` live chunks containing a query term
        (only among `allowed` cids if given; IDF stays corpus-wide)."""
        n_docs = self.n
        if n_docs <= 0 or self.total <= 0 or not qtokens or k <= 0:
            return []
        avgdl = self.total / n_docs
        k1, b = self.k1, self.b
        mean_idf = None
        all_c, all_s = [], []
        for q in qtokens:                      # repeated terms count again, as in rank_bm25
            rows = tf = None
            j = self._tid.get(q)
            if j is not None:
                a, e = self._indptr[j], self._indptr[j + 1]
                rows, tf = self._rows[a:e], self._tf[a:e]
                alive = self._alive[rows]
                if not alive.all():
                    rows, tf = rows[alive], tf[alive]
            p = self.post.get(q)
            df = (0 if rows is None else len(rows)) + (len(p) if p else 0)
            if not df:
                continue
            idf = math.log(n_docs - df + 0.5) - math.log(df + 0.5)
            if idf < 0:                        # term in > half the chunks
                if mean_idf is None:
                    mean_idf = self._mean_idf(n_docs)
                idf = self.epsilon * mean_idf
            if rows is not None and len(rows):
                f = tf.astype(np.float64)
                dl = self._row_dl[rows]
                all_c.append(self._row_cid[rows])
                all_s.append(idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl)))
            if p:
                keys = p.keys()
                f = np.fromiter(p.values(), np.float64, len(p))
                dl = np.fromiter(map(self.dl.__getitem__, keys), np.float64, len(p))
                all_c.append(np.fromiter(keys, np.int64, len(p)))
                all_s.append(idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl)))
        if not all_c:
            return []
        cids, sc = np.concatenate(all_c), np.concatenate(all_s)
        if allowed is not None:
            keep = np.isin(cids, allowed)
            cids, sc = cids[keep], sc[keep]
            if not len(cids):
                return []
        if len(all_c) > 1:
            cids, inv = np.unique(cids, return_inverse=True)
            sc = np.bincount(inv, weights=sc)
        n = min(k, len(cids))
        top = np.argpartition(-sc, n - 1)[:n]
        top = top[np.argsort(-sc[top], kind="stable")]
        return [(int(cids[i]), float(sc[i])) for i in top]
