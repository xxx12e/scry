import random

import numpy as np
import pytest

from localsearch.bm25 import BM25Index
from localsearch.index import DualIndex

rank_bm25 = pytest.importorskip("rank_bm25")


def _corpus(rnd, n, vocab=60):
    # small vocab -> plenty of terms in > half the docs (exercises the epsilon floor)
    return {c: [f"w{rnd.randrange(vocab)}" for _ in range(rnd.randint(0, 30))]
            for c in range(n)}


def _assert_same(ix, docs, rnd):
    cids = list(docs)
    ref = rank_bm25.BM25Okapi([docs[c] for c in cids])
    for _ in range(25):
        q = [f"w{rnd.randrange(70)}" for _ in range(rnd.randint(1, 5))]
        want = ref.get_scores(q)
        got = dict(ix.top(q, len(cids)))
        for i, c in enumerate(cids):
            matched = any(t in docs[c] for t in q)
            assert (c in got) == matched
            if matched:
                assert got[c] == pytest.approx(want[i], rel=1e-9, abs=1e-12)


def test_scores_identical_to_rank_bm25():
    rnd = random.Random(3)
    docs = _corpus(rnd, 400)
    _assert_same(BM25Index.build(list(docs), list(docs.values())), docs, rnd)


def test_incremental_updates_match_a_fresh_fit():
    rnd = random.Random(5)
    docs = _corpus(rnd, 300)
    ix = BM25Index.build(list(docs), list(docs.values()))
    nxt = 300
    for _ in range(200):
        if rnd.random() < 0.5 and docs:
            c = rnd.choice(list(docs))
            ix.remove(c, docs.pop(c))
        else:
            docs[nxt] = [f"w{rnd.randrange(60)}" for _ in range(rnd.randint(1, 30))]
            ix.add(nxt, docs[nxt])
            nxt += 1
    _assert_same(ix, docs, rnd)


def test_topk_order():
    ix = BM25Index.build([1, 2, 3], [["a", "a", "b"], ["a"], ["c"]])
    assert {c for c, _ in ix.top(["a"], 10)} == {1, 2}
    assert ix.top(["zzz"], 10) == [] and ix.top([], 10) == []
    assert len(ix.top(["a", "c"], 1)) == 1


def test_out_of_lock_build_replays_mutations():
    vec = lambda n: np.ones((n, 4), np.float32) / 2
    ix = DualIndex(4)
    ix.add_file("a.txt", [("alpha beta", 0, 10)], vec(1), {})
    ix.add_file("b.txt", [("gamma delta", 0, 11)], vec(1), {})
    assert ix.bm25_built()                         # a fresh index is live from the start
    ix._bm25 = None                                # ...a loaded one is not (as after load())
    bid, cids, toks, _ = ix.bm25_inputs()          # snapshot, build "elsewhere"
    assert ix.bm25_building()
    ix.add_file("c.txt", [("alpha omega", 0, 11)], vec(1), {})   # concurrent writes
    ix.remove_file("b.txt")
    assert ix.set_bm25(bid, DualIndex.build_bm25(cids, toks))
    hits = {ix.chunk_meta[c]["path"] for c, _ in ix.search_bm25(["alpha"], 10)}
    assert hits == {"a.txt", "c.txt"}
    assert ix.search_bm25(["gamma"], 10) == []
    # a superseded build is never installed over a live one
    assert not ix.set_bm25(bid, DualIndex.build_bm25(cids, toks))


def test_engine_search_during_background_build_is_dense_only(fake_embedder, corpus, tmp_path):
    import localsearch.engine as E
    from localsearch import Config
    a = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    a.build([corpus])
    a.close()
    b = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    assert not b.index.bm25_built()
    bid, cids, toks, _ = b.index.bm25_inputs()     # a background build is "running"
    assert b.search_semantic("reservoir water")    # answers, does not build under the lock
    assert b.index.bm25_building()
    assert b.index.set_bm25(bid, DualIndex.build_bm25(cids, toks))
    assert b.index.search_bm25(["reservoir"], 5)
    b.close()
