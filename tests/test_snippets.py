import localsearch.engine as E
from localsearch import Config
from localsearch.search import Hit, VecCache, lexical_snippet, make_snippets

TEXT = ("Background filler about many things. The Larkfield reservoir hit a record low. "
        "Unrelated closing remark here.")


def test_lexical_snippet_picks_overlapping_sentence():
    assert lexical_snippet(TEXT, ["reservoir", "low"]).startswith("…The Larkfield reservoir")
    assert lexical_snippet(TEXT, ["zzz"]).startswith("Background")      # no overlap: chunk start


def test_cached_snippets_identical_and_skip_the_model(fake_embedder):
    emb = fake_embedder()
    calls = []
    orig = emb.embed_docs
    emb.embed_docs = lambda t: calls.append(len(t)) or orig(t)
    q = emb.embed_query("reservoir record low")
    plain = [Hit("/a", 1, TEXT, 0, 0)]
    make_snippets(plain, q, emb)
    cache = VecCache()
    first, again = [Hit("/a", 1, TEXT, 0, 0)], [Hit("/a", 1, TEXT, 0, 0)]
    make_snippets(first, q, emb, cache=cache)
    n = len(calls)
    make_snippets(again, q, emb, cache=cache)
    assert plain[0].snippet == first[0].snippet == again[0].snippet
    assert len(calls) == n                    # second run fully served from cache


def test_progressive_equals_synchronous(fake_embedder, corpus, tmp_path):
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    eng.build([corpus])
    sync = {h.path: h.snippet for h in eng.search_semantic("reservoir water level", 3)}
    fast = eng.search_semantic("reservoir water level", 3, snippets="lexical")
    sid = eng.queue_snippets("semantic", "reservoir water level", fast)
    assert eng.resolve_snippets(sid) == sync
    assert eng.resolve_snippets(sid) == {}    # one-shot
    eng.close()


def test_api_search_then_snippets(fake_embedder, corpus, tmp_path):
    from localsearch import app as A
    app, engine, state = A.create_app(tmp_path / "idx")
    engine.build([corpus])
    c = app.test_client()
    r = c.post("/api/search", json={"query": "reservoir water", "mode": "semantic"}).get_json()
    assert r["results"] and r["snip_id"]
    snips = c.post("/api/snippets", json={"id": r["snip_id"]}).get_json()["snippets"]
    assert set(snips) <= {x["path"] for x in r["results"]} and snips
    assert c.post("/api/snippets", json={"id": "nope"}).get_json() == {"snippets": {}}
    engine.close()
