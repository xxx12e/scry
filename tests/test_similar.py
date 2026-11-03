from conftest import settle
import os

import localsearch.engine as E
from localsearch import Config


def _corpus(d):
    d.mkdir()
    (d / "r1.txt").write_text("reservoir water level drought record low dam", encoding="utf-8")
    (d / "r2.txt").write_text("drought reservoir dam water shortage level", encoding="utf-8")
    (d / "inv.txt").write_text("invoice reimbursement approval workflow finance", encoding="utf-8")
    return d


def test_similar_finds_the_neighbour_and_skips_itself(fake_embedder, tmp_path):
    d = _corpus(tmp_path / "docs")
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    eng.build([d])
    hits, q = eng.similar(str((d / "r1.txt").resolve()), top_k=5)
    assert [os.path.basename(h.path) for h in hits][0] == "r2.txt"
    assert all(not h.path.endswith("r1.txt") for h in hits)
    # a file outside the index is embedded on the fly
    ext = tmp_path / "outside.txt"
    ext.write_text("dam reservoir drought water", encoding="utf-8")
    hits2, _ = eng.similar(str(ext), top_k=5)
    assert os.path.basename(hits2[0].path) in ("r1.txt", "r2.txt")
    sid = eng.queue_snippets("semantic", q, hits)
    assert set(eng.resolve_snippets(sid)) == {h.path for h in hits}
    eng.close()


def test_api_similar_requires_a_shown_result(fake_embedder, tmp_path):
    from localsearch import app as A
    d = _corpus(tmp_path / "docs")
    app, engine, state = A.create_app(tmp_path / "idx")
    settle(state)
    engine.build([d])
    c = app.test_client()
    target = str((d / "r1.txt").resolve())
    assert c.post("/api/similar", json={"path": target}).status_code == 403
    c.post("/api/search", json={"query": "reservoir drought", "mode": "semantic"})
    r = c.post("/api/similar", json={"path": target}).get_json()
    assert r["similar_to"] == "r1.txt" and r["results"][0]["name"] == "r2.txt" and r["snip_id"]
    engine.close()
