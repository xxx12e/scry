import os
from datetime import datetime

import pytest

import localsearch.engine as E
from localsearch import Config
from localsearch.query import parse

TEXT = "The Larkfield reservoir dropped to a record low water level."


@pytest.fixture
def eng(fake_embedder, tmp_path):
    d = tmp_path / "docs"
    (d / "合同").mkdir(parents=True)
    files = {"old.txt": datetime(2024, 1, 10), "new.md": datetime(2026, 10, 20),
             "合同/legal.txt": datetime(2025, 6, 1)}
    for name, when in files.items():
        p = d / name
        p.write_text(TEXT + " " + name, encoding="utf-8")
        os.utime(p, (when.timestamp(), when.timestamp()))
    e = E.SearchEngine(tmp_path / "idx", Config(personalize=False))
    e.build([d])
    yield e
    e.close()


def names(hits):
    return sorted(os.path.basename(h.path) for h in hits)


@pytest.mark.parametrize("raw,want", [
    ("reservoir", ["legal.txt", "new.md", "old.txt"]),
    ("reservoir ext:md", ["new.md"]),
    ("reservoir type:text in:合同", ["legal.txt"]),
    ("reservoir before:2024-02", ["old.txt"]),
    ("reservoir time:2025", ["legal.txt"]),
    ("reservoir ext:pdf", []),
])
def test_deep_search_filters(eng, raw, want):
    qf = parse(raw)
    assert names(eng.search_semantic(qf.text, 10, filters=qf)) == want


def test_browse_newest_first(eng):
    hits = eng.browse(parse("type:text"), 10)
    assert [os.path.basename(h.path) for h in hits] == ["new.md", "legal.txt", "old.txt"]
    assert hits[0].snippet.startswith("The Larkfield")


def test_filename_mode_translates_and_rechecks(eng, monkeypatch, tmp_path):
    files = {"a/合同/x.md": datetime(2025, 1, 1), "b/x.md": datetime(2025, 1, 1),
             "a/合同/x.txt": datetime(2025, 1, 1), "a/合同/old.md": datetime(2023, 1, 1)}
    for rel, when in files.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
        os.utime(f, (when.timestamp(), when.timestamp()))
    seen = {}

    def fake_es(exe, q, limit, timeout):        # an Everything that ignores every filter
        seen["q"] = q
        return [str(tmp_path / r) for r in files]
    monkeypatch.setattr(E, "filename_search", fake_es)
    qf = parse("report ext:md in:合同 after:2024-03")
    out = eng.search_filename(qf.text, 10, filters=qf)
    assert "report" in seen["q"] and "ext:md" in seen["q"] and "dm:>=2024-03-01" in seen["q"]
    assert 'path:"合同"' in seen["q"]
    # re-checked locally: wrong folder, wrong type and too old are all dropped
    assert out == [str(tmp_path / "a/合同/x.md")]


def test_api_reports_filters_and_suggestion(fake_embedder, tmp_path, corpus):
    from localsearch import app as A
    app, engine, state = A.create_app(tmp_path / "i")
    engine.build([corpus])
    c = app.test_client()
    r = c.post("/api/search", json={"query": "reservoir type:text", "mode": "semantic"}).get_json()
    assert r["filters"][0]["kind"] == "type" and r["results"]
    assert all(x["ext"] in ("txt", "md") for x in r["results"])
    r = c.post("/api/search", json={"query": "去年的水库报告", "mode": "semantic"}).get_json()
    assert r["suggest"]["op"] == "time:去年" and not r["filters"]
    r = c.post("/api/search", json={"query": "type:md", "mode": "semantic"}).get_json()
    assert [x["name"] for x in r["results"]] == ["latency.md"] and r["mode_used"] == "semantic"
    engine.close()


def test_bm25_allowed_subset():
    from localsearch.bm25 import BM25Index
    ix = BM25Index.build([1, 2, 3], [["a", "b"], ["a"], ["a", "c"]])
    assert {c for c, _ in ix.top(["a"], 10, allowed=[2, 3])} == {2, 3}
    assert ix.top(["a"], 10, allowed=[]) == []
