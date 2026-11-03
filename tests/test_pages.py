import localsearch.engine as E
from localsearch import Config


def test_chunks_know_their_page(fake_embedder, tmp_path, monkeypatch):
    fitz = __import__("pytest").importorskip("pymupdf")
    d = tmp_path / "docs"
    d.mkdir()
    doc = fitz.open()
    for i, txt in enumerate(["Introduction and general background of the report.",
                             "Budget table for the second quarter.",
                             "The reservoir reached a record low water level in August."]):
        doc.new_page().insert_text((72, 72), txt)
    doc.save(d / "report.pdf")
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False, chunk_max_words=12))
    eng.build([d])
    hit = eng.search_semantic("reservoir record low water level", 1)[0]
    assert hit.page == 3
    eng.close()


def test_open_pdf_at_page_uses_the_browser_viewer(fake_embedder, tmp_path, monkeypatch):
    from conftest import settle
    from localsearch import app as A
    fitz = __import__("pytest").importorskip("pymupdf")
    d = tmp_path / "docs"
    d.mkdir()
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "cover page")
    doc.new_page().insert_text((72, 72), "reservoir record low water level")
    doc.save(d / "r.pdf")
    app, engine, state = A.create_app(tmp_path / "idx")
    settle(state)
    engine.build([d])
    c = app.test_client()
    res = c.post("/api/search", json={"query": "reservoir water level", "mode": "semantic"}).get_json()["results"]
    assert res[0]["page"] == 2
    monkeypatch.delenv("LOCALSEARCH_NO_LAUNCH", raising=False)
    opened = []
    monkeypatch.setattr(A.webbrowser, "open", lambda u: opened.append(u))
    r = c.post("/api/open", json={"path": res[0]["path"], "page": 2})
    assert r.get_json()["ok"] and opened[0].startswith("file://") and opened[0].endswith("#page=2")
    engine.close()
