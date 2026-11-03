import localsearch.engine as E
from localsearch import Config, winsearch
from localsearch.query import parse


def test_falls_back_to_windows_search_when_everything_is_missing(fake_embedder, tmp_path, monkeypatch):
    f = tmp_path / "季度报告 Q3.pdf"
    f.write_bytes(b"%PDF")
    seen = {}

    def fake_sql(sql):
        seen["sql"] = sql
        return [("file:" + str(f).replace("\\\\", "/"), "x")]
    monkeypatch.setattr(winsearch, "_run_sql", fake_sql)
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False, es_exe=str(tmp_path / "nope.exe")))
    qf = parse("季度报告 *.pdf type:pdf")
    paths = eng.search_filename(qf.text, 10, filters=qf)
    assert eng.filename_backend == "windows" and len(paths) == 1
    assert "System.FileName LIKE '%季度报告%'" in seen["sql"] and "LIKE '%%.pdf%'" in seen["sql"]
    assert "System.FileExtension = '.pdf'" in seen["sql"]
    monkeypatch.setattr(winsearch, "_run_sql", lambda sql: None)      # provider down too
    assert eng.search_filename("x", 10) is None
    eng.close()


def test_no_everything_and_no_windows_search_is_a_clean_error(fake_embedder, tmp_path, monkeypatch):
    """Regression: the fallback imported win32com outside its try -> HTTP 500."""
    import builtins
    from conftest import settle
    from localsearch import app as A
    real_import = builtins.__import__

    def no_pywin32(name, *a, **k):
        if name.startswith("win32com"):
            raise ImportError("no pywin32 here")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_pywin32)
    app, engine, state = A.create_app(tmp_path / "idx", Config(es_exe=str(tmp_path / "nope.exe")))
    settle(state)
    r = app.test_client().post("/api/search", json={"query": "report", "mode": "filename"})
    assert r.status_code == 200 and r.get_json()["code"] == "no_everything"
    engine.close()
