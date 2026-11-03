import localsearch.engine as E
from localsearch import Config


def test_build_reports_what_it_could_not_read(fake_embedder, tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "ok.txt").write_text("readable text", encoding="utf-8")
    (d / "thesis.caj").write_bytes(b"x")              # CNKI format: not supported
    (d / "a.png").write_bytes(b"x")                   # image while OCR is off
    (d / "broken.docx").write_bytes(b"not a zip")      # supported but unreadable
    (d / "empty.pdf").write_bytes(b"%PDF-1.4 broken")
    (d / "setup.exe").write_bytes(b"x")               # not document-like: not counted
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    st = eng.build([d])
    assert st["unsupported"] == {".caj": 1, ".png": 1}
    assert st["failed_by"] == {".docx": 1, ".pdf": 1}
    # cumulative: a diff rescan still reports the unreadable files it skipped
    st2 = eng.build([d])
    assert st2["skipped"] == 3 and st2["failed_by"] == {".docx": 1, ".pdf": 1}
    eng.close()
