import types

import localsearch.engine as E
from localsearch import Config


def test_cloud_only_attributes():
    st = lambda a: types.SimpleNamespace(st_file_attributes=a)
    assert E.cloud_only(st(0x00400000))            # RECALL_ON_DATA_ACCESS (OneDrive online-only)
    assert E.cloud_only(st(0x00040000))            # RECALL_ON_OPEN
    assert E.cloud_only(st(0x00001000))            # OFFLINE
    assert not E.cloud_only(st(0x20 | 0x80000))    # ARCHIVE | PINNED ("always keep on device")
    assert not E.cloud_only(types.SimpleNamespace())   # non-Windows stat


def test_placeholders_are_never_read(fake_embedder, tmp_path, monkeypatch):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "local.txt").write_text("reservoir water level report", encoding="utf-8")
    (d / "cloud.txt").write_text("x" * 4242, encoding="utf-8")       # marker size
    monkeypatch.setattr(E, "cloud_only", lambda st: st.st_size == 4242)
    read = []
    orig = E.parse_file
    monkeypatch.setattr(E, "parse_file", lambda p: read.append(p) or orig(p))
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    st = eng.build([d])
    assert st["added"] == 1 and st["cloud_skipped"] == 1
    assert not any(p.endswith("cloud.txt") for p in read)
    assert eng.update_file(str(d / "cloud.txt")) == 0               # watcher path too
    eng.close()


def test_indexed_file_survives_free_up_space(fake_embedder, tmp_path, monkeypatch):
    d = tmp_path / "docs"
    d.mkdir()
    f = d / "notes.txt"
    f.write_text("y" * 4242, encoding="utf-8")
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    assert eng.build([d])["added"] == 1                # indexed while local
    monkeypatch.setattr(E, "cloud_only", lambda st: st.st_size == 4242)   # now online-only
    st = eng.build([d])
    assert st["removed"] == 0 and st["skipped"] == 1 and eng.stats()["files"] == 1
    eng.close()
