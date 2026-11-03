import json
import ntpath
import os
import pickle
import types

import pytest

import localsearch.engine as E
import localsearch.index as I
from localsearch import Config, resources
from localsearch.chunker import TOKENIZER_VERSION
from localsearch.index import DualIndex


# ---- whole-drive roots ----------------------------------------------------
_NT = types.SimpleNamespace(path=ntpath, sep="\\")


def test_under_any_drive_root(monkeypatch):
    monkeypatch.setattr(E, "os", _NT)
    assert E._under_any(r"C:\docs\a.txt", ["C:\\"])
    assert E._under_any(r"C:\docs\a.txt", [r"C:\docs"])
    assert not E._under_any(r"D:\docs\a.txt", ["C:\\"])
    assert not E._under_any(r"C:\docs2\a.txt", [r"C:\docs"])


def test_under_any_posix_root():
    assert E._under_any("/home/u/a.txt", ["/"])


def test_dir_known_drive_root(monkeypatch):
    monkeypatch.setattr(I, "os", _NT)
    ix = DualIndex(4)
    ix.file_meta = {r"C:\Users\me\a.txt": {}}
    assert ix.dir_known("C:\\")
    assert ix.dir_known(r"C:\Users\me" + "\\")
    assert not ix.dir_known(r"C:\Program Files")


# ---- engine with a fake embedder -------------------------------------------
@pytest.fixture
def engine(fake_embedder, tmp_path):
    eng = E.SearchEngine(tmp_path / "idx", Config(personalize=False))
    yield eng
    eng.close()


def test_build_search_and_root_removal(engine, corpus):
    st = engine.build([corpus])
    assert st["added"] == 3 and st["failed"] == 0
    assert engine.search_semantic("reservoir water level")[0].path.endswith("reservoir.txt")
    assert engine.search_semantic("发票报销")[0].path.endswith("invoice.txt")
    assert engine.remove_tree(str(corpus)) > 0
    assert engine.stats()["files"] == 0


def test_deleted_file_dropped_on_rescan(engine, corpus):
    engine.build([corpus])
    (corpus / "latency.md").unlink()
    assert engine.build([corpus])["removed"] == 1


def test_noop_rescan_writes_nothing(engine, corpus):
    engine.build([corpus])
    store = engine.index_dir + "/store.pkl"
    before = os.stat(store).st_mtime_ns
    st = engine.build([corpus])
    assert st["skipped"] == 3 and not (st["added"] or st["updated"] or st["removed"])
    assert os.stat(store).st_mtime_ns == before
    assert engine._save_pending is False


def test_prepare_index_does_not_schedule_a_save(fake_embedder, corpus, tmp_path):
    a = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    a.build([corpus])
    a.close()
    b = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    b.prepare_index()
    assert b._save_pending is False and b._save_timer is None
    b.close()


def test_old_tokens_are_regenerated_and_saved(fake_embedder, corpus, tmp_path):
    a = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    a.build([corpus])
    a.close()
    p = tmp_path / "i" / "store.pkl"
    store = pickle.loads(p.read_bytes())
    store["tok_ver"] = 1
    store["tokens"] = {c: ["stale"] for c in store["tokens"]}
    p.write_bytes(pickle.dumps(store))
    b = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    assert b.index._tokens == {}
    b.prepare_index()
    assert b._save_pending is True                 # regenerated tokens get persisted
    assert all(t != ["stale"] for t in b.index._tokens.values())
    b._save_timer.cancel()
    b._save_timer = None
    b.close()
    assert pickle.loads(p.read_bytes())["tok_ver"] == TOKENIZER_VERSION


def test_watcher_filter_matches_walk(engine, tmp_path):
    root = tmp_path / "proj"
    for d in ("node_modules/pkg", ".git", "docs", "AppData/Local"):
        (root / d).mkdir(parents=True)
    ex = engine.in_excluded_dir
    assert ex(root / "node_modules" / "pkg" / "README.md", [root])
    assert ex(root / ".git" / "x.txt", [root])
    assert ex(root / "AppData" / "Local" / "a.txt", [root])
    assert not ex(root / "docs" / "a.txt", [root])
    # a folder the user added explicitly is never excluded by where it lives
    notes = root / "AppData" / "Local"
    assert not ex(notes / "a.txt", [root, notes])


# ---- model identity survives moving the portable app ------------------------
def _model_dir(base, cfg):
    d = base / "models" / "embed"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def test_index_survives_moving_the_app(fake_embedder, corpus, tmp_path):
    idx = tmp_path / "idx"
    a = E.SearchEngine(idx, Config(model_name=_model_dir(tmp_path / "v1", {"h": 384}),
                                   personalize=False))
    a.build([corpus])
    a.close()
    b = E.SearchEngine(idx, Config(model_name=_model_dir(tmp_path / "v2", {"h": 384}),
                                   personalize=False))
    assert b.load_error == "" and b.stats()["files"] == 3
    b.close()
    c = E.SearchEngine(idx, Config(model_name=_model_dir(tmp_path / "v3", {"h": 768}),
                                   personalize=False))
    assert c.load_error == "idx_version" and c.index is None
    c.close()


def test_legacy_absolute_path_tag_accepted(tmp_path):
    cur = _model_dir(tmp_path / "new", {"h": 384})
    assert resources.model_tag_matches(r"C:\Old\Scry-v1.3\_internal\models\embed", cur)
    assert not resources.model_tag_matches(r"C:\Users\me\.localsearch_index\dlc\model-large", cur)
    assert resources.model_tag_matches("intfloat/multilingual-e5-small",
                                       "intfloat/multilingual-e5-small")


def test_dim_mismatch_quarantines_instead_of_crashing(fake_embedder, corpus, tmp_path,
                                                      monkeypatch):
    a = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    a.build([corpus])
    a.close()

    class Wider(fake_embedder):
        dim = 32
    monkeypatch.setattr(E, "Embedder", Wider)
    b = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    assert b.embedder.dim == 32
    assert b.index is None and b.load_error == "idx_version"
    assert b.search_semantic("reservoir") == []
    b.close()
