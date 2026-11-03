from conftest import settle
import pytest

from localsearch import app as A


@pytest.fixture
def client(fake_embedder, tmp_path):
    app, engine, state = A.create_app(tmp_path / "idx")
    settle(state)
    app.testing = True
    yield app.test_client(), engine, state
    engine.close()


@pytest.mark.parametrize("host,origin,ok", [
    ("127.0.0.1:8731", None, True),
    ("localhost:8731", "http://localhost:8731", True),
    ("[::1]:8731", "http://[::1]:8731", True),
    ("evil.example:8731", None, False),                          # DNS rebinding
    ("evil.example:8731", "http://evil.example:8731", False),
    ("127.0.0.1:8731", "https://evil.example", False),           # cross-site POST
    ("127.0.0.1:8731", "null", False),                           # sandboxed iframe
])
def test_local_only_guard(client, host, origin, ok):
    c, _, _ = client
    headers = {"Host": host}
    if origin:
        headers["Origin"] = origin
    r = c.post("/api/forget", data="x", content_type="text/plain", headers=headers)
    assert (r.status_code == 200) is ok
    assert (c.get("/api/status", headers={"Host": host}).status_code == 200) is \
        (A._host_name(host) in A._LOOPBACK)


def test_open_only_search_results(client, corpus):
    import os
    import sys
    c, engine, state = client
    engine.build([corpus])
    target = str((corpus / "reservoir.txt").resolve())
    assert c.post("/api/open", json={"path": target}).status_code == 403
    # an existing executable on any OS: must never be launchable via the API
    assert c.post("/api/open", json={"path": sys.executable}).status_code == 403
    res = c.post("/api/search", json={"query": "reservoir water", "mode": "semantic"}).get_json()
    assert any(os.path.normcase(r["path"]) == os.path.normcase(target) for r in res["results"])
    r = c.post("/api/open", json={"path": target})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert c.post("/api/open", json={"path": str(corpus / "nope.txt")}).status_code == 404


@pytest.mark.parametrize("body", [{"query": "x", "top_k": "abc"}, {"query": "x", "top_k": 10**9},
                                  {"query": 5}, ["not", "a", "dict"]])
def test_search_input_is_validated(client, body):
    c, _, _ = client
    assert c.post("/api/search", json=body).status_code == 200


def test_add_folder_normalizes_and_dedupes(client, corpus):
    c, engine, state = client
    import time
    for spelling in (str(corpus), str(corpus) + "/", str(corpus / "." )):
        while state.building:
            time.sleep(0.01)
        assert c.post("/api/add_folder", json={"path": spelling}).status_code == 200
    while state.building:
        time.sleep(0.01)
    assert state.folders == [str(corpus)]
    assert c.post("/api/remove_folder", json={"path": str(corpus) + "/"}).get_json()["ok"]
    assert state.folders == []


def test_startup_catch_up_scan(fake_embedder, corpus, tmp_path):
    import json
    import time
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "folders.json").write_text(json.dumps([str(corpus)]))
    app, engine, state = A.create_app(idx)
    settle(state)
    deadline = time.time() + 10
    while time.time() < deadline and not (state.last_build and not state.building):
        time.sleep(0.02)
    assert state.last_build and state.last_build["added"] == 3
    engine.close()
