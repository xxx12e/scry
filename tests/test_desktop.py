import pytest

from localsearch import desktop


@pytest.mark.parametrize("spec,mods,vk", [
    ("ctrl+alt+space", 0x0002 | 0x0001 | 0x4000, 0x20),
    ("Ctrl + Shift + K", 0x0002 | 0x0004 | 0x4000, ord("K")),
    ("win+f2", 0x0008 | 0x4000, 0x71),
    ("alt+`", 0x0001 | 0x4000, 0xC0),
])
def test_parse_hotkey(spec, mods, vk):
    assert desktop.parse_hotkey(spec) == (mods, vk)


@pytest.mark.parametrize("bad", ["space", "ctrl+alt", "ctrl+banana"])
def test_parse_hotkey_rejects(bad):
    with pytest.raises(ValueError):
        desktop.parse_hotkey(bad)


def test_shell_unavailable_off_windows_falls_back():
    import sys
    if sys.platform != "win32":
        assert desktop.run("http://127.0.0.1:1", lambda: None, lambda: None) is False


def test_api_show_raises_the_window(fake_embedder, tmp_path):
    from conftest import settle
    from localsearch import app as A
    app, engine, state = A.create_app(tmp_path / "idx")
    settle(state)
    c = app.test_client()
    assert c.post("/api/show", json={}).get_json()["ok"] is False      # browser mode
    shown = []
    state.show_window = lambda: shown.append(1)
    assert c.post("/api/show", json={}).get_json()["ok"] is True and shown
    engine.close()
