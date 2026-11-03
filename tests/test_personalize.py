from localsearch import Config
from localsearch.personalize import Personalizer
from localsearch.search import Hit


def _hits(scores):
    return [Hit(path=f"/docs/{n}.txt", score=s, chunk_text="", char_start=0, char_end=0)
            for n, s in scores]


def _p(tmp_path, opened, times=6):
    p = Personalizer(tmp_path, Config())
    for _ in range(times):
        p.record_open(f"/docs/{opened}.txt")
    return p


def test_favorite_rises_but_never_displaces_the_best_match(tmp_path):
    scores = [("best", 0.90), ("b", 0.88), ("c", 0.86), ("d", 0.85), ("e", 0.84),
              ("f", 0.83), ("fav", 0.82), ("g", 0.80)]
    out = [h.path.split("/")[-1][:-4] for h in _p(tmp_path, "fav").rerank(_hits(scores))]
    assert out[0] == "best"                      # the best match keeps the top
    assert out.index("fav") < 6                  # ...while the favorite clearly rose


def test_practically_tied_favorite_may_take_the_top(tmp_path):
    scores = [("best", 0.900), ("fav", 0.899), ("c", 0.70), ("d", 0.60)]
    out = [h.path.split("/")[-1][:-4] for h in _p(tmp_path, "fav").rerank(_hits(scores))]
    assert out[0] == "fav"


def test_disabled_keeps_base_order(tmp_path):
    p = _p(tmp_path, "fav")
    p.set_enabled(False)
    hits = _hits([("best", 0.9), ("fav", 0.5)])
    assert [h.path for h in p.rerank(hits)] == [h.path for h in hits]
