import random
import re
import string

from localsearch.chunker import chunk_text, tokenize, _weight, _STOP

_OLD_WORD = re.compile(r"[a-z0-9]+")


def test_unpunctuated_text_is_capped():
    cells = " ".join(f"Product{i} Region{i % 7} Revenue{i}" for i in range(2000))
    chunks = chunk_text(cells, 150, 1)
    assert len(chunks) >= 40
    assert max(_weight(c[0]) for c in chunks) <= 150


def test_cjk_without_punctuation_is_capped_and_spans_exact():
    text = "季度报销流程" * 200
    chunks = chunk_text(text, 150, 1)
    assert max(len(c[0]) for c in chunks) <= 150
    assert all(text[a:b] == t for t, a, b in chunks)


def test_overlap_never_overflows():
    rnd = random.Random(1)
    text = " ".join(" ".join(rnd.choice(["alpha", "beta", "gamma"])
                             for _ in range(rnd.randint(5, 140))) + "." for _ in range(60))
    assert max(_weight(c[0]) for c in chunk_text(text, 150, 1)) <= 150


def test_normal_prose_unchanged():
    text = "First sentence here. Second one! Third? Fourth."
    assert chunk_text(text, 150, 1) == [(text, 0, len(text))]


def test_ascii_tokens_identical_to_previous_tokenizer():
    rnd = random.Random(7)
    alphabet = string.ascii_letters + string.digits + " .,;:'-_/()"
    for _ in range(300):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 80)))
        old = [w for w in _OLD_WORD.findall(s.lower()) if len(w) > 1 and w not in _STOP]
        assert tokenize(s) == old


def test_unicode_words_are_tokens():
    assert tokenize("café résumé naïve") == ["cafe", "resume", "naive"]
    assert tokenize("Привет мир документ") == ["привет", "мир", "документ"]
    assert tokenize("학교 숙제 제출") == ["학교", "숙제", "제출"]
    assert tokenize("Straße") == tokenize("STRASSE")
    assert tokenize("ＡＢＣ１２３") == ["abc123"]


def test_cjk_bigrams_kept():
    assert tokenize("发票报销") == ["发票", "票报", "报销"]
