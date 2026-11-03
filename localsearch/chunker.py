"""Sentence-accumulating, CJK-aware chunker + BM25 tokenizer."""
import re
import sys
import unicodedata

_SENT = re.compile(r"[^.!?。！？；\n]*[.!?。！？；\n]+|\S[^.!?。！？；\n]*$")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿]+")
# one "word" of chunk weight: a single CJK char, or a run of anything else
_UNIT = re.compile(r"[぀-ヿ㐀-䶿一-鿿]|[^\s぀-ヿ㐀-䶿一-鿿]+")

# Bump when tokenize() output changes: stored per-chunk tokens are then dropped
# on load and re-derived from chunk text (no re-embedding needed).
TOKENIZER_VERSION = 2
_WORD = re.compile(r"[^\W_]+")          # Unicode letters/digits (any script)
_STOP = frozenset((
    "a an and are as at be by for from has he in is it its of on that the to was were "
    "will with this these those i you your we our they them their his her she him my me "
    "but or not no if then than so such can could should would may might must do does did "
    "have had having been being am are what which who whom when where why how all any both "
    "each few more most other some only own same too very s t just don now about above below "
    "up down out off over under again further once here there into through during before after"
).split())


def _weight(s):
    """Chunk-size weight of a sentence: whitespace words + CJK chars."""
    return len(s.split()) + len(_CJK.findall(s))


def _split_long(s, st, max_words):
    """Cut a 'sentence' heavier than max_words (spreadsheet cells, unpunctuated
    HTML/PDF text, ...) into max_words-unit windows with exact char spans."""
    units = list(_UNIT.finditer(s))
    if len(units) <= max_words:
        return [(s, st, st + len(s))]
    out = []
    for i in range(0, len(units), max_words):
        a, b = units[i].start(), units[min(i + max_words, len(units)) - 1].end()
        out.append((s[a:b], st + a, st + b))
    return out


def chunk_text(text, max_words=150, overlap_sentences=1):
    """Return [(chunk_text, start_char, end_char)]; no chunk exceeds max_words
    (the embedding model silently truncates anything past ~512 tokens)."""
    sents = []
    for m in _SENT.finditer(text):
        g = m.group()
        s = g.strip()
        if not s:
            continue
        st = m.start() + (len(g) - len(g.lstrip()))
        sents.extend(_split_long(s, st, max_words) if _weight(s) > max_words
                     else [(s, st, st + len(s))])
    if not sents:
        return []
    chunks, cur, cur_w = [], [], 0
    for s, st, en in sents:
        w = _weight(s)
        if cur and cur_w + w > max_words:
            chunks.append((" ".join(x[0] for x in cur), cur[0][1], cur[-1][2]))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_w = sum(_weight(x[0]) for x in cur)
            while cur and cur_w + w > max_words:     # overlap must not overflow
                cur_w -= _weight(cur.pop(0)[0])
        cur.append((s, st, en))
        cur_w += w
    if cur:
        chunks.append((" ".join(x[0] for x in cur), cur[0][1], cur[-1][2]))
    return chunks


def _fold(s):
    """Compatibility + diacritic folding for the non-CJK part: 'Résumé' ->
    'resume', 'ＡＢＣ' -> 'abc'; Hangul is recomposed (NFC) so it stays intact."""
    d = unicodedata.normalize("NFKD", s)
    return unicodedata.normalize(
        "NFC", "".join(c for c in d if not unicodedata.combining(c)))


def tokenize(text):
    """BM25 tokens: words of any script (case/diacritic-folded, stopword-filtered)
    + CJK character bigrams."""
    low = text.casefold()
    rest = _CJK_RUN.sub(" ", low)
    if not rest.isascii():
        rest = _fold(rest)
    toks = [sys.intern(w) for w in _WORD.findall(rest)
            if w not in _STOP and len(w) > 1]
    for run in _CJK_RUN.findall(low):
        if len(run) == 1:
            toks.append(sys.intern(run))
        else:
            toks.extend(sys.intern(run[i:i + 2]) for i in range(len(run) - 1))
    return toks
