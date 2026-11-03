"""Robust file-text extraction. Office/HTML/RTF formats use only the stdlib
(zipfile + xml.etree + html.parser) -- no new dependencies, so the bundle never
grows for the extra coverage."""
import codecs
import io
import locale
import logging
import os
import posixpath
import re
import struct
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from . import ocr as _ocr

log = logging.getLogger("localsearch.parser")

# Page/slide boundary marker. Parsers that know pages join them with PAGE; the
# engine maps each chunk to its page by counting markers before it (str.strip
# and the tokenizer both treat \f as whitespace, so it never reaches the index).
PAGE = "\n\f\n"

_OCR_ON = False


def set_ocr(enabled):
    """Turn image OCR on/off. Returns the effective state (on only if the Windows
    OCR engine is actually available)."""
    global _OCR_ON
    _OCR_ON = bool(enabled) and _ocr.available()
    return _OCR_ON


def _ocr_active():
    return _OCR_ON

try:                         # PyMuPDF >= 1.24.3 is `pymupdf`; `fitz` is deprecated
    import pymupdf as fitz
except Exception:
    try:
        import fitz
    except Exception:
        fitz = None


# ---- text decoding -------------------------------------------------------
# Plain-text files on Windows are often NOT UTF-8: Notepad saved "ANSI" (GBK on
# Chinese Windows, cp1252 on Western) for decades, and "Unicode" means UTF-16.
# Decoding everything as UTF-8 turned those files into U+FFFD soup.

_GBK_FAMILY = {"cp936", "gbk", "gb2312", "gb_2312", "gb18030", "936", "ms936",
               "x-gbk", "csgb2312", "euc-cn"}


def _canon_codec(name):
    """Python codec name for `name` (GBK family -> its superset gb18030), or None."""
    if not name:
        return None
    n = str(name).strip().lower()
    if n in _GBK_FAMILY:
        return "gb18030"
    try:
        return codecs.lookup(n).name
    except LookupError:
        return None


def _system_ansi_codec():
    """The Windows "ANSI" code page (what Notepad called ANSI); utf-8 elsewhere."""
    try:
        enc = (locale.getencoding() if hasattr(locale, "getencoding")
               else locale.getpreferredencoding(False))
    except Exception:
        enc = None
    return _canon_codec(enc)


_ANSI = _system_ansi_codec()
_BOMS = ((codecs.BOM_UTF8, "utf-8-sig"),
         (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
         (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"))


_GB2312_PAIR = re.compile(rb"[\xa1-\xfe]{2}")
_HIGH = re.compile(rb"[\x80-\xff]")
# Only where the ANSI code page is Western/UTF-8 do we also sniff for GBK (Chinese
# users on English-locale Windows). On Cyrillic/Greek/... code pages a short
# all-letter file could pass the test, so there we trust the code page instead.
_SNIFF_GBK = _ANSI in (None, "utf-8", "cp1252", "latin-1", "iso8859-1", "iso8859-15")


def _as_gbk(raw):
    """Decode as GB18030 if the bytes look like real Chinese text: strict decode
    succeeds AND >= 90% of high bytes form GB2312-block pairs (lead+trail
    0xA1-0xFE). Western text fails this: its accented letters are lone high bytes."""
    high = len(_HIGH.findall(raw))
    if not high or 2 * len(_GB2312_PAIR.findall(raw)) < 0.9 * high:
        return None
    try:
        return raw.decode("gb18030")
    except UnicodeDecodeError:
        return None


def decode_bytes(raw, hint=None):
    """BOM -> declared `hint` -> strict UTF-8 -> (GBK sniff) -> system ANSI code
    page -> lossy UTF-8."""
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return raw.decode(enc, errors="replace")
    tried = set()
    for enc in (_canon_codec(hint), "utf-8", "gbk?", _ANSI):
        if not enc or enc in tried:
            continue
        tried.add(enc)
        if enc == "gbk?":
            text = _as_gbk(raw) if _SNIFF_GBK and _ANSI != "gb18030" else None
            if text is not None:
                return text
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_text(path, hint=None):
    return decode_bytes(Path(path).read_bytes(), hint)


def _parse_md(path):
    t = _read_text(path)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"[#>*_`]+", " ", t)
    return t


_OCR_PDF_MAX_PAGES = 100        # bound the cost of one scanned book
_OCR_DPI = 200


def _ocr_pdf_page(page):
    """Render one image-only PDF page and read it with the Windows OCR engine."""
    pix = page.get_pixmap(dpi=_OCR_DPI)
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pix.save(tmp)
        return _ocr.ocr_image(tmp) or ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _parse_pdf(path):
    """Text layer per page; with OCR on, pages that are just a scan (no text,
    but images) are rendered and recognized -- scanned contracts, receipts..."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    doc = fitz.open(path)
    try:
        pages, ocred = [], 0
        for page in doc:
            text = page.get_text()
            if (_OCR_ON and len(text.strip()) < 25 and ocred < _OCR_PDF_MAX_PAGES
                    and page.get_images()):
                ocred += 1
                text = _ocr_pdf_page(page) or text
            pages.append(text)
        return PAGE.join(pages)
    finally:
        doc.close()


# ---- OOXML (docx / xlsx / pptx) -----------------------------------------
# All three keep text in <*:t> nodes grouped into paragraphs (<w:p>, <a:p>) or
# string items (<si>, <is>). One line per paragraph keeps sentence structure for
# the chunker; joining every node with spaces made a whole spreadsheet or deck
# one giant "sentence" that the model silently truncated after ~500 tokens.

_OOXML_BLOCKS = frozenset({"p", "si", "is"})


def _natural_key(name):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def _ooxml_part_lines(xml_bytes, lines):
    stack = []
    for ev, el in ET.iterparse(io.BytesIO(xml_bytes), events=("start", "end")):
        tag = el.tag.rsplit("}", 1)[-1]
        if ev == "start":
            if tag in _OOXML_BLOCKS:
                stack.append([])
            continue
        if tag == "t":                         # w:t / a:t / t (not delText/instrText)
            if el.text:
                if stack:
                    stack[-1].append(el.text)
                else:
                    lines.append(el.text)
        elif tag in ("br", "cr") and stack:
            stack[-1].append("\n")
        elif tag == "tab" and stack:
            stack[-1].append(" ")
        elif tag in _OOXML_BLOCKS and stack:
            line = "".join(stack.pop()).strip()
            if line:
                lines.append(line)
            el.clear()


def _ooxml_text(path, patterns):
    """Paragraph-per-line text of the zip members matching `patterns`, in
    pattern order (natural sort within a pattern: slide2 before slide10)."""
    lines = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for pat in patterns:
            rx = re.compile(pat)
            for name in sorted((n for n in names if rx.match(n)), key=_natural_key):
                try:
                    _ooxml_part_lines(z.read(name), lines)
                except ET.ParseError:
                    continue
    return "\n".join(lines)


def _parse_docx(path):
    # body (incl. tables + text boxes, which python-docx's .paragraphs skipped),
    # then headers/footers and foot/endnotes
    return _ooxml_text(path, [r"word/document\.xml$", r"word/(header|footer)\d*\.xml$",
                              r"word/(footnotes|endnotes)\.xml$"])


def _parse_xlsx(path):
    return _ooxml_text(path, [r"xl/sharedStrings\.xml$", r"xl/worksheets/sheet.*\.xml$"])


def _parse_pptx(path):
    """One page per slide (in slide order); speaker notes after the slides."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=_natural_key)
        pages = []
        for n in slides:
            lines = []
            try:
                _ooxml_part_lines(z.read(n), lines)
            except ET.ParseError:
                pass
            pages.append("\n".join(lines))
    notes = _ooxml_text(path, [r"ppt/notesSlides/.*\.xml$"])
    return PAGE.join(pages) + ("\n" + notes if notes else "")


# ---- HTML ----------------------------------------------------------------

_HTML_SKIP = frozenset({"script", "style", "noscript", "template"})
_HTML_BLOCK = frozenset(("p div br li tr td th h1 h2 h3 h4 h5 h6 table ul ol dl dt dd "
                         "section article header footer nav aside main blockquote pre "
                         "title figcaption hr").split())
_META_CHARSET = re.compile(rb"""<meta[^>]{0,200}?charset\s*=\s*["']?\s*([a-zA-Z0-9_\-:.]+)""",
                           re.I)


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in _HTML_SKIP:
            self._skip += 1
        elif tag in _HTML_BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _HTML_SKIP and self._skip:
            self._skip -= 1
        elif tag in _HTML_BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _parse_html(path):
    raw = Path(path).read_bytes()
    m = _META_CHARSET.search(raw[:4096])
    p = _HTMLText()
    p.feed(decode_bytes(raw, m.group(1).decode("ascii", "ignore") if m else None))
    p.close()
    lines = (" ".join(ln.split()) for ln in "".join(p.parts).split("\n"))
    return "\n".join(ln for ln in lines if ln)


# ---- RTF -----------------------------------------------------------------
# Non-ASCII text in RTF is escaped: \'xx bytes in the document (\ansicpg) or the
# current font's (\fcharset) code page, and \uN code points followed by \ucN
# fallback characters. Stripping those escapes (the old approach) erased every
# Chinese character, while font tables and metadata leaked in as "text".

_RTF_TOKEN = re.compile(
    r"\\([a-zA-Z]+)(-?\d+)? ?"       # 1,2: control word (+ its delimiter space)
    r"|\\'([0-9a-fA-F]{2})"          # 3:   hex-escaped byte
    r"|\\([^a-zA-Z'])"               # 4:   control symbol
    r"|([{}])"                       # 5:   group open/close
    r"|[\r\n]+"                      #      raw newlines carry no meaning
    r"|([^\\{}\r\n]+)", re.S)        # 6:   plain text
_RTF_SKIP = frozenset((
    "fonttbl colortbl stylesheet info pict object objdata themedata "
    "colorschememapping latentstyles datastore xmlnstbl listtable listoverridetable "
    "rsidtbl generator fldinst pntext pntxta pntxtb filetbl revtbl bkmkstart bkmkend "
    "xe tc sp shppict nonshppict userprops docvar").split())
_RTF_BREAK = frozenset({"par", "line", "sect", "page", "row"})
_RTF_SPACE = frozenset({"tab", "cell", "emspace", "enspace", "qmspace"})
_RTF_CHAR = {"emdash": "\u2014", "endash": "\u2013", "bullet": "\u2022",
             "lquote": "\u2018", "rquote": "\u2019",
             "ldblquote": "\u201c", "rdblquote": "\u201d"}
_RTF_SYM = {"~": " ", "_": "-", "-": "", "\\": "\\", "{": "{", "}": "}"}
# \fcharset -> Windows code page (charset 0 = "ANSI": use the document's \ansicpg)
_RTF_CHARSET_CP = {128: 932, 129: 949, 130: 1361, 134: 936, 136: 950,
                   161: 1253, 162: 1254, 163: 1258, 177: 1255, 178: 1256,
                   186: 1257, 204: 1251, 222: 874, 238: 1250}


def rtf_to_text(rtf):
    out = []
    buf = bytearray()                 # consecutive \'xx bytes (DBCS pairs)
    stack = []
    skip, uc, font, dest = False, 1, None, None
    deff = None                       # \deffN: font used before any \fN
    doc_cp = "cp1252"
    font_cp = {}                      # font number -> codec (from \fonttbl)
    tbl_font = None
    pending = 0                       # \ucN fallback chars still to drop
    group_start = False

    def codec():
        return font_cp.get(font if font is not None else deff) or doc_cp

    def flush():
        if buf:
            if not skip:
                out.append(bytes(buf).decode(codec(), errors="replace"))
            buf.clear()

    for m in _RTF_TOKEN.finditer(rtf):
        word, arg, hexb, sym, brace, text = m.groups()
        if hexb is not None:
            if pending:
                pending -= 1
            elif not skip:
                buf.append(int(hexb, 16))
            group_start = False
            continue
        flush()
        first, group_start = group_start, False
        if brace == "{":
            stack.append((skip, uc, font, dest))
            group_start = True
        elif brace == "}":
            pending = 0
            skip, uc, font, dest = stack.pop() if stack else (False, 1, None, None)
        elif word is not None:
            if first and word in _RTF_SKIP:
                skip, dest = True, word
            elif dest == "fonttbl":
                if word == "f" and arg:
                    tbl_font = int(arg)
                elif word == "fcharset" and arg and tbl_font is not None:
                    cp = _RTF_CHARSET_CP.get(int(arg))
                    c = _canon_codec("cp%d" % cp) if cp else None
                    if c:
                        font_cp[tbl_font] = c
            elif word == "deff" and arg:
                deff = int(arg)
            elif word == "ansicpg" and arg:
                doc_cp = _canon_codec("cp" + arg) or doc_cp
            elif word == "f" and arg:
                font = int(arg)
            elif word == "uc" and arg:
                uc = max(0, int(arg))
            elif word == "u" and arg:
                if not skip:
                    n = int(arg)
                    out.append(chr(n + 65536 if n < 0 else n))
                pending = uc
            elif skip:
                pass
            elif word in _RTF_BREAK:
                out.append("\n")
            elif word in _RTF_SPACE:
                out.append(" ")
            elif word in _RTF_CHAR:
                out.append(_RTF_CHAR[word])
        elif sym is not None:
            if sym == "*":
                skip = True
            elif sym in "\r\n":
                if not skip:
                    out.append("\n")
            elif not skip and sym in _RTF_SYM:
                out.append(_RTF_SYM[sym])
        elif text is not None:
            if pending:
                k = min(pending, len(text))
                text, pending = text[k:], pending - k
            if text and not skip:
                if not text.isascii():           # raw 8-bit bytes (read as latin-1)
                    text = text.encode("latin-1", "replace").decode(codec(), "replace")
                out.append(text)
    flush()
    lines = (" ".join(ln.split()) for ln in "".join(out).split("\n"))
    return "\n".join(ln for ln in lines if ln)


def _parse_rtf(path):
    return rtf_to_text(Path(path).read_bytes().decode("latin-1"))


from .formats import (parse_ofd as _parse_ofd, parse_doc as _parse_doc,  # noqa: E402
                      parse_xls as _parse_xls, parse_ppt as _parse_ppt,
                      parse_epub as _parse_epub, parse_odf as _parse_odf,
                      parse_eml as _parse_eml)

_PARSERS = {".txt": _read_text, ".md": _parse_md, ".pdf": _parse_pdf, ".docx": _parse_docx,
            ".xlsx": _parse_xlsx, ".pptx": _parse_pptx, ".csv": _read_text,
            ".html": _parse_html, ".htm": _parse_html, ".rtf": _parse_rtf,
            # China: e-invoices / official documents (GB/T 33190)
            ".ofd": _parse_ofd,
            # legacy binary Office -- and WPS Office's own formats (best effort:
            # they are parsed as their Word/Excel/PowerPoint counterparts)
            ".doc": _parse_doc, ".xls": _parse_xls, ".ppt": _parse_ppt,
            ".wps": _parse_doc, ".et": _parse_xls, ".dps": _parse_ppt,
            ".epub": _parse_epub, ".odt": _parse_odf, ".ods": _parse_odf, ".odp": _parse_odf,
            ".eml": _parse_eml}

SUPPORTED = tuple(_PARSERS.keys())


def current_exts():
    """Extensions the engine should index right now (image types only while OCR
    is on) -- the single source of truth for both the walk and the watcher."""
    return SUPPORTED + _ocr.IMAGE_EXT if _OCR_ON else SUPPORTED


def is_supported(path):
    ext = Path(path).suffix.lower()
    return ext in _PARSERS or (_OCR_ON and ext in _ocr.IMAGE_EXT)


def parse_file(path):
    """Return extracted text, or None if unsupported / unreadable (logged)."""
    ext = Path(path).suffix.lower()
    fn = _PARSERS.get(ext)
    if fn is None and _OCR_ON and ext in _ocr.IMAGE_EXT:
        fn = _ocr.ocr_image
    if fn is None:
        log.debug("skip unsupported type: %s", path)
        return None
    try:
        text = fn(path)
        if not text or not text.strip():
            log.debug("empty after parse: %s", path)
            return None
        return text
    except Exception as e:
        log.warning("parse failed, skipping %s: %s: %s", path, type(e).__name__, e)
        return None
