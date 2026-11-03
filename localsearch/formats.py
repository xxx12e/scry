"""Extra document formats: OFD, legacy binary Office (+ WPS), EPUB, ODF, EML.

Pure stdlib except two small pure-Python readers for the OLE-based formats
(`olefile` for .doc/.ppt containers, `xlrd` for .xls). Every parser raises on
a file it cannot read; parser.parse_file() logs and skips it.
"""
import posixpath
import re
import struct
import zipfile
from xml.etree import ElementTree as ET


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _lines(text_lines):
    out = (" ".join(ln.split()) for ln in text_lines)
    return "\n".join(ln for ln in out if ln)


# ---- OFD (GB/T 33190: Chinese e-invoices, official documents) ------------

def _is_cjk(ch):
    return "　" <= ch <= "鿿" or "＀" <= ch <= "￯"


def _join_runs(parts):
    """Join text runs; no space between CJK neighbours, one between others."""
    out = ""
    for p in parts:
        if out and p and not (_is_cjk(out[-1]) or _is_cjk(p[0])) and not out[-1].isspace():
            out += " "
        out += p
    return out


def _ofd_page_lines(xml_bytes):
    root = ET.fromstring(xml_bytes)
    lines = []
    for el in root.iter():
        if _local(el.tag) == "TextObject":
            runs = [c.text for c in el.iter() if _local(c.tag) == "TextCode" and c.text]
            line = _join_runs(runs).strip()
            if line:
                lines.append(line)
    return lines


def parse_ofd(path):
    """OFD.xml -> DocRoot -> Document.xml page tree -> page Content.xml; the
    text lives in TextObject/TextCode. Template pages (an e-invoice's printed
    labels) come after the pages. Falls back to every Content.xml in the
    package for files that bend the standard (seen in the wild)."""
    from .parser import PAGE, _natural_key
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        lookup = {n.lower().lstrip("/"): n for n in names}

        def read(loc):
            n = lookup.get(loc.lower().lstrip("/"))
            return z.read(n) if n else None

        pages, tpls = [], []
        try:
            ofd = ET.fromstring(read("OFD.xml"))
            for dr in (e for e in ofd.iter() if _local(e.tag) == "DocRoot" and e.text):
                root = dr.text.strip().lstrip("/")
                base = posixpath.dirname(root)
                doc = ET.fromstring(read(root))

                def resolve(loc):
                    loc = loc.strip()
                    return loc.lstrip("/") if loc.startswith("/") else \
                        posixpath.normpath(posixpath.join(base, loc))
                for e in doc.iter():
                    tag, loc = _local(e.tag), e.get("BaseLoc")
                    if loc and tag == "Page":
                        pages.append(resolve(loc))
                    elif loc and tag == "TemplatePage":
                        tpls.append(resolve(loc))
        except (ET.ParseError, TypeError, KeyError, ValueError):
            pages, tpls = [], []
        if not pages:
            pages = sorted((n for n in names if n.lower().endswith("content.xml")
                            and "/tpls/" not in n.lower()), key=_natural_key)
        out = []
        for loc in pages:
            data = read(loc)
            try:
                out.append("\n".join(_ofd_page_lines(data)) if data else "")
            except ET.ParseError:
                out.append("")
        text = PAGE.join(out)
        tpl_lines = []
        for loc in dict.fromkeys(tpls):
            data = read(loc)
            if data:
                try:
                    tpl_lines += _ofd_page_lines(data)
                except ET.ParseError:
                    pass
        if tpl_lines:
            text += "\n" + "\n".join(dict.fromkeys(tpl_lines))
        return text


# ---- legacy Word (.doc; WPS .wps) -----------------------------------------
# [MS-DOC]: the text is a sequence of "pieces" described by the piece table in
# the CLX (FibRgFcLcb97.fcClx at FIB offset 0x1A2, stored in 0Table/1Table).
# Each piece is either cp1252 bytes ("compressed") or UTF-16LE.

_WORD_MAP = {"\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x07": "\t", "\x1e": "-",
             "\xa0": " "}


def _clean_word(raw):
    out, depth_instr = [], []
    for ch in raw:
        if ch == "\x13":                 # field begin: instruction follows
            depth_instr.append(True)
            continue
        if ch == "\x14":                 # field separator: result follows
            if depth_instr:
                depth_instr[-1] = False
            continue
        if ch == "\x15":                 # field end
            if depth_instr:
                depth_instr.pop()
            continue
        if depth_instr and depth_instr[-1]:
            continue                     # inside a field instruction (HYPERLINK "...")
        ch = _WORD_MAP.get(ch, ch)
        if ch < " " and ch not in "\n\t":
            continue                     # anchors, footnote refs, other controls
        out.append(ch)
    return _lines("".join(out).split("\n"))


def parse_doc(path):
    import olefile
    with olefile.OleFileIO(path) as ole:
        if not ole.exists("WordDocument"):
            raise ValueError("no WordDocument stream (not a Word 97+ file)")
        wd = ole.openstream("WordDocument").read()
        if len(wd) < 0x1AA or struct.unpack_from("<H", wd, 0)[0] != 0xA5EC:
            raise ValueError("not a Word 97-2003 document")
        flags = struct.unpack_from("<H", wd, 0x0A)[0]
        if flags & 0x0100:
            raise ValueError("encrypted document")
        table = "1Table" if flags & 0x0200 else "0Table"
        fc_clx, lcb_clx = struct.unpack_from("<II", wd, 0x1A2)
        tbl = ole.openstream(table).read()
    clx = tbl[fc_clx:fc_clx + lcb_clx]
    i = 0
    while i < len(clx) and clx[i] == 0x01:                    # Prc entries
        i += 3 + struct.unpack_from("<H", clx, i + 1)[0]
    if i + 5 > len(clx) or clx[i] != 0x02:
        raise ValueError("piece table not found")
    lcb = struct.unpack_from("<I", clx, i + 1)[0]
    plc = clx[i + 5:i + 5 + lcb]
    n = (lcb - 4) // 12
    cps = struct.unpack_from("<%dI" % (n + 1), plc, 0)
    parts = []
    for k in range(n):
        fcc = struct.unpack_from("<I", plc, 4 * (n + 1) + 8 * k + 2)[0]
        count = cps[k + 1] - cps[k]
        if count <= 0:
            continue
        fc = fcc & 0x3FFFFFFF
        if fcc & 0x40000000:                                  # 8-bit "compressed"
            parts.append(wd[fc // 2:fc // 2 + count].decode("cp1252", "replace"))
        else:
            parts.append(wd[fc:fc + 2 * count].decode("utf-16-le", "replace"))
    return _clean_word("".join(parts))


# ---- legacy Excel (.xls; WPS .et) ----------------------------------------

def _cell(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def parse_xls(path):
    import xlrd
    wb = xlrd.open_workbook(path, on_demand=True)
    try:
        lines = []
        for sh in wb.sheets():
            lines.append(sh.name)
            for r in range(sh.nrows):
                cells = [_cell(v) for v in sh.row_values(r) if v not in ("", None)]
                if cells:
                    lines.append(" ".join(c for c in cells if c))
        return _lines(lines)
    finally:
        wb.release_resources()


# ---- legacy PowerPoint (.ppt; WPS .dps) ----------------------------------
# [MS-PPT] records: 8-byte header (ver/instance, type, length); containers have
# ver 0xF. Text lives in TextCharsAtom (UTF-16) / TextBytesAtom (8-bit).

_PPT_TEXT_CHARS, _PPT_TEXT_BYTES = 0x0FA0, 0x0FA8
_PPT_SKIP = {0x03F8, 0x0FC9, 0x0FD9}      # MainMaster, Handout, NotesMaster ("Click to edit")


def _ppt_walk(data, start, end, out, depth=0):
    pos = start
    while pos + 8 <= end and depth < 64:
        verinst, rtype, rlen = struct.unpack_from("<HHI", data, pos)
        body = pos + 8
        if body + rlen > end:
            break
        if verinst & 0xF == 0xF:
            if rtype not in _PPT_SKIP:
                _ppt_walk(data, body, body + rlen, out, depth + 1)
        elif rtype == _PPT_TEXT_CHARS:
            out.append(data[body:body + rlen].decode("utf-16-le", "replace"))
        elif rtype == _PPT_TEXT_BYTES:
            out.append(data[body:body + rlen].decode("cp1252", "replace"))
        pos = body + rlen


def parse_ppt(path):
    import olefile
    with olefile.OleFileIO(path) as ole:
        if not ole.exists("PowerPoint Document"):
            raise ValueError("no PowerPoint Document stream")
        data = ole.openstream("PowerPoint Document").read()
    out = []
    _ppt_walk(data, 0, len(data), out)
    lines = []
    for t in out:
        lines += t.replace("\r", "\n").replace("\x0b", "\n").split("\n")
    lines = [ln for ln in lines if ln.strip() not in ("*", "")]   # "*" = slide-number field
    return _lines(dict.fromkeys(lines))      # placeholders repeat across views


# ---- EPUB ----------------------------------------------------------------

def parse_epub(path):
    from .parser import _HTMLText, decode_bytes
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        order = []
        try:
            cont = ET.fromstring(z.read("META-INF/container.xml"))
            opf_path = next(e.get("full-path") for e in cont.iter() if _local(e.tag) == "rootfile")
            opf = ET.fromstring(z.read(opf_path))
            base = posixpath.dirname(opf_path)
            items = {e.get("id"): e.get("href") for e in opf.iter() if _local(e.tag) == "item"}
            for e in opf.iter():
                if _local(e.tag) == "itemref" and items.get(e.get("idref")):
                    order.append(posixpath.normpath(posixpath.join(base, items[e.get("idref")])))
        except (KeyError, StopIteration, ET.ParseError):
            pass
        if not order:
            order = sorted(n for n in names if re.search(r"\.x?html?$", n, re.I))
        chapters = []
        for n in order:
            if n not in names:
                continue
            p = _HTMLText()
            p.feed(decode_bytes(z.read(n)))
            p.close()
            chapters.append(_lines("".join(p.parts).split("\n")))
        return "\n".join(c for c in chapters if c)


# ---- OpenDocument (.odt / .ods / .odp) -----------------------------------

def _odf_collect(el, buf, out):
    if el.text:
        buf.append(el.text)
    for c in el:
        t = _local(c.tag)
        if t in ("p", "h"):
            _odf_para(c, out)
        elif t == "s":
            buf.append(" " * int(c.get("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}c", "1")))
        elif t == "tab":
            buf.append("\t")
        elif t == "line-break":
            buf.append("\n")
        elif t not in ("note-citation", "annotation", "page-number", "page-count"):
            _odf_collect(c, buf, out)
        if c.tail:
            buf.append(c.tail)


def _odf_para(el, out):
    buf = []
    _odf_collect(el, buf, out)
    line = "".join(buf).strip()
    if line:
        out.append(line)


def _odf_walk(el, out, pages):
    for c in el:
        t = _local(c.tag)
        if t in ("p", "h"):
            _odf_para(c, out)
        elif t == "page":                       # draw:page (one slide)
            sub = []
            _odf_walk(c, sub, pages)
            pages.append("\n".join(sub))
        else:
            _odf_walk(c, out, pages)


def parse_odf(path):
    from .parser import PAGE
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    out, pages = [], []
    _odf_walk(root, out, pages)
    if pages:                                   # presentation: one page per slide
        return PAGE.join(pages) + ("\n" + "\n".join(out) if out else "")
    return "\n".join(out)


# ---- e-mail (.eml) -------------------------------------------------------

def parse_eml(path):
    from email import policy
    from email.parser import BytesParser
    from .parser import _HTMLText
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    lines = [str(msg.get(h, "")) for h in ("Subject", "From", "To", "Cc", "Date")]
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is not None:
        content = body.get_content()
        if body.get_content_type() == "text/html":
            p = _HTMLText()
            p.feed(content)
            p.close()
            content = "".join(p.parts)
        lines += content.split("\n")
    lines += [a.get_filename() or "" for a in msg.iter_attachments()]
    return _lines(lines)
