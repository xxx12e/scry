import io
import zipfile

import pytest

from localsearch import parser

ZH = "今年第三季度的发票报销流程已经改为线上审批"


def test_utf16_with_bom(tmp_path):
    f = tmp_path / "u16.txt"
    f.write_text(ZH, encoding="utf-16")
    assert parser.parse_file(f) == ZH


def test_gbk_on_chinese_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "_ANSI", "gb18030")        # zh-CN "ANSI" = cp936
    f = tmp_path / "gbk.txt"
    f.write_bytes(ZH.encode("gbk"))
    assert parser.parse_file(f) == ZH


def test_gbk_sniffed_on_western_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(parser, "_ANSI", "cp1252")
    monkeypatch.setattr(parser, "_SNIFF_GBK", True)
    f = tmp_path / "gbk.txt"
    f.write_bytes((ZH + "，金额 4200 元。Invoice OK").encode("gbk"))
    assert parser.parse_file(f).startswith(ZH)


@pytest.mark.parametrize("text", ["café résumé naïve à la carte", "Größe Straße über",
                                  "señor año pequeño", "déjà vu"])
def test_western_ansi_not_mistaken_for_gbk(text, monkeypatch):
    monkeypatch.setattr(parser, "_ANSI", "cp1252")
    monkeypatch.setattr(parser, "_SNIFF_GBK", True)
    assert parser.decode_bytes(text.encode("cp1252")) == text


def test_utf8_still_preferred(tmp_path):
    f = tmp_path / "u8.txt"
    f.write_text("naïve 发票", encoding="utf-8")
    assert parser.parse_file(f) == "naïve 发票"


def test_html_meta_charset_and_blocks(tmp_path):
    html = ('<html><head><meta charset="gb2312"><title>报销</title>'
            '<style>p{color:red}</style><script>var x=1;</script></head>'
            '<body><p>第一段</p><div>second block</div><ul><li>a</li><li>b</li></ul>'
            'R&amp;D</body></html>')
    f = tmp_path / "p.html"
    f.write_bytes(html.encode("gbk"))
    lines = parser.parse_file(f).split("\n")
    assert lines == ["报销", "第一段", "second block", "a", "b", "R&D"]


def _rtf(tmp_path, body):
    f = tmp_path / "d.rtf"
    f.write_text(body, encoding="ascii")
    return parser.parse_file(f)


def test_rtf_ansicpg_hex_bytes(tmp_path):
    hexes = "".join("\\'%02x" % b for b in "发票报销".encode("gbk"))
    rtf = ("{\\rtf1\\ansi\\ansicpg936\\deff0{\\fonttbl{\\f0\\fnil\\fcharset134 SimSun;}}"
           "{\\*\\generator Riched20 10.0.19041}\\viewkind4\\uc1\\pard\\f0\\fs22 "
           + hexes + "\\par second line\\par}")
    assert _rtf(tmp_path, rtf) == "发票报销\nsecond line"


def test_rtf_font_charset_on_western_codepage(tmp_path):
    hexes = "".join("\\'%02x" % b for b in "合同".encode("gbk"))
    rtf = ("{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0 Arial;}"
           "{\\f1\\fnil\\fcharset134 SimSun;}}\\pard\\f0 caf\\'e9 \\f1 " + hexes + "\\par}")
    assert _rtf(tmp_path, rtf) == "café 合同"


def test_rtf_unicode_escapes_skip_fallback(tmp_path):
    rtf = "{\\rtf1\\ansi\\uc1 \\u21512?\\u21516? and \\uc2\\u-28207\\'3f\\'3f end\\par}"
    # \u-28207 -> U+91D1 (金); \uc2 drops the two fallback bytes
    assert _rtf(tmp_path, rtf) == "合同 and 金 end"


def test_rtf_skips_metadata_and_keeps_literals(tmp_path):
    rtf = ("{\\rtf1{\\info{\\title Secret}{\\author Bob}}{\\colortbl;\\red0\\green0\\blue0;}"
           "{\\*\\bkmkstart x}Price \\{50\\}\\tab ok\\line next\\par}")
    assert _rtf(tmp_path, rtf) == "Price {50} ok\nnext"


def _zip(tmp_path, name, members):
    f = tmp_path / name
    with zipfile.ZipFile(f, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return f


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def test_docx_tables_headers_and_textboxes(tmp_path):
    doc = (f'<w:document {W}><w:body>'
           '<w:p><w:r><w:t>Body </w:t></w:r><w:r><w:t>text</w:t></w:r></w:p>'
           '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Invoice INV-7731</w:t></w:r></w:p></w:tc>'
           '<w:tc><w:p><w:r><w:t>报销金额 4200 元</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
           '<w:p><w:r><w:t>Outer</w:t></w:r><w:r><w:txbxContent><w:p><w:r><w:t>In box</w:t>'
           '</w:r></w:p></w:txbxContent></w:r></w:p>'
           '<w:p><w:r><w:delText>deleted</w:delText><w:instrText>HYPERLINK x</w:instrText></w:r></w:p>'
           '</w:body></w:document>')
    hdr = f'<w:hdr {W}><w:p><w:r><w:t>Confidential header</w:t></w:r></w:p></w:hdr>'
    f = _zip(tmp_path, "a.docx", {"word/document.xml": doc, "word/header1.xml": hdr})
    assert parser.parse_file(f).split("\n") == [
        "Body text", "Invoice INV-7731", "报销金额 4200 元", "In box", "Outer",
        "Confidential header"]


def test_pptx_paragraphs_in_slide_order(tmp_path):
    A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    sld = lambda t: f'<p:sld {A} xmlns:p="p"><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>'
    f = _zip(tmp_path, "d.pptx", {"ppt/slides/slide10.xml": sld("ten"),
                                  "ppt/slides/slide2.xml": sld("two")})
    assert parser.parse_file(f) == "two\nten"


def test_xlsx_one_line_per_string(tmp_path):
    ss = ('<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
          '<si><t>Region</t></si><si><r><t>North</t></r><r><t>east</t></r></si></sst>')
    f = _zip(tmp_path, "b.xlsx", {"xl/sharedStrings.xml": ss})
    assert parser.parse_file(f) == "Region\nNortheast"


def test_unreadable_file_is_skipped(tmp_path):
    f = tmp_path / "broken.docx"
    f.write_bytes(b"not a zip")
    assert parser.parse_file(f) is None
