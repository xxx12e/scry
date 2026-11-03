import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest

from localsearch import parser

FIX = Path(__file__).parent / "fixtures" / "office_samples.zip"


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    d = tmp_path_factory.mktemp("office")
    with zipfile.ZipFile(FIX) as z:
        z.extractall(d)
    return d


@pytest.mark.parametrize("name,expect", [
    ("sample.doc", ["季度报销流程说明", "reimbursement deadline", "INV-7731", "4200 元", "内部资料"]),
    ("sample.xls", ["预算", "研发部 120000 服务器采购", "Notes", "Quarterly budget review"]),
    ("sample.ppt", ["年度战略规划", "Roadmap 2027", "演讲备注：强调隐私"]),
    ("sample.odt", ["季度报销流程说明", "INV-7731"]),
    ("sample.ods", ["研发部", "120000", "Quarterly budget review"]),
    ("sample.odp", ["年度战略规划", "Launch the offline search product"]),
    ("sample.epub", ["季度报销流程说明", "4200 元"]),
])
def test_real_files(samples, name, expect):
    text = parser.parse_file(samples / name)
    assert text is not None
    for e in expect:
        assert e in text, (name, e, text[:300])


def test_ppt_has_no_placeholder_noise(samples):
    lines = parser.parse_file(samples / "sample.ppt").split("\n")
    assert "*" not in lines and len(lines) == len(set(lines))


def test_slides_are_pages(samples):
    assert parser.parse_file(samples / "sample.odp").count("\f") == 1      # 2 slides


def test_wps_formats_route_to_binary_office_parsers(samples, tmp_path):
    # WPS Office's .wps/.et/.dps are read as Word/Excel/PowerPoint binaries
    # (verified here only with MS-format files under the WPS extension)
    for src, dst in (("sample.doc", "a.wps"), ("sample.xls", "b.et"), ("sample.ppt", "c.dps")):
        (tmp_path / dst).write_bytes((samples / src).read_bytes())
        assert parser.parse_file(tmp_path / dst)


def _ofd(tmp_path, absolute=False, broken_root=False):
    ns = 'xmlns:ofd="http://www.ofdspec.org/2016"'
    page = lambda *objs: (f'<ofd:Page {ns}><ofd:Content><ofd:Layer ID="1">' + "".join(
        f'<ofd:TextObject ID="{i}">' + "".join(f'<ofd:TextCode X="0" Y="0">{t}</ofd:TextCode>'
                                                for t in runs) + '</ofd:TextObject>'
        for i, runs in enumerate(objs)) + '</ofd:Layer></ofd:Content></ofd:Page>')
    pre = "/Doc_0/" if absolute else ""
    doc = (f'<ofd:Document {ns}><ofd:CommonData><ofd:TemplatePage ID="9" BaseLoc="{pre}Tpls/Tpl_0/Content.xml"/>'
           f'</ofd:CommonData><ofd:Pages><ofd:Page ID="1" BaseLoc="{pre}Pages/Page_0/Content.xml"/>'
           f'<ofd:Page ID="2" BaseLoc="{pre}Pages/Page_1/Content.xml"/></ofd:Pages></ofd:Document>')
    root = "Doc_0/Document.xml" if not broken_root else "Nope/Document.xml"
    f = tmp_path / ("abs.ofd" if absolute else "inv.ofd")
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("OFD.xml", f'<ofd:OFD {ns} DocType="OFD"><ofd:DocBody><ofd:DocRoot>{root}</ofd:DocRoot>'
                              '</ofd:DocBody></ofd:OFD>')
        z.writestr("Doc_0/Document.xml", doc)
        z.writestr("Doc_0/Pages/Page_0/Content.xml", page(["电子发票", "（普通发票）"], ["发票号码", "24117000000123456789"]))
        z.writestr("Doc_0/Pages/Page_1/Content.xml", page(["Total", "4200.00"]))
        z.writestr("Doc_0/Tpls/Tpl_0/Content.xml", page(["购买方信息"], ["销售方信息"]))
    return f


@pytest.mark.parametrize("absolute", [False, True])
def test_ofd_invoice(tmp_path, absolute):
    text = parser.parse_file(_ofd(tmp_path, absolute))
    assert text.split("\n")[:2] == ["电子发票（普通发票）", "发票号码24117000000123456789"]
    assert "Total 4200.00" in text and text.count("\f") == 1         # 2 pages
    assert text.rstrip().endswith("购买方信息\n销售方信息")              # template labels after pages


def test_ofd_nonstandard_package_falls_back(tmp_path):
    text = parser.parse_file(_ofd(tmp_path, broken_root=True))
    assert "电子发票" in text and "Total 4200.00" in text


def test_eml(tmp_path):
    m = EmailMessage()
    m["Subject"] = "报销单已审批"
    m["From"] = "finance@example.com"
    m["To"] = "me@example.com"
    m.set_content("Your reimbursement INV-7731 was approved.")
    m.add_alternative("<p>Your <b>reimbursement</b> was approved.</p>", subtype="html")
    m.add_attachment(b"%PDF", maintype="application", subtype="pdf", filename="发票.pdf")
    f = tmp_path / "a.eml"
    f.write_bytes(bytes(m))
    text = parser.parse_file(f)
    for e in ("报销单已审批", "finance@example.com", "INV-7731", "发票.pdf"):
        assert e in text


def test_scanned_pdf_is_ocrd_when_ocr_is_on(tmp_path, monkeypatch):
    fitz = pytest.importorskip("pymupdf")
    f = tmp_path / "scan.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "A real text layer on page one.")
    p2 = doc.new_page()                                  # page two: only an image
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pix.clear_with(200)
    p2.insert_image(fitz.Rect(72, 72, 300, 300), pixmap=pix)
    doc.save(f)
    seen = []
    monkeypatch.setattr(parser._ocr, "ocr_image", lambda p: seen.append(p) or "扫描件：合同金额 4200 元")
    monkeypatch.setattr(parser, "_OCR_ON", False)
    assert "扫描件" not in parser.parse_file(f)           # OCR off: text layer only
    monkeypatch.setattr(parser, "_OCR_ON", True)
    text = parser.parse_file(f)
    assert "A real text layer" in text and "扫描件：合同金额 4200 元" in text
    assert text.index("\f") < text.index("扫描件") and len(seen) == 1   # only the image page


def test_ocr_retry_for_previously_empty_pdfs(fake_embedder, tmp_path, monkeypatch):
    import localsearch.engine as E
    from localsearch import Config
    fitz = pytest.importorskip("pymupdf")
    d = tmp_path / "docs"
    d.mkdir()
    doc = fitz.open()
    pg = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
    pg.insert_image(fitz.Rect(72, 72, 300, 300), pixmap=pix)
    doc.save(d / "scan.pdf")
    monkeypatch.setattr(parser._ocr, "ocr_image", lambda p: "合同 contract scanned text")
    monkeypatch.setattr(parser, "_OCR_ON", False)
    eng = E.SearchEngine(tmp_path / "i", Config(personalize=False))
    assert eng.build([d])["failed"] == 1                 # no text, OCR off
    assert eng.build([d])["skipped"] == 1                # unchanged -> skipped
    monkeypatch.setattr(parser, "_OCR_ON", True)         # user turns OCR on
    st = eng.build([d])
    assert st["updated"] == 1 and eng.stats()["chunks"] == 1
    assert eng.build([d])["skipped"] == 1                # and only once
    eng.close()
