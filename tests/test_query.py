from datetime import datetime

import pytest

from localsearch.query import parse, time_range

NOW = datetime(2026, 11, 3, 15, 30)          # a Tuesday


def d(*a):
    return datetime(*a)


@pytest.mark.parametrize("value,start,end", [
    ("今天", d(2026, 11, 3), None), ("today", d(2026, 11, 3), None),
    ("昨天", d(2026, 11, 2), d(2026, 11, 3)),
    ("本周", d(2026, 11, 2), None), ("上周", d(2026, 10, 26), d(2026, 11, 2)),
    ("lastweek", d(2026, 10, 26), d(2026, 11, 2)),
    ("上个月", d(2026, 10, 1), d(2026, 11, 1)), ("本月", d(2026, 11, 1), None),
    ("去年", d(2025, 1, 1), d(2026, 1, 1)), ("前年", d(2024, 1, 1), d(2025, 1, 1)),
    ("最近7天", d(2026, 10, 28), None), ("past30days", d(2026, 10, 5), None), ("2w", d(2026, 10, 21), None),
    ("最近3个月", d(2026, 8, 3), None),
    ("2024", d(2024, 1, 1), d(2025, 1, 1)), ("2024-03", d(2024, 3, 1), d(2024, 4, 1)),
    ("2024/03/05", d(2024, 3, 5), d(2024, 3, 6)), ("2024年3月", d(2024, 3, 1), d(2024, 4, 1)),
])
def test_time_range(value, start, end):
    assert time_range(value, NOW) == (start, end)


def test_jan_last_month_crosses_year():
    assert time_range("上个月", datetime(2026, 1, 15)) == (d(2025, 12, 1), d(2026, 1, 1))


def test_operators_are_filters_and_leave_the_words():
    q = parse("预算 type:sheet time:今年 in:\"My Docs\"", NOW)
    assert q.text == "预算"
    assert q.exts == frozenset({".xlsx", ".csv"})
    assert q.after == d(2026, 1, 1).timestamp() and q.before is None
    assert q.match_path("/x/My Docs/a.xlsx") and not q.match_path("/x/Other/a.xlsx")
    assert [f["kind"] for f in q.filters] == ["type", "time", "in"]


def test_match():
    q = parse("x ext:pdf,docx in:合同 after:2024-03 before:2024-05", NOW)
    ok = d(2024, 4, 1).timestamp()
    assert q.match("/a/合同/x.pdf", ok)
    assert not q.match("/a/合同/x.txt", ok)                     # wrong type
    assert not q.match("/a/其他/x.pdf", ok)                     # wrong folder
    assert not q.match("/a/合同/x.pdf", d(2024, 5, 1).timestamp())   # before is exclusive
    assert not q.match("/a/合同/x.pdf", d(2024, 2, 29).timestamp())


def test_only_filters_means_browse():
    q = parse("time:上周 type:pdf", NOW)
    assert q.text == "" and q.filtered


def test_bare_time_words_are_only_suggested():
    q = parse("去年的工作总结", NOW)
    assert q.text == "去年的工作总结" and not q.filtered      # NOT applied
    assert q.suggest["op"] == "time:去年" and q.suggest["start"] == "2025-01-01"
    assert parse("report from last month", NOW).suggest["op"] == "time:lastmonth"
    assert parse("最近7天的会议纪要", NOW).suggest["op"] == "time:最近7天"
    assert parse("budget 2024", NOW).suggest is None


def test_not_operators():
    for raw in ["C:\\Users\\me notes", "login:admin password", "http://x.org", "type:", "time:someday"]:
        q = parse(raw, NOW)
        assert not q.filtered
    assert parse("time:someday x", NOW).unknown == ["time:someday"]
