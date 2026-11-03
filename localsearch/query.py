"""Search operators and time phrases (Chinese + English).

    预算 type:sheet time:今年      budget spreadsheets modified this year
    合同 in:法务 after:2024-03     contracts under a path containing 法务, since March 2024
    time:上周 type:pdf             no words at all: list last week's PDFs, newest first

Operators are HARD filters. A bare time phrase among the words ("去年的工作总结")
is deliberately NOT applied: it usually describes the content (a summary OF last
year, saved this January), and a hard filter would hide exactly that file. It is
offered instead as a one-click suggestion (`Query.suggest`).

Times are local; ranges are half-open [after, before). Week starts Monday.
"""
import calendar
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .ocr import IMAGE_EXT

_DOCS = (".docx", ".rtf", ".txt", ".md", ".pdf")
_SHEETS = (".xlsx", ".csv")
_SLIDES = (".pptx",)
_TEXT = (".txt", ".md")
_WEB = (".html", ".htm")
KINDS = {
    "pdf": (".pdf",), "doc": _DOCS, "docs": _DOCS, "document": _DOCS, "文档": _DOCS,
    "word": (".docx", ".rtf"),
    "sheet": _SHEETS, "excel": _SHEETS, "spreadsheet": _SHEETS, "表格": _SHEETS,
    "slide": _SLIDES, "slides": _SLIDES, "ppt": _SLIDES, "幻灯片": _SLIDES, "演示": _SLIDES,
    "text": _TEXT, "文本": _TEXT, "web": _WEB, "html": _WEB, "网页": _WEB,
    "image": IMAGE_EXT, "images": IMAGE_EXT, "图片": IMAGE_EXT, "截图": IMAGE_EXT,
}

_OP = re.compile(r'(?<![^\s])(ext|type|in|path|after|since|before|time|date)'
                 r':("([^"]*)"|[^\s"]+)', re.I)

# relative periods: canonical key -> aliases
_REL = {
    "today": ("today", "今天", "今日"),
    "yesterday": ("yesterday", "昨天", "昨日"),
    "daybeforeyesterday": ("前天",),
    "thisweek": ("thisweek", "本周", "这周", "这星期", "本星期", "这个星期"),
    "lastweek": ("lastweek", "上周", "上星期", "上个星期", "上一周"),
    "thismonth": ("thismonth", "本月", "这个月", "这月"),
    "lastmonth": ("lastmonth", "上个月", "上月", "上一个月"),
    "thisyear": ("thisyear", "今年", "本年"),
    "lastyear": ("lastyear", "去年", "上一年"),
    "yearbeforelast": ("前年",),
}
_ALIAS = {a.lower(): k for k, al in _REL.items() for a in al}
_N_ZH = re.compile(r"^(?:最近|近|过去)(\d{1,3})(天|日|周|星期|个月|月|年)$")
_N_EN = re.compile(r"^(?:past|last)?(\d{1,3})(d|days?|w|weeks?|m|months?|y|years?)$", re.I)
_ABS = re.compile(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?$")
_ABS_ZH = re.compile(r"^(\d{4})年(?:(\d{1,2})月(?:(\d{1,2})[日号])?)?$")

# phrases to recognize in free text (suggestion only), longest first
_SUGGEST = sorted({a for al in _REL.values() for a in al if not a.isascii()}, key=len, reverse=True)
_SUGGEST_EN = [("today", "today"), ("yesterday", "yesterday"), ("this week", "thisweek"),
               ("last week", "lastweek"), ("this month", "thismonth"),
               ("last month", "lastmonth"), ("this year", "thisyear"), ("last year", "lastyear")]
_SUGGEST_N = re.compile(r"(?:最近|近|过去)\d{1,3}(?:天|日|周|星期|个月|月|年)"
                        r"|\b(?:past|last) \d{1,3} (?:days?|weeks?|months?|years?)\b", re.I)


def _midnight(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _add_months(dt, n):
    m = dt.month - 1 + n
    y, m = dt.year + m // 12, m % 12 + 1
    return dt.replace(year=y, month=m, day=min(dt.day, calendar.monthrange(y, m)[1]))


def time_range(value, now=None):
    """(start, end) datetimes for a period value, end exclusive (None = open);
    None if `value` is not a period."""
    now = now or datetime.now()
    v = value.strip().lower().replace(" ", "")
    day = _midnight(now)
    key = _ALIAS.get(v)
    if key == "today":
        return day, None
    if key == "yesterday":
        return day - timedelta(days=1), day
    if key == "daybeforeyesterday":
        return day - timedelta(days=2), day - timedelta(days=1)
    week = day - timedelta(days=day.weekday())
    if key == "thisweek":
        return week, None
    if key == "lastweek":
        return week - timedelta(days=7), week
    month = day.replace(day=1)
    if key == "thismonth":
        return month, None
    if key == "lastmonth":
        return _add_months(month, -1), month
    year = month.replace(month=1)
    if key == "thisyear":
        return year, None
    if key == "lastyear":
        return year.replace(year=year.year - 1), year
    if key == "yearbeforelast":
        return year.replace(year=year.year - 2), year.replace(year=year.year - 1)
    m = _N_ZH.match(v) or _N_EN.match(v)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit[0] in "天日d":
            return day - timedelta(days=n - 1), None        # "past 7 days" includes today
        if unit[0] in "周星w":
            return day - timedelta(days=7 * n - 1), None
        if unit[0] in "个月m":
            return _add_months(day, -n), None
        return day.replace(year=day.year - n), None
    m = _ABS.match(v) or _ABS_ZH.match(v)
    if m:
        y, mo, d = int(m.group(1)), m.group(2), m.group(3)
        try:
            if d:
                start = datetime(y, int(mo), int(d))
                return start, start + timedelta(days=1)
            if mo:
                start = datetime(y, int(mo), 1)
                return start, _add_months(start, 1)
            return datetime(y, 1, 1), datetime(y + 1, 1, 1)
        except ValueError:
            return None
    return None


@dataclass
class Query:
    text: str
    exts: frozenset = None
    paths: tuple = ()
    after: float = None               # epoch seconds, inclusive
    before: float = None              # epoch seconds, exclusive
    filters: list = field(default_factory=list)    # for the UI: {kind, value, token}
    suggest: dict = None              # time phrase found in the words, not applied
    unknown: list = field(default_factory=list)    # operator values we could not read

    @property
    def filtered(self):
        return bool(self.exts or self.paths or self.after is not None
                    or self.before is not None)

    def match_path(self, path):
        if self.exts is not None and os.path.splitext(path)[1].lower() not in self.exts:
            return False
        if self.paths:
            p = os.path.normcase(str(path))
            return all(f in p for f in self.paths)
        return True

    def match_time(self, mtime):
        if self.after is not None and mtime < self.after:
            return False
        return self.before is None or mtime < self.before

    def match(self, path, mtime=None):
        if not self.match_path(path):
            return False
        if self.after is None and self.before is None:
            return True
        if mtime is None:
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                return False
        return self.match_time(mtime)


def _fmt(dt):
    return dt.strftime("%Y-%m-%d")


def _clip(q, start, end):
    s, e = start.timestamp(), (end.timestamp() if end else None)
    q.after = s if q.after is None else max(q.after, s)
    if e is not None:
        q.before = e if q.before is None else min(q.before, e)


def parse(raw, now=None):
    """Split a raw query into free text + filters (see module docstring)."""
    now = now or datetime.now()
    q = Query(text="")
    exts = set()
    has_ext = False

    def take(m):
        nonlocal has_ext
        key, val = m.group(1).lower(), (m.group(3) if m.group(3) is not None else m.group(2))
        token = m.group(0)
        if not val:
            q.unknown.append(token)
            return token
        if key in ("ext", "type"):
            vals = [v for v in re.split(r"[,;，；|]", val) if v]
            got = []
            for v in vals:
                lv = v.lower().lstrip(".")
                if key == "type" and lv in KINDS:
                    got.extend(KINDS[lv])
                elif re.fullmatch(r"[a-z0-9]{1,8}", lv):
                    got.append("." + lv)
            if not got:
                q.unknown.append(token)
                return token
            exts.update(got)
            has_ext = True
            q.filters.append(dict(kind="type", value=" ".join(sorted(set(got))), token=token))
        elif key in ("in", "path"):
            q.paths += (os.path.normcase(val),)
            q.filters.append(dict(kind="in", value=val, token=token))
        else:
            rng = time_range(val, now)
            if rng is None:
                q.unknown.append(token)
                return token
            start, end = rng
            if key in ("after", "since"):
                _clip(q, start, None)
                q.filters.append(dict(kind="after", value=_fmt(start), token=token))
            elif key == "before":
                e = start.timestamp()
                q.before = e if q.before is None else min(q.before, e)
                q.filters.append(dict(kind="before", value=_fmt(start), token=token))
            else:
                _clip(q, start, end)
                q.filters.append(dict(kind="time", value=val, token=token, start=_fmt(start),
                                      end=_fmt(end - timedelta(seconds=1)) if end else _fmt(now)))
        return " "

    text = _OP.sub(take, raw)
    q.exts = frozenset(exts) if has_ext else None
    q.text = " ".join(text.split())
    q.suggest = _suggest(q.text, now) if not any(f["kind"] == "time" for f in q.filters) else None
    return q


def _suggest(text, now):
    low = text.lower()
    phrase = canon = None
    m = _SUGGEST_N.search(text)
    if m:
        phrase = m.group(0)
        canon = phrase.replace(" ", "")
    else:
        for a in _SUGGEST:
            if a in text:
                phrase, canon = a, a
                break
        else:
            for p, c in _SUGGEST_EN:
                if re.search(r"\b%s\b" % re.escape(p), low):
                    phrase, canon = p, c
                    break
    if not phrase:
        return None
    rng = time_range(canon, now)
    if rng is None:
        return None
    start, end = rng
    return dict(op="time:" + canon, phrase=phrase, start=_fmt(start),
                end=_fmt(end - timedelta(seconds=1)) if end else _fmt(now))

