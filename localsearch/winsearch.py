"""Instant GLOBAL content search by riding the OS's own content index."""
import logging
import re

log = logging.getLogger("localsearch.winsearch")
_PROVIDER = "Provider=Search.CollatorDSO;Extended Properties='Application=Windows';"
_TERM = re.compile(r"[^\W_]+")          # words of any script (CJK split out below)
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿]+")


def _url_to_path(url):
    """System.ItemUrl -> real filesystem path (canonical, non-localized names)."""
    if url and url.lower().startswith("file:"):
        url = url[5:]
    return (url or "").replace("/", "\\")


def _sanitize(text):
    t = text.replace("'", " ").replace('"', " ").replace("\n", " ")
    return " ".join(t.split())[:200]


def _or_terms(terms):
    """Split a sanitized query into OR-able terms: words + CJK bigrams."""
    out = list(_TERM.findall(_CJK_RUN.sub(" ", terms)))
    for run in _CJK_RUN.findall(terms):
        if len(run) <= 2:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))

    seen = set()
    return [t for t in out if not (t.lower() in seen or seen.add(t.lower()))]


def available():
    try:
        import win32com.client
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open(_PROVIDER)
        conn.Close()
        return True
    except Exception as e:
        log.info("Windows Search not available: %s", e)
        return False


def _run_sql(sql):
    """Run one SystemIndex query -> [(path, snippet)] via a single GetRows call;
    None if the query itself failed (vs [] for no matches)."""
    import win32com.client
    out = []
    conn = None
    try:
        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open(_PROVIDER)
        rs = win32com.client.Dispatch("ADODB.Recordset")
        rs.Open(sql, conn)
        if not rs.EOF:
            data = rs.GetRows()
            urls, snips = data[0], data[1]
            for url, snip in zip(urls, snips):
                path = _url_to_path(url)
                if path:
                    out.append((path, " ".join(str(snip or "").split())))
        rs.Close()
    except Exception as e:
        log.warning("Windows Search query failed: %s", e)
        out = None
    finally:
        try:
            if conn is not None:
                conn.Close()
        except Exception:
            pass
    return out


def _build_sql(where, limit):
    return (f"SELECT TOP {int(limit)} System.ItemUrl, System.Search.AutoSummary "
            f"FROM SystemIndex WHERE {where} ORDER BY System.Search.Rank DESC")


def _ext_sql(exts):
    if not exts:
        return ""
    return " AND (%s)" % " OR ".join(
        "System.FileExtension = '%s'" % e.replace("'", "''") for e in sorted(exts))


def query(text, limit=200, scope=None, min_results=None, exts=None):
    """Whole-disk content search via the OS index -> [(path, snippet)], best-first.
    `exts` narrows by extension in SQL when the provider accepts it; if that
    clause fails or matches nothing, the query is retried without it (callers
    post-filter), so an unexpected provider can never silently empty a search."""
    terms = _sanitize(text)
    if not terms:
        return []
    scope_sql = ""
    if scope:
        scope_sql = " AND SCOPE='file:%s'" % str(scope).replace("'", "''")
    extra = _ext_sql(exts)

    def run(where):
        rows = _run_sql(_build_sql(where + scope_sql + extra, limit))
        if extra and not rows:
            rows = _run_sql(_build_sql(where + scope_sql, limit))
        return rows or []

    strict = run(f"FREETEXT(System.Search.Contents, '{terms}')")
    want = min_results if min_results is not None else 0
    if len(strict) >= max(want, limit // 4):
        return strict
    ors = _or_terms(terms)
    if len(ors) < 2:
        return strict
    contains = " OR ".join(f'"{t}"' for t in ors)
    loose = run(f"CONTAINS(System.Search.Contents, '{contains}')")
    seen = {p for p, _ in strict}
    merged = strict + [(p, s) for p, s in loose if p not in seen]
    return merged[:limit]


def filename_query(text, limit=50, scope=None, exts=None):
    """Filename search through the OS index -- the fallback when Everything is
    not installed. Covers only indexed locations (usually the user folders),
    not the whole disk. Everything-style * and ? wildcards are honoured.
    -> [path] newest first, or None if the provider failed."""
    terms = [t for t in re.split(r"\s+", text.replace("'", " ").replace('"', " ")) if t]
    where = []
    for t in terms[:8]:
        like = t.replace("%", "").replace("_", "").replace("*", "%").replace("?", "_")
        where.append("System.FileName LIKE '%%%s%%'" % like)
    if scope:
        where.append("SCOPE='file:%s'" % str(scope).replace("'", "''"))
    if not where and not exts:
        return []
    sql = (f"SELECT TOP {int(limit)} System.ItemUrl, System.ItemName FROM SystemIndex "
           f"WHERE {' AND '.join(where) or '1=1'}{_ext_sql(exts)} "
           f"ORDER BY System.DateModified DESC")
    rows = _run_sql(sql)
    return None if rows is None else [p for p, _ in rows]


if __name__ == "__main__":
    import os
    import sys
    import time
    q = sys.argv[1] if len(sys.argv) > 1 else "Microsoft license copyright"
    print("Windows Search available:", available())
    t0 = time.perf_counter()
    res = query(q, limit=15)
    dt = (time.perf_counter() - t0) * 1000
    print(f"query={q!r} -> {len(res)} hits in {dt:.0f} ms (whole disk, no pre-indexing)")
    exists = 0
    for path, snip in res[:8]:
        ok = os.path.exists(path)
        exists += ok
        print(f"  [{'open-able' if ok else 'BAD-PATH'}] {path}")
        if snip:
            print(f"       snippet: {snip[:90]}")
    print(f"openable paths: {exists}/{min(8, len(res))}")
