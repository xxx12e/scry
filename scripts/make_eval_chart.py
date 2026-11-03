"""Render the README's eval chart (a dumbbell: v1.4 settings -> now) from the
JSON results that scripts/eval.py writes -- the numbers are never typed by hand.

    python scripts/make_eval_chart.py        # -> docs/img/eval-{en,zh}-{light,dark}.svg

Colours: one hue, two validated steps (dataviz ordinal check on GitHub's light
#ffffff and dark #0d1117 surfaces); text uses text tokens, never the data colour.
"""
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape as esc, quoteattr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval import metrics  # noqa: E402

THEMES = {
    "light": dict(before="#86b6ef", after="#2a78d6", surface="#ffffff", grid="#e6e8eb",
                  ink="#1f2328", muted="#59636e", line="#c9cfd6"),
    "dark": dict(before="#184f95", after="#5598e7", surface="#0d1117", grid="#262c36",
                 ink="#f0f6fc", muted="#9198a1", line="#3d444d"),
}
TEXT = {
    "en": dict(title="Finding the right file — bundled model, v1.4 settings vs. now",
               sub="Bilingual eval set (60 zh/en docs, 102 queries). Share of queries whose right file is ranked…",
               before="v1.4", after="now", same="unchanged",
               rows=["Cross-language · #1", "Cross-language · in the top 5", "Same-language paraphrase · #1",
                     "All 102 queries · #1", "Keywords & exact phrases · #1"],
               font="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"),
    "zh": dict(title="找到对的那个文件——内置模型，v1.4 设置 vs 现在",
               sub="中英双语评测集（60 篇中英文档，102 条查询）。目标文件排在……的查询占比",
               before="v1.4", after="现在", same="不变",
               rows=["跨语言 · 排第一", "跨语言 · 进前五", "同语言改写 · 排第一",
                     "全部 102 条查询 · 排第一", "关键词与精确短语 · 排第一"],
               font="'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', -apple-system, 'Segoe UI', sans-serif"),
}


def values():
    def load(name):
        res = json.loads((ROOT / "eval" / "results" / f"{name}.json").read_text(encoding="utf-8"))
        return next(iter(res["alpha"].values()))
    old, new = load("e5small_v1.4"), load("e5small_now")

    def pick(rows, kinds, key):
        return metrics([r for r in rows if kinds is None or r["kind"] in kinds])[key]
    spec = [(("cross",), "R1"), (("cross",), "R5"), (("para",), "R1"), (None, "R1"),
            (("kw", "exact"), "R1")]
    return [(pick(old, k, m), pick(new, k, m)) for k, m in spec]


def svg(lang, theme, vals):
    c, t = THEMES[theme], TEXT[lang]
    W, left, right, top, row_h = 820, 290, 740, 92, 46
    H = top + row_h * len(vals) + 40
    x = lambda v: left + v * (right - left)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family={quoteattr(t["font"])} role="img" aria-label={quoteattr(t["title"])}>',
           f'<rect width="{W}" height="{H}" rx="10" fill="{c["surface"]}"/>',
           f'<text x="24" y="34" font-size="17" font-weight="600" fill="{c["ink"]}">{esc(t["title"])}</text>',
           f'<text x="24" y="56" font-size="12.5" fill="{c["muted"]}">{esc(t["sub"])}</text>']
    # legend (two series -> always present)
    lx = left
    for key in ("before", "after"):
        out.append(f'<circle cx="{lx}" cy="76" r="5" fill="{c[key]}"/>')
        out.append(f'<text x="{lx + 10}" y="80" font-size="12" fill="{c["muted"]}">{esc(t[key])}</text>')
        lx += 70
    # recessive hairline grid + ticks
    gy0, gy1 = top - 6, top + row_h * len(vals) - 10
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(v):.1f}" y1="{gy0}" x2="{x(v):.1f}" y2="{gy1}" stroke="{c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x(v):.1f}" y="{gy1 + 18}" font-size="11" text-anchor="middle" '
                   f'fill="{c["muted"]}">{int(v * 100)}%</text>')
    for i, ((b, a), label) in enumerate(zip(vals, t["rows"])):
        y = top + i * row_h + 12
        out.append(f'<text x="24" y="{y + 4}" font-size="13" fill="{c["ink"]}">{esc(label)}</text>')
        if abs(a - b) > 1e-9:
            out.append(f'<line x1="{x(b):.1f}" y1="{y}" x2="{x(a):.1f}" y2="{y}" stroke="{c["line"]}" '
                       f'stroke-width="2" stroke-linecap="round"/>')
        # surface ring keeps each dot legible where it meets the connector
        out.append(f'<circle cx="{x(b):.1f}" cy="{y}" r="6" fill="{c["before"]}" stroke="{c["surface"]}" stroke-width="2"/>')
        out.append(f'<circle cx="{x(a):.1f}" cy="{y}" r="6" fill="{c["after"]}" stroke="{c["surface"]}" stroke-width="2"/>')
        before_label = f"{b * 100:.0f}%" if abs(a - b) > 1e-9 else t["same"]
        out.append(f'<text x="{x(b) - 12:.1f}" y="{y + 4}" font-size="12" text-anchor="end" '
                   f'fill="{c["muted"]}">{esc(before_label)}</text>')
        out.append(f'<text x="{x(a) + 12:.1f}" y="{y + 4}" font-size="13" font-weight="600" '
                   f'fill="{c["ink"]}">{a * 100:.0f}%</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    vals = values()
    d = ROOT / "docs" / "img"
    d.mkdir(parents=True, exist_ok=True)
    for lang in TEXT:
        for theme in THEMES:
            (d / f"eval-{lang}-{theme}.svg").write_text(svg(lang, theme, vals), encoding="utf-8")
    for (b, a), label in zip(vals, TEXT["en"]["rows"]):
        print(f"{label:36s} {b:.3f} -> {a:.3f}")


if __name__ == "__main__":
    main()
