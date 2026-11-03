"""Retrieval quality on the bilingual eval set (eval/scry_eval_v1.json).

    python scripts/eval.py --model models/embed                 # auto backend
    python scripts/eval.py --model models/embed --backend torch
    python scripts/eval.py --model path/to/int8_dir             # a quantized export

Reports Recall@1/@5 and MRR per query kind (paraphrase / cross-language /
keywords) and per query language, a hybrid-weight (alpha) sweep, and the
calibration data for the relevance floor: how far below the top hit's cosine
a RELEVANT hit can sit, versus irrelevant ones. The set is small and
hand-written (see its note): use it to compare configurations.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(model, backend, data, alphas=(0.7,), top_k=10, balances=(0.0,)):
    import logging
    logging.disable(logging.WARNING)
    from localsearch import Config, SearchEngine
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        for did, d in data["docs"].items():
            (corpus / f"{did}.txt").write_text(d["text"], encoding="utf-8")
        cfg = Config(model_name=model, backend=backend, personalize=False, device="cpu")
        eng = SearchEngine(Path(tmp) / "idx", cfg)
        t = time.perf_counter()
        eng.build([corpus])
        build_s = time.perf_counter() - t
        out = {"backend": eng.embedder.backend, "build_s": round(build_s, 2), "alpha": {}}
        for alpha, bal in [(a, b) for a in alphas for b in balances]:
            eng.cfg.alpha, eng.cfg.lang_balance = alpha, bal
            alpha = f"{alpha}/bal{bal}"
            rows = []
            for q in data["queries"]:
                hits = eng.search_semantic(q["q"], top_k=top_k, snippets="lexical")
                ids = [Path(h.path).stem for h in hits]
                rank = ids.index(q["rel"]) + 1 if q["rel"] in ids else None
                rows.append(dict(q, rank=rank, sims=[h.sim for h in hits],
                                 rel_idx=(rank - 1) if rank else None))
            out["alpha"][alpha] = rows
        eng.close()
        return out


def metrics(rows):
    n = len(rows)
    r1 = sum(1 for r in rows if r["rank"] == 1) / n
    r5 = sum(1 for r in rows if r["rank"] and r["rank"] <= 5) / n
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    return dict(n=n, R1=round(r1, 3), R5=round(r5, 3), MRR=round(mrr, 3))


def floor_calibration(rows):
    """Gap below the top cosine: relevant hits vs. irrelevant ones."""
    rel_gaps, irr_gaps = [], []
    for r in rows:
        s = [x for x in r["sims"] if x is not None]
        if not s:
            continue
        top = s[0]
        for i, x in enumerate(r["sims"]):
            if x is None:
                continue
            (rel_gaps if i == r["rel_idx"] else irr_gaps).append(top - x)
    out = {"max_gap_of_a_relevant_hit": round(max(rel_gaps), 4) if rel_gaps else None}
    for g in (0.03, 0.05, 0.08, 0.10):
        out[f"gap>{g}"] = dict(
            relevant_dimmed=sum(1 for x in rel_gaps if x > g),
            irrelevant_dimmed_pct=round(100 * sum(1 for x in irr_gaps if x > g) / max(1, len(irr_gaps)), 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"])
    ap.add_argument("--data", default=str(ROOT / "eval" / "scry_eval_v1.json"))
    ap.add_argument("--alphas", help="comma list, e.g. 0.5,0.7,0.9 (default: Config)")
    ap.add_argument("--balance", help="lang_balance values, comma list (default: Config)")
    ap.add_argument("--keep-en", type=int, default=None,
                    help="stress test: keep only the first N English topics (0 = Chinese-only corpus)")
    ap.add_argument("--qlang", choices=["zh", "en"], help="only queries in this language")
    ap.add_argument("--json", help="write the full result here")
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text(encoding="utf-8"))
    if a.keep_en is not None:          # language-imbalanced / monolingual corpora
        en = [t for t, d in data["docs"].items() if d["lang"] == "en" and "__" not in t][:a.keep_en]
        keep = {t for t, d in data["docs"].items() if d["lang"] != "en"} | \
               {t for t in data["docs"] if t.split("__")[0] in en}
        data = {"docs": {k: v for k, v in data["docs"].items() if k in keep},
                "queries": [q for q in data["queries"] if q["rel"] in keep]}
    if a.qlang:
        data = {"docs": data["docs"], "queries": [q for q in data["queries"] if q["qlang"] == a.qlang]}
    from localsearch import Config
    alphas = tuple(float(x) for x in a.alphas.split(",")) if a.alphas else (Config.alpha,)
    bals = (tuple(float(x) for x in a.balance.split(",")) if a.balance
            else (Config.lang_balance,))
    res = run(a.model, a.backend, data, alphas, balances=bals)
    print(f"backend={res['backend']}  index build {res['build_s']}s  "
          f"({len(data['docs'])} docs, {len(data['queries'])} queries)")
    for alpha, rows in res["alpha"].items():
        print(f"\nalpha={alpha}   overall {metrics(rows)}")
        by = defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r)
            by["q:" + r["qlang"]].append(r)
        for k in ("para", "cross", "kw", "exact", "q:zh", "q:en"):
            if by[k]:                      # a subset may lack a group (e.g. --qlang zh)
                print(f"   {k:6s} {metrics(by[k])}")
        miss = [r for r in rows if r["rank"] != 1 and r["kind"] != "cross"]
        if miss:
            print("   non-cross not at rank 1:", "; ".join(f"{r['q']} -> {r['rank']}" for r in miss[:10]))
    first = next(iter(res["alpha"].values()))
    print("\nrelevance floor calibration (alpha=%s):" % next(iter(res["alpha"])),
          json.dumps(floor_calibration(first), ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
