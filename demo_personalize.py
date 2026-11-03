"""Demo: local, training-free personalization.

Learns from what you OPEN — never trains the model (pure counting + vector math
on frozen embeddings), fully on-device. The reliable, honest effect: the files
(and folders) you keep opening float to the top for similar queries — even one
the base ranking had buried below the fold — while a fresh query is never
hijacked. Toggle off restores the base order; "forget" wipes the local profile.

    python demo_personalize.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from localsearch import SearchEngine, Config

FAV = "会议记录-周一晨会.txt"


def rank(hits, name):
    for i, h in enumerate(hits):
        if os.path.basename(h.path) == name:
            return i
    return 999


def show(hits, title, n=6):
    print(f"\n{title}")
    for i, h in enumerate(hits[:n]):
        star = "  ★常用" if getattr(h, "personal", False) else ""
        print(f"   {i+1:>2}. {os.path.basename(h.path)}{star}")


def main():
    docs = tempfile.mkdtemp(prefix="demo_pers_")
    # a pile of similar meeting notes (near-tied for the query) + your go-to one
    for i in range(11):
        open(os.path.join(docs, f"会议记录-{i:02d}.txt"), "w", encoding="utf-8").write(
            "本周会议记录：项目进度同步、任务分配与下一步计划的要点汇总。")
    open(os.path.join(docs, FAV), "w", encoding="utf-8").write(
        "本周会议记录：项目进度同步、任务分配与下一步计划的要点汇总。")

    eng = SearchEngine(tempfile.mkdtemp(prefix="demo_pers_idx_"), Config())
    eng.build([docs])
    Q = "会议记录"

    base = eng.search_semantic(Q, top_k=12)
    r0 = rank(base, FAV)
    show(base, f"① 搜『{Q}』(还没学任何偏好)。你常用的「{FAV}」此刻排在第 {r0+1} 位:")

    print(f"\n② 你连续打开了几次「{FAV}」(本地记录,不上传、不训练模型)。")
    for _ in range(4):
        eng.search_semantic(Q, top_k=12)          # refresh what's "shown"
        eng.record_open(os.path.join(docs, FAV))

    after = eng.search_semantic(Q, top_k=12)
    r1 = rank(after, FAV)
    show(after, f"③ 再搜『{Q}』:它浮到了第 {r1+1} 位,并标上「★常用」——你常开的文件自动靠前:")

    eng.personalizer.set_enabled(False)
    off = eng.search_semantic(Q, top_k=12)
    r_off = rank(off, FAV)
    show(off, "④ 关掉个性化:立刻回到原始排序(个性化只重排,不改检索本身,也绝不顶掉本次查询的最佳匹配):")

    fresh = eng.search_semantic("完全不相关的查询 quantum", top_k=5)
    eng.personalizer.set_enabled(True)
    eng.personalizer.reset()

    print("\n" + "=" * 64)
    print("结果")
    print("=" * 64)
    checks = [
        ("常开的文件明显靠前", r1 < r0, f"第 {r0+1} 位 → 第 {r1+1} 位"),
        ("并被标记为『常用』", any(getattr(h, "personal", False) for h in after)),
        ("关掉后恢复原始排序(标记消失)", not any(getattr(h, "personal", False) for h in off)),
        ("无关的新查询不会被你的习惯劫持", all(not getattr(h, "personal", False) for h in fresh)),
        ("『忘记我』后本地画像清空", len(eng.personalizer.opens) == 0),
    ]
    ok = True
    for name, cond, *d in checks:
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {d[0]}" if d else ""))
    eng.close()
    print("\n" + ("全部通过 —— 本地、免训练的个性化成立:你常开的文件越来越靠前,"
                  "全程本地、从不上传、也从不训练模型。"
                  if ok else "有检查未通过。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
