from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.retrieval.hybrid import retrieve_documents


# Each case: (query, expected_title_fragments)
# A hit = at least one returned document title contains one of the fragments (case-insensitive).
EVAL_CASES = [
    ("徒步 银座 表参道",               ["表银座"]),
    ("升职加薪 职业目标",               ["升职加薪", "Burnout"]),
    ("日本生活 试用期",                 ["日本试用期", "日本生活体验"]),
    ("播客 年终总结 迁徙",              ["年终总结", "Pod Luck"]),
    ("跨国求职 AI工具使用",             ["跨国找工", "AI工具"]),
    ("Kamala Harris 美国大选 播客",     ["Kamala"]),
    ("独居 幸福 生活",                  ["独居"]),
    ("美国 种族歧视 经历",              ["种族歧视"]),
    ("东京 交友 线下社交",              ["东京交友"]),
    ("尴尬 情绪 社交",                  ["尴尬"]),
]
TOP_K = 5


def main() -> None:
    if not (ROOT_DIR / "data" / "db" / "mebot.sqlite3").exists():
        print("DB not found. Run scripts/ingest_corpus.py first.")
        sys.exit(1)

    hits = 0
    for query, expected_fragments in EVAL_CASES:
        results = retrieve_documents(query, max_chunks_per_document=1)[:TOP_K]
        returned_titles = [doc.title for doc in results]

        hit = any(
            fragment.lower() in title.lower()
            for title in returned_titles
            for fragment in expected_fragments
        )
        if hit:
            hits += 1

        status = "HIT " if hit else "MISS"
        print(f"[{status}] {query}")
        print(f"       expected fragments : {expected_fragments}")
        for doc in results:
            print(f"       {doc.score:.3f}  {doc.title}")
        print()

    total = len(EVAL_CASES)
    print(f"Recall@{TOP_K}: {hits}/{total}  ({hits / total:.0%})")


if __name__ == "__main__":
    main()
