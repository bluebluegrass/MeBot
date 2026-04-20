from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.retrieval.hybrid import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 scripts/query_corpus.py '<query>'")
        raise SystemExit(1)

    query = sys.argv[1]
    results = retrieve(query)
    print(f"query={query!r} results={len(results)}")
    for index, result in enumerate(results, start=1):
        excerpt = " ".join(result.text.split())
        if len(excerpt) > 180:
            excerpt = f"{excerpt[:180].rstrip()}..."
        print(
            f"{index}. {result.title} | {result.source_type} | "
            f"{result.date_published or 'undated'} | score={result.score:.3f}"
        )
        if result.canonical_url:
            print(f"   {result.canonical_url}")
        print(f"   {excerpt}")


if __name__ == "__main__":
    main()
