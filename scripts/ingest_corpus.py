from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ingestion.ingest import ingest_corpus


def main() -> None:
    result = ingest_corpus()
    print(
        f"documents_seen={result.documents_seen} "
        f"documents_ingested={result.documents_ingested} "
        f"chunks_created={result.chunks_created}"
    )


if __name__ == "__main__":
    main()
