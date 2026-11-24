from __future__ import annotations

import argparse
from gst_rag.systems import SYSTEMS, RetrievalEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--system", choices=SYSTEMS, default="version_aware")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    engine = RetrievalEngine.from_disk()
    print(f"System: {args.system}")
    for rank, result in enumerate(engine.search(args.query, args.system, args.date, args.top_k), start=1):
        print(f"\n{rank}. {result['title']} | section {result['section_number']} | page {result['page_number']}")
        print(result["text"][:500].replace("\n", " "))
        print(result["source_url"])


if __name__ == "__main__":
    main()
