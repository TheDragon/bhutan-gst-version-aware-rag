from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute notebook code cells in one Python process")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    notebook = nbformat.read(args.notebook, as_version=4)
    namespace: dict = {"__name__": "__notebook__"}
    executed = 0
    for cell_number, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        code = compile(cell.source, f"{args.notebook}:cell-{cell_number}", "exec")
        exec(code, namespace, namespace)
        executed += 1
        print(f"Validated code cell {cell_number}")
    print(f"Validated {executed} code cells in {args.notebook}")


if __name__ == "__main__":
    main()

