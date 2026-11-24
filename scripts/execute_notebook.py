from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a notebook without loading optional nbconvert plugins.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}})
    client.execute()
    output = args.output or args.notebook
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed {args.notebook} -> {output}")


if __name__ == "__main__":
    main()

