"""Reproduce notebook cells and their GitHub-visible tables and figures.

This in-process runner also works where network-isolated environments prevent
the Jupyter kernel message sockets from starting. It executes each code cell
in sequence, captures stdout and generated figures, and validates the result.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _display(value: object) -> None:
    """Capture human-readable tables in the notebook's stdout output."""
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)


def reproduce_notebook(path: Path, *, save: bool = True) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    variables = {"__name__": "__main__", "display": _display}
    execution_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        execution_count += 1
        source = cell.source.replace("%matplotlib inline", "")
        text_output = io.StringIO()
        try:
            with redirect_stdout(text_output):
                exec(compile(source, str(path), "exec"), variables)
        except Exception as error:
            raise RuntimeError(f"Notebook {path.name}, cell {execution_count}") from error
        outputs = []
        if text_output.getvalue():
            outputs.append(nbformat.v4.new_output(
                "stream", name="stdout", text=text_output.getvalue(),
            ))
        for number in plt.get_fignums():
            figure = plt.figure(number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            outputs.append(nbformat.v4.new_output(
                "display_data", data={
                    "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "text/plain": "Experiment visualization",
                }, metadata={},
            ))
            plt.close(figure)
        cell.outputs = outputs
        cell.execution_count = execution_count
    nbformat.validate(notebook)
    if save:
        nbformat.write(notebook, path)
    print(f"Verified {path.name}: {execution_count} code cells, "
          f"{sum(len(c.get('outputs', [])) for c in notebook.cells)} saved outputs")
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Execute without overwriting notebooks")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    os.chdir(NOTEBOOKS)
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        reproduce_notebook(path, save=not args.check)


if __name__ == "__main__":
    main()
