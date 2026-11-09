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
import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _display(value: object) -> None:
    """Capture human-readable tables in the notebook's stdout output."""
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)


def _verify_outputs(previous: list, current: list, path: Path, cell_number: int) -> None:
    """Check numerical text and integrity/visual consistency of saved charts."""
    def text_results(outputs):
        return [(output.get("output_type"), output.get("name"),
                 str(output.get("text", "")).strip())
                for output in outputs if output.get("output_type") == "stream"]

    def figures(outputs):
        return [output["data"]["image/png"] for output in outputs
                if "image/png" in output.get("data", {})]

    if text_results(previous) != text_results(current):
        raise RuntimeError(f"Published results differ in {path.name}, cell {cell_number}")
    old_figures, new_figures = figures(previous), figures(current)
    if len(old_figures) != len(new_figures):
        raise RuntimeError(f"Published chart count differs in {path.name}, cell {cell_number}")
    def decoded_chart(encoded: str) -> tuple[tuple[int, int], np.ndarray]:
        try:
            binary = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(binary)) as chart:
                if chart.format != "PNG":
                    raise ValueError("Chart is not a PNG image")
                chart.load()  # Force decoding; checking the PNG header alone is insufficient.
                pixels = np.asarray(
                    chart.convert("RGB").resize((128, 96), Image.Resampling.LANCZOS),
                    dtype=np.int16,
                )
                return chart.size, pixels
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Invalid published chart in {path.name}, cell {cell_number}"
            ) from exc

    for old, new in zip(old_figures, new_figures):
        old_size, old_pixels = decoded_chart(old)
        new_size, new_pixels = decoded_chart(new)
        if (max(abs(a - b) for a, b in zip(old_size, new_size)) > 3
                or np.abs(old_pixels - new_pixels).mean() > 3.0):
            raise RuntimeError(f"Published chart differs in {path.name}, cell {cell_number}")


def reproduce_notebook(path: Path, *, save: bool = True,
                       check: bool = False) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    variables = {"__name__": "__main__", "display": _display}
    execution_count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        execution_count += 1
        original_outputs = list(cell.outputs)
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
        if check:
            _verify_outputs(original_outputs, outputs, path, execution_count)
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
        reproduce_notebook(path, save=not args.check, check=args.check)


if __name__ == "__main__":
    main()
