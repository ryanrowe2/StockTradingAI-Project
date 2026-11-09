"""Reproduce notebook cells and their GitHub-visible tables and figures.

This in-process runner also works where network-isolated environments prevent
the Jupyter kernel message sockets from starting. It executes each code cell
in sequence, captures text streams, rich displays, and generated figures,
and validates the result.
"""

from __future__ import annotations

import argparse
import ast
import base64
import codeop
import io
import os
import sys
import tokenize
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from IPython.core.formatters import DisplayFormatter
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import label


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class _StreamCapture(io.TextIOBase):
    """Collect ordered stdout/stderr writes as notebook stream outputs."""

    def __init__(self, name: str, events: list[nbformat.NotebookNode]):
        self.name = name
        self.events = events

    def write(self, message: str) -> int:
        if message:
            if (self.events and self.events[-1].output_type == "stream"
                    and self.events[-1].name == self.name):
                self.events[-1].text += message
            else:
                self.events.append(nbformat.v4.new_output(
                    "stream", name=self.name, text=message,
                ))
        return len(message)


def _remove_inline_magic(source: str) -> str:
    """Remove standalone notebook magic without changing multiline string data."""
    string_lines: set[int] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
                string_lines.update(range(token.start[0], token.end[0] + 1))
    except tokenize.TokenError:
        # Parsing the remaining source will report malformed Python below.
        pass
    return "".join(
        "\n" if number not in string_lines and line.strip() == "%matplotlib inline" else line
        for number, line in enumerate(source.splitlines(keepends=True), 1)
    )


def _execute_cell(source: str, path: Path, variables: dict,
                  compiler: codeop.Compile | None = None) -> object:
    """Execute a cell and return its final expression for Jupyter-style display."""
    compiler = compiler if compiler is not None else codeop.Compile()
    tree = ast.parse(source, filename=str(path))
    meaningful = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
                  if token.type not in (tokenize.ENDMARKER, tokenize.NEWLINE,
                                        tokenize.NL, tokenize.COMMENT, tokenize.DEDENT)]
    suppressed = bool(meaningful and meaningful[-1].type == tokenize.OP
                      and meaningful[-1].string == ";")
    if (not tree.body or not isinstance(tree.body[-1], ast.Expr)
            or suppressed):
        exec(compiler(tree, str(path), "exec"), variables)
        return None

    capture_name = "__notebook_runner_last_expression__"
    missing = object()
    previous = variables.get(capture_name, missing)
    last = tree.body[-1]
    tree.body[-1] = ast.copy_location(
        ast.Assign(targets=[ast.Name(id=capture_name, ctx=ast.Store())], value=last.value),
        last,
    )
    ast.fix_missing_locations(tree)
    try:
        exec(compiler(tree, str(path), "exec"), variables)
        return variables[capture_name]
    finally:
        if previous is missing:
            variables.pop(capture_name, None)
        else:
            variables[capture_name] = previous


def _verify_outputs(previous: list, current: list, path: Path, cell_number: int) -> None:
    """Check numerical text and integrity/visual consistency of saved charts."""
    def non_image_results(outputs):
        """Check every output, including rich text, errors, and output ordering."""
        comparable = []
        for output in outputs:
            saved = dict(output)
            saved.pop("execution_count", None)
            if "data" in saved:
                data = dict(saved["data"])
                data.pop("image/png", None)  # Compare PNG pixels separately.
                saved["data"] = data
            comparable.append(saved)
        return comparable

    def figures(outputs):
        return [output["data"]["image/png"] for output in outputs
                if "image/png" in output.get("data", {})]

    old_figures, new_figures = figures(previous), figures(current)
    if len(old_figures) != len(new_figures):
        raise RuntimeError(f"Published chart count differs in {path.name}, cell {cell_number}")
    if non_image_results(previous) != non_image_results(current):
        raise RuntimeError(f"Published results differ in {path.name}, cell {cell_number}")
    def decoded_chart(encoded: str) -> tuple[tuple[int, int], np.ndarray]:
        try:
            binary = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(binary)) as chart:
                if chart.format != "PNG":
                    raise ValueError("Chart is not a PNG image")
                chart.load()  # Force decoding; checking the PNG header alone is insufficient.
                pixels = np.asarray(
                    chart.convert("RGB").resize((384, 288), Image.Resampling.LANCZOS),
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
        difference = np.abs(old_pixels - new_pixels)
        # A mean-only check misses a thin graph line that moves dramatically
        # across a mostly blank chart. Also count strongly changed pixels.
        strong_difference = difference.max(axis=2) > 40
        significant_fraction = np.mean(strong_difference)
        components, count = label(strong_difference, structure=np.ones((3, 3), dtype=int))
        largest_change = (np.bincount(components.ravel())[1:].max()
                          if count else 0)
        if (max(abs(a - b) for a, b in zip(old_size, new_size)) > 3
                or difference.mean() > 3.0 or significant_fraction > 0.001
                or largest_change >= 30):
            raise RuntimeError(f"Published chart differs in {path.name}, cell {cell_number}")


def reproduce_notebook(path: Path, *, save: bool | None = None,
                       check: bool = False) -> nbformat.NotebookNode:
    if save is None:
        save = not check
    if check and save:
        raise ValueError("Notebook verification cannot save or overwrite files")
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    variables = {"__name__": "__main__"}
    execution_count = 0
    previous_directory = Path.cwd()
    previous_sys_path = sys.path.copy()
    existing_figures = set(plt.get_fignums())
    formatter = DisplayFormatter()
    compiler = codeop.Compile()
    try:
        # Relative data paths inside a notebook resolve beside that notebook,
        # regardless of where a caller invokes this helper.
        os.chdir(path.parent)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            original_outputs = list(cell.outputs)
            source = _remove_inline_magic(cell.source)
            outputs: list[nbformat.NotebookNode] = []

            def capture_display(*values: object) -> None:
                for value in values:
                    data, metadata = formatter.format(value)
                    if data:
                        outputs.append(nbformat.v4.new_output(
                            "display_data", data=data, metadata=metadata,
                        ))

            def capture_figures() -> None:
                for number in list(plt.get_fignums()):
                    if number in existing_figures:
                        continue
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

            variables["display"] = capture_display
            previous_show = plt.show
            try:
                try:
                    plt.show = lambda *args, **kwargs: capture_figures()
                    with (redirect_stdout(_StreamCapture("stdout", outputs)),
                          redirect_stderr(_StreamCapture("stderr", outputs))):
                        expression_result = _execute_cell(source, path, variables, compiler)
                        if expression_result is not None:
                            data, metadata = formatter.format(expression_result)
                            if data:
                                outputs.append(nbformat.v4.new_output(
                                    "execute_result", data=data, metadata=metadata,
                                    execution_count=execution_count,
                                ))
                        # Inline Matplotlib also publishes remaining open figures
                        # automatically at the end of a notebook cell.
                        capture_figures()
                finally:
                    plt.show = previous_show
            except Exception as error:
                raise RuntimeError(f"Notebook {path.name}, cell {execution_count}") from error
            if check:
                _verify_outputs(original_outputs, outputs, path, execution_count)
            cell.outputs = outputs
            cell.execution_count = execution_count
    finally:
        try:
            # Failed cells can leave figures open. Never capture or close
            # figures that belonged to the caller before this notebook ran.
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)
        finally:
            os.chdir(previous_directory)
            sys.path[:] = previous_sys_path
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
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        reproduce_notebook(path, save=not args.check, check=args.check)


if __name__ == "__main__":
    main()
