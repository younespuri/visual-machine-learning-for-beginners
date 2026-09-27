"""Build each lesson's notebook.ipynb from the ```python blocks in its README.

The README is the single source of truth: every python block becomes a code
cell, the heading above it becomes a markdown cell, and the bullet notes right
after it follow as markdown. Notebooks are executed so GitHub shows real output.

    python tools/build_notebooks.py            # build + execute every lesson
    python tools/build_notebooks.py 02 05      # only lessons starting with 02 or 05
    python tools/build_notebooks.py --verify   # execute in memory, write nothing (CI)
"""
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REPO = "younespuri/visual-machine-learning-for-beginners"
FENCE = re.compile(r"^```(\w*)\s*$")


def parse(readme: str):
    """Yield ('heading', text), ('code', src) and ('notes', md) in document order."""
    lines = readme.splitlines()
    i, n = 0, len(lines)
    in_code_section = False
    while i < n:
        line = lines[i]
        if line.startswith("## "):
            in_code_section = line.strip() == "## Code it"
        if in_code_section and line.startswith("### "):
            yield "heading", line[4:].strip()
        m = FENCE.match(line)
        if m:
            lang = m.group(1)
            j = i + 1
            while j < n and not lines[j].startswith("```"):
                j += 1
            if lang == "python" and in_code_section:
                yield "code", "\n".join(lines[i + 1:j])
                k = j + 1
                notes = []
                while k < n and (not lines[k].strip() or lines[k].startswith(("- ", "  "))):
                    notes.append(lines[k])
                    k += 1
                if any(s.strip() for s in notes):
                    yield "notes", "\n".join(notes).strip()
                i = k
                continue
            i = j + 1
            continue
        i += 1


def build(lesson_dir: Path):
    readme = (lesson_dir / "README.md").read_text(encoding="utf-8")
    title = next(l[2:].strip() for l in readme.splitlines() if l.startswith("# "))
    link = f"https://github.com/{REPO}/blob/main/lessons/{lesson_dir.name}/README.md"
    cells = [nbformat.v4.new_markdown_cell(
        f"# {title}\n\nCompanion notebook for the [lesson]({link}). "
        "Run the cells from top to bottom with **Shift + Enter**.")]
    for kind, text in parse(readme):
        if kind == "heading":
            cells.append(nbformat.v4.new_markdown_cell(f"### {text}"))
        elif kind == "code":
            cells.append(nbformat.v4.new_code_cell(text))
        else:
            cells.append(nbformat.v4.new_markdown_cell(text))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}})
    return nb


def execute(nb, cwd: Path):
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(cwd)}}).execute()
    return nb


def main(argv):
    verify = "--verify" in argv
    prefixes = [a for a in argv if not a.startswith("--")]
    lessons = sorted(p for p in (ROOT / "lessons").iterdir()
                     if (p / "README.md").exists() and (not prefixes or p.name.startswith(tuple(prefixes))))
    failed = []
    for lesson in lessons:
        nb = build(lesson)
        n_code = sum(c.cell_type == "code" for c in nb.cells)
        if n_code == 0:
            print(f"-  {lesson.name}: no code cells, skipped")
            continue
        try:
            execute(nb, lesson)
        except Exception as e:  # noqa: BLE001 - report which lesson broke, then keep going
            failed.append(lesson.name)
            print(f"X  {lesson.name}: {type(e).__name__}: {str(e).splitlines()[-1][:200]}")
            continue
        if not verify:
            nbformat.write(nb, lesson / "notebook.ipynb")
        print(f"OK {lesson.name}: {n_code} code cells ran")
    if failed:
        sys.exit(f"{len(failed)} notebook(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main(sys.argv[1:])
