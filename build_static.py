"""Build the static GitHub Pages version of the web app into site/.

Same page and same Python game code as the Flask app; Pyodide runs that Python in
the visitor's browser instead of on a server. Run `python build_static.py`, then
preview with `python -m http.server -d site`. The GitHub Actions workflow in
.github/workflows/pages.yml runs this on every push to main and publishes site/.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.js"
PLACEHOLDER = "<!-- static-build: scripts -->"

# Everything the browser needs to import webapp.api and webapp.game (no Flask).
PY_SOURCES = ["tictactoe/**/*.py", "webapp/__init__.py", "webapp/api.py", "webapp/game.py", "saved_models/*.pkl"]


def main():
    shutil.rmtree(SITE, ignore_errors=True)
    SITE.mkdir()

    files = sorted({p.relative_to(ROOT).as_posix() for pattern in PY_SOURCES for p in ROOT.glob(pattern)})
    for rel in files:
        dest = SITE / "py" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    (SITE / "py" / "files.json").write_text(json.dumps(files))

    html = (ROOT / "webapp" / "templates" / "index.html").read_text()
    if PLACEHOLDER not in html:
        raise SystemExit(f"index.html is missing the {PLACEHOLDER!r} marker")
    scripts = f'<script src="{PYODIDE_URL}"></script>\n<script src="pyodide_api.js"></script>'
    (SITE / "index.html").write_text(html.replace(PLACEHOLDER, scripts))
    shutil.copy2(ROOT / "static" / "pyodide_api.js", SITE / "pyodide_api.js")
    (SITE / ".nojekyll").touch()  # serve files as-is, skip GitHub's Jekyll processing

    print(f"Built {SITE.relative_to(ROOT)}/ ({len(files)} Python/model files)")


if __name__ == "__main__":
    main()
