"""Bundle ``explorer_data.json`` into a single self-contained HTML page.

    python -m exploratory.som.explorer exploratory/som/output/niusde_2025_H1

writes ``explorer.html`` next to the data. Open it in any browser; it needs
no server and loads only fonts from the web.

    python -m exploratory.som.explorer exploratory/som/output/niusde_2025_H1 --site docs

also writes the page to ``docs/atlas/index.html`` for GitHub Pages
(Settings -> Pages -> Deploy from a branch -> main, /docs).
"""

import argparse
from pathlib import Path

TEMPLATE = Path(__file__).parent / "templates" / "explorer.html"


def build(output_dir):
    output_dir = Path(output_dir)
    data = (output_dir / "explorer_data.json").read_text()
    # A literal "</script>" inside the JSON would end the script block early.
    data = data.replace("</", "<\\/")
    # Broken characters from the captions stay escaped, not raw U+FFFD.
    data = data.replace("\ufffd", "\\ufffd")
    html = TEMPLATE.read_text().replace("__DATA__", data)
    path = output_dir / "explorer.html"
    path.write_text(html)
    return path


PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
body{margin:0;font-size:14px}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def publish_site(page, site_dir, name="atlas"):
    """Copy a built page into a GitHub Pages folder as ``<name>/index.html``."""
    site_dir = Path(site_dir)
    target = site_dir / name / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    html = Path(page).read_text()
    # The template starts with its <title>, links and <style>; they go into
    # <head>, the rest into <body>.
    cut = html.index("</style>") + len("</style>")
    target.write_text(PAGE_HEAD + html[:cut] + "\n</head>\n<body>\n"
                      + html[cut:] + "\n</body>\n</html>\n")
    (site_dir / ".nojekyll").touch()  # serve files as they are
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--site", type=Path,
                        help="also write <site>/atlas/index.html for GitHub Pages")
    args = parser.parse_args()
    page = build(args.output_dir)
    print(page)
    if args.site:
        print(publish_site(page, args.site))


if __name__ == "__main__":
    main()
