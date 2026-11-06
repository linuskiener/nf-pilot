"""Bundle ``explorer_data.json`` into a single self-contained HTML page.

    python -m exploratory.som.explorer exploratory/som/output/niusde_2025_H1

writes ``explorer.html`` next to the data. Open it in any browser; it needs
no server and loads only fonts from the web.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", type=Path)
    print(build(parser.parse_args().output_dir))


if __name__ == "__main__":
    main()
