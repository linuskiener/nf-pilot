"""Read codebook items from the Markdown files in codebook/.

The Markdown tables are the single source of truth: coding sheets and LLM
prompts are built from them, so a wording change only happens in one place.
"""

import re
from pathlib import Path

CODEBOOK_DIR = Path(__file__).resolve().parents[2] / "codebook"

# A table row whose first cell is an item id such as "RESP1" or "HUM4".
ITEM_ROW = re.compile(r"^\|\s*([A-Z]+\d+[a-z]?)\s*\|\s*(.+?)\s*\|\s*$")


def load_items(name="generic_frames.md"):
    """Return ``{item_id: question}`` in codebook order."""
    items = {}
    for line in (CODEBOOK_DIR / name).read_text(encoding="utf-8").splitlines():
        match = ITEM_ROW.match(line)
        if match:
            items[match.group(1)] = match.group(2)
    return items
