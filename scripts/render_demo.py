"""Render docs/demo.svg: a real terminal session against the bundled sample library.

uv run python scripts/render_demo.py
"""

from __future__ import annotations

import io
import shlex
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.terminal_theme import MONOKAI

import bookworm.cli as cli_module

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    "bookworm index examples/library",
    'bookworm search "a small golden key and a tiny door" -k 2',
    'bookworm ask "Why does Mrs. Bennet want her husband to call on Bingley?"',
]


def main() -> None:
    # Not a terminal, so progress bars stay out of the recording; colours are still kept.
    console = Console(
        record=True, width=104, file=io.StringIO(), force_terminal=False, color_system="truecolor"
    )
    cli_module.console = console
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "demo.duckdb"
        for command in COMMANDS:
            console.print(f"[bold green]$[/bold green] [bold]{command}[/bold]")
            args = shlex.split(command)[1:]
            if args[0] == "index":
                args[1] = str(ROOT / args[1])
            try:
                cli_module.cli.main(["--db", str(db), *args], standalone_mode=False)
            except SystemExit as exc:  # pragma: no cover - surfaced in the image
                print(f"command exited with {exc.code}", file=sys.stderr)
            console.print()
    svg = console.export_svg(title="bookworm", theme=MONOKAI)
    svg = svg.replace(str(ROOT) + "/", "")
    (ROOT / "docs" / "demo.svg").write_text(svg, encoding="utf-8")
    print("wrote docs/demo.svg")


if __name__ == "__main__":
    main()
