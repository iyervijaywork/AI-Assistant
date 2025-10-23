"""Utility script to bundle the Live Conversation AI Assistant into a zip file."""
from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile
from typing import Iterable, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
ARCHIVE_NAME = "live-conversation-assistant"


INCLUDE_PATTERNS: Tuple[tuple[pathlib.Path, str], ...] = (
    (ROOT / "src", "src"),
    (ROOT / "requirements.txt", "requirements.txt"),
    (ROOT / "README.md", "README.md"),
    (ROOT / ".env.example", ".env.example"),
)


def _iter_existing(paths: Iterable[tuple[pathlib.Path, str]]):
    for source, arcname in paths:
        if source.exists():
            yield source, arcname
        else:
            print(f"[warn] Skipping missing path: {source}", file=sys.stderr)


def build_archive(include_pyinstaller: bool = False) -> pathlib.Path:
    """Create the zip archive and return its path."""
    DIST_DIR.mkdir(exist_ok=True)
    archive_path = DIST_DIR / f"{ARCHIVE_NAME}.zip"

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source, arcname in _iter_existing(INCLUDE_PATTERNS):
            if source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file():
                        zf.write(file, pathlib.Path(arcname) / file.relative_to(source))
            else:
                zf.write(source, arcname)

        if include_pyinstaller:
            bundle = ROOT / "dist" / "LiveConversationAssistant.app"
            if bundle.exists():
                for file in bundle.rglob("*"):
                    if file.is_file():
                        zf.write(file, pathlib.Path("LiveConversationAssistant.app") / file.relative_to(bundle))
            else:
                print(
                    "[warn] Requested PyInstaller bundle but dist/LiveConversationAssistant.app was not found. Run pyinstaller first.",
                    file=sys.stderr,
                )

    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-pyinstaller",
        action="store_true",
        help="Include the PyInstaller app bundle if it has already been built.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = build_archive(include_pyinstaller=args.include_pyinstaller)
    print(f"Created archive: {archive}")


if __name__ == "__main__":
    main()
