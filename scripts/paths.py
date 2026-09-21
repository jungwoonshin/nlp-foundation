"""Repo-root paths for scripts nested under scripts/<model>/<stage>/."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
