"""Run scripts/<model>/<stage>/run.py as __main__."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_ROOT.parent
STAGES = ("prepare", "train", "eval")


def available_models(stage: str) -> list[str]:
    models: list[str] = []
    for path in sorted(SCRIPTS_ROOT.iterdir()):
        if path.name.startswith("_") or not path.is_dir():
            continue
        if (path / stage / "run.py").is_file():
            models.append(path.name)
    return models


def script_path(model: str, stage: str) -> Path:
    return SCRIPTS_ROOT / model / stage / "run.py"


def main(stage: str, argv: list[str] | None = None) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")

    argv = list(sys.argv[1:] if argv is None else argv)
    models = available_models(stage)
    if not argv or argv[0] in {"-h", "--help"}:
        joined = ", ".join(models) or "(none)"
        extra = "  python prepare.py all\n" if stage == "prepare" else ""
        print(f"Usage: python {stage}.py <model> [args...]")
        print(extra, end="")
        print(f"Models: {joined}")
        return

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    selected = argv[0]
    extra = argv[1:]
    targets = models if selected == "all" and stage == "prepare" else [selected]
    if selected == "all" and stage != "prepare":
        raise SystemExit(f"'all' is only valid for prepare, not {stage}.")

    for model in targets:
        target = script_path(model, stage)
        if not target.is_file():
            joined = ", ".join(models) or "(none)"
            raise SystemExit(f"No {stage} script for {model!r}. Available: {joined}")
        sys.argv = [str(target), *extra]
        runpy.run_path(str(target), run_name="__main__")
