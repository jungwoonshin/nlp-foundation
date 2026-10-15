# Codex preferences

## Commits

- Do not create commits unless the user explicitly requests a commit for the current task.
- Earlier requests about commit formatting or committing previous work do not authorize commits for later tasks.
- Leave edits uncommitted when the current request does not explicitly ask for a commit.

## Temporary files

- Store agent-created temporary files, scratch scripts, intermediate outputs, render previews, and task-specific caches under `.tmp/` at the workspace root.
- Use descriptive task subfolders when needed. Do not create `tmp/`, `temp/`, or `.temp/` for temporary work.
- Keep final deliverables in their appropriate project locations.
