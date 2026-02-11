"""Inlocuieste blocul de note in limba engleza cu varianta in romana."""

from __future__ import annotations

from pathlib import Path

REPLACEMENTS = {
    "# Notes:\n": "# Nota:\n",
    "# Note:\n": "# Nota:\n",
    "# - Uses repo-relative paths defined in this module.\n": "# - Foloseste path-uri relative la repo definite in acest modul.\n",
    "# - Intended for use within the project pipeline.\n": "# - Este destinat folosirii in pipeline-ul proiectului.\n",
    "# - Outputs artifacts to project folders when applicable.\n": "# - Genereaza artefacte in folderele proiectului cand este cazul.\n",
    "# - Assumes inputs follow schema from data/README.md (when applicable).\n": "# - Presupune schema de intrare din data/README.md (cand este cazul).\n",
    "# - Determinism is applied when a seed is defined.\n": "# - Determinismul este aplicat cand exista un seed definit.\n",
    "# - Keeps console output minimal for clarity.\n": "# - Pastreaza output-ul in consola minim pentru claritate.\n",
}


def replace_block(text: str) -> str:
    """Aplica inlocuirile pentru blocul de note."""
    updated = text
    for old, new in REPLACEMENTS.items():
        updated = updated.replace(old, new)
    return updated


def main() -> None:
    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        updated = replace_block(text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
