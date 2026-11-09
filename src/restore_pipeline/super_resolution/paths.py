"""Czytelne opisy ścieżek bez lokalnej lokalizacji repozytorium."""

from pathlib import Path


def display_path(path: Path) -> str:
    """Zwróć ścieżkę względem projektu, a dla pliku spoza niego samą nazwę."""

    project_root = Path(__file__).resolve().parents[3]
    resolved = path.resolve()

    if resolved.is_relative_to(project_root):
        return resolved.relative_to(project_root).as_posix()

    return resolved.name
