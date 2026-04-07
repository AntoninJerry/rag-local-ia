from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def scan_documents(source_dir: Path, supported_extensions: set[str] | None = None) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    extensions = supported_extensions or SUPPORTED_EXTENSIONS
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
