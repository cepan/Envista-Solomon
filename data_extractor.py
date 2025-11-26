from __future__ import annotations

import shutil
import sys
from pathlib import Path


def build_target_path(image_path: Path, captures_root: Path, destination_root: Path) -> Path:
    """Build destination filename by joining relative parts with underscores."""
    relative_parts = image_path.relative_to(captures_root).parts
    filename = "_".join(relative_parts)
    return destination_root / filename


def ensure_unique_path(target_path: Path) -> Path:
    """If the target path exists, append an incrementing suffix to avoid collisions."""
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    captures_root = base_dir / "captures"
    destination_root = base_dir / "captures_extracted"
    patterns = (
        "step-01_top_raw.png",
        "step-02_top_*.png",
    )

    if not captures_root.exists():
        print(f"Source folder not found: {captures_root}", file=sys.stderr)
        return 1

    destination_root.mkdir(exist_ok=True)

    copied_files = 0
    for pattern in patterns:
        for image_path in captures_root.rglob(pattern):
            target_path = build_target_path(image_path, captures_root, destination_root)
            target_path = ensure_unique_path(target_path)
            shutil.copy2(image_path, target_path)
            copied_files += 1

    print(f"Copied {copied_files} files into {destination_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
