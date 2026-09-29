from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = ROOT / name
        actual = digest(path)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch: {name}: {actual} != {expected}")

    split_path = ROOT / "splits" / "slide_3fold.csv"
    rows = split_path.read_text(encoding="utf-8").splitlines()
    required = {"slide_0165", "slide_0166", "slide_0167"}
    observed = {slide for slide in required if any(slide in row for row in rows)}
    if observed != required:
        raise SystemExit(f"Incomplete slide assignment: {sorted(observed)}")
    print(f"PASS: artifact v{manifest['version']} ({len(manifest['sha256'])} files verified)")


if __name__ == "__main__":
    main()

