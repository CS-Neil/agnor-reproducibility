from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    files = [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "reproducibility_v1.0.0.zip", "reproducibility_v1.0.1.zip"}
        and not path.name.startswith("._")
        and path.name != ".DS_Store"
        and "__pycache__" not in path.parts
    ]
    manifest = {
        "artifact": "agnor-lopo-reproducibility",
        "version": "1.0.1",
        "created": "2026-08-13T17:25:00+08:00",
        "documentation_revised": "2026-09-20",
        "revision_scope": "Environment-provenance documentation only; frozen results unchanged",
        "distribution": "GitHub release v1.1.0",
        "training_performed": False,
        "formal_protocol": (
            "strict 3-fold leave-one-patient/slide-out, seed 42, "
            "validation-only checkpoint and threshold selection"
        ),
        "test_oracle_policy": "diagnostic only; never used for formal results",
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
