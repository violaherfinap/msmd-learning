"""Build GitHub Release zip bundles for preprocessing artifacts."""

from __future__ import annotations

import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.msmd_preprocess.bundles import artifact_groups
from src.msmd_preprocess.config import load_config
from src.msmd_preprocess.io import sha256_file
from src.msmd_preprocess.paths import relative_posix

LOGGER = logging.getLogger(__name__)
BUNDLES = {
    "core": "preproc-v1-core.zip",
    "task1": "preproc-v1-task1.zip",
    "task2": "preproc-v1-task2.zip",
    "meta": "preproc-v1-meta.zip",
}


def build_zip(output: Path, paths: list[Path]) -> None:
    tmp = output.with_suffix(output.suffix + ".part")
    with zipfile.ZipFile(
        tmp,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for path in paths:
            archive.write(path, arcname=relative_posix(path, ROOT))
    os.replace(tmp, output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config: dict[str, Any] = load_config()
    dist = ROOT / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    groups = artifact_groups(config)

    manifest = ROOT / config["processed_dir"] / "manifest_preprocessing.json"
    if not manifest.exists():
        raise SystemExit("Run scripts/build_manifest.py before building release bundles")

    records = []
    for stage, filename in BUNDLES.items():
        paths = groups.get(stage, [])
        if not paths:
            LOGGER.warning("Skipping empty bundle: %s", stage)
            continue
        output = dist / filename
        LOGGER.info("Building %s (%d files)", output, len(paths))
        build_zip(output, paths)
        digest = sha256_file(output)
        records.append(
            {
                "asset": filename,
                "bytes": int(output.stat().st_size),
                "sha256": digest,
            }
        )
        print(f"{filename}: {output.stat().st_size / 1e9:.3f} GB")

    sums = "\n".join(f"{record['sha256']}  {record['asset']}" for record in records)
    (dist / "SHA256SUMS.txt").write_text(sums + "\n", encoding="utf-8")
    (dist / "release_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
