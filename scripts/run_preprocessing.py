"""CLI orchestrator for the MSMD preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.msmd_preprocess.config import load_config, resolve_paths
from src.msmd_preprocess.core import run_core
from src.msmd_preprocess.dataset import exclude_no_mung, load_metadata
from src.msmd_preprocess.task1 import run_task1
from src.msmd_preprocess.task2 import run_task2
from src.msmd_preprocess.validate import validate_preprocessing


def configure_logging(config: dict) -> Path:
    paths = resolve_paths(config)
    log_dir = paths["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MSMD preprocessing runner")
    parser.add_argument(
        "--stage",
        default="core",
        choices=["core", "task1", "task2", "all", "verify"],
        help="Stage to run",
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--force", action="store_true", help="Reserved for cache invalidation")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing artifacts without loading raw data",
    )
    args = parser.parse_args()
    if args.verify_only:
        args.stage = "verify"

    config = load_config(args.config)
    paths = resolve_paths(config)
    if args.dry_run:
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return

    log_path = configure_logging(config)
    logger = logging.getLogger("run_preprocessing")
    logger.info("Config: %s", config["_config_path"])
    logger.info("Log: %s", log_path)

    if args.stage == "verify":
        report = validate_preprocessing(config)
        print(json.dumps(report, indent=2, default=str))
        if not report["ok"]:
            raise SystemExit(1)
        logger.info("Validation passed")
        return

    pieces, perfs, _ = load_metadata(config)
    pieces, perfs, excluded = exclude_no_mung(
        pieces, perfs, paths["raw_dir"]
    )
    logger.info(
        "Metadata: pieces=%d performances=%d excluded_no_mung=%s",
        len(pieces),
        len(perfs),
        excluded,
    )

    if args.stage in {"core", "all"}:
        summary = run_core(config, pieces, perfs, excluded)
        logger.info("Core summary: %s", json.dumps(summary, indent=2))
    if args.stage in {"task1", "all"}:
        summary = run_task1(config)
        logger.info("Task 1 summary: %s", json.dumps(summary, indent=2))
    if args.stage in {"task2", "all"}:
        summary = run_task2(config)
        logger.info("Task 2 summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
