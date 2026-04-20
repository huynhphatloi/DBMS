"""Phase 1: Data Pipeline End-to-End Audit

Runs the full data pipeline:
  1. Load all 4 source datasets
  2. Harmonize labels
  3. Assign / verify splits
  4. Run data quality audit
  5. Save unified JSONL files to data/unified/
  6. Print audit findings and save audit report to results/phase1/

Usage:
    python experiments/phase1_data_audit.py

Config is read from configs/data.yaml.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audit import DataAuditor
from src.data.harmonizer import LabelHarmonizer
from src.data.loaders import (
    load_data_generate,
    load_data_scraping,
    load_mohler,
    load_scientsbank,
)
from src.data.schema import UnifiedRecord
from src.data.splitter import SplitIntegrityError, SplitManager
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1_data_audit")

CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def record_to_dict(rec: UnifiedRecord) -> dict:
    """Convert a UnifiedRecord to a JSON-serialisable dict."""
    return dataclasses.asdict(rec)


def save_jsonl(records: list[UnifiedRecord], output_path: Path) -> None:
    """Write records to a JSONL file (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(record_to_dict(rec), ensure_ascii=False) + "\n")
    logger.info("Saved %d records -> %s", len(records), output_path)


def print_section(title: str) -> None:
    bar = "=" * 70
    logger.info(bar)
    logger.info("  %s", title)
    logger.info(bar)


def main() -> None:
    # 0. Load config
    if not CONFIG_PATH.exists():
        logger.error("Config file not found: %s", CONFIG_PATH)
        sys.exit(1)

    cfg = load_config(CONFIG_PATH)
    seed: int = cfg.get("seed", 42)
    set_seed(seed)
    logger.info("Loaded config from %s  (seed=%d)", CONFIG_PATH, seed)

    raw_cfg: dict = cfg.get("raw_data", {})
    unified_dir = PROJECT_ROOT / cfg.get("unified_output_dir", "data/unified")
    results_dir = PROJECT_ROOT / "results" / "phase1"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    print_section("STEP 1 — Loading raw datasets")

    all_records: list[UnifiedRecord] = []
    source_records: dict[str, list[UnifiedRecord]] = {}

    # SciEntsBank
    seb_path = PROJECT_ROOT / raw_cfg.get("scientsbank", {}).get("path", "data/raw/scientsbank")
    logger.info("Loading SciEntsBank from %s ...", seb_path)
    seb_records = load_scientsbank(seb_path)
    if seb_records:
        source_records["scientsbank"] = seb_records
        all_records.extend(seb_records)
        logger.info("  -> %d SciEntsBank records loaded", len(seb_records))
    else:
        logger.warning("  -> No SciEntsBank records loaded (raw data missing or empty)")

    # MohlerASAG
    moh_path = PROJECT_ROOT / raw_cfg.get("mohler", {}).get("path", "data/raw/mohler")
    logger.info("Loading MohlerASAG from %s ...", moh_path)
    moh_records = load_mohler(moh_path)
    if moh_records:
        source_records["mohler"] = moh_records
        all_records.extend(moh_records)
        logger.info("  -> %d MohlerASAG records loaded", len(moh_records))
    else:
        logger.warning("  -> No MohlerASAG records loaded (raw data missing or empty)")

    # Data_Generate
    gen_path = PROJECT_ROOT / raw_cfg.get("data_generate", {}).get("path", "data-generate.csv")
    logger.info("Loading Data_Generate from %s ...", gen_path)
    gen_records = load_data_generate(gen_path)
    if gen_records:
        source_records["data_generate"] = gen_records
        all_records.extend(gen_records)
        logger.info("  -> %d Data_Generate records loaded", len(gen_records))
    else:
        logger.warning("  -> No Data_Generate records loaded (file missing or empty)")

    # Data_Scraping
    scr_path = PROJECT_ROOT / raw_cfg.get("data_scraping", {}).get("path", "data-scraping.json")
    logger.info("Loading Data_Scraping from %s ...", scr_path)
    scr_records = load_data_scraping(scr_path)
    if scr_records:
        source_records["data_scraping"] = scr_records
        all_records.extend(scr_records)
        logger.info("  -> %d Data_Scraping records loaded", len(scr_records))
    else:
        logger.warning("  -> No Data_Scraping records loaded (file missing or empty)")

    logger.info("Total records loaded: %d", len(all_records))

    if not all_records:
        logger.error("No records loaded from any source. Exiting.")
        sys.exit(1)

    # 2. Harmonize labels
    print_section("STEP 2 — Harmonizing labels")

    harm_cfg = cfg.get("harmonization", {})
    threshold_2way: float = harm_cfg.get("mohler_2way_threshold", 2.5)
    harmonizer = LabelHarmonizer(threshold_2way=threshold_2way)
    harmonizer.harmonize_all(all_records)
    logger.info("Label harmonization complete for %d records", len(all_records))

    # 3. Assign / verify splits
    print_section("STEP 3 — Checking splits")

    split_manager = SplitManager(seed=seed)
    try:
        split_manager.assign_splits(all_records)
        logger.info("Split assignment / verification passed")
    except SplitIntegrityError as e:
        logger.error("Split integrity violation: %s", e)
        logger.warning("Continuing despite split integrity error (see above)")

    # 4. Run data quality audit
    print_section("STEP 4 — Running data quality audit")

    audit_report = DataAuditor.full_audit(all_records)
    logger.info("Audit complete")

    # 5. Save unified JSONL files
    print_section("STEP 5 — Saving unified JSONL files")

    source_to_filename = {
        "scientsbank": "scientsbank.jsonl",
        "mohler": "mohler.jsonl",
        "data_generate": "data_generate.jsonl",
        "data_scraping": "data_scraping.jsonl",
    }

    saved_files: dict[str, int] = {}
    for source, filename in source_to_filename.items():
        recs = source_records.get(source, [])
        out_path = unified_dir / filename
        if recs:
            save_jsonl(recs, out_path)
            saved_files[source] = len(recs)
        else:
            logger.warning("Skipping %s — no records available", filename)
            saved_files[source] = 0

    # 6. Print and save audit findings
    print_section("STEP 6 — Audit findings")

    # 6a. Label distributions
    logger.info("--- Label Distributions ---")
    label_dist_report: list[dict] = []
    for dist in audit_report.label_distributions:
        logger.info("Source: %s", dist.source_dataset)
        if dist.label_5way:
            logger.info("  label_5way counts : %s", dist.label_5way)
            logger.info("  label_5way pct    : %s",
                        {k: f"{v:.1f}%" for k, v in dist.label_5way_pct.items()})
        if dist.label_3way:
            logger.info("  label_3way counts : %s", dist.label_3way)
            logger.info("  label_3way pct    : %s",
                        {k: f"{v:.1f}%" for k, v in dist.label_3way_pct.items()})
        if dist.label_2way:
            logger.info("  label_2way counts : %s", dist.label_2way)
            logger.info("  label_2way pct    : %s",
                        {k: f"{v:.1f}%" for k, v in dist.label_2way_pct.items()})
        label_dist_report.append({
            "source_dataset": dist.source_dataset,
            "label_5way": dist.label_5way,
            "label_5way_pct": dist.label_5way_pct,
            "label_3way": dist.label_3way,
            "label_3way_pct": dist.label_3way_pct,
            "label_2way": dist.label_2way,
            "label_2way_pct": dist.label_2way_pct,
        })

    # 6b. Low-confidence records
    low_conf = audit_report.low_confidence_records
    logger.info("--- Low-Confidence Records (annotation_confidence < 0.85) ---")
    logger.info("  Count: %d", len(low_conf))
    if low_conf:
        logger.info("  Sample IDs (first 10): %s", [r.sample_id for r in low_conf[:10]])

    # 6c. Data_Scraping "Not found" reference answers
    not_found = audit_report.not_found_reference_records
    logger.info("--- Data_Scraping 'Not found' reference answers ---")
    logger.info("  Count: %d", len(not_found))
    if not_found:
        logger.info("  Sample IDs (first 10): %s", [r.sample_id for r in not_found[:10]])

    # 6d. Short student answers
    short_answers = audit_report.short_answer_records
    logger.info("--- Short Student Answers (fewer than 3 tokens) ---")
    logger.info("  Count: %d", len(short_answers))
    if short_answers:
        logger.info("  Sample IDs (first 10): %s", [r.sample_id for r in short_answers[:10]])

    # 6e. Numerical vs conceptual questions (Data_Scraping)
    logger.info("--- Data_Scraping Question Types ---")
    logger.info("  Numerical/computational questions : %d", audit_report.numerical_question_count)
    logger.info("  Conceptual questions              : %d", audit_report.conceptual_question_count)

    # 7. Save audit report JSON
    print_section("STEP 7 — Saving audit report")

    audit_json: dict = {
        "seed": seed,
        "total_records": len(all_records),
        "records_per_source": {src: len(recs) for src, recs in source_records.items()},
        "saved_jsonl_files": saved_files,
        "label_distributions": label_dist_report,
        "low_confidence_count": len(low_conf),
        "low_confidence_sample_ids": [r.sample_id for r in low_conf],
        "not_found_reference_count": len(not_found),
        "not_found_reference_sample_ids": [r.sample_id for r in not_found],
        "short_answer_count": len(short_answers),
        "short_answer_sample_ids": [r.sample_id for r in short_answers],
        "data_scraping_numerical_questions": audit_report.numerical_question_count,
        "data_scraping_conceptual_questions": audit_report.conceptual_question_count,
    }

    report_path = results_dir / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2, ensure_ascii=False)
    logger.info("Audit report saved -> %s", report_path)

    print_section("DONE")
    logger.info("Phase 1 data audit complete.")
    logger.info("  Unified JSONL files : %s", unified_dir)
    logger.info("  Audit report        : %s", report_path)


if __name__ == "__main__":
    main()
