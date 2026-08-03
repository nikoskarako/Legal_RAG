#!/usr/bin/env python3
"""
Manual QA Pairs Runner (hardcoded doc IDs)
-----------------------------------------
Run QA generation for a fixed list of document IDs, skipping clustering. Configuration is read from .env.

Usage:
  python QA_Generation_Manual.py
"""

import argparse
import logging
import time
import sys
import os
from dotenv import load_dotenv
from typing import List

import pandas as pd  # for shuffling

# ---------------------------
# Dependencies (required)
# ---------------------------
# - clustering_qa_harvester_openai_10  or  clustering_qa_harvester:
#     Provides functions:
#       fetch_document_texts(db_uri, text_table, doc_ids)
#       dynamic_generate_qa_pairs_for_document(doc)
#       export_qa_pairs_to_jsonl(records, out_path)
#       QAPairList (class for QA storage)
#
# - Database access:
#     Requires psycopg2 (for Postgres) installed and working DB connection.
#
# - OpenAI API (used inside clustering_qa_harvester_openai_10):
#     Requires environment variable OPENAI_API_KEY set.
#     Model is defined inside clustering_qa_harvester_openai_10 (e.g., gpt-4o).
#
# - dotenv (.env file):
#     Contains DB_URI or PG_* variables and optional parameters.
# ---------------------------

# --- Load configuration from .env ---
load_dotenv()
# Prefer full DB_URI; otherwise construct from PG_* pieces
_RAW_DB_URI = (os.getenv("DB_URI") or "").strip()
if _RAW_DB_URI:
    DB_URI = _RAW_DB_URI
else:
    PG_DATABASE = os.getenv("PG_DATABASE", "").strip()
    PG_HOST = os.getenv("PG_HOST", "localhost").strip()
    PG_PORT = os.getenv("PG_PORT", "5432").strip()
    PG_USER = os.getenv("PG_USER", "").strip()
    PG_PASSWORD = (os.getenv("PG_PASSWORD", "") or "").strip()
    if PG_DATABASE and PG_USER:
        auth = PG_USER if PG_PASSWORD == "" else f"{PG_USER}:{PG_PASSWORD}"
        DB_URI = f"postgresql://{auth}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
    else:
        DB_URI = None
TEXT_TABLE = os.getenv("TEXT_TABLE", "laws_text_table")
_TP_ENV = os.getenv("TARGET_PAIRS", "10")
_OUT_ENV = os.getenv("OUTPUT_PATH", "qa_pairs_openai_manual.jsonl")
_SLEEP_ENV = os.getenv("SLEEP_BETWEEN_DOCS", "1.0")
_SEED_ENV = os.getenv("SEED", "")
TARGET_PAIRS_DEFAULT = int(_TP_ENV) if _TP_ENV else 10
OUTPUT_DEFAULT = _OUT_ENV or "qa_pairs_openai_manual.jsonl"
SLEEP_DEFAULT = float(_SLEEP_ENV) if _SLEEP_ENV else 1.0
SEED_DEFAULT = int(_SEED_ENV) if _SEED_ENV.strip() else None

# -------- Manual flag (easy toggle) --------
# Set to True → generate answers normally
# Set to False → questions only, answers = "---"
GENERATE_ANSWER = False
# -------------------------------------------

# ---------- EDIT THESE DOC IDS ----------
MANUAL_DOC_IDS: List[str] = [
    "doc-311",
    "doc-312",
    "doc-313",
    "doc-384",
    "doc-389",
    "doc-83",
    "doc-84",
    "doc-85"
]
# ----------------------------------------

log = logging.getLogger("manual-qa")
def dbg(msg: str): log.debug(msg)
def warn(msg: str): log.warning(msg)
def err(msg: str): log.error(msg)

_fetch = _gen = _export = _QAPairList = None
def _try_imports():
    global _fetch, _gen, _export, _QAPairList
    tried = []
    try:
        from clustering_qa_harvester_openai_10 import (  # type: ignore
            fetch_document_texts,
            dynamic_generate_qa_pairs_for_document,
            export_qa_pairs_to_jsonl,
            QAPairList,
        )
        _fetch = fetch_document_texts
        _gen = dynamic_generate_qa_pairs_for_document
        _export = export_qa_pairs_to_jsonl
        _QAPairList = QAPairList
        return
    except Exception as e:
        tried.append(("clustering_qa_harvester_openai_10", e))
    try:
        from clustering_qa_harvester import (  # type: ignore
            fetch_document_texts,
            dynamic_generate_qa_pairs_for_document,
            export_qa_pairs_to_jsonl,
            QAPairList,
        )
        _fetch = fetch_document_texts
        _gen = dynamic_generate_qa_pairs_for_document
        _export = export_qa_pairs_to_jsonl
        _QAPairList = QAPairList
        return
    except Exception as e:
        tried.append(("clustering_qa_harvester", e))
    # Could not import required pipeline functions. Tried:
    #   - clustering_qa_harvester_openai_10
    #   - clustering_qa_harvester
    raise ImportError("Required pipeline functions not found. Please install clustering_qa_harvester_openai_10 or clustering_qa_harvester.")

def run_manual(
    db_uri: str,
    text_table: str,
    target_pairs: int,
    output_path: str,
    sleep_between_docs: float = 1.0,
    seed: int | None = None,
) -> None:
    if not db_uri:
        raise RuntimeError("Database connection is not configured. Set DB_URI or provide PG_DATABASE, PG_HOST, PG_PORT, PG_USER (and optional PG_PASSWORD) in your .env.")
    _try_imports()

    rep_ids = [str(x) for x in MANUAL_DOC_IDS]
    if not rep_ids:
        warn("MANUAL_DOC_IDS is empty — nothing to process.")
        return

    dbg(f"Requested manual doc_ids (n={len(rep_ids)}).")

    id_to_text = _fetch(db_uri, text_table=text_table, doc_ids=rep_ids)
    print(f"[DIAG] (manual) fetched_texts={len(id_to_text)} / requested={len(rep_ids)}")

    missing = [rid for rid in rep_ids if rid not in id_to_text or not id_to_text[rid]]
    if missing:
        warn(f"(manual) Missing texts for {len(missing)} provided ids; sample={missing[:10]}")

    order = pd.Series(rep_ids).sample(frac=1, random_state=seed).tolist()

    qa_records = []
    total_pairs = 0

    for idx, doc_id in enumerate(order, start=1):
        dbg(f"(manual) Progress: {idx}/{len(order)} docs processed (remaining_pairs=unknown)")
        doc_text = id_to_text.get(doc_id, "")
        dbg(f"(manual) QA for doc_id={doc_id}: text_len={len(doc_text) if doc_text else 0}")

        if not doc_text:
            warn(f"(manual) Skipping QA (no text) for doc_id={doc_id}")
            qa_records.append({"source_file": doc_id, "qa_list": _QAPairList(qa_pairs=[])})
            time.sleep(sleep_between_docs)
            continue

        try:
            doc = {"document_text": doc_text, "law_metadata": {}, "source_file": doc_id}
            qa_list = _gen(doc)
            if not GENERATE_ANSWER:
                # Preserve generated questions (and metadata if present), blank out answers
                new_pairs = []
                for p in qa_list.qa_pairs:
                    if isinstance(p, dict):
                        q = p.get("question", "---")
                        meta = p.get("metadata", {})
                    else:
                        q = getattr(p, "question", "---")
                        meta = getattr(p, "metadata", {})
                    new_pairs.append({
                        "question": q or "---",
                        "metadata": meta or {},
                        "answer": "---",
                    })
                qa_list = _QAPairList(qa_pairs=new_pairs)

            # Always keep exactly 3 QA pairs per document (if available)
            qa_list.qa_pairs = qa_list.qa_pairs[:3]
            n_pairs = len(qa_list.qa_pairs)
            total_pairs += n_pairs

            dbg(f"(manual) doc_id={doc_id}: produced pairs={n_pairs}; total_so_far={total_pairs}")
            if n_pairs == 0:
                warn(f"(manual) LLM returned 0 pairs for doc_id={doc_id}")

            qa_records.append({"source_file": doc_id, "qa_list": qa_list})
        except Exception as e:
            err(f"(manual) Exception generating QA for doc_id={doc_id}: {e}")
            qa_records.append({"source_file": doc_id, "qa_list": _QAPairList(qa_pairs=[])})

        time.sleep(sleep_between_docs)

    _export(qa_records, output_path)
    print(f"\n✅ (manual) QA exported for {len(qa_records)} docs; total pairs={total_pairs} (target={target_pairs}).")
    print("Manual mode active: skipping visualization and cluster exports.")

def parse_args():
    p = argparse.ArgumentParser(description="Run manual QA generation for hardcoded document IDs.")
    p.add_argument("--target-pairs", type=int, default=TARGET_PAIRS_DEFAULT, help="Total QA pairs to generate (cap).")
    p.add_argument("--output", default=OUTPUT_DEFAULT, help="Output JSONL path.")
    p.add_argument("--sleep", type=float, default=SLEEP_DEFAULT, help="Seconds to sleep between docs.")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Optional seed for reproducible shuffling.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    run_manual(
        db_uri=DB_URI,
        text_table=TEXT_TABLE,
        target_pairs=args.target_pairs,
        output_path=args.output,
        sleep_between_docs=args.sleep,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()