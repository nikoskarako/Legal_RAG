"""Shared project paths and sibling-module imports.

The scripts in this repository were originally written as a flat pile of files
that all ran from the project root. They are now grouped by pipeline stage
(``src/indexing``, ``src/rag``, ``src/retrieval``, ...), so two things have to
be re-established: where the data lives, and how a script in one stage imports
a module from another.

Import this module first in any script that needs either:

    import paths                        # registers sibling stages on sys.path
    from paths import DATASETS_DIR      # ...and gives absolute data locations

After importing it, the historical flat imports keep working unchanged, e.g.
``import chat_openrouter`` from a script that lives in ``src/retrieval``.
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))

# Pipeline stages. Adding each to sys.path keeps the original flat imports
# (``import chat_openrouter``, ``from run_bm25_eval import retrieve_bm25``)
# working now that the files sit in different directories.
_STAGES = ("indexing", "qa_generation", "rag", "api", "retrieval", "evaluation", "analysis")

for _stage in _STAGES:
    _stage_dir = os.path.join(SRC_DIR, _stage)
    if os.path.isdir(_stage_dir) and _stage_dir not in sys.path:
        sys.path.insert(0, _stage_dir)

# --- Data locations -------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
QA_PAIRS_DIR = os.path.join(DATA_DIR, "qa_pairs")
CLUSTERS_DIR = os.path.join(DATA_DIR, "clusters")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Per-system retrieval datasets (one JSON per evaluated question).
DENSE_DATASET = os.path.join(DATASETS_DIR, "RAGAs_Dataset")
BM25_DATASET = os.path.join(DATASETS_DIR, "BM25_Dataset")
HYBRID_DATASET = os.path.join(DATASETS_DIR, "Hybrid_Dataset")
BASELINE_DATASET = os.path.join(DATASETS_DIR, "RAGAS_baseline_dataset")

# The human-reviewed question bank, and the accepted-question CSV derived from
# it that the retrieval experiments read.
QA_REVIEW_JSON = os.path.join(QA_PAIRS_DIR, "qa_review.json")
QUESTIONS_CSV = os.path.join(QA_PAIRS_DIR, "Questions_for_Evaluation.csv")

# Large artefacts that are NOT in the repository (see README): the SQLite
# harvester dump and the persisted LlamaIndex storage directory. Override
# either with an environment variable.
HARVESTER_DB = os.getenv("HARVESTER_DB", os.path.join(PROJECT_ROOT, "harvester_full.db"))
STORAGE_DIR = os.getenv("STORAGE_DIR", os.path.join(PROJECT_ROOT, "storage"))
