#!/usr/bin/env python3
"""
Batch RAGAS Export from qa_review.json (Legal_Chatbot)
-----------------------------------------------------

Goal
- Produce ONE RAGAS-style JSON file per *accepted* question found in qa_review.json.
- Each output file is named after the question id (e.g., "2.json", "15.json").

How it works
1) Loads qa_review.json and filters questions by review_status in {"accept", "accepted"}.
2) Loads the PGVector-backed LlamaIndex index ONCE (fast + RAM-friendly).
3) For each accepted question:
   - calls process_query() from chat_cli_ragas_builder.py
   - writes a JSON file containing exactly what RAGAS needs:
       {
         "id": "<question_id>",
         "question": "...",
         "answer": "...",
         "context": {"1": "...", "2": "...", ...},
         "doc_ids": ["doc-...", "doc-...", ...]
       }

Important
- This script imports chat_cli_ragas_builder.py, so it will use:
  - your VECTOR_TABLE / PG_* env vars
  - your dynamic chunk selection thresholds
  - your OpenRouter model + fallbacks
- You can control retrieval/reranking via env vars (recommended) without editing code:
  SIMILARITY_TOP_K, RERANK_CANDIDATES, RERANK_MIN_CHUNKS, RERANK_MAX_CHUNKS, etc.

Usage
python batch_ragas_export.py

# or override paths
python batch_ragas_export.py \
  --qa-review QA_Pairs_Evaluation/qa_review.json \
  --out-dir  RAGAs_Dataset

Optional:
  --limit 50        (process only first N accepted)
  --overwrite       (overwrite existing output files)
  --model ...       (override DEFAULT_MODEL)
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts src/rag on sys.path)

# Default locations
DEFAULT_QA_REVIEW = Path(paths.QA_REVIEW_JSON)
DEFAULT_OUT_DIR = Path(paths.DENSE_DATASET)

# The retrieval + reranking + answering pipeline this exporter drives.
import chat_cli_ragas_builder as rag  # noqa: E402


def load_accepted_questions(qa_review_path: Path):
    data = json.loads(qa_review_path.read_text(encoding="utf-8"))

    # Expected structure: {"questions":[...]} but we tolerate list too
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError("qa_review.json has unexpected structure. Expected dict with 'questions' list.")

    accepted_statuses = {"accept", "accepted"}
    accepted = [q for q in questions if str(q.get("review_status", "")).lower() in accepted_statuses]

    return accepted


def build_ragas_json(question_obj: dict, result: dict) -> dict:
    """
    Convert the output of rag.process_query(...) into a per-question JSON payload.
    """
    qid = str(question_obj.get("id", "unknown"))
    question_text = question_obj.get("question", "")

    # result["retrieved_chunks"] is what the LLM saw (after rerank + dynamic selection)
    chunks = result.get("retrieved_chunks", []) or []

    context = {}
    doc_ids = []

    for i, ch in enumerate(chunks, start=1):
        context[str(i)] = ch.get("text", "") or ""
        doc_ids.append(ch.get("doc_id", "") or "")

    payload = {
        "id": qid,
        "question": question_text,
        "answer": result.get("answer", "") or "",
        "context": context,
        "doc_ids": doc_ids,
    }
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-review",
        default=str(DEFAULT_QA_REVIEW),
        help="Path to qa_review.json (default: QA_Pairs_Evaluation/qa_review.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for per-question JSON files (default: RAGAs_Dataset/)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N accepted questions (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--model", default="", help="Override DEFAULT_MODEL (otherwise uses env/default inside builder)")
    args = parser.parse_args()

    qa_review_path = Path(args.qa_review).expanduser().resolve()
    if not qa_review_path.exists():
        raise FileNotFoundError(
            f"qa_review.json not found at {qa_review_path}. "
            f"Pass --qa-review to point to the correct file."
        )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted = load_accepted_questions(qa_review_path)
    if args.limit and args.limit > 0:
        accepted = accepted[: args.limit]

    print(f"✅ Found {len(accepted)} accepted questions in {qa_review_path}")
    print(f"📥 Input:  {qa_review_path}")
    print(f"📤 Output: {out_dir}")

    # Load index once (this is the big win for speed + RAM)
    print("⏳ Loading index once...")
    index = rag.load_index()
    print("✅ Index loaded.")

    # Model choice
    model = args.model.strip() or os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    print(f"ℹ️ Using model: {model}")

    processed = 0
    skipped = 0

    for q in accepted:
        qid = str(q.get("id", "unknown"))
        question_text = (q.get("question") or "").strip()
        if not question_text:
            print(f"⚠️ Skipping id={qid}: empty question text")
            skipped += 1
            continue

        out_path = out_dir / f"{qid}.json"
        if out_path.exists() and not args.overwrite:
            print(f"⏭️  Skip id={qid} (exists). Use --overwrite to regenerate.")
            skipped += 1
            continue

        print(f"\n▶️  Processing id={qid}")
        result = rag.process_query(index, question_text, model=model)

        payload = build_ragas_json(q, result)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Wrote {out_path}")

        processed += 1

    print(f"\n✅ Done. processed={processed}, skipped={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    main()