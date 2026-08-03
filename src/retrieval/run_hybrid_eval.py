"""
Experiment 2 — Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Fuses the dense (LlamaIndex/PGVector) and BM25 (PostgreSQL) ranked lists
with RRF, feeds the fused top-N chunks to the SAME answer generator, and
writes per-question JSON to Hybrid_Dataset/ in the exact schema that
evaluate_ragas.py consumes (same as BM25_Dataset/).

Reuses, unchanged:
  * dense retriever  — `index` from chat_openrouter.py
  * BM25 retriever   — `retrieve_bm25`, `PG_CONFIG`, question loader from run_bm25_eval.py
  * scoring          — evaluate_ragas.py (run afterwards)
  * significance     — stats.py (add a "hybrid" entry to its file map)

Usage:
    python run_hybrid_eval.py

Then:
    python evaluate_ragas.py --data_dir Hybrid_Dataset --out_file ragas_scores_hybrid.json

RRF: score(chunk) = sum_lists 1 / (k_rrf + rank), rank 1-based.
Fusion key = chunk text (dense and BM25 read the same physical table, so the
same chunk surfaces with identical text in both lists).
"""
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts src/rag and src/retrieval on sys.path)

# --- reuse BM25 side (import does NOT connect; main() is __main__-guarded) ---
from run_bm25_eval import load_accepted_questions, retrieve_bm25, PG_CONFIG, QUESTIONS_CSV
import psycopg2

# --- reuse dense side (importing chat_openrouter builds/loads the index once) ---
from chat_openrouter import index

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL      = os.getenv("EVAL_MODEL", "deepseek/deepseek-chat-v3-0324")
CAND_K     = 50    # candidates pulled from EACH retriever before fusion
TOP_N      = 10    # fused chunks kept as final context (matches dense/BM25 k)
K_RRF      = 60    # RRF damping constant (standard default)

OUTPUT_DIR = os.getenv("HYBRID_DATASET_DIR", paths.HYBRID_DATASET)

# Numbered-citation prompt (same wording as the dense system in chat_openrouter.py)
ANSWER_PROMPT = (
    "Βάσει ΜΟΝΟ των παρακάτω ελληνικών νομικών κειμένων, απάντησε στα Ελληνικά.\n"
    "Απάντησε αποκλειστικά από τα παρακάτω κείμενα, χωρίς εξωτερική γνώση.\n\n"
    "Ερώτηση:\n{query}\n\n"
    "Νομικά κείμενα:\n{context}\n"
)

# Build the dense retriever once
_dense_retriever = index.as_retriever(similarity_top_k=CAND_K)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_dense(query: str, k: int = CAND_K) -> list[dict]:
    """Dense candidates, best-first. Returns [{text, doc_id}]."""
    nodes = _dense_retriever.retrieve(query)
    out = []
    for n in nodes[:k]:
        meta = getattr(n, "metadata", {}) or {}
        out.append({
            "text":   n.text or "",
            "doc_id": str(meta.get("doc_id", "")),
        })
    return out


def rrf_fuse(dense_list: list[dict], bm25_list: list[dict],
             k_rrf: int = K_RRF, top_n: int = TOP_N) -> list[dict]:
    """Reciprocal Rank Fusion keyed on chunk text. Returns fused top-N chunks."""
    scores: dict[str, float] = {}
    meta:   dict[str, dict]  = {}
    for lst in (dense_list, bm25_list):
        for rank, item in enumerate(lst, start=1):
            key = (item.get("text") or "").strip()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            meta.setdefault(key, item)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [meta[key] for key, _ in ranked]


# ---------------------------------------------------------------------------
# Generation (same call pattern as run_bm25_eval.generate_answer)
# ---------------------------------------------------------------------------
def generate_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return ""
    context = "\n\n".join(f"[S{i+1}] {c['text']}" for i, c in enumerate(chunks))
    prompt  = ANSWER_PROMPT.format(query=query, context=context)
    for attempt in range(1, 4):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    LLM attempt {attempt} failed: {e}")
            if attempt < 3:
                time.sleep(attempt * 2)
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    questions = load_accepted_questions(QUESTIONS_CSV)
    print(f"Loaded {len(questions)} accepted questions")
    print(f"Model: {MODEL}  cand_k={CAND_K}  top_n={TOP_N}  k_rrf={K_RRF}")
    print(f"Output: {OUTPUT_DIR}/\n")

    conn = psycopg2.connect(**PG_CONFIG)
    empty = 0

    for i, q in enumerate(questions, start=1):
        out_path = os.path.join(OUTPUT_DIR, f"{q['id']}.json")
        if os.path.exists(out_path):
            print(f"[{i:3d}/{len(questions)}] skip (exists): {q['id']}.json")
            continue

        print(f"[{i:3d}/{len(questions)}] id={q['id']} | {q['question'][:60]}...")

        dense_list = retrieve_dense(q["question"], CAND_K)
        bm25_list  = retrieve_bm25(conn, q["question"], CAND_K)
        fused      = rrf_fuse(dense_list, bm25_list)

        if not fused:
            empty += 1
            print("           ⚠  hybrid: 0 chunks")

        answer = generate_answer(q["question"], fused)

        record = {
            "id":       str(q["id"]),
            "question": q["question"],
            "answer":   answer,
            "context":  {str(j + 1): c["text"] for j, c in enumerate(fused)},
            "doc_ids":  [c["doc_id"] for c in fused],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    conn.close()

    completed = len([q for q in questions
                     if os.path.exists(os.path.join(OUTPUT_DIR, f"{q['id']}.json"))])
    print(f"\nDone. {completed}/{len(questions)} files in {OUTPUT_DIR}/")
    print(f"Hybrid zero-match (this run): {empty}")
    print("\nNext: python evaluate_ragas.py --data_dir Hybrid_Dataset --out_file ragas_scores_hybrid.json")


if __name__ == "__main__":
    main()
