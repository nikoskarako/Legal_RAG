"""
BM25 (sparse retrieval) baseline — standalone, no LlamaIndex dependency.

Reads the accepted questions from Questions_for_Evaluation.csv, retrieves
top-k chunks via PostgreSQL BM25, generates answers via
OpenRouter (same model as student's dense RAG), and writes per-question
JSON files to BM25_Dataset/ in the same format as RAGAs_Dataset/.

Usage:
    python run_bm25_eval.py

Output:
    BM25_Dataset/{row_id}.json  — one file per accepted question

Then run RAGAS:
    python evaluate_ragas.py --data_dir BM25_Dataset --out_file ragas_scores_bm25.json
"""
import csv
import json
import os
import sys
import time

import psycopg2
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (also registers sibling stages on sys.path)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PG_CONFIG = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   os.getenv("PG_DATABASE", "legalchatbot2"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Same model as the dense RAG system in src/rag/chat_openrouter.py
MODEL = os.getenv("EVAL_MODEL", "deepseek/deepseek-chat-v3-0324")
K     = 10   # chunks to retrieve

QUESTIONS_CSV = os.getenv("QUESTIONS_CSV", paths.QUESTIONS_CSV)
print(f"Loading questions from {QUESTIONS_CSV}")
OUTPUT_DIR    = os.getenv("BM25_DATASET_DIR", paths.BM25_DATASET)

ANSWER_PROMPT = (
    "Βάσει ΜΟΝΟ των παρακάτω ελληνικών νομικών κειμένων, απάντησε στα Ελληνικά.\n"
    "Απάντησε αποκλειστικά από τα παρακάτω κείμενα, χωρίς εξωτερική γνώση.\n\n"
    "Ερώτηση:\n{query}\n\n"
    "Νομικά κείμενα:\n{context}\n"
)


# ---------------------------------------------------------------------------
# Data loading — preserves original CSV row number as ID (matches RAGAs_Dataset)
# ---------------------------------------------------------------------------

def load_accepted_questions(path: str) -> list[dict]:
    """
    Returns accepted questions with their 1-based CSV row number as id
    (matching the filenames in RAGAs_Dataset/).
    """
    questions = []
    with open(path, encoding="utf-8-sig") as f:  # -sig strips a leading BOM
        # Comma-delimited: the Greek question mark is ';', so every question ends
        # in a semicolon and a ';' delimiter survives only via quoting.
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            if row.get("Review", "").strip() == "Accept":
                questions.append({
                    "id":            row_idx,
                    "question":      row["question"].strip(),
                    "question_type": row.get("question_type", "").strip(),
                    "difficulty":    row.get("difficulty", "").strip(),
                    "source_doc":    row.get("source_file", "").strip(),
                })
    return questions


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def retrieve_bm25_was(conn, query: str, k: int = K) -> list[dict]:
    """
    PostgreSQL BM25 retrieval.
    Returns [] on no match — reported as zero-coverage in stats.
    """
    cur = conn.cursor()
    sql = """
        SELECT id,
               text,
               metadata_,
               text::text <&> to_bm25query(%s, 'idx_bm25') AS bm25_score
        FROM   data_laws_vector_table_v2_512
        WHERE  text::text <&> to_bm25query(%s, 'idx_bm25') > 0
        ORDER  BY bm25_score ASC
        LIMIT  %s 
    """  
    # sql = """
    #     SELECT id,
    #            text,
    #            metadata_,
    #            ts_rank_cd(text_tsv, websearch_to_tsquery('greek', %s)) AS rank
    #     FROM   data_laws_vector_table_v2_512
    #     WHERE  text_tsv @@ websearch_to_tsquery('greek', %s)
    #     ORDER  BY rank DESC
    #     LIMIT  %s
    # """
    try:
        cur.execute(sql, (query, query, k))
        rows = cur.fetchall()
    except Exception as e:
        print(f"    BM25 error: {e}")
        conn.rollback()
        rows = []
    finally:
        cur.close()

    chunks = []
    for row in rows:
        raw_meta = row[2]
        if isinstance(raw_meta, dict):
            meta = raw_meta
        elif raw_meta:
            try:
                meta = json.loads(raw_meta)
            except Exception:
                meta = {}
        else:
            meta = {}
        chunks.append({
            "text":   row[1],
            "doc_id": meta.get("doc_id", str(row[0])),
            "rank":   float(row[3]),
        })
    return chunks

def retrieve_bm25(conn, query: str, k: int = K) -> list[dict]:
    """
    PostgreSQL lexical retrieval via tsvector full-text rank (the retriever
    that produced the paper's BM25 numbers; matches test_bm25.py). Uses the
    existing GIN index idx_text_tsv on text_tsv with the 'greek' config.
    """
    cur = conn.cursor()
    sql = """
        SELECT id,
               text,
               metadata_,
               ts_rank(text_tsv, websearch_to_tsquery('greek', %s)) AS rank
        FROM   data_laws_vector_table_v2_512
        WHERE  text_tsv @@ websearch_to_tsquery('greek', %s)
        ORDER  BY rank DESC
        LIMIT  %s
    """
    try:
        cur.execute(sql, (query, query, k))
        rows = cur.fetchall()
    except Exception as e:
        print(f"    BM25 error: {e}")
        conn.rollback()
        rows = []
    finally:
        cur.close()

    chunks = []
    for row in rows:
        raw_meta = row[2]
        if isinstance(raw_meta, dict):
            meta = raw_meta
        elif raw_meta:
            try:
                meta = json.loads(raw_meta)
            except Exception:
                meta = {}
        else:
            meta = {}
            
        chunks.append({
            "text":   row[1],
            "doc_id": meta.get("doc_id", str(row[0])),
            "rank":   float(row[3]), # This is the bm25 distance score
        })
    return chunks


# ---------------------------------------------------------------------------
# Generation
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
                    "model":       MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
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
    print(f"DB: {PG_CONFIG['host']}  Model: {MODEL}  k={K}")
    print(f"Output: {OUTPUT_DIR}/\n")

    conn      = psycopg2.connect(**PG_CONFIG)
    bm25_empty = 0

    for i, q in enumerate(questions, start=1):
        out_path = os.path.join(OUTPUT_DIR, f"{q['id']}.json")

        # Skip already-completed files (resume support)
        if os.path.exists(out_path):
            print(f"[{i:3d}/{len(questions)}] skip (exists): {q['id']}.json")
            continue

        print(f"[{i:3d}/{len(questions)}] id={q['id']} | {q['question'][:65]}...")

        chunks = retrieve_bm25(conn, q["question"])

        if not chunks:
            bm25_empty += 1
            print(f"           ⚠  BM25: 0 chunks")

        answer = generate_answer(q["question"], chunks)

        # Build context dict matching RAGAs_Dataset schema: {"1": "...", "2": "..."}
        context_dict = {str(j + 1): c["text"] for j, c in enumerate(chunks)}
        doc_ids      = [c["doc_id"] for c in chunks]

        record = {
            "id":       str(q["id"]),
            "question": q["question"],
            "answer":   answer,
            "context":  context_dict,
            "doc_ids":  doc_ids,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    conn.close()

    total     = len(questions)
    completed = len([q for q in questions
                     if os.path.exists(os.path.join(OUTPUT_DIR, f"{q['id']}.json"))])
    print(f"\nDone. {completed}/{total} files in {OUTPUT_DIR}/")
    print(f"BM25 zero-match (this run): {bm25_empty} questions returned 0 chunks")
    print(f"\nNext: python evaluate_ragas.py --data_dir BM25_Dataset --out_file ragas_scores_bm25.json")


if __name__ == "__main__":
    main()
