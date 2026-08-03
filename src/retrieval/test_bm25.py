"""Smoke test for the PostgreSQL lexical (BM25) retriever.

Runs a few Greek queries straight against the tsvector index so you can check
that the full-text configuration and GIN index are in place before running the
full BM25 experiment in ``run_bm25_eval.py``.
"""
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

PG_CONFIG = {
    "dbname":   os.getenv("PG_DATABASE", "legalchatbot2"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
}

def test_bm25_query(query_text: str, limit: int = 5):
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()
    
    sql = """
        SELECT id, 
               LEFT(text, 200) AS preview, 
               ts_rank(text_tsv, websearch_to_tsquery('greek', %s)) AS rank
        FROM data_laws_vector_table_v2_512
        WHERE text_tsv @@ websearch_to_tsquery('greek', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    cur.execute(sql, (query_text, query_text, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    print(f"\n🔍 Query: {query_text}\n")
    for row in rows:
        print(f"id={row[0]} | rank={row[2]:.4f}")
        print(f"  {row[1]}...\n")
    return rows

if __name__ == "__main__":
    # Use a question from your CSV
    test_bm25_query("κοινωνικό μέρισμα 2013")
    test_bm25_query("προθεσμία ανάθεσης προγραμμάτων στέγασης αστέγων")
    test_bm25_query("πώς υπολογίζεται εφάπαξ χρηματική ενίσχυση στελέχη Ενόπλων Δυνάμεων")


    