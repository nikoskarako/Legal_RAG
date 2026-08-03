#!/usr/bin/env python3
"""
RAG CLI + RAGAS export (Legal_Chatbot)
----------------------------------

What this script does
- Provides a terminal (CLI) loop that:
  1) retrieves candidate chunks from a PGVector-backed LlamaIndex index,
  2) reranks them with a cross-encoder reranker (BGE Reranker),
  3) dynamically selects ONLY the most relevant chunks,
  4) sends those chunks to an OpenRouter LLM,
  5) exports EXACTLY what the LLM saw to JSONL for RAGAS evaluation.

Why it exists
- Your RAGAS scores (especially context_relevancy) depend heavily on the quality of
  the context that you send to the LLM.
- The goal here is not to preserve “diverse doc_ids”, but to maximize relevance.
  It is completely OK if all selected chunks come from the same document, as long
  as they are the best matches for the user question.

Data sources / storage
- Vector store: PostgreSQL + pgvector via `PGVectorStore`.
  IMPORTANT:
  - `PGVectorStore` uses an internal physical table named:
      public.data_<BASE_TABLE_NAME>
  - In code we configure the *base* name via `VECTOR_TABLE`.
    Example:
      VECTOR_TABLE=laws_vector_table_v2_512
    which maps to:
      public.data_laws_vector_table_v2_512

- Optional persisted LlamaIndex storage (docstore/index store) on disk:
  - Controlled by `STORAGE_DIR`.
  - If `${STORAGE_DIR}/docstore.json` exists, we load the full persisted index via
    `load_index_from_storage(...)`.
  - If it does NOT exist, we fall back to a “vector-store-only” index:
    `VectorStoreIndex.from_vector_store(...)`.
  - This fallback is safe and is often enough for retrieval/reranking. Persisted
    storage is mainly useful for certain index features and faster loading.

High-level pipeline (runtime)
1) Vector retrieval (broad net)
   - We fetch `SIMILARITY_TOP_K` candidates from the vector store.
   - This stage is recall-oriented: it tries not to miss relevant text.

2) Cross-encoder reranking (precision, CPU/RAM optimized)
   - Vector retrieval is cheap; cross-encoder reranking is expensive.
   - To keep quality but reduce cost, we first **pre-filter** the retrieved candidates to the
     best `RERANK_CANDIDATES` *by vector similarity score*.
   - Then we rerank only that smaller pool using `FlagEmbeddingReranker` (BAAI/bge-reranker-*).
   - Reranking returns NodeWithScore objects (each has `.node` and a `.score` from the reranker).

3) Dynamic chunk selection (key for Context Relevance)
   - Instead of always sending a fixed Top-N, we apply a qualification rule:
     * Keep chunks whose score is within `RERANK_SCORE_GAP` of the best score.
     * Optionally require an absolute threshold (`RERANK_ABS_THRESHOLD`).
     * Always send at least `RERANK_MIN_CHUNKS` (if available).
     * Never send more than `RERANK_MAX_CHUNKS`.

4) LLM answer
   - We build a prompt that forces the model to answer using ONLY the provided
     chunks and to cite them using [S#] markers.
   - The OpenRouter model is selected via `DEFAULT_MODEL` and has fallbacks.

5) RAGAS export (ground truth for evaluation)
   - We export a JSONL record per question containing:
     - question
     - answer (the final answer text)
     - context (the exact chunks that were sent to the LLM, in order)
     - doc_ids (doc_id extracted from node metadata for each chunk)
   - This is critical: for RAGAS, “retrieved context” must match what the LLM
     actually saw (after reranking + filtering).

Main knobs to tune (env vars)
- VECTOR_TABLE: base table name (WITHOUT data_ prefix)
- STORAGE_DIR: path to persisted LlamaIndex files (optional)
- SIMILARITY_TOP_K: how many candidates to retrieve from the vector DB
- RERANK_CANDIDATES: how many candidates reranker considers (upper bound)
- RERANK_MIN_CHUNKS / RERANK_MAX_CHUNKS: bounds for context size
- RERANK_SCORE_GAP: relative threshold from best reranker score
- RERANK_ABS_THRESHOLD: optional absolute score cutoff (blank disables)

Tip for debugging
- If you see “❗ Δεν βρέθηκαν σχετικά νομικά κείμενα.” while you know the doc exists,
  confirm:
  1) `VECTOR_TABLE` points to the correct base name,
  2) the internal table `public.data_<VECTOR_TABLE>` actually contains rows,
  3) you are using the same embedding model family at query time as at build time.
"""

import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# ----------------------------
# CONFIG / ENV
# ----------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

load_dotenv()  # read .env if present

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing in environment")

PG_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DATABASE", "legalchatbot2"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

STORAGE_DIR = paths.STORAGE_DIR

# IMPORTANT: PGVectorStore expects the *base* table name; it uses data_<base> internally.
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "laws_vector_table_v2_512")

RAGAS_EXPORT_PATH = os.getenv("RAGAS_EXPORT_PATH", os.path.join(paths.RESULTS_DIR, "ragas_exports.jsonl"))

# OpenRouter retry + model fallbacks
RETRY_COUNT = 3
BACKOFF_SECONDS = 1.0
REQUEST_TIMEOUT = 60
MODEL_FALLBACKS = [
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
]

# Retrieval / Reranking tuning
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 80))

# We rerank a larger pool and then select "qualified" chunks dynamically.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 30))
RERANK_MIN_CHUNKS = int(os.getenv("RERANK_MIN_CHUNKS", 3))
RERANK_MAX_CHUNKS = int(os.getenv("RERANK_MAX_CHUNKS", 8))

# Score-based qualification (works with FlagEmbeddingReranker scores).
# Keep nodes whose score is close enough to the best score.
RERANK_SCORE_GAP = float(os.getenv("RERANK_SCORE_GAP", 0.05))

# Optional absolute threshold. If set to a number, nodes must also pass it.
# Leave empty / unset to disable.
_RERANK_ABS_THRESH_RAW = os.getenv("RERANK_ABS_THRESHOLD", "").strip()
RERANK_ABS_THRESHOLD = float(_RERANK_ABS_THRESH_RAW) if _RERANK_ABS_THRESH_RAW else None

def _persisted_index_exists(persist_dir: str) -> bool:
    """Return True if LlamaIndex persisted files exist in persist_dir."""
    if not persist_dir:
        return False
    return os.path.isfile(os.path.join(persist_dir, "docstore.json"))


def _normalize_vector_table_base(name: str) -> str:
    """Normalize a PGVector table name to the *base* name (strip data_/index_ prefixes)."""
    n = (name or "").strip()
    if n.startswith("data_"):
        n = n[len("data_"):]
    if n.startswith("index_"):
        n = n[len("index_"):]
    return n

VECTOR_TABLE = _normalize_vector_table_base(VECTOR_TABLE)
print(f"ℹ️ Using PGVector base table: {VECTOR_TABLE} (internal data table: public.data_{VECTOR_TABLE})")

# LlamaIndex settings: use same embedding family as index build
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

# ----------------------------
# Reranker Initialization (Load once to save time)
# ----------------------------
print("Φόρτωση μοντέλου Reranker (BAAI/bge-reranker-large)...")
# Χρησιμοποιούμε το 'large' για μέγιστη ακρίβεια (Context Relevance).
# Αν το σύστημα είναι αργό, αλλάξτε το σε 'BAAI/bge-reranker-base'.
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-large",
    top_n=RERANK_CANDIDATES,  # rerank a larger pool; we will filter dynamically for the LLM
)

def _get_score(nws) -> float:
    """Best-effort extract score from a NodeWithScore-like object."""
    return float(getattr(nws, "score", 0.0) or 0.0)


def select_qualified_nodes(reranked_nodes: list) -> list:
    """Select a dynamic subset of reranked nodes to send to the LLM.

    Policy:
    - Always return at least RERANK_MIN_CHUNKS (if available).
    - Prefer nodes whose score is within RERANK_SCORE_GAP of the best score.
    - If RERANK_ABS_THRESHOLD is set, nodes must also pass that absolute threshold.
    - Never exceed RERANK_MAX_CHUNKS.
    """
    if not reranked_nodes:
        return []

    # Ensure sorted by score desc (defensive)
    reranked_nodes = sorted(reranked_nodes, key=_get_score, reverse=True)

    best = _get_score(reranked_nodes[0])
    qualified = []

    for nws in reranked_nodes:
        sc = _get_score(nws)

        # Relative-to-best filter
        if sc < best - RERANK_SCORE_GAP:
            continue

        # Optional absolute threshold
        if RERANK_ABS_THRESHOLD is not None and sc < RERANK_ABS_THRESHOLD:
            continue

        qualified.append(nws)
        if len(qualified) >= RERANK_MAX_CHUNKS:
            break

    # Fallback: ensure at least min chunks
    if len(qualified) < min(RERANK_MIN_CHUNKS, len(reranked_nodes)):
        qualified = reranked_nodes[: min(RERANK_MIN_CHUNKS, len(reranked_nodes))]

    return qualified

# ----------------------------
# Index loader (depends on PGVectorStore)
# ----------------------------
def load_index() -> VectorStoreIndex:
    """
    Load the index from local storage + Postgres vector store.
    Returns a LlamaIndex VectorStoreIndex object ready for retrieval.
    """
    async_str = (
        f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@"
        f"{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"
    ).replace("postgresql://", "postgresql+asyncpg://")

    vector_store = PGVectorStore.from_params(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
        database=PG_CONFIG["database"],
        table_name=VECTOR_TABLE,
        embed_dim=768,
        async_connection_string=async_str,
        schema_name="public",
    )

    # If persisted index files exist, load them; otherwise build an index object from the vector store only.
    if _persisted_index_exists(STORAGE_DIR):
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=STORAGE_DIR,
        )
        return load_index_from_storage(storage_context)

    print(
        f"⚠️ Persisted index not found at {os.path.join(STORAGE_DIR, 'docstore.json')}. "
        "Falling back to vector-store-only load. "
        "(Set STORAGE_DIR to the correct folder if you want to use persisted storage.)"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# ----------------------------
# Simple OpenRouter client (with retries + fallbacks)
# ----------------------------
def chat_with_openrouter(prompt: str, model: str) -> str:
    """
    Call OpenRouter and return assistant message text.
    On API error returns a string starting with "⚠️ API ERROR:".
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    def call_model(cur_model: str):
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": cur_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                if "choices" not in data:
                    err = data.get("error", {}).get("message", "Unknown API error.")
                    return f"⚠️ API ERROR: {err}"
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retriable = status in (429, 500, 502, 503, 504) or status is None
                if retriable and attempt < RETRY_COUNT:
                    time.sleep(BACKOFF_SECONDS * attempt)
                    continue
                if status:
                    try:
                        data = e.response.json()
                        msg = data.get("error", {}).get("message", str(e))
                    except Exception:
                        msg = str(e)
                    return f"⚠️ API ERROR: {msg}"
                return f"⚠️ API ERROR: {str(e)}"

    tried = [model] + [m for m in MODEL_FALLBACKS if m != model]
    last = None
    for m in tried:
        last = call_model(m)
        if not (isinstance(last, str) and last.startswith("⚠️ API ERROR:")):
            return last
    return last

def export_ragas_record(question: str, answer: str, retrieved_chunks: list):
    """
    Append a single RAGAs-style record to a JSONL file.
    Each record contains:
      - question: the user query
      - answer: the final LLM answer
      - context: the EXACT chunks sent to the LLM (reranked top-3)
      - doc_ids: a list of doc_id for each chunk
    """
    export_path = RAGAS_EXPORT_PATH
    if os.path.isdir(export_path):
        export_path = os.path.join(export_path, "ragas_exports.json")

    # Build context entries
    context_obj = {}
    doc_ids = []
    # retrieved_chunks here contains the already filtered, reranked nodes
    for idx, c in enumerate(retrieved_chunks, start=1):
        key = str(idx)
        context_obj[key] = c.get("text", "")
        doc_ids.append(c.get("doc_id", ""))

    record = {
        "question": question,
        "answer": answer,
        "context": context_obj,
        "doc_ids": doc_ids,
    }
    try:
        with open(export_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Σφάλμα κατά την αποθήκευση των δεδομένων RAGAs: {e}")

# ----------------------------
# Query processing (Retrieval -> Reranking -> LLM)
# ----------------------------
def process_query(index: VectorStoreIndex, query: str, model: str = "deepseek/deepseek-chat-v3-0324:free") -> dict:
    """
    Runs a 2-stage retrieval:
    1. Vector Search (Top-50)
    2. Cross-Encoder Reranking (Top-3)
    Returns answer + sources.
    """
    
    # --- STAGE 1: Broad Retrieval (Vector Search) ---
    # Ζητάμε SIMILARITY_TOP_K για να είμαστε σίγουροι ότι η απάντηση υπάρχει κάπου στη λίστα
    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    initial_nodes = retriever.retrieve(query)

    if not initial_nodes:
        return {
            "answer": "❗ Δεν βρέθηκαν σχετικά νομικά κείμενα.",
            "sources": [],
            "cited_sources": [],
            "sources_numbered": [],
            "retrieved_chunks": [],
        }

    # --- STAGE 1.5: Cheap pre-filter before reranking (CPU/RAM friendly) ---
    # Cross-encoder reranking is the expensive step.
    # We keep recall high by retrieving SIMILARITY_TOP_K from the vector DB,
    # then we rerank ONLY the best RERANK_CANDIDATES by *vector similarity score*.
    # This usually preserves quality while making the reranker much lighter.
    def _vec_score(nws) -> float:
        return float(getattr(nws, "score", 0.0) or 0.0)

    if len(initial_nodes) > RERANK_CANDIDATES:
        initial_nodes = sorted(initial_nodes, key=_vec_score, reverse=True)[:RERANK_CANDIDATES]

    # --- STAGE 2: Reranking (Cross-Encoder) ---
    # Το μοντέλο συγκρίνει Ερώτηση <-> Κείμενο και επαναταξινομεί το μικρότερο pool.
    reranked_nodes = reranker.postprocess_nodes(nodes=initial_nodes, query_str=query)

    # Dynamically choose how many chunks to send to the LLM
    selected_nodes = select_qualified_nodes(reranked_nodes)

    # These are the final chunks that the LLM (and thus RAGAS) will see
    top_nodes = selected_nodes

    # Debug: show how many survived qualification and their score range
    if top_nodes:
        scores = [_get_score(x) for x in top_nodes]
        print(
            f"ℹ️ Selected {len(top_nodes)} chunks for LLM "
            f"(retrieved={SIMILARITY_TOP_K}, reranked_pool={len(reranked_nodes)}, best={max(scores):.4f}, worst={min(scores):.4f})"
        )

    # Prepare export data (RAGAS requires exactly what the LLM sees)
    retrieved_chunks = []
    for rank, node in enumerate(top_nodes, start=1):
        # Note: postprocess_nodes returns NodeWithScore objects, accessing .node gives the TextNode
        text_node = node.node 
        text = getattr(text_node, "text", "") or ""
        metadata = getattr(text_node, "metadata", {}) or {}
        doc_id = metadata.get("doc_id") or metadata.get("id") or metadata.get("docid") or ""
        
        retrieved_chunks.append({
            "rank": rank,
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata,
            "score": node.score # Reranker score (utility for debugging)
        })

    # Prepare prompt context
    numbered_ctx = []
    sources_numbered = []
    
    for idx, r_node in enumerate(top_nodes, start=1):
        node = r_node.node
        text = getattr(node, "text", "") or ""
        metadata = getattr(node, "metadata", {}) or {}
        doc_id = metadata.get("doc_id") or metadata.get("id") or metadata.get("docid") or ""
        
        numbered_ctx.append(f"[S{idx}] (doc_id: {doc_id})\n{text}\n")
        
        label = metadata.get("title") or metadata.get("law_title") or metadata.get("fek") or doc_id or f"S{idx}"
        sources_numbered.append({
            "s_index": idx,
            "doc_id": doc_id,
            "label": label,
            "raw_meta": metadata,
        })

    # Count sources for deduping later
    doc_chunk_counts = {}
    for s in sources_numbered:
        did = s.get("doc_id")
        if did:
            doc_chunk_counts[did] = doc_chunk_counts.get(did, 0) + 1

    ctx = "\n\n".join(numbered_ctx)
    prompt = (
        "Απάντησε στα Ελληνικά.\n"
        "Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ τις πληροφορίες που βρίσκονται στα παρακάτω αποσπάσματα.\n"
        "ΜΗΝ χρησιμοποιείς γνώσεις εκτός των αποσπασμάτων.\n"
        "ΜΗΝ προσθέτεις σχόλια, ερμηνείες ή περιττές πληροφορίες.\n"
        "Όταν χρησιμοποιείς πληροφορία από συγκεκριμένο απόσπασμα, πρόσθεσε στο τέλος της πρότασης μια παραπομπή της μορφής [S#].\n"
        "Δώσε σύντομη, άμεση και ακριβή απάντηση στο ερώτημα.\n"
        "Εάν τα αποσπάσματα δεν παρέχουν αρκετές πληροφορίες, απάντησε μόνο: 'Δεν γνωρίζω'.\n\n"
        f"Ερώτηση:\n{query}\n\n"
        f"Αποσπάσματα (ταξινομημένα κατά συνάφεια):\n{ctx}\n"
    )

    final_answer = chat_with_openrouter(prompt, model)

    # Handle API errors
    if isinstance(final_answer, str) and final_answer.startswith("⚠️ API ERROR:"):
        return {
            "answer": final_answer,
            "sources": [],
            "cited_sources": [],
            "sources_numbered": sources_numbered,
            "retrieved_chunks": retrieved_chunks,
        }

    # Find cited S# markers
    import re
    cited_idx = []
    for m in re.finditer(r"\[S(\d+)\]", final_answer):
        k = int(m.group(1))
        if 1 <= k <= len(top_nodes) and k not in cited_idx:
            cited_idx.append(k)

    cited_sources = [s for s in sources_numbered if s["s_index"] in cited_idx]

    # Clean answer
    cleaned_answer = re.sub(r"\[S\d+[^\]]*\]", "", final_answer)
    cleaned_answer = re.sub(r"\s{2,}", " ", cleaned_answer).strip()

    # Build deduped source list
    seen = set()
    sources_retr = []
    for s in sources_numbered:
        key = (s.get("doc_id"), s.get("label"))
        if key in seen:
            continue
        seen.add(key)
        did = s.get("doc_id")
        s["chunk_count"] = doc_chunk_counts.get(did, 1)
        sources_retr.append(s)

    return {
        "answer": cleaned_answer,
        "sources": sources_retr,
        "cited_sources": cited_sources,
        "sources_numbered": sources_numbered,
        "retrieved_chunks": retrieved_chunks, # Contains ONLY the Top-3 Reranked
    }

# ----------------------------
# CLI
# ----------------------------
def main():
    print("Φόρτωση ευρετηρίου... (αυτό μπορεί να πάρει λίγα δευτερόλεπτα)")
    idx = load_index()
    print("Το ευρετήριο και ο reranker φορτώθηκαν.")
    print("Πληκτρολογήστε την ερώτησή σας (πληκτρολογήστε 'exit' για έξοδο).")

    default_model = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat-v3-0324:free")

    try:
        while True:
            q = input("\nΕρώτηση: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break

            print("Ανάκτηση (Vector Search + BGE Reranking) και υποβολή...")
            out = process_query(idx, q, default_model)

            print("\n--- Απάντηση ---\n")
            print(out.get("answer", ""))

            if out.get("sources"):
                print(f"\n--- Πηγές (Top {len(out['sources'])} Selected) ---")
                for i, s in enumerate(out["sources"], 1):
                    label = s.get("label") or s.get("doc_id") or "Πηγή"
                    doc_id = s.get("doc_id")
                    chunk_count = s.get("chunk_count", 1)
                    print(f" {i}. {label} (doc_id={doc_id})")

            # Export data for RAGAs
            export_ragas_record(q, out.get("answer", ""), out.get("retrieved_chunks", []))

    except KeyboardInterrupt:
        print("\nΈξοδος από το πρόγραμμα.")

if __name__ == "__main__":
    main()