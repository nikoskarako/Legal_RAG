import os
import shutil
import logging
import sys
import sqlite3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tqdm import tqdm  # ✅ progress bars

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

# --- Resolve storage path for index files ---
# Separate storage dir so the 1024-token index in storage/ stays untouched.
DEFAULT_STORAGE_DIR_512 = os.path.join(paths.PROJECT_ROOT, "storage_512")

# --- Load environment variables ---
load_dotenv()

# --- Configure logging for SQL debug ---
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# --- PostgreSQL configuration ---
PG_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DATABASE", "legalchatbot2"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

# Build connection strings
sync_str = (
    f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@"
    f"{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"
)
async_str = sync_str.replace("postgresql://", "postgresql+asyncpg://")

# ✅ SAFE: only ensure extension exists, DO NOT drop any tables globally
engine = create_engine(sync_str)
with engine.begin() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

# --- Embedding model (same as before) ---
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")


# ---------------------------------------------------------------------------
# Space-safety + resumability knobs
# ---------------------------------------------------------------------------
# IMPORTANT:
# LlamaIndex will (by default) store a very large metadata JSON per chunk
# (e.g., `_node_content`), which can explode Postgres size.
# We explicitly trim metadata to the minimum we need (doc_id + a few fields)
# before writing to PGVector.
#
# Also, we support a simple "resume" mode: if the target data table already
# contains chunks for a doc_id, we skip re-indexing that doc.

# Set to "1" to skip documents already present in the target table.
RESUME_IF_EXISTS = os.getenv("RESUME_IF_EXISTS", "1") == "1"

# Optional: limit number of documents during debugging.
DOC_LIMIT = os.getenv("DOC_LIMIT")
DOC_LIMIT = int(DOC_LIMIT) if DOC_LIMIT else None

# Optional: start offset for batching/runs.
DOC_OFFSET = os.getenv("DOC_OFFSET")
DOC_OFFSET = int(DOC_OFFSET) if DOC_OFFSET else 0

# If true, do NOT drop the target table (safer default).
DROP_TARGET_TABLE = os.getenv("DROP_TARGET_TABLE", "0") == "1"


def _resolve_data_table_name(pg_table: str) -> str:
    """LlamaIndex PGVectorStore prefixes tables with `data_` by default."""
    # You observed tables like: public.data_laws_vector_table_v2_512
    return f"data_{pg_table}"


def _fetch_existing_doc_ids(engine: Engine, pg_table: str) -> set[str]:
    """Return set of doc_id already present in the target data table."""
    data_table = _resolve_data_table_name(pg_table)
    q = text(f"""
        SELECT DISTINCT metadata_->>'doc_id' AS doc_id
        FROM public.{data_table}
        WHERE metadata_->>'doc_id' IS NOT NULL
    """)
    try:
        with engine.begin() as conn:
            rows = conn.execute(q).fetchall()
        return {r[0] for r in rows if r and r[0]}
    except Exception:
        # Table might not exist yet.
        return set()


def _minimize_metadata(doc_id: str, chunk_size: int, chunk_overlap: int) -> dict:
    """Keep ONLY small, useful metadata to reduce Postgres footprint."""
    return {
        "doc_id": doc_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_documents_from_sqlite(
    sqlite_path: str,
    et_table: str = "et",
) -> list[Document]:
    """
    Loads rows from et, pulling only the ID and fekTEXT columns.
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    limit_clause = ""
    offset_clause = ""
    if DOC_LIMIT is not None:
        limit_clause = f"\nLIMIT {DOC_LIMIT}"
    if DOC_OFFSET:
        offset_clause = f"\nOFFSET {DOC_OFFSET}"

    query = f"""
    SELECT
      ID      AS doc_id,
      fekTEXT AS content
    FROM {et_table}
    WHERE fekTEXT IS NOT NULL
    ORDER BY ID
    {limit_clause}
    {offset_clause}
    """
    cur.execute(query)

    rows = cur.fetchall()

    docs: list[Document] = []
    for row in tqdm(rows, desc="Loading SQLite docs", unit="doc"):
        rec = dict(row)
        raw = rec.get("content") or ""
        doc_id = str(rec.get("doc_id"))
        node_id = f"doc-{doc_id}"
        docs.append(Document(text=raw, id_=node_id, metadata={"doc_id": doc_id}))

    conn.close()
    print(f"✅ Loaded {len(docs)} documents from {sqlite_path}")
    return docs


def build_index(
    sqlite_path: str = "laws.sqlite",
    pg_table: str = "laws_vector_table_v2_512",   # ✅ NEW table (old remains untouched)
    storage_dir: str = DEFAULT_STORAGE_DIR_512,   # ✅ NEW storage dir (old remains untouched)
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    # ✅ Set chunking for THIS build (512/64)
    Settings.node_parser = SimpleNodeParser(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Ensure storage dir exists (even if the job crashes later)
    _ensure_dir(storage_dir)

    # Decide whether to drop target table (default: NO)
    if DROP_TARGET_TABLE:
        print(f"⚠️ DROP_TARGET_TABLE=1 -> dropping target vector table: {pg_table}")
        engine = create_engine(sync_str)
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {pg_table};"))
            # Also try dropping the actual data_ table if it exists
            conn.execute(text(f"DROP TABLE IF EXISTS public.{_resolve_data_table_name(pg_table)};"))
    else:
        print(f"ℹ️ Target table will NOT be dropped (set DROP_TARGET_TABLE=1 to force drop).")

    print("1/3 Loading documents from SQLite...")

    # 1) Load docs
    all_docs = load_documents_from_sqlite(sqlite_path)

    # Resume mode: skip doc_ids already present in the target data table
    if RESUME_IF_EXISTS:
        pg_engine = create_engine(sync_str)
        existing = _fetch_existing_doc_ids(pg_engine, pg_table)
        if existing:
            before = len(all_docs)
            all_docs = [d for d in all_docs if d.metadata.get("doc_id") not in existing]
            after = len(all_docs)
            print(f"🔁 RESUME_IF_EXISTS=1 -> skipping {before - after} already-indexed docs; remaining: {after}")

    # IMPORTANT: minimize metadata to avoid huge DB bloat (drops _node_content etc.)
    for d in all_docs:
        did = d.metadata.get("doc_id") or ""
        d.metadata = {"doc_id": did}

    print("2/3 Inserting raw texts into Postgres (laws_text_table)...")

    # Insert raw document texts into separate table (safe, unchanged behavior)
    text_engine = create_engine(sync_str)
    with text_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS laws_text_table (
                doc_id TEXT PRIMARY KEY,
                content TEXT
            );
        """))
        for doc in tqdm(all_docs, desc="Inserting texts", unit="doc"):
            conn.execute(
                text("""
                    INSERT INTO laws_text_table (doc_id, content)
                    VALUES (:id, :content)
                    ON CONFLICT (doc_id) DO NOTHING;
                """),
                {"id": doc.node_id, "content": doc.text}
            )
    print(f"📝 Inserted {len(all_docs)} document texts into laws_text_table")

    # 2) Instantiate PGVectorStore in NEW table
    vector_store = PGVectorStore.from_params(
        database=PG_CONFIG['database'],
        host=PG_CONFIG['host'],
        port=PG_CONFIG['port'],
        user=PG_CONFIG['user'],
        password=PG_CONFIG['password'],
        table_name=pg_table,
        embed_dim=768,
        async_connection_string=async_str,
        hnsw_kwargs={
            'hnsw_m': 16,
            'hnsw_ef_construction': 64,
            'hnsw_ef_search': 40,
            'hnsw_dist_method': 'vector_cosine_ops'
        }
    )

    # 3) Build & persist into NEW storage dir
    print("3/3 Building vector index (embedding + upsert)...")
    print(f"🔧 table={pg_table}, storage_dir={storage_dir}, chunk={chunk_size}/{chunk_overlap}")

    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        index_store=SimpleIndexStore(),
        vector_store=vector_store,
        persist_dir=storage_dir
    )
    # Build nodes explicitly so we can trim per-chunk metadata before insert
    print("🔪 Creating nodes (chunking) ...")
    nodes = Settings.node_parser.get_nodes_from_documents(all_docs)
    for n in nodes:
        did = (n.metadata or {}).get("doc_id") or ""
        n.metadata = _minimize_metadata(did, chunk_size, chunk_overlap)

    print(f"📦 Nodes to embed+upsert: {len(nodes)}")
    _ = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    storage_context.persist()
    print("✅ Index built and stored in PostgreSQL and local storage (new table, old untouched).")

    # Sanity check: confirm persisted files exist
    expected = ["docstore.json", "index_store.json"]
    existing_files = [p for p in expected if os.path.exists(os.path.join(storage_dir, p))]
    print(f"📁 Persisted in {storage_dir}: {existing_files}")


if __name__ == '__main__':
    build_index(
        sqlite_path=paths.HARVESTER_DB,
        pg_table="laws_vector_table_v2_512",
        storage_dir=DEFAULT_STORAGE_DIR_512,
        chunk_size=512,
        chunk_overlap=64
    )