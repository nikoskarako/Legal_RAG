import os
import sys
import json
import requests
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
import psycopg2
import time
import sqlite3
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

# --- runtime label caches ---
_LABEL_CACHE = {}       # vector id -> {label,title,url,ref_doc_id}

# --- Large artefacts that live outside the repo (see README) ---
# Both resolve to the project root by default; override with STORAGE_DIR /
# HARVESTER_DB in .env (e.g. STORAGE_DIR=storage_512 for the 512-token index).
STORAGE_DIR  = paths.STORAGE_DIR
HARVESTER_DB = paths.HARVESTER_DB

# Vector table name in Postgres (LlamaIndex PGVectorStore often prefixes with `data_`).
# Override via env if you ever change it again.
VECTOR_TABLE_NAME = os.getenv("PG_VECTOR_TABLE", "laws_vector_table_v2_512")

# --- Load API key ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ API Key not found. Check your .env file.")

# --- OpenRouter retry & fallback config ---
RETRY_COUNT = 3
BACKOFF_SECONDS = 1.0   # linear backoff per attempt
REQUEST_TIMEOUT = 60    # seconds
MODEL_FALLBACKS = [
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
]

# --- Settings ---
PG_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "database": os.getenv("PG_DATABASE", "legalchatbot2"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
Settings.node_parser = SimpleNodeParser(chunk_size=4096)


def build_index() -> VectorStoreIndex:
    "Loads the existing PGVector index from PostgreSQL + local storage."
    sync_str = (
        f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}@"
        f"{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"
    )
    async_str = sync_str.replace("postgresql://", "postgresql+asyncpg://")

    vector_store = PGVectorStore.from_params(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
        database=PG_CONFIG["database"],
        table_name=VECTOR_TABLE_NAME,
        embed_dim=768,
        async_connection_string=async_str,
        schema_name="public"
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=STORAGE_DIR
    )
    return load_index_from_storage(storage_context)


# Load index once
index = build_index()


def chat_with_openrouter(prompt: str, model: str) -> str:
    """
    Call OpenRouter with retries on transient errors and fallback models.
    Returns either the assistant message string or an error string that starts with '⚠️ Σφάλμα API:'.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    def call_model(cur_model: str):
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "Legal ChatBot",
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
                    print("⚠️ OpenRouter error payload:", json.dumps(data, ensure_ascii=False))
                    err = data.get("error", {}).get("message", "Unknown API error.")
                    return f"⚠️ Σφάλμα API: {err}"
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
                        print("⚠️ OpenRouter error payload:", json.dumps(data, ensure_ascii=False))
                        msg = data.get("error", {}).get("message", str(e))
                    except Exception:
                        msg = str(e)
                    return f"⚠️ Σφάλμα API: {msg}"
                else:
                    return f"⚠️ Σφάλμα API: {str(e)}"

    tried = [model] + [m for m in MODEL_FALLBACKS if m != model]
    for m in tried:
        result = call_model(m)
        if not (isinstance(result, str) and result.startswith("⚠️ Σφάλμα API:")):
            return result
    return result


def _sqlite_conn():
    return sqlite3.connect(HARVESTER_DB)


def _safe_str(v):
    """Return a clean string for DB values.

    - Accepts str/int/float/None
    - Never raises
    """
    if v is None:
        return None
    try:
        s = str(v)
    except Exception:
        return None
    s = s.strip()
    return s if s else None


def _parse_fek_date(v):
    """Parse fekDate stored as Unix timestamp in milliseconds.

    Example: 915487200000 -> '1999-01-01'

    Returns YYYY-MM-DD or None.
    """
    if v is None:
        return None
    try:
        ts_ms = int(v)
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def _lookup_nomos_info(ref_doc_id: str):
    """Lookup nomosTitle, fekDate and nomosCategory from SQLite harvester_full.db (table et).

    We accept ids like:
      - "384"
      - "doc-384"
      - values stored in et.ID or et.fekID

    Returns:
      {"nomosTitle": <str|None>, "fekDate": <str|None>, "nomosCategory": <str|None>}  or None
    """
    if not ref_doc_id:
        return None

    rid = _safe_str(ref_doc_id)
    if not rid:
        return None

    # Normalize: allow both "doc-384" and "384"
    digits = "".join(ch for ch in rid if ch.isdigit())

    try:
        with _sqlite_conn() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # Detect date column name if present
            try:
                cur.execute("PRAGMA table_info(et);")
                cols = {row[1] for row in cur.fetchall()}  # column names
            except Exception:
                cols = set()

            date_col = None
            for candidate in ("fekDate", "fek_date", "fekDATE", "FEKDate", "fekdate"):
                if candidate in cols:
                    date_col = candidate
                    break

            if date_col:
                select_sql = f"SELECT nomosTitle, {date_col} AS fekDate, nomosCategory FROM et WHERE {{col}} = ? LIMIT 1"
            else:
                select_sql = "SELECT nomosTitle, NULL AS fekDate, nomosCategory FROM et WHERE {col} = ? LIMIT 1"

            candidates = []
            # exact
            candidates.append(("ID", rid))
            candidates.append(("fekID", rid))

            # numeric-only variants
            if digits:
                candidates.append(("ID", digits))
                candidates.append(("fekID", digits))
                candidates.append(("ID", f"doc-{digits}"))
                candidates.append(("fekID", f"doc-{digits}"))

            for col, val in candidates:
                try:
                    cur.execute(select_sql.format(col=col), (val,))
                    row = cur.fetchone()
                except Exception as qerr:
                    print("SQLite query error:", qerr)
                    continue

                if row:
                    title = _safe_str(row["nomosTitle"]) if "nomosTitle" in row.keys() else None
                    # fekDate is a numeric timestamp in milliseconds (e.g., 915487200000)
                    fek_date = _parse_fek_date(row["fekDate"]) if "fekDate" in row.keys() else None
                    category = _safe_str(row["nomosCategory"]) if "nomosCategory" in row.keys() else None

                    if title or fek_date or category:
                        return {
                            "nomosTitle": title or None,
                            "fekDate": fek_date or None,
                            "nomosCategory": category or None,
                        }

            return None
    except Exception as e:
        print("SQLite lookup error:", e)
        return None

# --- runtime cache for full-text lookups ---
_FULL_TEXT_CACHE = {}


def lookup_full_law_text(ref_doc_id):
    """Look up a law's full text (et.fekTEXT) from HARVESTER_DB by doc id.

    Accepts ids like "384" or "doc-384"; tries both ID and fekID columns.

    Returns: {"nomosTitle", "fekDate", "nomosCategory", "text"} or None.
    """
    if not ref_doc_id:
        return None
    rid = _safe_str(ref_doc_id)
    if not rid:
        return None
    if rid in _FULL_TEXT_CACHE:
        return _FULL_TEXT_CACHE[rid]

    digits = "".join(ch for ch in rid if ch.isdigit())
    candidates = [("ID", rid), ("fekID", rid)]
    if digits:
        candidates += [
            ("ID", digits), ("fekID", digits),
            ("ID", f"doc-{digits}"), ("fekID", f"doc-{digits}"),
        ]
    select_sql = "SELECT nomosTitle, fekDate, nomosCategory, fekTEXT FROM et WHERE {col} = ? LIMIT 1"

    try:
        with _sqlite_conn() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            for col, val in candidates:
                try:
                    cur.execute(select_sql.format(col=col), (val,))
                    row = cur.fetchone()
                except Exception as qerr:
                    print("Full-text SQLite query error:", qerr)
                    continue
                if row:
                    info = {
                        "nomosTitle":    _safe_str(row["nomosTitle"]),
                        "fekDate":       _parse_fek_date(row["fekDate"]),
                        "nomosCategory": _safe_str(row["nomosCategory"]),
                        "text":          _safe_str(row["fekTEXT"]),
                    }
                    _FULL_TEXT_CACHE[rid] = info
                    return info
            return None
    except Exception as e:
        print("Full-text SQLite lookup error:", e)
        return None


def _pg_conn():
    return psycopg2.connect(
        host=PG_CONFIG["host"],
        port=PG_CONFIG["port"],
        dbname=PG_CONFIG["database"],
        user=PG_CONFIG["user"],
        password=PG_CONFIG["password"],
    )

def _extract_title_from_content(txt: str):
    if not txt:
        return None
    # Use first meaningful short-ish line as a title
    lines = [l.strip() for l in txt.splitlines()[:30]]
    lines = [l for l in lines if l]
    if not lines:
        return None
    # Prefer lines that look like headings
    for l in lines:
        up = l.upper()
        if up.startswith(("ΝΟΜΟΣ", "Π.Δ", "Π. Δ", "ΑΠΟΦ", "ΑΡΘΡΟ", "ΚΕΦΑΛΑΙΟ", "ΝΟΜΟΣ ΥΠ")):
            return l[:180]
    # Fallback: first non-empty reasonably sized line
    for l in lines:
        if 5 <= len(l) <= 180:
            return l
    return lines[0][:180]

def _lookup_label_runtime(vector_id: str):
    """
    Enrich a source with legal metadata using ONLY SQLite (harvester_full.db, table et).

    Returns dict:
      {
        "label": <str>,
        "nomosTitle": <str|None>,
        "fekDate": <str|None>,
        "nomosCategory": <str|None>,
        "url": None,
        "ref_doc_id": None
      }
    """
    vid = str(vector_id).strip()

    if vid in _LABEL_CACHE:
        return _LABEL_CACHE[vid]

    nm = _lookup_nomos_info(vid)

    if nm and (nm.get("nomosTitle") or nm.get("fekDate") or nm.get("nomosCategory")):
        lbl_parts = [p for p in [nm.get("nomosTitle"), nm.get("fekDate"), nm.get("nomosCategory")] if p]
        label = " | ".join(lbl_parts) if lbl_parts else vid
    else:
        label = vid

    info = {
        "label": label,
        "nomosTitle": nm.get("nomosTitle") if nm else None,
        "fekDate": nm.get("fekDate") if nm else None,
        "nomosCategory": nm.get("nomosCategory") if nm else None,
        "url": None,
        "ref_doc_id": None,
    }

    _LABEL_CACHE[vid] = info
    return info

def _node_to_source(node):
    """Extract a compact source record from a retrieved node, enriched via runtime DB lookup.

    NOTE: In our PGVector table, node.metadata['doc_id'] is usually the *source document id* (e.g. doc-384).
    We use that id to lookup nomosTitle/fekDate/nomosCategory from SQLite harvester_full.db (table et).
    """
    meta = getattr(node, "metadata", {}) or {}

    # Prefer the actual source doc id (e.g. doc-384)
    raw_id = meta.get("doc_id") or meta.get("document_id") or meta.get("ref_doc_id") or ""
    source_doc_id = _safe_str(raw_id) or ""
    enriched = _lookup_label_runtime(source_doc_id)

    return {
        "doc_id": source_doc_id,
        "nomosTitle": enriched.get("nomosTitle"),
        "fekDate": enriched.get("fekDate"),
        "nomosCategory": enriched.get("nomosCategory"),
        "url": enriched.get("url"),
        "label": enriched.get("label"),
    }

def _unique_sources(nodes, limit=5):
    """Preserve order, dedupe by doc_id/title/url, truncate to limit."""
    seen = set()
    out = []
    for n in nodes:
        s = _node_to_source(n)
        key = (s.get("doc_id"), s.get("nomosTitle"), s.get("url"))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out

def _label_from_meta(meta: dict) -> str:
    """
    Pick the best available label for display.
    Preference order: title > law_title > law_no > fek > doc_id.
    """
    if not meta:
        return "Πηγή"
    for k in ("title", "law_title", "law_no", "fek"):
        v = meta.get(k)
        if v:
            return str(v)
    return str(meta.get("doc_id") or "Πηγή")


def process_query(query: str, settings_json: str) -> str:
    settings = json.loads(settings_json)
    model   = settings.get("model", "deepseek/deepseek-chat-v3-0324")

    retriever = index.as_retriever(similarity_top_k=50)
    results   = retriever.retrieve(query)

    print("🔍 Raw retrieval:")
    for i, node in enumerate(results):
        txt = node.text or ""
        print(f" #{i}: len={len(txt)}, meta={node.metadata}, head={repr(txt[:30])}")

    top = results[:10]
    doc_ids = [getattr(node, "metadata", {}).get("doc_id") for node in top]
    print(f"📝 Retrieved doc_ids: {doc_ids}")

    if not top:
        payload = {
            "answer": "❗ Δεν βρέθηκαν σχετικά νομικά κείμενα.",
            "sources": [],
            "cited_sources": [],
            "sources_numbered": []  # S-legend
        }
        return json.dumps(payload, ensure_ascii=False)

    # Build numbered context S1..Sk so the LLM can cite with [S#]
    numbered_ctx = []

    sources_numbered = []  # legend aligned with S#
    for idx, node in enumerate(top, start=1):
        doc_id_val = getattr(node, "metadata", {}).get("doc_id", "")
        numbered_ctx.append(
            f"[S{idx}] (doc_id: {str(doc_id_val)})\n{node.text}\n"
        )
        s = _node_to_source(node)
        s["s_index"] = idx
        s["chunk_text"] = node.text or ""
        sources_numbered.append(s)

    ctx = "\n\n".join(numbered_ctx)
    prompt = (
        "Βάσει ΜΟΝΟ των παρακάτω ελληνικών νομικών κειμένων, απάντησε στα Ελληνικά.\n"
        "Όταν χρησιμοποιείς πληροφορία από συγκεκριμένη πηγή, βάλε παραπομπή στο τέλος της πρότασης με τη μορφή [S#], "
        "όπου # ο αριθμός της πηγής όπως δίνεται.\n\n"
        f"Ερώτηση:\n{query}\n\n"
        f"Νομικά κείμενα:\n{ctx}\n"
    )
    final_answer = chat_with_openrouter(prompt, model)

    # If provider returned an error string, wrap it in JSON so the UI can render gracefully
    if isinstance(final_answer, str) and final_answer.startswith("⚠️ Σφάλμα API:"):
        payload = {
            "answer": final_answer,
            "sources": [],
            "cited_sources": [],
            "sources_numbered": []
        }
        return json.dumps(payload, ensure_ascii=False)

    # Retrieved top (dedup & trimmed)
    seen, sources_retr = set(), []
    for n in top:
        s = _node_to_source(n)
        key = (s.get("doc_id"), s.get("nomosTitle"), s.get("url"))
        if key in seen:
            continue
        seen.add(key)
        sources_retr.append(s)
        if len(sources_retr) >= 5:
            break

    # Parse cited markers — accept [S1], [S1,S2], or [S1, S2, S3] (any whitespace).
    # Each bracket is one citation token; we extract every S# inside it.
    import re
    cited_idx = []
    for m in re.finditer(r"\[\s*S\d+(?:\s*,\s*S?\d+)*\s*\]", final_answer):
        for s_match in re.finditer(r"\d+", m.group(0)):
            k = int(s_match.group(0))
            if 1 <= k <= len(top) and k not in cited_idx:
                cited_idx.append(k)

    cited_sources = []
    # Use the same objects as sources_numbered so S# -> citation stays consistent
    by_s = {s.get("s_index"): s for s in sources_numbered}
    for k in cited_idx:
        s = by_s.get(k)
        if not s:
            continue
        cited_sources.append(s)

    # Deduplicate citations by underlying document (doc_id) while preserving which [S#] were used.
    # This is what the UI should typically render (one entry per document).
    cited_sources_dedup = []
    _seen_docs = {}
    for s in cited_sources:
        doc_key = _safe_str(s.get("doc_id")) or ""
        if doc_key not in _seen_docs:
            _seen_docs[doc_key] = {
                "doc_id": doc_key,
                "nomosTitle": s.get("nomosTitle"),
                "fekDate": s.get("fekDate"),
                "nomosCategory": s.get("nomosCategory"),
                "url": s.get("url"),
                "label": s.get("label"),
                "s_indices": [s.get("s_index")],
            }
        else:
            _seen_docs[doc_key]["s_indices"].append(s.get("s_index"))

            # Prefer first non-empty metadata (in case some entries miss fields)
            for key in ("nomosTitle", "fekDate", "nomosCategory", "url", "label"):
                if not _seen_docs[doc_key].get(key) and s.get(key):
                    _seen_docs[doc_key][key] = s.get(key)

    # Preserve insertion order
    for _k in _seen_docs:
        # keep S indices sorted and unique
        _seen_docs[_k]["s_indices"] = sorted({i for i in _seen_docs[_k]["s_indices"] if isinstance(i, int)})
        cited_sources_dedup.append(_seen_docs[_k])

    payload = {
        "answer": final_answer,            # includes inline [S#]
        "sources": sources_retr,           # retrieved (ranked)
        "cited_sources": cited_sources,    # actually cited
        "cited_sources_dedup": cited_sources_dedup,  # deduplicated by doc_id, with s_indices
        "sources_numbered": sources_numbered  # legend: S# → doc
    }
    return json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    print("🔎 Greek Legal QA CLI")
    while True:
        q = input("\n💬 Ερώτηση: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        settings = {"model": "deepseek/deepseek-chat-v3-0324:free"}
        ans_json = process_query(q, json.dumps(settings))
        try:
            parsed = json.loads(ans_json)
            print("\n📜 Απάντηση:\n", parsed.get("answer", ""))
            if parsed.get("sources"):
                print("\n🔗 Πηγές:")
                for i, s in enumerate(parsed["sources"], 1):
                    label = s.get("label") or s.get("title") or s.get("doc_id") or s.get("url") or "Source"
                    url = s.get("url")
                    if url:
                        print(f"  {i}. {label} – {url}")
                    else:
                        print(f"  {i}. {label}")
        except Exception:
            # Fallback if something unexpected happens
            print("\n📜 Απάντηση:\n", ans_json)