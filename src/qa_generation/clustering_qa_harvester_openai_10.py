import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community.community_louvain as community_louvain
from sqlalchemy import create_engine, inspect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import re
import time
from pydantic import BaseModel, Field, ValidationError
from enum import Enum
from typing import List, Dict
from dotenv import load_dotenv

from openai import OpenAI
load_dotenv(dotenv_path=".env")

OPENAI_MODEL_DEFAULT = "gpt-4.1-mini"
client = OpenAI()

# ---------------- Debug helper ----------------
def dbg(msg: str):
    print(f"[DBG] {msg}", flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)

def err(msg: str):
    print(f"[ERR] {msg}", flush=True)

# Maximum characters of document to send in prompt (avoid LLM context limits)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))

# --- QA Schema Definitions --
class QTypeEnum(str, Enum):
    factual = 'factual'
    inferential = 'inferential'
    analytical = 'analytical'

class DifficultyEnum(str, Enum):
    easy = 'easy'
    medium = 'medium'
    hard = 'hard'

class QAQualityEnum(str, Enum):
    good = 'good'
    fair = 'fair'
    poor = 'poor'

class QAPairMetaData(BaseModel):
    question_type: QTypeEnum = Field(description="Question type")
    difficulty: DifficultyEnum = Field(description="Question difficulty")
    required_context: str = Field(description="Specific quote string from the document needed to answer the question")
    reasoning: str = Field(description="A brief description of how you arrived at the answer from the context")
    q_a_quality: QAQualityEnum = Field(description="Your assessment of the quality of the Q-A pair")

class QAPair(BaseModel):
    question: str
    metadata: QAPairMetaData
    answer: str

class QAPairList(BaseModel):
    qa_pairs: List[QAPair]

# Pydantic v2 provides model_json_schema(); Pydantic v1 uses schema()
try:
    QA_SCHEMA = QAPairList.model_json_schema()  # type: ignore[attr-defined]
except AttributeError:
    QA_SCHEMA = QAPairList.schema()  # Pydantic v1

# --- Prompt Templates and LLM Interaction ---
def generate_question_prompt(text: str) -> str:
    return f"""
You are a Greek legal assistant AI helping diverse users understand legal documents.

Generate exactly 3 realistic questions in Greek, each from a different user perspective and difficulty level.
The 3 questions should vary in **difficulty** (easy, medium, hard), with at least one encouraging deeper reasoning or interpretation.
Also the questions should vary in type (factual, inferential, analytical).
═══════════════════════════════════════════════════

PERSONAS:
• ΠΟΛΙΤΗΣ (Citizen): Practical concerns, simple language
• ΕΠΑΓΓΕΛΜΑΤΙΑΣ (Professional): Specific procedures, compliance
• ΕΠΙΧΕΙΡΗΜΑΤΙΑΣ (Business): Requirements, obligations, timelines

═══════════════════════════════════════════════════

GENERATE 3 QUESTIONS (one per level):

[1] ΕΎΚΟΛΟ (EASY) - Απλή πληροφορία
Ask about a specific fact: number, date, name, definition.
Examples:
✓ "Πόσα ευρώ κοστίζει η ανανέωση της άδειας οδήγησης;"
✓ "Ποιος φορέας εκδίδει τις άδειες λειτουργίας εστιατορίων στην Αθήνα;"
✓ "Μέχρι ποια ημερομηνία πρέπει να καταβάλω τα τέλη κυκλοφορίας;"

[2] ΜΕΤΡΙΟ (MEDIUM) - Κατανόηση/Εφαρμογή  
Ask WHY something is required, or HOW a process works.
Examples:
✓ "Γιατί η βεβαίωση από το ΙΚΑ είναι υποχρεωτική για την αίτηση σύνταξης;"
✓ "Πώς υπολογίζεται το πρόστιμο για καθυστέρηση καταβολής ΦΠΑ;"
✓ "Ποια η διαφορά μεταξύ της κύριας σύνταξης και της επικουρικής για δημοσίους υπαλλήλους;"

[3] ΔΥΣΚΟΛΟ (HARD) - Σενάριο με συγκεκριμένες λεπτομέρειες
Ask about a specific real-world situation with concrete details.
Include: WHO exactly, WHAT specifically, WHERE/WHEN if relevant.
Examples:
✓ "Αν η εταιρεία μου καθυστερήσει την καταβολή ΦΠΑ κατά 3 μήνες, πόσο πρόστιμο θα επιβληθεί;"
✓ "Μπορώ να ανοίξω κομμωτήριο στην Αθήνα χωρίς άδεια από το Επιμελητήριο Αθηνών;"
✓ "Τι γίνεται αν ο εργοδότης μου δεν κατέβαλε τις εισφορές μου στο ΙΚΑ για 2 χρόνια;"

═══════════════════════════════════════════════════

❌ AVOID:
- "Ποια είναι η διαδικασία για..."
- "Ποιες είναι οι προϋποθέσεις..."
- "Σύμφωνα με το έγγραφο..."
- "Με βάση τον νόμο X..."

═══════════════════════════════════════════════════

REQUIREMENTS:
- All questions answerable from the document
- Natural Greek phrasing
- Independent questions
- Vary persona and difficulty

═══════════════════════════════════════════════════

<start of document>
{text}
<end of document>

Output: Valid JSON array of exactly 3 strings.
"""

def generate_answer_prompt(text: str, questions_json: str) -> str:
    return f"""
You are a highly reliable Greek legal assistant AI.  
Provide precise, well-reasoned answers in Greek to each question based strictly on the document.
Answer only from the document; do not use outside knowledge.
Answers should be concise yet complete. The user should be able to fully understand the answer and be left with no further questions.

Metadata guidance:
- question_type: factual | inferential | analytical
- difficulty: easy | medium | hard
- required_context: short verbatim excerpt from the doc
- reasoning: how the answer follows
- q_a_quality: good | fair | poor

Output format:
Return only a valid JSON object:
{{
  "qa_pairs": [
    {{
      "question": "<string>",
      "answer": "<string>",
      "metadata": {{
        "question_type": "<factual|inferential|analytical>",
        "difficulty": "<easy|medium|hard>",
        "required_context": "<exact text excerpt from the document>",
        "reasoning": "<brief explanation of how the answer was derived>",
        "q_a_quality": "<good|fair|poor>"
      }}
    }},
    ... (exactly 3 total items) ...
  ]
}}

<start of document>
{text}
<end of document>

<questions>
{questions_json}
</questions>
"""

def _extract_json_substr(raw: str) -> str:
    start = raw.find('{')
    end = raw.rfind('}')
    return raw[start:end+1] if start != -1 and end != -1 else raw


# --- OpenAI Chat Completions API chat helper ---
def chat_with_openai(prompt: str, model: str = OPENAI_MODEL_DEFAULT, force_json: bool = False) -> str:
    """
    Calls the OpenAI Chat Completions API.
    If force_json=True, requests a strict JSON object.
    Includes stronger backoff and waits indefinitely on rate limits.
    """
    while True:  # keep trying until success
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            time.sleep(2.0)  # 2-second pacing per request
            return text.strip()
        except Exception as e:
            msg = str(e)
            # Detailed diagnostic logging
            err("LLM request failed with exception:")
            err(f"Exception type: {type(e).__name__}")
            err(f"Exception message: {msg}")

            # Attempt to surface OpenAI-style error metadata if present
            if hasattr(e, "response") and e.response is not None:
                try:
                    err(f"HTTP status: {e.response.status_code}")
                except Exception:
                    pass
                try:
                    err(f"Headers: {e.response.headers}")
                except Exception:
                    pass
                try:
                    body = e.response.json() if hasattr(e.response, "json") else None
                    if body:
                        err(f"Response JSON: {body}")
                except Exception:
                    err("Response body exists but could not be parsed as JSON.")

            if "429" in msg or "rate limit" in msg.lower():
                wait_time = 60
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After") if hasattr(e.response, "headers") else None
                    if retry_after and retry_after.isdigit():
                        wait_time = int(retry_after)
                warn(f"Rate limit reached for {model}. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            raise  # other errors should not loop



# --- LLM-assisted JSON repair helper ---
def repair_json_with_llm(raw: str, schema_snippet: str, model_name: str, debug_tag: str = "unknown") -> str | None:
    """
    Ask the LLM to strictly convert raw text into a valid JSON object matching the expected schema.
    Returns the repaired JSON string or None if it fails.
    """
    dbg(f"[{debug_tag}] Attempting LLM-assisted JSON repair")
    fix_prompt = f"""
    You will be given some text that is *intended* to be a JSON object but may have formatting errors (unescaped quotes, line-breaks, etc.).
    Your task:
    - Return a VALID JSON OBJECT, and nothing else (no code fences).
    - It MUST match this schema shape (keys and value types):
    {schema_snippet}

    Here is the text to repair (do NOT invent content; only fix formatting):
    <raw>
    {raw}
    </raw>
    """
    try:
        fixed = chat_with_openai(fix_prompt, model=model_name, force_json=True)
        return fixed.strip()
    except Exception as e:
        warn(f"[{debug_tag}] LLM JSON repair failed: {e}")
        return None

def get_structured_qa_pairs(text: str, model_name: str = OPENAI_MODEL_DEFAULT) -> QAPairList:
    if not text or not text.strip():
        warn("get_structured_qa_pairs: empty document_text")
        return QAPairList(qa_pairs=[])

    dbg(f"Document length before truncation: {len(text)}")
    if len(text) > MAX_CONTEXT_CHARS:
        warn(f"Truncating document from {len(text)} to {MAX_CONTEXT_CHARS}")
        text = text[:MAX_CONTEXT_CHARS]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        err("OPENAI_API_KEY not set — returning empty QA list")
        return QAPairList(qa_pairs=[])

    question_prompt = generate_question_prompt(text)

    # Generate questions with simple retry on empty body
    raw_questions = ''
    for attempt in range(2):
        try:
            dbg(f"Generating questions attempt={attempt+1}")
            raw = chat_with_openai(question_prompt, model=model_name)
        except Exception as e:
            warn(f"Error during OpenAI call (questions) attempt {attempt+1}: {e}")
            continue
        raw = raw.strip()
        cleaned = re.sub(r'^>\s*', '', raw, flags=re.MULTILINE)
        cleaned = re.sub(r'```(?:json)?', '', cleaned).strip()
        dbg(f"Questions raw length={len(raw)} cleaned length={len(cleaned)}")
        if cleaned:
            raw_questions = cleaned
            break
        warn(f"Empty questions response on attempt {attempt+1}, retrying...")

    if not raw_questions:
        warn("No questions generated after retries, returning empty")
        return QAPairList(qa_pairs=[])

    if raw_questions.startswith("```"):
        raw_questions = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_questions.strip(), flags=re.IGNORECASE)
    try:
        questions = json.loads(raw_questions)
        if isinstance(questions, dict) and "questions" in questions:
            questions = questions["questions"]
        if not isinstance(questions, list) or len(questions) != 3:
            raise ValueError("Questions JSON is not a list of length 3")
        dbg(f"Parsed {len(questions)} questions successfully")
    except Exception as e:
        warn(f"Error parsing questions JSON: {e}")
        return QAPairList(qa_pairs=[])

    # Answers + metadata
    answer_prompt = generate_answer_prompt(text, questions)
    try:
        dbg("Requesting answers+metadata")
        raw_answers = chat_with_openai(answer_prompt, model=model_name, force_json=True).strip()
        raw_answers = re.sub(r'^>\s*', '', raw_answers, flags=re.MULTILINE)
        raw_answers = re.sub(r'```(?:json)?', '', raw_answers).strip()
        dbg(f"Answers raw length after cleaning={len(raw_answers)}")
    except Exception as e:
        warn(f"Error during OpenAI call (answers): {e}")
        return QAPairList(qa_pairs=[])

    # Try strict Pydantic parse
    try:
        result = QAPairList.model_validate_json(raw_answers)
        dbg(f"Pydantic validate_json success; pairs={len(result.qa_pairs)}")
        return result
    except ValidationError as ve:
        warn(f"Pydantic validate_json failed: {ve.errors() if hasattr(ve, 'errors') else ve}")

    # Fallback: extract inner JSON
    cleaned = _extract_json_substr(raw_answers)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "qa_pairs" in obj:
            obj["qa_pairs"] = [q for q in obj["qa_pairs"] if isinstance(q, dict) and "question" in q and "answer" in q]
        result = QAPairList.model_validate(obj)
        dbg(f"Pydantic model_validate success; pairs={len(result.qa_pairs)}")
        return result
    except (json.JSONDecodeError, ValidationError) as e:
        warn(f"Could not parse QA JSON even after substring extraction: {e}")

    # Final attempt: try an LLM-based repair to produce strict JSON
    repaired = repair_json_with_llm(raw_answers, json.dumps(QA_SCHEMA, ensure_ascii=False, indent=2), model_name, debug_tag="answers-repair")
    if repaired:
        try:
            result = QAPairList.model_validate_json(repaired)
            dbg(f"LLM repair succeeded; pairs={len(result.qa_pairs)}")
            return result
        except ValidationError as ve:
            warn(f"LLM repair returned invalid JSON again: {ve}")

    warn("Returning empty QA list after parse failures")
    return QAPairList(qa_pairs=[])

def dynamic_generate_qa_pairs_for_document(document: Dict) -> QAPairList:
    text = document.get("document_text")
    if not text or not isinstance(text, str) or not text.strip():
        warn("dynamic_generate_qa_pairs_for_document: missing/empty document_text")
        return QAPairList(qa_pairs=[])
    return get_structured_qa_pairs(text.strip())

def export_qa_pairs_to_jsonl(qa_records: List[Dict], output_file: str):
    all_pairs = []
    for rec in qa_records:
        src = rec["source_file"]
        for pair in rec["qa_list"].qa_pairs:
            obj = pair.model_dump()
            obj["source_file"] = src
            all_pairs.append(obj)
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump({"qa_pairs": all_pairs}, fout, ensure_ascii=False, indent=2)
    dbg(f"Wrote {len(all_pairs)} QA pairs to {output_file}")

# --- Load and Aggregate Document Embeddings (Embeddings only; no texts) ---
def load_doc_embeddings_ids_only(db_uri: str, vector_table: str) -> pd.DataFrame:
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    available = inspector.get_table_names()
    if vector_table not in available:
        raise RuntimeError(f"Vector table '{vector_table}' not found. Available tables: {available}")
    query = f"""
        SELECT
            vec.embedding,
            vec.metadata_->>'ref_doc_id' AS doc_id
        FROM {vector_table} vec
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    dbg(f"Loaded {len(df)} rows from {vector_table}")

    # Parse vectors (pgvector → list) to numpy arrays
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(json.loads(x)) if isinstance(x, str) else np.array(x)
    )

    grouped = df.groupby('doc_id').agg({
        'embedding': lambda embs: np.mean(np.vstack(embs.values), axis=0),
    }).reset_index().rename(columns={'embedding': 'doc_embedding'})

    dbg(f"Aggregated to {len(grouped)} unique doc_ids")
    grouped["doc_embedding"] = grouped["doc_embedding"].apply(
        lambda v: np.asarray(v, dtype=np.float32)
    )
    dbg(f"Doc embeddings dtype sample: {grouped['doc_embedding'].iloc[0].dtype if len(grouped) else 'N/A'}")
    return grouped

# --- Fetch texts only for specific doc_ids (laws_text_table) ---
def fetch_document_texts(db_uri: str, text_table: str, doc_ids: List[str]) -> Dict[str, str]:
    if not doc_ids:
        return {}
    engine = create_engine(db_uri)
    # Use ANY(%(ids)s) with a single array parameter (works with psycopg2)
    query = f"SELECT doc_id, content FROM {text_table} WHERE doc_id = ANY (%(ids)s)"
    params = {"ids": doc_ids}  # list[str] adapts to TEXT[] via psycopg2
    with engine.connect() as conn:
        rows = pd.read_sql(query, conn, params=params)
    dbg(f"Fetched {len(rows)} rows from {text_table} for {len(doc_ids)} ids")
    return dict(zip(rows["doc_id"].astype(str), rows["content"]))

# --- Clustering with Louvain on Documents ---
def cluster_with_louvain_embeddings(df: pd.DataFrame, k: int = 10):
    arr = np.vstack(df['doc_embedding'].values).astype(np.float32, copy=False)
    dbg(f"Clustering on arr shape={arr.shape}, dtype={arr.dtype}, k={k}")
    sim_matrix = cosine_similarity(arr)
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))
    for i in range(len(df)):
        neighbors = np.argsort(sim_matrix[i])[::-1][1:k+1]
        for j in neighbors:
            G.add_edge(i, j, weight=float(sim_matrix[i, j]))
    partition = community_louvain.best_partition(G, weight='weight')
    df['cluster'] = df.index.map(lambda idx: partition.get(idx, -1))
    counts = df['cluster'].value_counts().sort_index().to_dict()
    dbg(f"Cluster label counts: {counts}")
    return df, arr  # arr is float32

# --- Representative Document Selection ---
def sample_representative_docs(df: pd.DataFrame, arr: np.ndarray) -> pd.DataFrame:
    reps = []
    for lbl in np.unique(df['cluster']):
        if lbl == -1:
            continue
        idxs = df.index[df['cluster'] == lbl].tolist()
        centroid = arr[idxs].mean(axis=0)
        best = idxs[np.argmin(np.linalg.norm(arr[idxs] - centroid, axis=1))]
        reps.append(df.loc[best])
    rep_df = pd.DataFrame(reps)
    dbg(f"Representatives selected: {len(rep_df)}; sample ids={rep_df['doc_id'].head(5).tolist() if not rep_df.empty else []}")
    return rep_df

# --- t-SNE Visualization ---
def visualize_embeddings(arr: np.ndarray, df: pd.DataFrame, rep_df: pd.DataFrame, perplexity: int = 30, random_state: int = 42):
    n = arr.shape[0]
    if n > 500:
        warn(f"Skipping t-SNE visualization for {n} samples (too many to plot)")
        return
    arr = arr.astype(np.float32, copy=False)
    n_samples = arr.shape[0]
    perp = perplexity if perplexity < n_samples else max(1, n_samples - 1)
    dbg(f"t-SNE with perplexity={perp}, samples={n_samples}")
    two_d = TSNE(n_components=2, perplexity=perp, random_state=random_state).fit_transform(arr)
    plt.figure(figsize=(12, 10))
    mask = df['cluster'] != -1
    scatter = plt.scatter(two_d[mask,0], two_d[mask,1], c=df.loc[mask, 'cluster'], cmap='viridis', alpha=0.7)
    for _, row in rep_df.iterrows():
        x, y = two_d[row.name]
        plt.text(x, y, str(row['doc_id']), fontsize=8, weight='bold')
    plt.title("Document Distribution in 2D (t-SNE)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# --- Main Script ---

PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DATABASE")
VECTOR_TABLE = os.getenv("PG_VECTOR_TABLE", "data_laws_vector_table_v2")
TEXT_TABLE = os.getenv("PG_TEXT_TABLE", "laws_text_table")
DB_URI = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

if __name__ == "__main__":
    dbg(f"Config: DB={PG_DB}@{PG_HOST}:{PG_PORT}, VECTOR_TABLE={VECTOR_TABLE}, TEXT_TABLE={TEXT_TABLE}")
    dbg(f"Env: OPENAI_API_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'MISSING'}; MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}")

    # --- Manual override flags ---
    USE_MANUAL_SELECTION = True  # Set True to bypass clustering and use MANUAL_DOC_IDS
     # e.g., ["doc-12345", "doc-67890"]
     # ["doc-72925","doc-74019","doc-11218","doc-74200","doc-72843"] -> v1
    MANUAL_DOC_IDS = ["doc-15054","doc-74354","doc-74283","doc-27344"]

    if USE_MANUAL_SELECTION:
        # --- Manual path: skip clustering, use provided doc_ids ---
        rep_ids = [str(x) for x in MANUAL_DOC_IDS]
        if not rep_ids:
            warn("USE_MANUAL_SELECTION=True but MANUAL_DOC_IDS is empty — nothing to process.")
            rep_ids = []

        id_to_text = fetch_document_texts(DB_URI, text_table=TEXT_TABLE, doc_ids=rep_ids)
        print(f"[DIAG] (manual) fetched_texts={len(id_to_text)} / requested={len(rep_ids)}")
        missing = [rid for rid in rep_ids if rid not in id_to_text or not id_to_text[rid]]
        if missing:
            warn(f"(manual) Missing texts for {len(missing)} provided ids; sample={missing[:10]}")

        # 5) Generate QAs for the manually provided ids and export (STOP AFTER 10 PAIRS, RANDOM ORDER)
        TARGET_PAIRS = 10
        remaining = TARGET_PAIRS
        qa_records = []
        total_pairs = 0

        # Randomize manual ids order
        order = pd.Series(rep_ids).sample(frac=1, random_state=None).tolist()

        for idx, doc_id in enumerate(order, start=1):
            if remaining <= 0:
                break

            dbg(f"(manual) Progress: {idx}/{len(order)} docs processed (remaining_pairs={remaining})")
            doc_text = id_to_text.get(doc_id, "")
            dbg(f"(manual) QA for doc_id={doc_id}: text_len={len(doc_text) if doc_text else 0}")

            if not doc_text:
                warn(f"(manual) Skipping QA (no text) for doc_id={doc_id}")
                qa_records.append({"source_file": doc_id, "qa_list": QAPairList(qa_pairs=[])})
                time.sleep(1.0)
                continue

            try:
                doc = {"document_text": doc_text, "law_metadata": {}, "source_file": doc_id}
                qa_list = dynamic_generate_qa_pairs_for_document(doc)

                # Cap pairs to not exceed TARGET_PAIRS
                if len(qa_list.qa_pairs) > remaining:
                    qa_list.qa_pairs = qa_list.qa_pairs[:remaining]

                n_pairs = len(qa_list.qa_pairs)
                total_pairs += n_pairs
                remaining -= n_pairs

                dbg(f"(manual) doc_id={doc_id}: produced pairs={n_pairs}; total_so_far={total_pairs}")
                if n_pairs == 0:
                    warn(f"(manual) LLM returned 0 pairs for doc_id={doc_id}")

                qa_records.append({"source_file": doc_id, "qa_list": qa_list})
            except Exception as e:
                err(f"(manual) Exception generating QA for doc_id={doc_id}: {e}")
                qa_records.append({"source_file": doc_id, "qa_list": QAPairList(qa_pairs=[])})

            # Gentle throttle between documents to avoid RPM spikes
            time.sleep(1.0)

        export_qa_pairs_to_jsonl(qa_records, "qa_pairs_openai_10_manual.jsonl")
        print(f"\n✅ (manual) QA exported for {len(qa_records)} docs; total pairs={total_pairs} (target={TARGET_PAIRS}).")

        # Skip visualization & cluster exports in manual mode
        print("Manual mode active: skipping visualization and cluster exports.")
    else:
        # 1) Load ONLY embeddings/doc_ids (avoid loading full texts)
        docs_df = load_doc_embeddings_ids_only(DB_URI, vector_table=VECTOR_TABLE)
        print(f"Loaded {len(docs_df)} documents with embeddings.")

        # 2) Cluster on embeddings
        clustered_df, doc_arr = cluster_with_louvain_embeddings(docs_df, k=10)
        print("Clustering complete.")

        # 3) Pick representative docs per cluster
        rep_df = sample_representative_docs(clustered_df, doc_arr)
        print(f"Selected {len(rep_df)} representative docs.")

        # 4) Fetch texts ONLY for the representative docs
        rep_ids = [str(x) for x in rep_df["doc_id"].tolist()]
        id_to_text = fetch_document_texts(DB_URI, text_table=TEXT_TABLE, doc_ids=rep_ids)
        print(f"[DIAG] fetched_texts={len(id_to_text)} / reps={len(rep_ids)}")
        missing = [rid for rid in rep_ids if rid not in id_to_text or not id_to_text[rid]]
        if missing:
            warn(f"Missing texts for {len(missing)} reps; sample={missing[:10]}")

        # 5) Generate QAs for representatives and export  (STOP AFTER 10 PAIRS, RANDOM ORDER)
        TARGET_PAIRS = 10
        remaining = TARGET_PAIRS
        qa_records = []
        total_pairs = 0

        # Randomize representative order while keeping indices (for visualize_embeddings indexing)
        order = rep_df.sample(frac=1, random_state=None).index.tolist()

        for idx, rep_idx in enumerate(order, start=1):
            if remaining <= 0:
                break
            row = rep_df.loc[rep_idx]

            dbg(f"Progress: {idx}/{len(order)} representative docs processed (remaining_pairs={remaining})")
            doc_id = str(row["doc_id"])
            doc_text = id_to_text.get(doc_id, "")
            dbg(f"QA for doc_id={doc_id}: text_len={len(doc_text) if doc_text else 0}")

            if not doc_text:
                warn(f"Skipping QA (no text) for doc_id={doc_id}")
                qa_records.append({"source_file": doc_id, "qa_list": QAPairList(qa_pairs=[])})
                time.sleep(1.0)
                continue

            try:
                doc = {"document_text": doc_text, "law_metadata": {}, "source_file": doc_id}
                qa_list = dynamic_generate_qa_pairs_for_document(doc)

                # Cap pairs to not exceed TARGET_PAIRS
                if len(qa_list.qa_pairs) > remaining:
                    qa_list.qa_pairs = qa_list.qa_pairs[:remaining]

                n_pairs = len(qa_list.qa_pairs)
                total_pairs += n_pairs
                remaining -= n_pairs

                dbg(f"doc_id={doc_id}: produced pairs={n_pairs}; total_so_far={total_pairs}")
                if n_pairs == 0:
                    warn(f"LLM returned 0 pairs for doc_id={doc_id}")

                qa_records.append({"source_file": doc_id, "qa_list": qa_list})
            except Exception as e:
                err(f"Exception generating QA for doc_id={doc_id}: {e}")
                qa_records.append({"source_file": doc_id, "qa_list": QAPairList(qa_pairs=[])})

            # Gentle throttle between documents to avoid RPM spikes
            time.sleep(1.0)

        export_qa_pairs_to_jsonl(qa_records, "qa_pairs_openai_10_v2.jsonl")
        print(f"\n✅ QA exported for {len(qa_records)} docs; total pairs={total_pairs} (target={TARGET_PAIRS}).")

        # 6) Visualization & cluster counts
        visualize_embeddings(doc_arr, clustered_df, rep_df)
        print(clustered_df['cluster'].value_counts().sort_index())

        # 7) Export clusters to individual JSON files
        EXPORT_CLUSTERS = False
        if EXPORT_CLUSTERS:
            # Export each cluster's member doc_ids ordered by proximity to the cluster centroid
            # (closest to centroid first)
            clusters = {}
            for cid, group in clustered_df.groupby('cluster'):
                if cid == -1:
                    continue
                # Stack embeddings for this cluster
                embs = np.vstack(group['doc_embedding'].values).astype(np.float32, copy=False)
                # Compute centroid and distances
                centroid = embs.mean(axis=0, dtype=np.float32)
                dists = np.linalg.norm(embs - centroid, axis=1)
                # Order by increasing distance (closest first)
                order = np.argsort(dists)
                ordered_doc_ids = group.iloc[order]['doc_id'].astype(str).tolist()
                clusters[int(cid)] = ordered_doc_ids

            # Write each ordered cluster to its own JSON file
            for cid, ordered_docs in clusters.items():
                filename = f"cluster_{cid}_openai.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump({str(cid): ordered_docs}, f, ensure_ascii=False, indent=2)
                print(f"Wrote {len(ordered_docs)} docs for cluster {cid} to {filename} (closest to centroid first)")