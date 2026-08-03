from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import glob
import argparse
import uuid
from pathlib import Path
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
#from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRelevance
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRelevance
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import llm_factory
# from ragas.embeddings import embedding_factory
# from ragas.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # (or from langchain_huggingface import HuggingFaceEmbeddings)
from ragas.embeddings import LangchainEmbeddingsWrapper

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI




# -----------------------------
# Defaults (edit if needed)
# -----------------------------
DEFAULT_DATA_DIR = paths.DENSE_DATASET  # any folder of per-question JSONs; see --data_dir
OUT_DIR = paths.RESULTS_DIR             # output folder
OUT_FILE = "ragas_scores2.json"  #"ragas_scores_baseline.json" or "ragas_scores.json"       # single file with averages + items

# Evaluator backends
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL = "gpt-4.1-mini" #"gpt-4o-mini"  # choose any OpenAI model you prefer
EMBED_MODEL = "intfloat/multilingual-e5-base"


# -----------------------------
# Helpers
# -----------------------------
def load_docs(folder: str):
    """
    Read per-document JSON files with schema:
      {
        "question": "...",
        "answer": "...",
        "context": { "1": "...", "2": "...", ... },
        "doc_ids": ["7610", "14637", "18556"]  # optional, user-helpful only
      }

    Notes:
    - `context` is expected to be a mapping from string keys to string chunks.
    - `doc_ids` is ignored by the evaluator and used only for user reference.

    Returns a flat list of rows:
      {
        "id": "<filename_stem>#1",
        "question": "...",
        "answer": "...",
        "contexts": ["...", "..."]
      }
    """
    rows = []
    for path in sorted(glob.glob(str(Path(folder) / "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        # Extract core fields from the new schema
        q = doc.get("question")
        a = doc.get("answer")
        ctx = doc.get("context", {})  # can be dict, list, or string

        # Normalize contexts into a list of strings
        contexts: list[str] = []

        if isinstance(ctx, dict):
            # Sort by key for deterministic order
            for k in sorted(ctx.keys(), key=str):
                v = ctx[k]
                if isinstance(v, str) and v.strip():
                    contexts.append(v)
        elif isinstance(ctx, list):
            for v in ctx:
                if isinstance(v, str) and v.strip():
                    contexts.append(v)
        elif isinstance(ctx, str):
            if ctx.strip():
                contexts.append(ctx)

        # Minimal validation: require non-empty question, answer, and contexts
        if not q or not a or not contexts:
            continue

        # Use filename stem as a stable document identifier
        doc_id = Path(path).stem

        rows.append({
            "id": f"{doc_id}#1",
            "question": q,
            "answer": a,
            "contexts": contexts,
        })

    return rows


def ensure_env_for_openai():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or not key.strip():
        raise RuntimeError("OPENROUTER_API_KEY not set in environment (.env).")
    # openai library and langchain_openai look for OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = key


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Folder containing per-document JSON files")
    ap.add_argument("--out_dir", default=OUT_DIR, help="Output folder")
    ap.add_argument("--out_file", default=OUT_FILE, help="Single JSON file to write (averages + items)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of items (0 = all)")
    ap.add_argument("--max_parallel", type=int, default=1, help="Max concurrent evaluation tasks (keep at 1 to minimize rate-limit risk)")
    ap.add_argument("--rpm", type=int, default=6, help="Target requests per minute (conservative by default to avoid rate limits)")
    ap.add_argument("--max_retries", type=int, default=10, help="Max retries per item on 429s with exponential backoff")
    ap.add_argument("--max_backoff", type=int, default=300, help="Max seconds to sleep between retries on 429s (cap)")
    ap.add_argument("--metric_retries", type=int, default=2, help="Max extra retries per item if any metric is NaN (0 = no extra retries)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Load & flatten dataset
    rows = load_docs(args.data_dir)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("No valid QA entries found. Check your folder and schema.")

    ds = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        for r in rows
    ])

    # 2) Configure evaluator backends (LLM + embeddings)
    ensure_env_for_openai()

    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    # llm = llm_factory(OPENAI_MODEL, client=openai_client)
    langchain_llm = ChatOpenAI(
            model=OPENAI_MODEL, 
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    llm = LangchainLLMWrapper(langchain_llm)

    
    # embeddings = embedding_factory(EMBED_MODEL)
    # embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # embeddings = HuggingFaceEmbeddings(model=EMBED_MODEL)
    # hf_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # hf_embeddings = HuggingFaceEmbeddings(model=EMBED_MODEL)
    # embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    lc_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

    metrics_list = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRelevance(llm=llm),
    ]
    # metrics_list = [
    #     faithfulness,
    #     answer_relevancy,
    #     context_relevance,
    # ]    

    # 3) Run RAGAS sequentially with rate limiting to avoid RPM caps
    # Compute a conservative delay between items based on desired RPM
    rpm = max(1, int(args.rpm))
    base_delay = max(1, int(math.ceil(60 / rpm)))  # seconds between items

    per_item_rows = []

    def _is_rate_limit_error(e: Exception) -> bool:
        msg = str(e)
        return (
            "429" in msg
            or "Rate limit" in msg
            or "Too Many Requests" in msg
            or "rate_limit" in msg.lower()
            or "ratelimit" in msg.lower()
        )

    for idx, r in enumerate(rows):
        if (idx % 10) == 0:
            print(f"[PROGRESS] {idx}/{len(rows)} ({100*idx/len(rows):.1f}%) items evaluated...")
            # print(f"[PROGRESS] {idx}/{len(rows)} items evaluated...")

        item_ds = Dataset.from_list([
            {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        ])

        attempts = 0
        delay = base_delay
        metric_attempts = 0  # extra retries when metrics come back as NaN

        while True:
            try:
                result = evaluate(
                    item_ds,
                    metrics=metrics_list,
                    raise_exceptions=False,
                )
                df1 = result.to_pandas()
                # print(f"Available columns: {df1.columns.tolist()}")
                # Available columns: ['user_input', 'retrieved_contexts', 'response', 'faithfulness', 'answer_relevancy', 'nv_context_relevance']


                # Check for NaN metrics and optionally retry a few times
                def _is_nan(v):
                    try:
                        return math.isnan(float(v))
                    except (TypeError, ValueError):
                        return False

                has_nan = any(
                    _is_nan(df1.loc[0, col])
                    for col in ["faithfulness", "answer_relevancy", "nv_context_relevance"]
                    # context_relevancy
                )

                if has_nan and metric_attempts < args.metric_retries:
                    metric_attempts += 1
                    print(
                        f"[WARN] NaN metric on item {idx+1}/{len(rows)}; "
                        f"retrying metrics ({metric_attempts}/{args.metric_retries})..."
                    )
                    time.sleep(delay)
                    continue

                # Store whatever metrics we have (may still contain NaNs if retries exhausted)
                per_item_rows.append({
                    "id": r["id"],
                    "question": r["question"],
                    "faithfulness": float(df1.loc[0, "faithfulness"]),
                    "answer_relevancy": float(df1.loc[0, "answer_relevancy"]),
                    "context_relevance": float(df1.loc[0, "nv_context_relevance"]),
                    # "context_relevance": float(df1.loc[0, "context_relevance"]),
                })
                print(f"  [{r['id']}] F={df1.loc[0,'faithfulness']:.3f}  AR={df1.loc[0,'answer_relevancy']:.3f}  CR={df1.loc[0,'nv_context_relevance']:.3f}")      

                # If we had to retry, add a small cool-down to reduce the chance of immediate re-limit
                if attempts > 0:
                    time.sleep(min(5, base_delay))

                break
            except Exception as e:
                attempts += 1
                # On rate limits, WAIT and retry (this run is meant to be long-running and self-healing)
                if _is_rate_limit_error(e):
                    if attempts <= args.max_retries:
                        # jitter helps avoid synchronized retry bursts
                        jitter = 0.25 + (0.25 * (attempts % 3))
                        sleep_for = min(int(delay * jitter), int(args.max_backoff))
                        sleep_for = max(1, sleep_for)
                        print(
                            f"[WARN] 429/rate-limit on item {idx+1}/{len(rows)}; sleeping {sleep_for}s and retrying "
                            f"(attempt {attempts}/{args.max_retries})..."
                        )
                        time.sleep(sleep_for)
                        delay = min(delay * 2, args.max_backoff)
                        continue
                    else:
                        print(
                            f"[ERROR] Rate-limited too many times on item {idx+1}/{len(rows)}; "
                            "recording NaNs and continuing."
                        )
                # Any other exception or exhausted retries: record NaNs and continue
                print(f"[ERROR] Item {idx+1} failed permanently: {e}")
                per_item_rows.append({
                    "id": r["id"],
                    "question": r["question"],
                    "faithfulness": float("nan"),
                    "answer_relevancy": float("nan"),
                    "context_relevance": float("nan"),
                })
                break

        # Respect base pacing even after a success
        if idx < len(rows) - 1:
            time.sleep(base_delay)

    # 4) Build single-file output (overall + per-item)
    # Compute averages ignoring NaNs
    import pandas as pd
    df = pd.DataFrame(per_item_rows)
    averages = {
        "faithfulness": float(df["faithfulness"].mean(skipna=True)),
        "answer_relevancy": float(df["answer_relevancy"].mean(skipna=True)),
        "context_relevance": float(df["context_relevance"].mean(skipna=True)),
    }

    combined = {
        "run_id": str(uuid.uuid4()),
        "count": int(len(df)),
        "averages": averages,
        "items": per_item_rows,
    }

    out_path = str(Path(args.out_dir) / args.out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Saved combined results → {out_path}")
    print("Averages:", combined["averages"])


if __name__ == "__main__":
    main()