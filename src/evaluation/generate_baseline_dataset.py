from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import glob
import time
import math
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
# The dense dataset is read only to reuse its question list; answers are then
# generated zero-shot, with no retrieved context.
INPUT_DIR  = paths.DENSE_DATASET
OUTPUT_DIR = paths.BASELINE_DATASET
MODEL      = "deepseek/deepseek-chat-v3-0324"
RPM        = 10          # requests per minute (conservative)
RETRY_COUNT    = 3
BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT = 60

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in .env")


# ── OpenRouter call (same pattern as chat_openrouter.py) ─────────────────────
def call_llm_zero_shot(question: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Legal ChatBot Baseline",
    }
    prompt = (
        "Απάντησε στα Ελληνικά στην παρακάτω νομική ερώτηση "
        "βάσει των γνώσεών σου:\n\n"
        f"Ερώτηση: {question}"
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data:
                err = data.get("error", {}).get("message", "Unknown API error")
                raise RuntimeError(f"OpenRouter error: {err}")
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retriable = status in (429, 500, 502, 503, 504) or status is None
            if retriable and attempt < RETRY_COUNT:
                sleep = BACKOFF_SECONDS * attempt
                print(f"  [retry {attempt}] sleeping {sleep}s...")
                time.sleep(sleep)
                continue
            raise


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(INPUT_DIR) / "*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in {INPUT_DIR}")

    base_delay = max(1, int(math.ceil(60 / RPM)))
    total = len(files)

    print(f"Generating zero-shot baseline for {total} questions → {OUTPUT_DIR}/")
    print(f"Model: {MODEL}  |  RPM: {RPM}  |  delay: {base_delay}s\n")

    for idx, path in enumerate(files):
        fname = Path(path).name

        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        question = doc.get("question", "")
        context  = doc.get("context", {})
        doc_ids  = doc.get("doc_ids", [])

        if not question:
            print(f"[SKIP] {fname} — no question field")
            continue

        if (idx % 10) == 0:
            print(f"[PROGRESS] {idx}/{total} ({100*idx/total:.1f}%)...")

        try:
            answer = call_llm_zero_shot(question)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            answer = ""

        out = {
            "question": question,
            "answer":   answer,
            "context":  context,
            "doc_ids":  doc_ids,
        }

        out_path = Path(OUTPUT_DIR) / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"  [{fname}] done — answer length: {len(answer)} chars")

        if idx < total - 1:
            time.sleep(base_delay)

    print(f"\nDone. {total} files written to {OUTPUT_DIR}/")
    print(f"Next step: python evaluate_ragas.py --data_dir {OUTPUT_DIR} --out_file ragas_scores_baseline_new.json")


if __name__ == "__main__":
    main()
