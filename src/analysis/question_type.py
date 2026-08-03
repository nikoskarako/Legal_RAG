"""Break RAGAS scores down by question type (factual / analytical / inferential)
and by difficulty."""
import json, csv, os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

# Load question metadata from CSV, keyed by question text.
#
# 12 of the 253 questions share their wording with another row, so this dict
# holds 240 keys rather than 253. That is safe for the breakdown below: every
# duplicated text that involves an *accepted* question agrees on both type and
# difficulty, and the single conflicting pair (rows 48 and 156, inferential vs
# analytical) is rejected on both rows and so is never scored. Join on `id`
# instead (as oracle.py does) if you extend this to the rejected questions.
meta = {}
with open(paths.QUESTIONS_CSV, newline="", encoding="utf-8-sig") as f:
    # Comma-delimited: the Greek question mark is ';', so every question ends
    # in a semicolon and a ';' delimiter survives only via quoting.
    for row in csv.DictReader(f):
        meta[row["question"].strip()] = {
            "question_type": row["question_type"].strip(),
            "difficulty": row["difficulty"].strip(),
        }

print(f"CSV rows loaded: {len(meta)}")
if meta:
    sample_key = next(iter(meta))
    print(f"  sample key: {repr(sample_key[:60])}")

def stats_by(path, label):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    items = d.get("items", d if isinstance(d, list) else [])
    print(f"{label}: top-level keys={list(d.keys()) if isinstance(d, dict) else 'list'}, n_items={len(items)}")
    if items:
        print(f"  first item keys: {list(items[0].keys())}")
    by_type_f  = defaultdict(list)
    by_type_ar = defaultdict(list)
    by_diff_f  = defaultdict(list)
    by_diff_ar = defaultdict(list)
    unmatched = []
    for item in items:
        q = item["question"].strip()
        if q not in meta:
            unmatched.append(item["id"])
            continue
        qt   = meta[q]["question_type"]
        diff = meta[q]["difficulty"]
        by_type_f[qt].append(item["faithfulness"])
        by_type_ar[qt].append(item["answer_relevancy"])
        by_diff_f[diff].append(item["faithfulness"])
        by_diff_ar[diff].append(item["answer_relevancy"])
    if unmatched:
        print(f"  UNMATCHED ({len(unmatched)}): {unmatched[:5]}")
    print(f"=== {label} — by question type ===")
    for t in ["factual", "analytical", "inferential"]:
        f  = np.array(by_type_f.get(t, []))
        ar = np.array(by_type_ar.get(t, []))
        if len(f):
            print(f"  {t}: n={len(f)}  F mean={f.mean():.3f} median={np.median(f):.3f}  AR mean={ar.mean():.3f}")
    print(f"=== {label} — by difficulty ===")
    for diff in ["easy", "medium", "hard"]:
        f  = np.array(by_diff_f.get(diff, []))
        ar = np.array(by_diff_ar.get(diff, []))
        if len(f):
            print(f"  {diff}: n={len(f)}  F mean={f.mean():.3f} median={np.median(f):.3f}  AR mean={ar.mean():.3f}")

stats_by(os.path.join(paths.RESULTS_DIR, "ragas_scores_dense_new.json"), "Dense")
stats_by(os.path.join(paths.RESULTS_DIR, "ragas_scores_bm25.json"), "BM25")
