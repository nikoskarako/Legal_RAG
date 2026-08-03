"""Break the per-question RAGAS scores into interpretable failure categories."""
import json, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

base = paths.RESULTS_DIR

def load(fname):
    items = json.load(open(os.path.join(base, fname), encoding="utf-8"))["items"]
    return {r["id"]: r for r in items}

dense  = load("ragas_scores_dense_new.json")
bm25   = load("ragas_scores_bm25.json")
hybrid = load("ragas_scores_hybrid.json")

def cr(r): return r.get("context_relevance") or r.get("context_relevancy") or 0.0

def cat1(d): return r["faithfulness"] < 0.5 and r["answer_relevancy"] > 0.5 if (r := d) else False
def cat2(d): return cr(d) <= 0.5

ids = sorted(set(dense) & set(bm25) & set(hybrid))
n = len(ids)
print(f"Common ids: {n}")

c1d = sum(1 for i in ids if dense[i]["faithfulness"]  < 0.5 and dense[i]["answer_relevancy"]  > 0.5)
c1b = sum(1 for i in ids if bm25[i]["faithfulness"]   < 0.5 and bm25[i]["answer_relevancy"]   > 0.5)
c1h = sum(1 for i in ids if hybrid[i]["faithfulness"] < 0.5 and hybrid[i]["answer_relevancy"] > 0.5)

c2d = sum(1 for i in ids if cr(dense[i])  <= 0.5)
c2b = sum(1 for i in ids if cr(bm25[i])   <= 0.5)
c2h = sum(1 for i in ids if cr(hybrid[i]) <= 0.5)

print(f"\nCat1 (F<0.5 & AR>0.5):  Dense={c1d}/{n}={100*c1d/n:.1f}%  BM25={c1b}/{n}={100*c1b/n:.1f}%  Hybrid={c1h}/{n}={100*c1h/n:.1f}%")
print(f"Cat2 (CR<=0.5):          Dense={c2d}/{n}={100*c2d/n:.1f}%  BM25={c2b}/{n}={100*c2b/n:.1f}%  Hybrid={c2h}/{n}={100*c2h/n:.1f}%")

# Cat3: inferential failures — join scores with CSV on question text
meta = {}
with open(paths.QUESTIONS_CSV, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):  # comma-delimited; see question_type.py
        if row.get("Review", "").strip().lower() == "accept":
            meta[row["question"].strip()] = row["question_type"].strip().lower()

def inferential_fail(scores, threshold_f=0.5):
    fail = ok = 0
    for r in scores.values():
        qt = meta.get(r["question"].strip())
        if qt == "inferential":
            if r["faithfulness"] < threshold_f:
                fail += 1
            else:
                ok += 1
    return fail, ok

for label, scores in [("Dense", dense), ("BM25", bm25), ("Hybrid", hybrid)]:
    fail, ok = inferential_fail(scores)
    total = fail + ok
    print(f"Cat3 inferential (F<0.5): {label} {fail}/{total} = {100*fail/total:.1f}%" if total else f"Cat3 {label}: no inferential found")
