"""Descriptive statistics and paired Wilcoxon tests across retrieval systems."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

R = paths.RESULTS_DIR
files = {
     "dense":    os.path.join(R, "ragas_scores_dense_new.json"),
     "bm25":     os.path.join(R, "ragas_scores_bm25.json"),
     "baseline": os.path.join(R, "ragas_scores_baseline_new.json"),
 }

data = {}
for name, path in files.items():
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    data[name] = {
        "f":  [r["faithfulness"]    for r in d["items"]],
        "ar": [r["answer_relevancy"] for r in d["items"]],
        "cr": [r.get("context_relevance", r.get("context_relevancy", float("nan"))) for r in d["items"]],
    }

for name, m in data.items():
    print(f"\n=== {name} ===")
    for metric, vals in m.items():
        a = np.array(vals)
        print(f"  {metric}: mean={np.nanmean(a):.4f}  median={np.nanmedian(a):.4f}  std={np.nanstd(a):.4f}")

print("\n=== Wilcoxon (faithfulness) ===")
for a, b in [("dense","baseline"), ("bm25","baseline"), ("bm25","dense")]:
    fa, fb = np.array(data[a]["f"]), np.array(data[b]["f"])
    stat, p = wilcoxon(fa, fb)
    print(f"  {a} vs {b}: W={stat:.0f}, p={p:.3e}")

print("\n=== Wilcoxon (answer_relevancy) ===")
for a, b in [("dense","baseline"), ("bm25","baseline"), ("bm25","dense")]:
    fa, fb = np.array(data[a]["ar"]), np.array(data[b]["ar"])
    stat, p = wilcoxon(fa, fb)
    print(f"  {a} vs {b}: W={stat:.0f}, p={p:.3e}")