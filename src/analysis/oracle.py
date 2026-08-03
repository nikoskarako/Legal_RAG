"""
Experiment 1 — Oracle "best-of-both" upper bound.

Pure post-hoc analysis on the per-question RAGAS scores already on disk
(no retrieval, no LLM calls). Joins the dense and BM25 score files by
item id and reports:

  * per-metric oracle ceiling (max over the two systems, per question);
  * a realistic faithfulness-routed oracle (per question pick the system
    with higher faithfulness, then report THAT system's F and AR);
  * the win split (which system wins each question);
  * correlation of per-question faithfulness (low ⇒ complementary);
  * Wilcoxon significance of the oracle gain over BM25.

Usage:
    python oracle.py
    python oracle.py --dense ragas_results/ragas_scores_dense_new.json \
                     --bm25  ragas_results/ragas_scores_bm25.json

Reuses the score schema written by evaluate_ragas.py:
    {"items": [{"id", "question", "faithfulness", "answer_relevancy",
                "context_relevance"}, ...]}
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import wilcoxon, pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402


def _get_cr(r: dict) -> float:
    return r.get("context_relevance", r.get("context_relevancy", float("nan")))


def load_items(path: str) -> dict:
    """Return {id: {f, ar, cr}} keyed by item id, dropping NaN/None faithfulness.

    Items are joined across files on their stable ``id`` field. Joining on the
    question *text* is wrong: a few prompts share identical wording but are
    distinct items (different ids, different source provisions), so a text key
    silently collapses and mis-pairs them (118 items -> 115 keys).
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for r in d["items"]:
        key = r.get("id")
        if key is None:
            continue
        fv = r.get("faithfulness", float("nan"))
        if fv is None or (isinstance(fv, float) and math.isnan(fv)):
            continue
        out[key] = {
            "f": float(fv),
            "ar": float(r.get("answer_relevancy", float("nan"))),
            "cr": float(_get_cr(r)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default=os.path.join(paths.RESULTS_DIR, "ragas_scores_dense_new.json"))
    ap.add_argument("--bm25", default=os.path.join(paths.RESULTS_DIR, "ragas_scores_bm25.json"))
    ap.add_argument("--hybrid", default=None,
                    help="optional hybrid (RRF) score file; reports its row + Wilcoxon vs BM25")
    args = ap.parse_args()

    dense = load_items(args.dense)
    bm25 = load_items(args.bm25)

    common = sorted(set(dense) & set(bm25))
    only_d = set(dense) - set(bm25)
    only_b = set(bm25) - set(dense)
    print(f"dense items: {len(dense)} | bm25 items: {len(bm25)} | matched: {len(common)}")
    if only_d or only_b:
        print(f"  unmatched — dense-only: {len(only_d)}, bm25-only: {len(only_b)} "
              f"(joined on id)")
    if not common:
        raise SystemExit("No overlapping questions — check that both files cover the same set.")

    fd = np.array([dense[q]["f"] for q in common])
    fb = np.array([bm25[q]["f"] for q in common])
    ad = np.array([dense[q]["ar"] for q in common])
    ab = np.array([bm25[q]["ar"] for q in common])

    # --- per-metric oracle ceiling (optimistic upper bound) ---
    oracle_f = np.maximum(fd, fb)
    oracle_ar = np.maximum(ad, ab)

    # --- realistic faithfulness-routed oracle: pick the higher-F system per Q,
    #     then report that system's own F and AR (one system per question) ---
    pick_dense = fd >= fb
    routed_f = np.where(pick_dense, fd, fb)
    routed_ar = np.where(pick_dense, ad, ab)

    def line(name, f, ar):
        print(f"  {name:<32} F={np.mean(f):.4f}   AR={np.mean(ar):.4f}")

    print("\n=== Means over matched questions ===")
    line("Dense", fd, ad)
    line("BM25", fb, ab)
    line("Oracle (per-metric max)", oracle_f, oracle_ar)
    line("Oracle (faithfulness-routed)", routed_f, routed_ar)

    best_single_f = max(np.mean(fd), np.mean(fb))
    print(f"\nFaithfulness gap over best single system ({best_single_f:.4f}):")
    print(f"  per-metric-max oracle : +{np.mean(oracle_f) - best_single_f:.4f}")
    print(f"  routed oracle         : +{np.mean(routed_f) - best_single_f:.4f}")

    # --- win split (faithfulness) ---
    eps = 1e-9
    d_win = int(np.sum(fd > fb + eps))
    b_win = int(np.sum(fb > fd + eps))
    tie = len(common) - d_win - b_win
    n = len(common)
    print("\n=== Faithfulness win split (complementarity) ===")
    print(f"  dense wins: {d_win} ({100*d_win/n:.1f}%)  "
          f"bm25 wins: {b_win} ({100*b_win/n:.1f}%)  tie: {tie} ({100*tie/n:.1f}%)")

    # --- correlation (low/moderate ⇒ systems fail on different questions) ---
    try:
        r, p = pearsonr(fd, fb)
        print(f"  per-question faithfulness corr: r={r:.3f} (p={p:.3e})")
    except Exception as e:
        print(f"  correlation unavailable: {e}")

    # --- significance of oracle gain vs BM25 ---
    print("\n=== Wilcoxon: oracle faithfulness vs BM25 ===")
    for name, vec in [("per-metric-max", oracle_f), ("routed", routed_f)]:
        try:
            stat, pv = wilcoxon(vec, fb)
            print(f"  {name:<14} W={stat:.0f}  p={pv:.3e}")
        except Exception as e:
            print(f"  {name:<14} n/a ({e})")

    # --- optional Hybrid (RRF) row, over the same id-joined paired set ---
    if args.hybrid:
        hyb = load_items(args.hybrid)
        ch = sorted(set(common) & set(hyb))
        missing = sorted(set(common) - set(hyb))
        print("\n=== Hybrid (RRF) ===")
        print(f"  hybrid items: {len(hyb)} | paired with dense∩bm25: {len(ch)}"
              + (f" | {len(missing)} common id(s) absent from hybrid run" if missing else ""))
        fh = np.array([hyb[q]["f"] for q in ch])
        ah = np.array([hyb[q]["ar"] for q in ch])
        fb_p = np.array([bm25[q]["f"] for q in ch])
        ab_p = np.array([bm25[q]["ar"] for q in ch])
        line("Hybrid (RRF)", fh, ah)
        try:
            _, pf = wilcoxon(fh, fb_p)
            _, pa = wilcoxon(ah, ab_p)
            print(f"  vs BM25 (paired n={len(ch)}): "
                  f"faithfulness p={pf:.3e} (≈BM25), answer-rel p={pa:.3e}")
        except Exception as e:
            print(f"  Wilcoxon vs BM25 unavailable: {e}")


if __name__ == "__main__":
    main()
