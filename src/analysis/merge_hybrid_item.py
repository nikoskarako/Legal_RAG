"""Merge per-item RAGAS scores from one score file into another, in place.

Use after scoring a single newly-regenerated question (e.g. id 105#1) in an
isolated folder, to fold it into the main hybrid score file WITHOUT re-running
the LLM judge over the items that were already scored (which would perturb
every existing number through judge stochasticity).

    python merge_hybrid_item.py \
        --into  ragas_results/ragas_scores_hybrid.json \
        --from  ragas_results/ragas_scores_hybrid_105.json

Items are matched on `id`; an incoming item replaces an existing one with the
same id, otherwise it is appended. Averages (faithfulness, answer_relevancy,
context_relevance) are recomputed over all items, skipping NaN/None.
"""
import argparse
import json
import math


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _avg(items, key):
    vals = [_num(it.get(key)) for it in items]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True, help="main score file, edited in place")
    ap.add_argument("--from", dest="src", required=True, help="score file to fold in")
    args = ap.parse_args()

    dst = json.load(open(args.into, encoding="utf-8"))
    src = json.load(open(args.src, encoding="utf-8"))

    by_id = {it.get("id"): it for it in dst["items"]}
    added, replaced = 0, 0
    for it in src["items"]:
        i = it.get("id")
        if i in by_id:
            replaced += 1
        else:
            added += 1
        by_id[i] = it

    dst["items"] = list(by_id.values())
    dst.setdefault("averages", {})
    for k in ("faithfulness", "answer_relevancy", "context_relevance"):
        dst["averages"][k] = _avg(dst["items"], k)

    json.dump(dst, open(args.into, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"merged: +{added} new, {replaced} replaced -> {len(dst['items'])} items")
    print("new averages:", {k: round(v, 4) for k, v in dst["averages"].items()})


if __name__ == "__main__":
    main()
