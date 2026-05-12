#!/usr/bin/env python3
"""
Rescore models on the v2 dataset.

For clean items (59): uses existing v1 results.
For regenerated items (27): uses v2 evaluation results (if available).

Usage:
    python3 rescore_v2.py              # Score with whatever data is available
    python3 rescore_v2.py --clean-only # Score only clean items (no new eval needed)
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

V1_MODEL_FILES = {
    "llama8b": "results_llama-3.1-8b-instant_t1.json",
    "scout": "results_meta-llama_llama-4-scout-17b-16e-instruct_t1.json",
    "qwen3": "results_qwen_qwen3-32b_t1.json",
    "gptoss": "results_openai_gpt-oss-120b_t1.json",
    "llama70b": "results_llama-3.3-70b-versatile_t1.json",
}

V2_MODEL_FILES = {
    "llama8b": "v2_results_llama-3.1-8b-instant.json",
    "scout": "v2_results_meta-llama_llama-4-scout-17b-16e-instruct.json",
    "qwen3": "v2_results_qwen_qwen3-32b.json",
    "gptoss": "v2_results_openai_gpt-oss-120b.json",
    "llama70b": "v2_results_llama-3.3-70b-versatile.json",
}

TOLERANCE = 0.01


def answers_match(a, b, tol=TOLERANCE):
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (ValueError, TypeError):
        return False
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / max(abs(b), 1e-9) < tol


def bootstrap_ci(deltas, n_boot=10000, alpha=0.05):
    deltas = np.array(deltas)
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(deltas, size=len(deltas), replace=True))
                  for _ in range(n_boot)]
    return np.percentile(boot_means, 100 * alpha / 2), np.percentile(boot_means, 100 * (1 - alpha / 2))


def classify(delta, p, acc_orig):
    if p > 0.05:
        return "INVARIANT"
    if delta > 0.3 and acc_orig > 0.85:
        return "PURE_MEMORIZER"
    if delta > 0.15:
        return "PARTIAL_MEMORIZER"
    if delta > 0.05:
        return "MILD_GAP"
    return "INVARIANT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()

    with open(DATA_DIR / "eval_verified_v2.json") as f:
        v2_data = json.load(f)

    clean_ids = set()
    regen_ids = set()
    for item in v2_data["items"]:
        if item.get("regenerated_v2"):
            regen_ids.add(item["item_id"])
        else:
            clean_ids.add(item["item_id"])

    # Build expected-answer lookup from the dataset for re-scoring v2 results
    expected_answers = {}
    for item in v2_data["items"]:
        expected_answers[item["item_id"]] = float(item["answer"])
        for var in item.get("variants", []):
            expected_answers[var["item_id"]] = float(var["answer"])

    item_ids = clean_ids if args.clean_only else clean_ids | regen_ids

    print(f"Dataset: {len(item_ids)} items ({len(clean_ids)} clean + {len(regen_ids)} regenerated)")
    if args.clean_only:
        print("Mode: CLEAN-ONLY (no new eval needed)")
    print(f"{'='*70}")

    for model_name in V1_MODEL_FILES:
        v1_path = RESULTS_DIR / V1_MODEL_FILES[model_name]
        if not v1_path.exists():
            continue

        with open(v1_path) as f:
            v1_data = json.load(f)
        v1_raw = v1_data["raw_results"]

        # Load v2 results if available
        v2_raw = {}
        v2_path = RESULTS_DIR / V2_MODEL_FILES.get(model_name, "")
        if v2_path.exists() and not args.clean_only:
            with open(v2_path) as f:
                v2_raw = json.load(f).get("raw_results", {})

        # Score each item
        item_deltas = []
        n_orig_correct = 0
        n_orig_total = 0
        n_var_correct = 0
        n_var_total = 0
        n_items_scored = 0

        for iid in sorted(item_ids):
            # Get original result (from v1)
            orig_key = f"{iid}_t0"
            if orig_key not in v1_raw:
                continue

            orig = v1_raw[orig_key]
            n_orig_total += 1
            if orig["correct"]:
                n_orig_correct += 1

            # Get variant results
            var_correct = 0
            var_total = 0

            for vi in range(5):
                var_key = f"{iid}_v{vi:02d}_t0"

                if iid in clean_ids:
                    # Use v1 results for clean items
                    if var_key in v1_raw:
                        var_total += 1
                        n_var_total += 1
                        if v1_raw[var_key]["correct"]:
                            var_correct += 1
                            n_var_correct += 1
                elif iid in regen_ids and not args.clean_only:
                    # Use v2 results for regenerated items
                    # Re-score against current dataset expected answers
                    var_item_id = f"{iid}_v{vi:02d}"
                    if var_key in v2_raw and var_item_id in expected_answers:
                        var_total += 1
                        n_var_total += 1
                        extracted = v2_raw[var_key].get("extracted")
                        expected = expected_answers[var_item_id]
                        if answers_match(extracted, expected):
                            var_correct += 1
                            n_var_correct += 1

            if var_total > 0:
                n_items_scored += 1
                orig_score = 1.0 if orig["correct"] else 0.0
                var_score = var_correct / var_total
                item_deltas.append(orig_score - var_score)

        if n_items_scored == 0:
            print(f"\n{model_name}: no scored items")
            continue

        acc_orig = n_orig_correct / n_orig_total
        acc_iso = n_var_correct / n_var_total
        delta = acc_orig - acc_iso
        se = np.std(item_deltas) / np.sqrt(len(item_deltas)) if item_deltas else 0
        t_stat = np.mean(item_deltas) / se if se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(1, len(item_deltas) - 1)))
        boot_lo, boot_hi = bootstrap_ci(item_deltas) if item_deltas else (0, 0)
        archetype = classify(delta, p_val, acc_orig)

        print(f"\n{'='*60}")
        print(f"  Model:     {v1_data['model']}")
        print(f"  N_scored:  {n_items_scored}")
        print(f"  Acc_orig:  {acc_orig*100:.1f}%    Acc_iso: {acc_iso*100:.1f}%")
        print(f"  Delta_raw: {delta:+.3f}   SE: {se:.3f}")
        print(f"  Boot CI:   [{boot_lo:+.3f}, {boot_hi:+.3f}]")
        print(f"  p={p_val:.2e}")
        print(f"  Verdict:   {archetype}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
