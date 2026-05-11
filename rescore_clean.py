#!/usr/bin/env python3
"""
Task 4: Rescore existing model results on verified-clean items only.
No new API calls needed — uses existing raw_results.
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

MODEL_FILES = {
    "llama8b": "results_llama-3.1-8b-instant_t1.json",
    "scout": "results_meta-llama_llama-4-scout-17b-16e-instruct_t1.json",
    "qwen3": "results_qwen_qwen3-32b_t1.json",
    "gptoss": "results_openai_gpt-oss-120b_t1.json",
    "llama70b": "results_llama-3.3-70b-versatile_t1.json",
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
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(deltas, size=len(deltas), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper


def classify_archetype(delta, p, acc_orig, acc_iso):
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
    with open(DATA_DIR / "eval_verified_clean.json") as f:
        clean_dataset = json.load(f)

    clean_ids = set()
    for item in clean_dataset["items"]:
        clean_ids.add(item["item_id"])

    print(f"Clean items: {len(clean_ids)}")
    print(f"{'='*70}")

    all_results = {}

    for model_name, fname in MODEL_FILES.items():
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"\n{model_name}: FILE NOT FOUND")
            continue

        with open(fpath) as f:
            data = json.load(f)

        raw = data["raw_results"]

        # Collect per-item results
        item_deltas = []
        n_orig_correct = 0
        n_orig_total = 0
        n_var_correct = 0
        n_var_total = 0
        n_items_with_data = 0
        items_detail = []

        for item_id in sorted(clean_ids):
            orig_key = f"{item_id}_t0"
            if orig_key not in raw:
                continue

            orig = raw[orig_key]
            orig_correct = orig["correct"]
            n_orig_total += 1
            if orig_correct:
                n_orig_correct += 1

            var_correct_count = 0
            var_total_count = 0
            for vi in range(5):
                var_key = f"{item_id}_v{vi:02d}_t0"
                if var_key in raw:
                    var_total_count += 1
                    n_var_total += 1
                    if raw[var_key]["correct"]:
                        var_correct_count += 1
                        n_var_correct += 1

            if var_total_count > 0:
                n_items_with_data += 1
                orig_score = 1.0 if orig_correct else 0.0
                var_score = var_correct_count / var_total_count
                delta_i = orig_score - var_score
                item_deltas.append(delta_i)
                items_detail.append({
                    "item_id": item_id,
                    "orig_correct": orig_correct,
                    "var_acc": var_score,
                    "delta": delta_i,
                })

        if n_items_with_data == 0:
            print(f"\n{model_name}: No data for clean items")
            continue

        acc_orig = n_orig_correct / n_orig_total
        acc_iso = n_var_correct / n_var_total
        delta_raw = acc_orig - acc_iso
        se = np.std(item_deltas) / np.sqrt(len(item_deltas))

        # p-value (one-sample t-test, H0: delta = 0)
        t_stat = np.mean(item_deltas) / se if se > 0 else 0
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(item_deltas) - 1))

        boot_lo, boot_hi = bootstrap_ci(item_deltas)

        # Count flagged items (delta > 0.5)
        n_flagged = sum(1 for d in item_deltas if d > 0.5)

        archetype = classify_archetype(delta_raw, p_value, acc_orig, acc_iso)

        # Extraction rate
        n_extracted_orig = sum(1 for iid in clean_ids if f"{iid}_t0" in raw and raw[f"{iid}_t0"]["extracted"] is not None)
        n_extracted_var = 0
        n_var_entries = 0
        for iid in clean_ids:
            for vi in range(5):
                vk = f"{iid}_v{vi:02d}_t0"
                if vk in raw:
                    n_var_entries += 1
                    if raw[vk]["extracted"] is not None:
                        n_extracted_var += 1

        extr_orig = n_extracted_orig / n_orig_total * 100 if n_orig_total > 0 else 0
        extr_var = n_extracted_var / n_var_entries * 100 if n_var_entries > 0 else 0

        print(f"\n{'='*60}")
        print(f"  Model:     {data['model']}")
        print(f"  N_clean:   {n_items_with_data}")
        print(f"  Acc_orig:  {acc_orig*100:.1f}%    Acc_iso: {acc_iso*100:.1f}%")
        print(f"  Delta_raw: {delta_raw:+.3f}   SE: {se:.3f}")
        print(f"  Boot CI:   [{boot_lo:+.3f}, {boot_hi:+.3f}]")
        print(f"  p={p_value:.2e}")
        print(f"  Flagged:   {n_flagged}/{n_items_with_data}")
        print(f"  Extr%:     orig={extr_orig:.1f}%  var={extr_var:.1f}%")
        print(f"  Verdict:   {archetype}")
        print(f"{'='*60}")

        all_results[model_name] = {
            "model": data["model"],
            "n_clean_items": n_items_with_data,
            "acc_orig": round(acc_orig, 4),
            "acc_iso": round(acc_iso, 4),
            "delta_raw": round(delta_raw, 4),
            "se": round(se, 4),
            "p_value": p_value,
            "boot_ci": [round(boot_lo, 4), round(boot_hi, 4)],
            "n_flagged": n_flagged,
            "archetype": archetype,
            "items": items_detail,
        }

    # Save rescored results
    output_path = RESULTS_DIR / "rescored_clean.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved rescored results to {output_path}")

    # Summary comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON: Full Dataset vs Clean-Only")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'N_full':>6} {'Δ_full':>8} {'N_clean':>7} {'Δ_clean':>8} {'p_clean':>10} {'Verdict':>15}")
    print(f"{'-'*70}")

    full_deltas = {
        "llama8b": (100, 0.417),
        "scout": (100, 0.402),
        "qwen3": (70, 0.465),
        "gptoss": (20, 0.430),
        "llama70b": (14, 0.443),
    }

    for model_name in MODEL_FILES:
        if model_name in all_results:
            r = all_results[model_name]
            n_full, d_full = full_deltas.get(model_name, (0, 0))
            print(f"{r['model']:<20} {n_full:>6} {d_full:>+8.3f} {r['n_clean_items']:>7} {r['delta_raw']:>+8.3f} {r['p_value']:>10.2e} {r['archetype']:>15}")


if __name__ == "__main__":
    main()
