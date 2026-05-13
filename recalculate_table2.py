#!/usr/bin/env python3
"""Recalculate Table 2 metrics on the entity-clean v3 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MODEL_FILES = {
    "Llama 3.1 8B": {
        "v1": "results_llama-3.1-8b-instant_t1.json",
        "v2": "v2_results_llama-3.1-8b-instant.json",
    },
    "Llama 4 Scout 17B": {
        "v1": "results_meta-llama_llama-4-scout-17b-16e-instruct_t1.json",
        "v2": "v2_results_meta-llama_llama-4-scout-17b-16e-instruct.json",
    },
    "Qwen3 32B": {
        "v1": "results_qwen_qwen3-32b_t1.json",
        "v2": "v2_results_qwen_qwen3-32b.json",
    },
    "GPT-OSS 120B": {
        "v1": "results_openai_gpt-oss-120b_t1.json",
        "v2": "v2_results_openai_gpt-oss-120b.json",
    },
    "Llama 3.3 70B": {
        "v1": "results_llama-3.3-70b-versatile_t1.json",
        "v2": "v2_results_llama-3.3-70b-versatile.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_raw(filename: str) -> dict[str, Any]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    return load_json(path).get("raw_results", {})


def answers_match(extracted: Any, expected: Any, tol: float = 0.01) -> bool:
    if extracted is None or expected is None:
        return False
    try:
        got = float(extracted)
        exp = float(expected)
    except (TypeError, ValueError):
        return False
    if exp == 0:
        return abs(got) <= tol
    return abs(got - exp) / max(abs(exp), 1e-9) <= tol


def result_correct(result: dict[str, Any], expected: float) -> bool:
    if "extracted" in result:
        return answers_match(result.get("extracted"), expected)
    return bool(result.get("correct"))


def bootstrap_ci(deltas: list[float], n_boot: int = 10000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = np.random.default_rng(42)
    arr = np.array(deltas, dtype=float)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(lo), float(hi)]


def p_value(deltas: list[float]) -> float:
    nonzero = [d for d in deltas if abs(d) > 1e-12]
    if not nonzero:
        return 1.0
    try:
        return float(stats.wilcoxon(deltas, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def classify(delta: float, p: float) -> str:
    if p > 0.05:
        return "Inv"
    if delta < 0.15:
        return "Mild"
    return "Mem"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def pick_result(item_id: str, raw_v1: dict[str, Any], raw_v2: dict[str, Any]) -> dict[str, Any] | None:
    key = f"{item_id}_t0"
    return raw_v2.get(key) or raw_v1.get(key)


def main() -> None:
    dataset = load_json(DATA_DIR / "eval_verified_v3.json")
    items = dataset["items"]
    table: dict[str, Any] = {}

    for model_name, files in MODEL_FILES.items():
        raw_v1 = load_raw(files["v1"])
        raw_v2 = load_raw(files["v2"])

        n_orig_correct = 0
        n_orig_total = 0
        n_var_correct = 0
        n_var_total = 0
        item_deltas: list[float] = []
        per_item: list[dict[str, Any]] = []

        for item in items:
            orig_result = pick_result(item["item_id"], raw_v1, raw_v2)
            if not orig_result:
                continue

            variant_scores: list[float] = []
            for variant in item.get("variants", []):
                variant_result = pick_result(variant["item_id"], raw_v1, raw_v2)
                if not variant_result:
                    continue
                correct = result_correct(variant_result, float(variant["answer"]))
                variant_scores.append(1.0 if correct else 0.0)
                n_var_total += 1
                n_var_correct += int(correct)

            if not variant_scores:
                continue

            orig_correct = result_correct(orig_result, float(item["answer"]))
            orig_score = 1.0 if orig_correct else 0.0
            n_orig_total += 1
            n_orig_correct += int(orig_correct)
            p_iso = float(np.mean(variant_scores))
            delta_i = orig_score - p_iso
            item_deltas.append(delta_i)
            per_item.append({
                "item_id": item["item_id"],
                "p_orig": orig_score,
                "p_iso": p_iso,
                "delta": delta_i,
                "n_variants": len(variant_scores),
            })

        if not item_deltas:
            continue

        acc_orig = n_orig_correct / n_orig_total
        acc_iso = n_var_correct / n_var_total
        delta = acc_orig - acc_iso
        ci = bootstrap_ci(item_deltas)
        p = p_value(item_deltas)
        archetype = classify(delta, p)

        table[model_name] = {
            "n": n_orig_total,
            "n_variants": n_var_total,
            "acc_orig": acc_orig,
            "acc_iso": acc_iso,
            "delta_iso": delta,
            "ci_95": ci,
            "p_value": p,
            "archetype": archetype,
            "per_item": per_item,
        }

    output = {
        "dataset": "data/eval_verified_v3.json",
        "n_dataset_items": len(items),
        "n_dataset_variants": sum(len(item.get("variants", [])) for item in items),
        "models": table,
    }
    out_path = DATA_DIR / "table2_v3.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_path}")
    print("Model & N & Acc_orig & Acc_iso & Delta [95% CI] & p & Type")
    for model, row in table.items():
        ci = row["ci_95"]
        print(
            f"{model} & {row['n']} & {format_percent(row['acc_orig'])} & "
            f"{format_percent(row['acc_iso'])} & {row['delta_iso']:+.3f} "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] & {format_p(row['p_value'])} & "
            f"{row['archetype']}"
        )


if __name__ == "__main__":
    main()
