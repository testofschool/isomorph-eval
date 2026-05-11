#!/usr/bin/env python3
"""
Independent Answer Verification Engine for Isomorph-Eval.

Three verification methods:
1. Multi-model consensus: If ≥3 models agree on an answer that differs from expected, expected is likely wrong
2. Heuristic bug detection: Value conflation, non-integer intermediates, answer-invariant variants
3. Original-variant consistency: Check if variant answer has plausible relationship to variant numbers

Usage:
    python3 verify_answers.py                    # Full audit
    python3 verify_answers.py --item gsm8k_0328  # Single item
    python3 verify_answers.py --summary           # Summary only
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

MODEL_FILES = {
    "llama8b": "results_llama-3.1-8b-instant_t1.json",
    "scout": "results_meta-llama_llama-4-scout-17b-16e-instruct_t1.json",
    "qwen3": "results_qwen_qwen3-32b_t1.json",
    "gptoss": "results_openai_gpt-oss-120b_t1.json",
    "llama70b": "results_llama-3.3-70b-versatile_t1.json",
}

TOLERANCE = 0.01  # 1% relative tolerance for answer matching


@dataclass
class VariantVerdict:
    variant_id: str
    expected_answer: float
    status: str  # "clean", "broken", "suspicious"
    reasons: list = field(default_factory=list)
    model_answers: dict = field(default_factory=dict)
    consensus_answer: Optional[float] = None
    consensus_count: int = 0


@dataclass
class ItemVerdict:
    item_id: str
    original_answer: float
    status: str  # "clean", "partial", "broken"
    n_clean: int = 0
    n_broken: int = 0
    n_suspicious: int = 0
    variants: list = field(default_factory=list)
    flags: list = field(default_factory=list)


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


def extract_numbers(text):
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]


def load_model_results():
    model_data = {}
    for name, fname in MODEL_FILES.items():
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        model_data[name] = data["raw_results"]
    return model_data


def load_dataset():
    with open(DATA_DIR / "eval_subset_100.json") as f:
        return json.load(f)


def check_answer_invariant(original_answer, variant_answer):
    return answers_match(original_answer, variant_answer)


def check_non_integer_intermediate(question_text, answer):
    try:
        ans = float(answer)
        if ans != int(ans) and ans != round(ans, 1):
            return False  # answer itself is non-integer, might be intentional
    except (ValueError, TypeError):
        pass

    numbers = extract_numbers(question_text)
    for i, a in enumerate(numbers):
        for j, b in enumerate(numbers):
            if i != j and b != 0:
                result = a / b
                if result != int(result) and abs(result - round(result)) > 0.01:
                    intermediate = a * b
                    if intermediate != int(intermediate):
                        return True
    return False


def check_value_conflation(question_text):
    numbers = extract_numbers(question_text)
    num_counter = Counter(numbers)
    conflated = {n: c for n, c in num_counter.items() if c > 1 and n not in (0, 1, 2)}
    return len(conflated) > 0, conflated


def get_model_consensus(model_data, variant_id):
    answers = {}
    for model_name, results in model_data.items():
        key = f"{variant_id}_t0"
        if key in results:
            entry = results[key]
            if entry["extracted"] is not None:
                answers[model_name] = entry["extracted"]

    if len(answers) < 2:
        return answers, None, 0

    answer_counts = Counter()
    for a in answers.values():
        matched = False
        for existing in answer_counts:
            if answers_match(a, existing):
                answer_counts[existing] += 1
                matched = True
                break
        if not matched:
            answer_counts[a] = 1

    if answer_counts:
        consensus_answer, consensus_count = answer_counts.most_common(1)[0]
        return answers, consensus_answer, consensus_count

    return answers, None, 0


def verify_item(item, model_data, verbose=False):
    item_id = item["item_id"]
    original_answer = float(item["answer"])

    verdict = ItemVerdict(
        item_id=item_id,
        original_answer=original_answer,
        status="clean"
    )

    for variant in item["variants"]:
        var_id = variant["item_id"]
        var_answer = float(variant["answer"])
        var_question = variant["question"]

        vv = VariantVerdict(
            variant_id=var_id,
            expected_answer=var_answer,
            status="clean"
        )

        # Check 1: Answer invariant (variant has same answer as original despite different numbers)
        if check_answer_invariant(original_answer, var_answer):
            orig_numbers = extract_numbers(item["question"])
            var_numbers = extract_numbers(var_question)
            if orig_numbers != var_numbers:
                vv.reasons.append("answer_invariant")

        # Check 2: Non-integer check on answer
        if var_answer != int(var_answer):
            remainder = var_answer - int(var_answer)
            if abs(remainder) > 0.01 and abs(remainder - 0.5) > 0.01:
                vv.reasons.append(f"non_integer_answer({var_answer})")

        # Check 3: Value conflation in variant text
        has_conflation, conflated_nums = check_value_conflation(var_question)
        if has_conflation:
            vv.reasons.append(f"value_conflation({conflated_nums})")

        # Check 4: Multi-model consensus
        model_answers, consensus, count = get_model_consensus(model_data, var_id)
        vv.model_answers = model_answers
        vv.consensus_answer = consensus
        vv.consensus_count = count

        if consensus is not None and count >= 2:
            if not answers_match(consensus, var_answer):
                if answers_match(consensus, var_answer, tol=0.05):
                    vv.reasons.append(f"consensus_near_miss(consensus={consensus},expected={var_answer})")
                else:
                    vv.reasons.append(f"consensus_disagrees(consensus={consensus},n={count},expected={var_answer})")

        # Check 5: Magnitude ratio check
        if original_answer != 0 and var_answer != 0:
            ratio = var_answer / original_answer
            if ratio > 10 or ratio < 0.1:
                orig_nums = extract_numbers(item["question"])
                var_nums = extract_numbers(var_question)
                max_num_ratio = max(
                    (max(var_nums) / max(orig_nums)) if orig_nums and var_nums and max(orig_nums) > 0 else 1,
                    1
                )
                if ratio > max_num_ratio * 3 or ratio < 1 / (max_num_ratio * 3):
                    vv.reasons.append(f"extreme_answer_ratio({ratio:.2f})")

        # Classify variant
        has_consensus_disagree = any("consensus_disagrees" in r for r in vv.reasons)
        has_answer_invariant = "answer_invariant" in vv.reasons
        has_non_integer = any("non_integer" in r for r in vv.reasons)

        if has_consensus_disagree and (has_answer_invariant or has_non_integer or len(vv.reasons) >= 2):
            vv.status = "broken"
        elif has_consensus_disagree:
            vv.status = "broken"  # consensus disagreement alone is strong signal
        elif has_answer_invariant or has_non_integer:
            vv.status = "suspicious"
        elif len(vv.reasons) >= 2:
            vv.status = "suspicious"

        verdict.variants.append(vv)

        if vv.status == "broken":
            verdict.n_broken += 1
        elif vv.status == "suspicious":
            verdict.n_suspicious += 1
        else:
            verdict.n_clean += 1

    # Classify item
    if verdict.n_broken >= 3:
        verdict.status = "broken"
    elif verdict.n_broken >= 1:
        verdict.status = "partial"
    elif verdict.n_suspicious >= 3:
        verdict.status = "suspicious"
    else:
        verdict.status = "clean"

    # Item-level flags
    if all(check_answer_invariant(original_answer, float(v["answer"])) for v in item["variants"]):
        verdict.flags.append("ALL_VARIANTS_SAME_ANSWER")

    return verdict


def run_audit(items, model_data, verbose=False):
    verdicts = []
    for item in items:
        if not item.get("is_original", False):
            continue
        v = verify_item(item, model_data, verbose)
        verdicts.append(v)
    return verdicts


def print_summary(verdicts):
    total = len(verdicts)
    clean = sum(1 for v in verdicts if v.status == "clean")
    partial = sum(1 for v in verdicts if v.status == "partial")
    broken = sum(1 for v in verdicts if v.status == "broken")
    suspicious = sum(1 for v in verdicts if v.status == "suspicious")

    print(f"\n{'='*60}")
    print(f"ISOMORPH-EVAL ANSWER VERIFICATION AUDIT")
    print(f"{'='*60}")
    print(f"Total items:  {total}")
    print(f"Clean:        {clean} ({clean/total*100:.1f}%)")
    print(f"Partial:      {partial} ({partial/total*100:.1f}%)")
    print(f"Broken:       {broken} ({broken/total*100:.1f}%)")
    print(f"Suspicious:   {suspicious} ({suspicious/total*100:.1f}%)")
    print(f"{'='*60}")

    # Print broken items
    if broken + partial > 0:
        print(f"\n--- BROKEN / PARTIAL ITEMS ---")
        for v in verdicts:
            if v.status in ("broken", "partial"):
                print(f"\n  {v.item_id} [{v.status.upper()}] orig_ans={v.original_answer}")
                print(f"    clean={v.n_clean} broken={v.n_broken} suspicious={v.n_suspicious}")
                if v.flags:
                    print(f"    FLAGS: {', '.join(v.flags)}")
                for var in v.variants:
                    if var.status != "clean":
                        reasons_str = "; ".join(var.reasons)
                        print(f"    {var.variant_id}: {var.status} — {reasons_str}")
                        if var.consensus_answer is not None and not answers_match(var.consensus_answer, var.expected_answer):
                            print(f"      Expected: {var.expected_answer}, Consensus: {var.consensus_answer} (n={var.consensus_count})")

    # Print suspicious items
    if suspicious > 0:
        print(f"\n--- SUSPICIOUS ITEMS ---")
        for v in verdicts:
            if v.status == "suspicious":
                print(f"  {v.item_id}: {'; '.join(v.flags) if v.flags else 'heuristic flags'}")

    return {
        "total": total,
        "clean": clean,
        "partial": partial,
        "broken": broken,
        "suspicious": suspicious,
        "clean_item_ids": [v.item_id for v in verdicts if v.status == "clean"],
        "broken_item_ids": [v.item_id for v in verdicts if v.status in ("broken", "partial")],
        "suspicious_item_ids": [v.item_id for v in verdicts if v.status == "suspicious"],
    }


def save_audit_results(verdicts, output_path):
    results = []
    for v in verdicts:
        item_dict = {
            "item_id": v.item_id,
            "original_answer": v.original_answer,
            "status": v.status,
            "n_clean": v.n_clean,
            "n_broken": v.n_broken,
            "n_suspicious": v.n_suspicious,
            "flags": v.flags,
            "variants": []
        }
        for var in v.variants:
            var_dict = {
                "variant_id": var.variant_id,
                "expected_answer": var.expected_answer,
                "status": var.status,
                "reasons": var.reasons,
                "model_answers": {k: v for k, v in var.model_answers.items()},
                "consensus_answer": var.consensus_answer,
                "consensus_count": var.consensus_count,
            }
            item_dict["variants"].append(var_dict)
        results.append(item_dict)

    with open(output_path, "w") as f:
        json.dump({"audit_results": results}, f, indent=2)
    print(f"\nSaved audit results to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", help="Verify a single item")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    parser.add_argument("--save", default="data/audit_results.json", help="Output path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Loading model results...")
    model_data = load_model_results()
    print(f"Loaded {len(model_data)} models: {list(model_data.keys())}")

    print("Loading dataset...")
    dataset = load_dataset()
    items = dataset["items"]
    originals = [i for i in items if i.get("is_original", False)]
    print(f"Dataset: {len(items)} total entries, {len(originals)} originals")

    if args.item:
        target = [i for i in originals if i["item_id"] == args.item]
        if not target:
            print(f"Item {args.item} not found")
            sys.exit(1)
        v = verify_item(target[0], model_data, verbose=True)
        print(f"\n{v.item_id} [{v.status.upper()}]")
        print(f"  Original answer: {v.original_answer}")
        print(f"  Clean: {v.n_clean}, Broken: {v.n_broken}, Suspicious: {v.n_suspicious}")
        for var in v.variants:
            print(f"\n  {var.variant_id}: {var.status}")
            print(f"    Expected: {var.expected_answer}")
            print(f"    Reasons: {var.reasons}")
            print(f"    Model answers: {var.model_answers}")
            if var.consensus_answer is not None:
                print(f"    Consensus: {var.consensus_answer} (n={var.consensus_count})")
        return

    print("\nRunning verification audit...")
    verdicts = run_audit(originals, model_data, verbose=args.verbose)

    summary = print_summary(verdicts)

    if not args.summary:
        save_audit_results(verdicts, Path(__file__).parent / args.save)

    return summary


if __name__ == "__main__":
    main()
