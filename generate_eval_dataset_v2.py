#!/usr/bin/env python3
"""
Isomorph-Eval Dataset Regeneration v2
=====================================
Fixes three root causes of broken variant answers:
1. Value conflation: numbers serving dual roles (variable + constant) are protected
2. Magnitude inflation: replacement values stay in the same digit class with ≥20% change
3. Non-integer intermediates: all intermediate computation results must be integers

For the 59 verified-clean items: keeps existing variants unchanged.
For the 41 broken/partial items: regenerates variants with all fixes applied.

Usage:
    python3 generate_eval_dataset_v2.py
    python3 generate_eval_dataset_v2.py --verify-only  # just run verification
"""

import json
import math
import re
import random
import hashlib
import sys
import argparse
from pathlib import Path
from collections import Counter
from datasets import load_dataset

ANNO_RE = re.compile(r"<<([^>]+?)=([^>]+?)>>")
ANSWER_RE = re.compile(r"####\s*(.+)")
TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?|[+\-*/()%])")
NUMBER_RE = re.compile(r'\b(\d+(?:\.\d+)?)\b')

DATA_DIR = Path(__file__).parent / "data"

KNOWN_CONSTANTS = {
    60,     # minutes/hour
    24,     # hours/day
    7,      # days/week
    12,     # months/year, eggs/dozen, inches/foot
    100,    # percent
    365,    # days/year
    52,     # weeks/year
    1000,   # g/kg, m/km, ml/L
    16,     # ounces/pound
    4,      # quarters/dollar
    10,     # dimes/dollar
    2,      # halves
    3,      # thirds
    0,      # never substitute zero
    1,      # multiplicative identity
}

AGENTS = [
    "Tomás", "Aisha", "Kenji", "Priya", "Oluwaseun", "Fatima", "Dmitri",
    "Yuki", "Ingrid", "Kofi", "Xiulan", "Rashid", "Svetlana", "Hiroshi",
    "Amara", "Bjorn", "Nalini", "Emeka", "Saoirse", "Tariq", "Mei-Lin",
    "Andrei", "Zara", "Kwame", "Linnea",
]

GSM8K_NAMES = [
    "Janet", "Mark", "James", "John", "Mary", "Tom", "Sarah", "Bob",
    "Alice", "Peter", "Jane", "Bill", "Mike", "Lisa", "Anna", "David",
    "Emily", "Alex", "Sam", "Chris", "Tim", "Dan", "Amy", "Beth", "Carl",
    "Weng", "Natalia", "Elaine", "Julie", "Martha", "Toby", "Henry",
    "Josh", "Jack", "Jill", "Gerald", "Cecelia", "Rex", "Gail",
    "Carla", "Joy", "Ruby", "Oliver", "Ryan", "Megan", "Phil", "Max",
    "Ken", "Ben", "Grace", "Steve", "Luke", "Tina", "Victor", "Hannah",
    "Kylar", "Marissa", "Tobias", "Leah", "Sofia", "Brennan",
]


# ============================================================================
# PARSING
# ============================================================================

def parse_number(s):
    s = s.strip().replace(",", "").replace("$", "").replace("€", "")
    if "/" in s:
        parts = s.split("/")
        try:
            return float(parts[0]) / float(parts[1])
        except Exception:
            return float('nan')
    return float(s)


def parse_expression(expr):
    tokens = TOKEN_RE.findall(expr.strip())
    numbers, operators = [], []
    for t in tokens:
        if t in "+-*/%":
            operators.append(t)
        elif t not in "()":
            numbers.append(parse_number(t))
    if not operators:
        return None, numbers
    ops = set(operators)
    if ops == {"*"}:
        return "multiply", numbers
    elif ops == {"/"}:
        return "divide", numbers
    elif ops == {"+"}:
        return "add", numbers
    elif ops == {"-"}:
        return "subtract", numbers
    elif ops <= {"+", "-"}:
        return ("subtract" if operators.count("-") >= operators.count("+") else "add"), numbers
    elif "*" in ops:
        return "multiply", numbers
    elif "/" in ops:
        return "divide", numbers
    return "add", numbers


def digit_class(v):
    return len(str(abs(int(v)))) if v != 0 else 1


def parse_gsm8k(question, solution):
    answer_match = ANSWER_RE.search(solution)
    if not answer_match:
        return None, None
    answer = parse_number(answer_match.group(1))

    annotations = ANNO_RE.findall(solution)
    if not annotations:
        return None, None

    nodes = []
    val_to_id = {}
    nid = 0

    all_literals = {}
    for expr, _ in annotations:
        for t in TOKEN_RE.findall(expr):
            if t not in "+-*/()%":
                v = parse_number(t)
                if v not in all_literals:
                    all_literals[v] = True

    computed = set()
    for v in all_literals:
        node_id = f"n{nid}"
        nodes.append({
            "id": node_id, "op": "assign", "refs": [], "value": v,
            "dc": digit_class(v),
        })
        val_to_id[v] = node_id
        nid += 1

    for expr, result_str in annotations:
        result = parse_number(result_str)
        op, operand_vals = parse_expression(expr)
        if op is None:
            continue

        refs = []
        for ov in operand_vals:
            if ov in val_to_id:
                refs.append(val_to_id[ov])
            else:
                node_id = f"n{nid}"
                nodes.append({
                    "id": node_id, "op": "assign", "refs": [], "value": ov,
                    "dc": digit_class(ov),
                })
                val_to_id[ov] = node_id
                nid += 1
                refs.append(node_id)

        comp_id = f"n{nid}"
        nodes.append({
            "id": comp_id, "op": op, "refs": refs, "value": result,
            "dc": digit_class(result),
        })
        val_to_id[result] = comp_id
        computed.add(result)
        nid += 1

    remove_ids = set()
    for n in nodes:
        if n["op"] == "assign" and n["value"] in computed:
            comp_node = next(
                (x for x in nodes if x["op"] != "assign" and x["value"] == n["value"]),
                None,
            )
            if comp_node:
                for other in nodes:
                    other["refs"] = [
                        comp_node["id"] if r == n["id"] else r for r in other["refs"]
                    ]
                remove_ids.add(n["id"])

    final = [n for n in nodes if n["id"] not in remove_ids]

    old_to_new = {}
    for i, n in enumerate(final):
        old_to_new[n["id"]] = f"n{i}"
        n["id"] = f"n{i}"
        n["refs"] = [old_to_new.get(r, r) for r in n["refs"]]

    try:
        computed_answer = execute_graph(final)
        if computed_answer is None or abs(computed_answer - answer) > 0.01:
            return None, None
    except Exception:
        return None, None

    return final, answer


# ============================================================================
# FORWARD EXECUTION WITH INTEGER INTERMEDIATE CHECKING
# ============================================================================

def execute_graph(nodes):
    vals = {}
    for n in nodes:
        if n["op"] == "assign":
            vals[n["id"]] = n["value"]
        else:
            operands = [vals[r] for r in n["refs"]]
            if n["op"] == "add":
                vals[n["id"]] = sum(operands)
            elif n["op"] == "subtract":
                vals[n["id"]] = operands[0] - sum(operands[1:])
            elif n["op"] == "multiply":
                vals[n["id"]] = math.prod(operands)
            elif n["op"] == "divide":
                vals[n["id"]] = operands[0] / operands[1] if operands[1] != 0 else float("inf")
            else:
                vals[n["id"]] = 0
    return vals.get(nodes[-1]["id"], None)


def execute_graph_checked(nodes):
    """Forward-execute with full intermediate checking.

    Returns (answer, intermediates, all_ok) where:
    - answer: final computed value
    - intermediates: dict of node_id -> computed value
    - all_ok: True if all intermediates are non-negative integers
    """
    vals = {}
    all_ok = True
    for n in nodes:
        if n["op"] == "assign":
            vals[n["id"]] = n["value"]
        else:
            operands = [vals[r] for r in n["refs"]]
            if n["op"] == "add":
                v = sum(operands)
            elif n["op"] == "subtract":
                v = operands[0] - sum(operands[1:])
            elif n["op"] == "multiply":
                v = math.prod(operands)
            elif n["op"] == "divide":
                if operands[1] == 0:
                    return None, vals, False
                v = operands[0] / operands[1]
            else:
                v = 0

            vals[n["id"]] = v

            if v != int(v):
                all_ok = False
            if v < 0:
                all_ok = False
            if abs(v) > 1e8:
                all_ok = False

    answer = vals.get(nodes[-1]["id"], None)
    if answer is not None and answer != int(answer):
        all_ok = False
    return answer, vals, all_ok


# ============================================================================
# TASK 1: VALUE CONFLATION PREVENTION
# ============================================================================

def build_semantic_role_map(nodes, question_text):
    """Determine which leaf values are VARIABLE (mutable) vs CONSTANT (protected).

    A value is tagged CONSTANT if:
    1. It appears in the KNOWN_CONSTANTS set AND appears in a context
       suggesting it's a conversion factor (not a problem quantity)
    2. It appears multiple times in the question with different semantic roles

    A value is tagged VARIABLE if it's a problem-specific quantity that
    the question varies.
    """
    role_map = {}  # node_id -> "VARIABLE" | "CONSTANT"

    # Get all leaf values and their positions in the question text
    leaf_nodes = [n for n in nodes if n["op"] == "assign"]
    leaf_values = [n["value"] for n in leaf_nodes]
    value_counts = Counter(leaf_values)

    # Find all number occurrences in question text
    question_numbers = [float(m) for m in NUMBER_RE.findall(question_text)]
    question_counts = Counter(question_numbers)

    for node in leaf_nodes:
        val = node["value"]
        int_val = int(val) if val == int(val) else val

        # Rule 0: Values not present in question text must NOT be mutated.
        # These are intermediate values from annotation algebra (e.g., the
        # solver writes <<40=40>> for a value derived from question numbers).
        val_str = str(int_val) if val == int(val) else str(val)
        if val_str not in question_text:
            role_map[node["id"]] = "CONSTANT"
            continue

        # Rule 1: Known constants that appear exactly once as a leaf
        # but could be confused with other occurrences
        if int_val in KNOWN_CONSTANTS:
            # Check if this value appears more times in the question text
            # than in the graph leaves — suggests it has multiple roles
            text_count = question_counts.get(val, 0) + question_counts.get(float(int_val), 0)
            graph_count = value_counts[val]

            if text_count > graph_count:
                role_map[node["id"]] = "CONSTANT"
                continue

        # Rule 2: Values that appear as both a leaf and would conflict
        # with a different semantic use (e.g., percentages where the
        # denominator 100 is implicit)
        if int_val in KNOWN_CONSTANTS and value_counts[val] > 1:
            role_map[node["id"]] = "CONSTANT"
            continue

        # Rule 3: Values appearing ≥2 times in question text likely serve
        # different semantic roles (e.g., "5 cars" and "$5 each"). Mutating
        # one occurrence while the graph uses both produces wrong answers.
        text_occurrences = len(re.findall(r'\b' + re.escape(val_str) + r'\b', question_text))
        if text_occurrences >= 2:
            role_map[node["id"]] = "CONSTANT"
            continue

        # Default: treat as VARIABLE (safe to mutate)
        role_map[node["id"]] = "VARIABLE"

    # Safety: if ALL leaves are CONSTANT, try to promote the largest value
    # that appears EXACTLY ONCE in the question text (safe to mutate).
    # Do NOT promote values that appear multiple times (conflation risk)
    # or values that don't appear in the question (phantom values).
    n_variable = sum(1 for v in role_map.values() if v == "VARIABLE")
    if n_variable == 0 and leaf_nodes:
        candidates = sorted(
            leaf_nodes,
            key=lambda n: abs(n["value"]),
            reverse=True,
        )
        for cand in candidates:
            if abs(cand["value"]) <= 1:
                continue
            val_str = str(int(cand["value"])) if cand["value"] == int(cand["value"]) else str(cand["value"])
            text_count = len(re.findall(r'\b' + re.escape(val_str) + r'\b', question_text))
            if text_count == 1:
                role_map[cand["id"]] = "VARIABLE"
                break

    return role_map


def detect_conflation_risk(nodes, question_text):
    """Check if the problem has values that serve dual roles.

    Returns a set of node_ids that should NOT be mutated because
    their string representation appears in contexts that suggest
    different semantic roles.
    """
    protected = set()
    leaf_nodes = [n for n in nodes if n["op"] == "assign"]

    # Extract all numbers from question with their surrounding context
    for match in NUMBER_RE.finditer(question_text):
        num_str = match.group(1)
        num_val = float(num_str)
        start, end = match.start(), match.end()

        # Get surrounding context (20 chars each side)
        ctx_before = question_text[max(0, start - 20):start].lower()
        ctx_after = question_text[end:end + 20].lower()

        # Check for conversion-factor contexts
        conversion_contexts = [
            "per hour", "per minute", "per day", "per week", "per month",
            "per year", "percent", "%", "minutes in", "hours in",
            "days in", "out of", "in a dozen", "in a foot",
            "in a pound", "in a kilogram", "in a meter",
        ]
        is_conversion = any(ctx in ctx_after for ctx in conversion_contexts)
        is_conversion = is_conversion or any(
            ctx in ctx_before for ctx in ["every ", "each "]
        )

        if is_conversion and int(num_val) in KNOWN_CONSTANTS:
            # Find the leaf node with this value and protect it
            for node in leaf_nodes:
                if node["value"] == num_val:
                    protected.add(node["id"])

    return protected


# ============================================================================
# TASK 2: MAGNITUDE-PRESERVING VALUE SAMPLING
# ============================================================================

def sample_same_magnitude(original_value, rng, exclude=None):
    """Sample a replacement value in the same digit class.

    Guarantees:
    - Same number of digits as original
    - Differs by at least 20%
    - Not in the exclude set
    """
    if exclude is None:
        exclude = set()

    int_val = int(abs(original_value))
    n_digits = len(str(int_val)) if int_val > 0 else 1

    if n_digits == 1:
        low, high = 2, 9
    else:
        low = 10 ** (n_digits - 1)
        high = 10 ** n_digits - 1

    min_diff = max(1, int(abs(original_value) * 0.2))

    for _ in range(200):
        candidate = rng.randint(low, high)
        if candidate in exclude:
            continue
        if abs(candidate - abs(int_val)) < min_diff:
            continue
        if int(candidate) in KNOWN_CONSTANTS:
            continue
        result = float(candidate)
        if original_value < 0:
            result = -result
        return result

    return None


def sample_values_for_graph(nodes, role_map, rng, max_attempts=50, original_answer=None):
    """Sample a complete set of replacement values that produce integer intermediates.

    If original_answer is provided, rejects variants where the answer magnitude
    differs by more than 3x from the original (prevents magnitude inflation).

    Returns new_nodes (with mutated values) or None if no valid set found.
    """
    leaf_nodes = [n for n in nodes if n["op"] == "assign"]
    variable_nodes = [n for n in leaf_nodes if role_map.get(n["id"]) == "VARIABLE"]

    for attempt in range(max_attempts):
        new_nodes = [dict(n) for n in nodes]

        # Collect all values being used (for collision avoidance)
        used_values = set()
        for n in new_nodes:
            if n["op"] == "assign" and role_map.get(n["id"]) == "CONSTANT":
                used_values.add(int(n["value"]))

        # Sample new values for VARIABLE nodes only
        valid = True
        for new_n in new_nodes:
            if new_n["op"] != "assign":
                continue
            if role_map.get(new_n["id"]) != "VARIABLE":
                continue

            new_val = sample_same_magnitude(
                new_n["value"], rng, exclude=used_values,
            )
            if new_val is None:
                valid = False
                break
            new_n["value"] = new_val
            new_n["dc"] = digit_class(new_val)
            used_values.add(int(abs(new_val)))

        if not valid:
            continue

        # Forward-execute with integer checking
        answer, intermediates, all_ok = execute_graph_checked(new_nodes)
        if answer is None or not all_ok:
            continue

        if math.isinf(answer) or math.isnan(answer):
            continue

        # Verify answer is a reasonable integer
        if answer != int(answer):
            continue

        # Magnitude preservation: reject if answer is >3x or <1/3x of original
        if original_answer is not None and original_answer != 0:
            ratio = abs(answer) / abs(original_answer)
            if ratio > 3.0 or ratio < 1 / 3.0:
                continue

        return new_nodes, int(answer)

    return None, None


# ============================================================================
# SURFACE TEXT GENERATION (improved)
# ============================================================================

def generate_variant_text(original_question, original_nodes, new_nodes, variant_idx):
    """Generate variant question text by simultaneous positional replacement.

    Finds all number positions in the original text, matches each to a
    graph leaf by value (first-match, greedy), then replaces all at once
    from right to left to avoid position-shift collisions.
    """
    leaf_orig = [n for n in original_nodes if n["op"] == "assign"]
    leaf_new = [n for n in new_nodes if n["op"] == "assign"]

    # Build a map: for each leaf that changed, record (orig_val_str, new_val_str)
    change_map = {}
    for orig_n, new_n in zip(leaf_orig, leaf_new):
        if orig_n["value"] != new_n["value"]:
            orig_str = str(int(orig_n["value"])) if orig_n["value"] == int(orig_n["value"]) else str(orig_n["value"])
            new_str = str(int(new_n["value"])) if new_n["value"] == int(new_n["value"]) else str(new_n["value"])
            if orig_str not in change_map:
                change_map[orig_str] = []
            change_map[orig_str].append(new_str)

    if not change_map:
        return original_question

    # Find all number positions in the original text
    positions = []  # (start, end, matched_str, replacement_str)
    for match in NUMBER_RE.finditer(original_question):
        num_str = match.group(1)
        if num_str in change_map and change_map[num_str]:
            replacement = change_map[num_str].pop(0)
            positions.append((match.start(), match.end(), num_str, replacement))

    # Replace from right to left so positions don't shift
    variant_q = original_question
    for start, end, _, replacement in reversed(positions):
        variant_q = variant_q[:start] + replacement + variant_q[end:]

    # Swap agent name — use word boundary regex and replace ALL occurrences
    agent = AGENTS[(variant_idx * 7 + 3) % len(AGENTS)]
    for name in sorted(GSM8K_NAMES, key=len, reverse=True):
        name_pattern = re.compile(r'\b' + re.escape(name) + r'\b')
        if name_pattern.search(variant_q):
            variant_q = name_pattern.sub(agent, variant_q)
            break

    return variant_q


# ============================================================================
# STRUCTURAL FINGERPRINT
# ============================================================================

def structural_fingerprint(nodes):
    sig = json.dumps({
        "n": len(nodes),
        "ops": [n["op"] for n in nodes],
        "dcs": [n["dc"] for n in nodes],
        "depths": [len(n["refs"]) for n in nodes],
    }, sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


# ============================================================================
# MAIN: REGENERATE DATASET
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--n-variants", type=int, default=5)
    args = parser.parse_args()

    # Load audit results to know which items are clean vs broken
    with open(DATA_DIR / "audit_results.json") as f:
        audit = json.load(f)
    clean_ids = set()
    broken_ids = set()
    for item in audit["audit_results"]:
        if item["status"] == "clean":
            clean_ids.add(item["item_id"])
        else:
            broken_ids.add(item["item_id"])

    print(f"Audit: {len(clean_ids)} clean, {len(broken_ids)} broken/partial/suspicious")

    # Load existing dataset
    with open(DATA_DIR / "eval_subset_100.json") as f:
        existing = json.load(f)

    existing_map = {
        item["item_id"]: item
        for item in existing["items"]
        if item.get("is_original", False)
    }

    # Load GSM8K for re-parsing broken items
    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Build index: gsm8k_XXXX -> dataset row
    gsm8k_index = {}
    for idx, row in enumerate(ds):
        item_id = f"gsm8k_{idx:04d}"
        gsm8k_index[item_id] = row

    K = args.n_variants

    # Process each item
    v2_items = []
    regen_success = 0
    regen_fail = 0
    kept_clean = 0

    for item_id in sorted(existing_map.keys()):
        item = existing_map[item_id]

        if item_id in clean_ids:
            # Keep existing clean variants
            v2_items.append(item)
            kept_clean += 1
            continue

        # Regenerate this item's variants
        if item_id not in gsm8k_index:
            print(f"  WARNING: {item_id} not in GSM8K index, skipping")
            regen_fail += 1
            continue

        row = gsm8k_index[item_id]
        nodes, answer = parse_gsm8k(row["question"], row["answer"])
        if nodes is None:
            print(f"  WARNING: {item_id} failed to parse, skipping")
            regen_fail += 1
            continue

        # Build semantic role map
        role_map = build_semantic_role_map(nodes, row["question"])
        conflation_protected = detect_conflation_risk(nodes, row["question"])

        # Merge protections
        for nid in conflation_protected:
            role_map[nid] = "CONSTANT"

        # Reject items where any leaf value doesn't appear in question text.
        # These have incomplete computation graphs (annotation starts with
        # pre-computed intermediates) that produce wrong answers when mutated.
        leaf_nodes = [n for n in nodes if n["op"] == "assign"]
        has_phantom = False
        for ln in leaf_nodes:
            val = ln["value"]
            val_str = str(int(val)) if val == int(val) else str(val)
            if val_str not in row["question"]:
                has_phantom = True
                break
        if has_phantom:
            print(f"  WARNING: {item_id} has phantom leaf values, skipping")
            regen_fail += 1
            continue

        n_variable = sum(1 for v in role_map.values() if v == "VARIABLE")
        n_constant = sum(1 for v in role_map.values() if v == "CONSTANT")

        if n_variable == 0:
            print(f"  WARNING: {item_id} has no variable leaves, skipping")
            regen_fail += 1
            continue

        new_variants = []
        for k in range(K):
            seed = int(hashlib.sha256(f"{item_id}_{k}_v2".encode()).hexdigest(), 16) % (2 ** 32)
            rng = random.Random(seed)

            new_nodes, new_answer = sample_values_for_graph(nodes, role_map, rng, original_answer=answer)
            if new_nodes is None:
                continue

            # Verify fingerprint
            new_fp = structural_fingerprint(new_nodes)
            orig_fp = structural_fingerprint(nodes)
            if new_fp != orig_fp:
                continue

            # Double-check answer via independent forward execution
            check_answer, _, check_ok = execute_graph_checked(new_nodes)
            if not check_ok or check_answer != new_answer:
                continue

            # Generate variant text
            variant_q = generate_variant_text(
                row["question"], nodes, new_nodes, k,
            )

            # Verify all new leaf values appear in variant text
            new_leaves = [n["value"] for n in new_nodes if n["op"] == "assign"]
            all_present = all(
                str(int(v)) in variant_q if v == int(v) else str(v) in variant_q
                for v in new_leaves
            )

            # Reject if variant text is identical to original
            if variant_q.strip() == row["question"].strip():
                continue

            new_variants.append({
                "item_id": f"{item_id}_v{k:02d}",
                "question": variant_q,
                "answer": float(new_answer),
                "verified": True,
                "role_map": {nid: role for nid, role in role_map.items()},
                "all_values_present": all_present,
            })

        if len(new_variants) >= 3:
            # Reject answer-invariant items (all variants same answer as original)
            var_answers = set(v["answer"] for v in new_variants)
            if len(var_answers) == 1 and float(answer) in var_answers:
                print(f"  WARNING: {item_id} is answer-invariant, skipping")
                regen_fail += 1
                continue

            new_item = dict(item)
            new_item["variants"] = new_variants
            new_item["regenerated_v2"] = True
            v2_items.append(new_item)
            regen_success += 1
        else:
            print(f"  WARNING: {item_id} only got {len(new_variants)} variants, skipping")
            regen_fail += 1

    # Build output dataset
    v2_dataset = {
        "items": v2_items,
        "metadata": {
            "source": "eval_subset_100.json + regenerated broken items",
            "n_items": len(v2_items),
            "n_kept_clean": kept_clean,
            "n_regenerated": regen_success,
            "n_failed": regen_fail,
            "n_variants_per_item": K,
            "fixes_applied": [
                "value_conflation_prevention",
                "magnitude_preserving_sampling",
                "integer_intermediate_checking",
                "independent_forward_execution",
            ],
        },
    }

    output_path = DATA_DIR / "eval_verified_v2.json"
    with open(output_path, "w") as f:
        json.dump(v2_dataset, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DATASET REGENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Kept clean:     {kept_clean}")
    print(f"  Regenerated:    {regen_success}")
    print(f"  Failed:         {regen_fail}")
    print(f"  Total items:    {len(v2_items)}")
    print(f"  Saved to:       {output_path}")

    # Run structural audit on regenerated items
    print(f"\n--- STRUCTURAL AUDIT ---")
    audit_regenerated(v2_items)


def audit_regenerated(items):
    """Run magnitude and structural checks on the regenerated dataset."""
    import numpy as np
    from scipy import stats

    orig_answers = []
    var_answers = []
    orig_max_nums = []
    var_max_nums = []

    for item in items:
        if not item.get("is_original"):
            continue
        orig_answers.append(float(item["answer"]))

        orig_nums = [float(x) for x in NUMBER_RE.findall(item["question"])]
        if orig_nums:
            orig_max_nums.append(max(orig_nums))

        for var in item.get("variants", []):
            var_answers.append(float(var["answer"]))
            var_nums = [float(x) for x in NUMBER_RE.findall(var["question"])]
            if var_nums:
                var_max_nums.append(max(var_nums))

    if not orig_answers or not var_answers:
        print("  No data for audit")
        return

    orig_mag = [math.log10(abs(a) + 1) for a in orig_answers]
    var_mag = [math.log10(abs(a) + 1) for a in var_answers]

    t_mag, p_mag = stats.ttest_ind(orig_mag, var_mag)
    t_max, p_max = stats.ttest_ind(
        [math.log10(x + 1) for x in orig_max_nums],
        [math.log10(x + 1) for x in var_max_nums],
    ) if orig_max_nums and var_max_nums else (0, 1)

    print(f"  answer_magnitude: t={t_mag:.3f}, p={p_mag:.3f} {'OK' if p_mag > 0.05 else 'SIGNIFICANT'}")
    print(f"  max_number:       t={t_max:.3f}, p={p_max:.3f} {'OK' if p_max > 0.05 else 'SIGNIFICANT'}")

    # Check all variant answers are integers
    non_int = sum(1 for a in var_answers if a != int(a))
    print(f"  non-integer answers: {non_int} {'OK' if non_int == 0 else 'FAIL'}")

    # Check regenerated items specifically
    regen_items = [i for i in items if i.get("regenerated_v2")]
    if regen_items:
        regen_var_answers = []
        for item in regen_items:
            for var in item.get("variants", []):
                regen_var_answers.append(float(var["answer"]))
        regen_non_int = sum(1 for a in regen_var_answers if a != int(a))
        print(f"  regenerated non-integer: {regen_non_int} / {len(regen_var_answers)}")


if __name__ == "__main__":
    main()
