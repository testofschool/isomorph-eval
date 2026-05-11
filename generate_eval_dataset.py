#!/usr/bin/env python3
"""
Generate the REAL isomorphic GSM8K evaluation dataset.
No LLM needed — uses rule-based parser + deterministic mutation.

Output: isomorph_gsm8k_eval.json with 50 originals × 5 variants = 300 items
"""
import json, math, re, random, hashlib, sys
import numpy as np
from datasets import load_dataset

# ── GSM8K annotation parser (inline, no imports needed) ──────────

ANNO_RE = re.compile(r"<<([^>]+?)=([^>]+?)>>")
ANSWER_RE = re.compile(r"####\s*(.+)")
TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?|[+\-*/()%])")

def parse_number(s):
    s = s.strip().replace(",","").replace("$","").replace("€","")
    if "/" in s:
        parts = s.split("/")
        try: return float(parts[0]) / float(parts[1])
        except: return float('nan')
    return float(s)

def parse_expression(expr):
    tokens = TOKEN_RE.findall(expr.strip())
    numbers, operators = [], []
    for t in tokens:
        if t in "+-*/%": operators.append(t)
        elif t not in "()": numbers.append(parse_number(t))
    if not operators: return None, numbers
    ops = set(operators)
    if ops == {"*"}: return "multiply", numbers
    elif ops == {"/"}: return "divide", numbers
    elif ops == {"+"}: return "add", numbers
    elif ops == {"-"}: return "subtract", numbers
    elif ops <= {"+","-"}:
        return ("subtract" if operators.count("-") >= operators.count("+") else "add"), numbers
    elif "*" in ops: return "multiply", numbers
    elif "/" in ops: return "divide", numbers
    return "add", numbers

def execute_graph(nodes):
    """Forward-execute and return final answer."""
    vals = {}
    for n in nodes:
        if n["op"] == "assign":
            vals[n["id"]] = n["value"]
        else:
            operands = [vals[r] for r in n["refs"]]
            if n["op"] == "add": vals[n["id"]] = sum(operands)
            elif n["op"] == "subtract": vals[n["id"]] = operands[0] - sum(operands[1:])
            elif n["op"] == "multiply": vals[n["id"]] = math.prod(operands)
            elif n["op"] == "divide": vals[n["id"]] = operands[0] / operands[1] if operands[1] != 0 else float('inf')
            else: vals[n["id"]] = 0
    return vals.get(nodes[-1]["id"], None)

def digit_class(v):
    return len(str(abs(int(v)))) if v != 0 else 1

def parse_gsm8k(question, solution):
    """Parse a GSM8K problem → graph nodes list + answer."""
    answer_match = ANSWER_RE.search(solution)
    if not answer_match: return None, None
    answer = parse_number(answer_match.group(1))

    annotations = ANNO_RE.findall(solution)
    if not annotations: return None, None

    nodes = []
    val_to_id = {}
    nid = 0

    # Collect all leaf values
    all_literals = {}
    for expr, _ in annotations:
        for t in TOKEN_RE.findall(expr):
            if t not in "+-*/()%":
                v = parse_number(t)
                if v not in all_literals:
                    all_literals[v] = True

    # Create assign nodes for leaves
    computed = set()
    for v in all_literals:
        node_id = f"n{nid}"
        nodes.append({"id": node_id, "op": "assign", "refs": [], "value": v,
                       "dc": digit_class(v)})
        val_to_id[v] = node_id
        nid += 1

    # Create computation nodes
    for expr, result_str in annotations:
        result = parse_number(result_str)
        op, operand_vals = parse_expression(expr)
        if op is None: continue

        refs = []
        for ov in operand_vals:
            if ov in val_to_id:
                refs.append(val_to_id[ov])
            else:
                node_id = f"n{nid}"
                nodes.append({"id": node_id, "op": "assign", "refs": [], "value": ov,
                               "dc": digit_class(ov)})
                val_to_id[ov] = node_id
                nid += 1
                refs.append(node_id)

        comp_id = f"n{nid}"
        nodes.append({"id": comp_id, "op": op, "refs": refs, "value": result,
                       "dc": digit_class(result)})
        val_to_id[result] = comp_id
        computed.add(result)
        nid += 1

    # Remove redundant assign nodes (values that are computed)
    final = []
    remove_ids = set()
    for n in nodes:
        if n["op"] == "assign" and n["value"] in computed:
            comp_node = next((x for x in nodes if x["op"] != "assign" and x["value"] == n["value"]), None)
            if comp_node:
                for other in nodes:
                    other["refs"] = [comp_node["id"] if r == n["id"] else r for r in other["refs"]]
                remove_ids.add(n["id"])

    final = [n for n in nodes if n["id"] not in remove_ids]

    # Re-index
    old_to_new = {}
    for i, n in enumerate(final):
        old_to_new[n["id"]] = f"n{i}"
        n["id"] = f"n{i}"
        n["refs"] = [old_to_new.get(r, r) for r in n["refs"]]

    # Verify
    try:
        computed_answer = execute_graph(final)
        if computed_answer is None or abs(computed_answer - answer) > 0.01:
            return None, None
    except:
        return None, None

    return final, answer

def structural_fingerprint(nodes):
    """Compute τ-signature fingerprint."""
    sig = json.dumps({
        "n": len(nodes),
        "ops": [n["op"] for n in nodes],
        "dcs": [n["dc"] for n in nodes],
        "depths": [len(n["refs"]) for n in nodes],
    }, sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest()[:12]

def mutate_graph(nodes, answer, seed):
    """Generate a τ-isomorphic variant by mutating leaf values."""
    rng = random.Random(seed)
    new_nodes = []
    for n in nodes:
        nn = dict(n)
        if nn["op"] == "assign":
            dc = nn["dc"]
            lo = max(2, 10**(dc-1))
            hi = 10**dc - 1
            new_val = rng.randint(lo, hi)
            while new_val == int(nn["value"]):
                new_val = rng.randint(lo, hi)
            nn["value"] = float(new_val)
            nn["dc"] = digit_class(new_val)
        new_nodes.append(nn)

    # Forward-execute to get new answer
    max_attempts = 20
    for attempt in range(max_attempts):
        try:
            new_answer = execute_graph(new_nodes)
            if new_answer is not None and not math.isinf(new_answer) and not math.isnan(new_answer):
                # Check no intermediate goes negative or infinite
                vals = {}
                ok = True
                for n in new_nodes:
                    if n["op"] == "assign":
                        vals[n["id"]] = n["value"]
                    else:
                        ops = [vals[r] for r in n["refs"]]
                        if n["op"] == "subtract":
                            v = ops[0] - sum(ops[1:])
                        elif n["op"] == "divide":
                            if ops[1] == 0: ok = False; break
                            v = ops[0] / ops[1]
                        elif n["op"] == "multiply":
                            v = math.prod(ops)
                        else:
                            v = sum(ops)
                        if v < 0 or abs(v) > 1e8: ok = False; break
                        vals[n["id"]] = v
                if ok:
                    return new_nodes, vals[new_nodes[-1]["id"]]
        except:
            pass
        # Retry with different values
        for n in new_nodes:
            if n["op"] == "assign":
                dc = n["dc"]
                lo = max(2, 10**(dc-1))
                hi = 10**dc - 1
                n["value"] = float(rng.randint(lo, hi))
    return None, None

# Names and themes for surface generation
AGENTS = ["Tomás","Aisha","Kenji","Priya","Oluwaseun","Fatima","Dmitri",
          "Yuki","Ingrid","Kofi","Xiulan","Rashid","Svetlana","Hiroshi",
          "Amara","Bjorn","Nalini","Emeka","Saoirse","Tariq","Mei-Lin",
          "Andrei","Zara","Kwame","Linnea"]

def generate_surface(original_question, nodes, new_answer, variant_idx):
    """Simple surface: inject mutated values into a template."""
    # Extract all leaf values
    leaf_vals = [n["value"] for n in nodes if n["op"] == "assign"]
    leaf_strs = [str(int(v)) if v == int(v) else str(v) for v in leaf_vals]

    # Build a simple problem statement listing the values
    agent = AGENTS[variant_idx % len(AGENTS)]
    ans_str = str(int(new_answer)) if new_answer == int(new_answer) else f"{new_answer:.2f}"

    # Template: state the values and ask for the answer
    ops = [n["op"] for n in nodes if n["op"] != "assign"]
    op_desc = ", ".join(ops)

    text = (f"Problem for {agent}: Given the values {', '.join(leaf_strs)}, "
            f"apply the operations [{op_desc}] in the same structure as the "
            f"original problem. What is the final answer?")

    # Better: reconstruct from original question pattern
    # Replace numbers in original with new numbers
    q = original_question
    original_leaves = []
    for n in nodes:
        if n["op"] == "assign":
            original_leaves.append(n)

    return text, ans_str


# ── Main: Load GSM8K, parse, mutate, export ──────────────────────

def main():
    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} problems.\n")

    # Parse all problems
    parsed = []
    for idx, item in enumerate(ds):
        try:
            nodes, answer = parse_gsm8k(item["question"], item["answer"])
        except (ValueError, ZeroDivisionError, IndexError):
            continue
        if nodes is not None and len(nodes) >= 3:
            # Skip problems with fractional leaf values (mutator generates integers)
            has_fraction = any(n["value"] != int(n["value"]) for n in nodes if n["op"] == "assign")
            if has_fraction:
                continue
            parsed.append({
                "idx": idx,
                "question": item["question"],
                "solution": item["answer"],
                "nodes": nodes,
                "answer": answer,
                "fingerprint": structural_fingerprint(nodes),
                "n_nodes": len(nodes),
                "depth": max(len(n["refs"]) for n in nodes),
            })

    print(f"Successfully parsed: {len(parsed)}/{len(ds)}")

    # Select 50 diverse problems (stratified by difficulty)
    # Sort by node count to get a range of complexities
    parsed.sort(key=lambda x: (x["n_nodes"], x["idx"]))
    step = max(1, len(parsed) // 50)
    selected = parsed[::step][:50]
    print(f"Selected {len(selected)} problems for evaluation")

    # Generate 5 variants per problem
    K = 5
    eval_items = {"items": [], "metadata": {
        "source": "GSM8K test set (openai/gsm8k)",
        "n_originals": len(selected),
        "n_variants_per_item": K,
        "total_items": len(selected) * (1 + K),
        "generation_method": "rule-based parse + deterministic mutation",
        "verification": "forward execution match on all items",
    }}

    success = 0
    for prob in selected:
        item = {
            "item_id": f"gsm8k_{prob['idx']:04d}",
            "question": prob["question"],
            "answer": prob["answer"],
            "is_original": True,
            "fingerprint": prob["fingerprint"],
            "n_nodes": prob["n_nodes"],
            "tau_signature": [n["op"] for n in prob["nodes"]],
            "variants": [],
        }

        for k in range(K):
            seed = prob["idx"] * 1000 + k * 137 + 2026
            new_nodes, new_answer = mutate_graph(prob["nodes"], prob["answer"], seed)
            if new_nodes is None:
                continue

            # Verify fingerprint preservation
            new_fp = structural_fingerprint(new_nodes)
            if new_fp != prob["fingerprint"]:
                continue

            # Verify answer via forward execution
            check = execute_graph(new_nodes)
            if check is None or abs(check - new_answer) > 0.01:
                continue

            # Build variant question (values only — model must reason)
            leaf_vals = [n["value"] for n in new_nodes if n["op"] == "assign"]
            leaf_strs = [str(int(v)) if v == int(v) else str(v) for v in leaf_vals]
            agent = AGENTS[(prob["idx"] + k) % len(AGENTS)]

            # Reconstruct a natural-language variant
            # Replace numbers in original question with new values
            orig_q = prob["question"]
            orig_leaves = [n["value"] for n in prob["nodes"] if n["op"] == "assign"]

            variant_q = orig_q
            replacements = list(zip(orig_leaves, leaf_vals))
            # Sort by length of original number string (longest first to avoid partial matches)
            replacements.sort(key=lambda x: -len(str(int(x[0])) if x[0]==int(x[0]) else str(x[0])))

            for orig_val, new_val in replacements:
                orig_str = str(int(orig_val)) if orig_val == int(orig_val) else str(orig_val)
                new_str = str(int(new_val)) if new_val == int(new_val) else str(new_val)
                # Replace first occurrence only
                variant_q = variant_q.replace(orig_str, new_str, 1)

            # Swap agent name if present
            # Find common names in GSM8K
            for name in ["Janet","Mark","James","John","Mary","Tom","Sarah","Bob",
                         "Alice","Peter","Jane","Bill","Mike","Lisa","Anna","David",
                         "Emily","Alex","Sam","Chris","Tim","Dan","Amy","Beth","Carl",
                         "Weng","Natalia","Elaine","Julie","Martha","Toby","Henry",
                         "Josh","Jack","Jill","Gerald","Cecelia","Rex","Gail"]:
                if name in variant_q:
                    variant_q = variant_q.replace(name, agent, 1)
                    break

            ans_val = new_answer
            if ans_val == int(ans_val):
                ans_val = int(ans_val)

            item["variants"].append({
                "item_id": f"gsm8k_{prob['idx']:04d}_v{k:02d}",
                "question": variant_q,
                "answer": float(ans_val),
                "fingerprint": new_fp,
                "leaf_values": [float(v) for v in leaf_vals],
            })

        if len(item["variants"]) >= 3:  # Need at least 3 variants
            eval_items["items"].append(item)
            success += 1

    eval_items["metadata"]["n_originals"] = success
    eval_items["metadata"]["total_items"] = sum(
        1 + len(item["variants"]) for item in eval_items["items"]
    )

    # Save
    outpath = "/home/claude/isomorph_eval/isomorph_gsm8k_eval.json"
    with open(outpath, "w") as f:
        json.dump(eval_items, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATASET GENERATED")
    print(f"{'='*60}")
    print(f"  Original problems:    {success}")
    print(f"  Variants per problem: ~{K}")
    print(f"  Total evaluation items: {eval_items['metadata']['total_items']}")
    print(f"  All fingerprints verified: ✓")
    print(f"  All answers forward-executed: ✓")
    print(f"  Saved to: {outpath}")

    # Show example
    if eval_items["items"]:
        ex = eval_items["items"][0]
        print(f"\n  Example original:")
        print(f"    {ex['question'][:100]}...")
        print(f"    Answer: {ex['answer']}")
        if ex["variants"]:
            v = ex["variants"][0]
            print(f"\n  Example variant:")
            print(f"    {v['question'][:100]}...")
            print(f"    Answer: {v['answer']}")
            print(f"    Fingerprint match: {v['fingerprint'] == ex['fingerprint']}")

if __name__ == "__main__":
    main()
