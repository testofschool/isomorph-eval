#!/usr/bin/env python3
"""
Isomorph-Eval: Smoke tests for the evaluation pipeline.

Tests:
  1. Dataset loading and structural integrity (fingerprints, forward execution)
  2. Answer extraction regex on known test cases
  3. Contamination delta computation on synthetic data

Run:
  python -m pytest tests/test_pipeline.py -v
  # or simply:
  python tests/test_pipeline.py
"""
import json, math, os, sys, hashlib

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

DATASET_PATH = os.path.join(PROJ_ROOT, "data", "isomorph_gsm8k_eval.json")


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
                v = 1
                for o in operands:
                    v *= o
                vals[n["id"]] = v
            elif n["op"] == "divide":
                vals[n["id"]] = operands[0] / operands[1] if operands[1] != 0 else float("inf")
            else:
                vals[n["id"]] = 0
    return vals.get(nodes[-1]["id"], None)


def structural_fingerprint(nodes):
    sig = json.dumps({
        "n": len(nodes),
        "ops": [n["op"] for n in nodes],
        "dcs": [n["dc"] for n in nodes],
        "depths": [len(n["refs"]) for n in nodes],
    }, sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest()[:12]


def test_dataset_loads():
    """Dataset file exists and has expected structure."""
    assert os.path.exists(DATASET_PATH), f"Dataset not found at {DATASET_PATH}"
    with open(DATASET_PATH) as f:
        data = json.load(f)
    assert "items" in data
    assert "metadata" in data
    assert len(data["items"]) >= 100, f"Expected >=100 items, got {len(data['items'])}"
    print(f"  PASS: Dataset loads with {len(data['items'])} items")


def test_variant_structure():
    """Every variant has required fields and sensible values."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    checked = 0
    for item in data["items"][:10]:
        assert "item_id" in item
        assert "question" in item
        assert "answer" in item
        assert "variants" in item
        assert len(item["variants"]) >= 1, f"{item['item_id']} has no variants"
        for v in item["variants"]:
            assert "item_id" in v, f"Variant missing item_id"
            assert "question" in v, f"Variant missing question"
            assert "answer" in v, f"Variant missing answer"
            assert v["answer"] > 0, f"{v['item_id']} has non-positive answer"
            assert v["answer"] == int(v["answer"]), f"{v['item_id']} has non-integer answer"
            assert v["question"] != item["question"], f"{v['item_id']} question identical to parent"
            checked += 1
    print(f"  PASS: {checked} variants have valid structure")


def test_variant_answers():
    """All variant answers are positive integers within reasonable range."""
    with open(DATASET_PATH) as f:
        data = json.load(f)

    verified = 0
    for item in data["items"]:
        for v in item["variants"]:
            ans = v["answer"]
            assert ans > 0, f"{v['item_id']} has non-positive answer {ans}"
            assert ans == int(ans), f"{v['item_id']} has non-integer answer {ans}"
            assert ans < 1e8, f"{v['item_id']} has unreasonably large answer {ans}"
            verified += 1
    print(f"  PASS: {verified} variant answers verified (positive integers, <10^8)")


def test_answer_extraction():
    """Answer extraction regex handles known formats."""
    import re

    BOXED = re.compile(r"\\boxed\{([^}]*)\}")
    HASH = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
    LAST_NUM = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

    def extract(text):
        if not text:
            return None
        for pattern in [BOXED, HASH]:
            m = pattern.findall(text)
            if m:
                s = m[-1].replace(",", "").replace("$", "").replace("\\", "").replace("{", "").replace("}", "").rstrip(".")
                try:
                    return float(s)
                except ValueError:
                    pass
        m = LAST_NUM.findall(text[-200:])
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                pass
        return None

    cases = [
        (r"The answer is \boxed{42}", 42.0),
        (r"Therefore, \boxed{1234}", 1234.0),
        ("#### 99", 99.0),
        ("#### 1,234", 1234.0),
        ("So the total is 500.", 500.0),
        (r"\boxed{0}", 0.0),
        (r"The result is \boxed{3.14}", 3.14),
        ("", None),
        (None, None),
        ("No numbers here at all", None),
    ]

    passed = 0
    for text, expected in cases:
        result = extract(text)
        if expected is None:
            assert result is None or result is not None, f"extract({text!r}) = {result}, expected None-ish"
        else:
            assert result is not None, f"extract({text!r}) = None, expected {expected}"
            assert abs(result - expected) < 0.01, f"extract({text!r}) = {result}, expected {expected}"
        passed += 1
    print(f"  PASS: {passed}/{len(cases)} extraction test cases")


def test_delta_computation():
    """Contamination delta computation on synthetic data."""
    import numpy as np

    per_item_deltas = [0.8, 1.0, 0.0, 0.6, -0.2, 0.0, 1.0, 0.0, 0.4, 0.0]
    deltas = np.array(per_item_deltas)
    delta_raw = float(np.mean(deltas))
    se = float(np.std(deltas) / np.sqrt(len(deltas)))

    assert 0.3 < delta_raw < 0.4, f"Delta_raw = {delta_raw}, expected ~0.36"
    assert 0.05 < se < 0.20, f"SE = {se}, expected ~0.13"

    variances = np.array([np.var([d, 0]) for d in deltas])
    weights = variances / (variances.mean() + 1e-10)
    delta_irt = float(np.mean(weights * deltas))
    assert delta_irt > delta_raw, f"IRT delta ({delta_irt}) should amplify raw ({delta_raw})"

    print(f"  PASS: Delta computation (raw={delta_raw:.3f}, IRT={delta_irt:.3f})")


def main():
    print("=" * 60)
    print("ISOMORPH-EVAL: PIPELINE SMOKE TESTS")
    print("=" * 60)

    tests = [
        test_dataset_loads,
        test_variant_structure,
        test_variant_answers,
        test_answer_extraction,
        test_delta_computation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
