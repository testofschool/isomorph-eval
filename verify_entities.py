#!/usr/bin/env python3
"""Audit entity consistency in verified Isomorph-Eval datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMMON_WORDS = {
    "A", "An", "After", "All", "And", "At", "Before", "By", "During",
    "Each", "Every", "For", "From", "He", "Her", "His", "How", "If",
    "In", "It", "Later", "On", "One", "She", "The", "Then", "They",
    "This", "To", "What", "When", "Where", "While", "With",
}

NUMBER_WORDS = {
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
    "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
}

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError(f"{path} does not contain an items list")
    return data


def word_counts(text: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(text))


def sentence_initial_spans(text: str) -> set[int]:
    spans: set[int] = set()
    for match in re.finditer(r"(^|[.!?]\s+)([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)", text):
        spans.add(match.start(2))
    return spans


def extract_proper_nouns(text: str) -> list[str]:
    initial_starts = sentence_initial_spans(text)
    names: list[str] = []
    seen: set[str] = set()
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).strip("'’")
        if len(token) < 3:
            continue
        if not token[0].isupper():
            continue
        if token in COMMON_WORDS or token in NUMBER_WORDS:
            continue
        if match.start() in initial_starts and token not in seen:
            # Sentence-initial common words are filtered above; keep repeated
            # capitalized names even when they happen to start a sentence.
            later = re.search(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(token)}(?![A-Za-zÀ-ÖØ-öø-ÿ])", text[match.end():])
            if not later:
                continue
        if token not in seen:
            names.append(token)
            seen.add(token)
    return names


def whole_word_count(text: str, word: str) -> int:
    return len(re.findall(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(word)}(?![A-Za-zÀ-ÖØ-öø-ÿ])", text))


def shared_suffix_artifact(original: str, candidate: str) -> str | None:
    if candidate == original or len(original) < 4 or len(candidate) <= len(original):
        return None
    max_suffix = math.ceil(len(original) / 2)
    for size in range(max_suffix, 1, -1):
        suffix = original[-size:]
        if candidate.endswith(suffix) and not candidate.startswith(original):
            stem = candidate[:-size]
            if len(stem) >= 3 and stem[0].isupper():
                return suffix
    return None


def numeric_values(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in NUMBER_RE.finditer(text)}


def audit_dataset(data: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    clean_items: list[str] = []
    items_with_failures: set[str] = set()
    total_variants = 0

    for item in data["items"]:
        item_id = item["item_id"]
        original_question = item.get("question", "")
        original_names = extract_proper_nouns(original_question)
        original_counts = {name: whole_word_count(original_question, name) for name in original_names}
        variants = item.get("variants", [])
        total_variants += len(variants)

        seen_text: dict[str, str] = {}
        item_failed = False

        for variant in variants:
            variant_id = variant["item_id"]
            variant_question = variant.get("question", "")
            variant_words = word_counts(variant_question)
            variant_names = extract_proper_nouns(variant_question)
            variant_name_set = set(variant_names)
            variant_failed = False

            if variant_question in seen_text:
                failures.append({
                    "item_id": item_id,
                    "variant_id": variant_id,
                    "check": "DUPLICATE",
                    "details": f"Duplicate variant text also used by {seen_text[variant_question]}",
                })
                item_failed = variant_failed = True
            else:
                seen_text[variant_question] = variant_id

            for original_name, original_count in original_counts.items():
                current_count = whole_word_count(variant_question, original_name)
                artifacts = [
                    word for word in variant_words
                    if shared_suffix_artifact(original_name, word)
                ]
                candidates = [
                    name for name in variant_names
                    if name not in original_counts and not shared_suffix_artifact(original_name, name)
                ]
                replacement_attempted = current_count < original_count or bool(artifacts)

                for artifact in artifacts:
                    suffix = shared_suffix_artifact(original_name, artifact)
                    failures.append({
                        "item_id": item_id,
                        "variant_id": variant_id,
                        "check": "SUBSTRING_ARTIFACT",
                        "details": (
                            f"Variant token '{artifact}' appears to append suffix "
                            f"'{suffix}' from original name '{original_name}'"
                        ),
                    })
                    item_failed = variant_failed = True

                if replacement_attempted and current_count > 0:
                    failures.append({
                        "item_id": item_id,
                        "variant_id": variant_id,
                        "check": "LEFTOVER_NAME",
                        "details": f"Original name '{original_name}' found {current_count} time(s) in variant text",
                    })
                    item_failed = variant_failed = True

                if replacement_attempted and candidates:
                    candidate = candidates[0]
                    candidate_count = whole_word_count(variant_question, candidate)
                    if current_count + candidate_count != original_count:
                        failures.append({
                            "item_id": item_id,
                            "variant_id": variant_id,
                            "check": "CONSISTENCY",
                            "details": (
                                f"Original name '{original_name}' appears {original_count} time(s); "
                                f"candidate replacement '{candidate}' appears {candidate_count} "
                                f"time(s), with {current_count} leftover original occurrence(s)"
                            ),
                        })
                        item_failed = variant_failed = True

            if variant.get("all_values_present") is False:
                failures.append({
                    "item_id": item_id,
                    "variant_id": variant_id,
                    "check": "NUMBER_GRAPH_MISMATCH",
                    "details": "Variant metadata reports all_values_present=false",
                })
                item_failed = variant_failed = True

            if not variant_failed:
                variant["_entity_clean"] = True

        if item_failed:
            items_with_failures.add(item_id)
        else:
            clean_items.append(item_id)

    breakdown = Counter(f["check"] for f in failures)
    return {
        "total_items": len(data["items"]),
        "total_variants": total_variants,
        "failures": failures,
        "failure_breakdown": dict(sorted(breakdown.items())),
        "items_with_failures": sorted(items_with_failures),
        "clean_items": sorted(clean_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_verified_v2.json")
    parser.add_argument("--output", default="data/entity_audit_results.json")
    args = parser.parse_args()

    data = load_dataset(Path(args.input))
    report = audit_dataset(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    clean = len(report["clean_items"])
    total = report["total_items"]
    failed = len(report["items_with_failures"])
    print(f"Entity audit complete: {clean}/{total} items clean, {failed} items have failures")
    print(f"Failure breakdown: {report['failure_breakdown']}")
    if not report["failures"]:
        print("0 failures found")
    else:
        print(f"{len(report['failures'])} failures found")


if __name__ == "__main__":
    main()
