#!/usr/bin/env python3
"""Create entity-clean eval_verified_v3.json from eval_verified_v2.json."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from verify_entities import (
    audit_dataset,
    extract_proper_nouns,
    shared_suffix_artifact,
    whole_word_count,
    word_counts,
)


DATA_DIR = Path(__file__).parent / "data"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def regex_replace_name(text: str, old_name: str, new_name: str) -> str:
    return re.sub(
        rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(old_name)}(?![A-Za-zÀ-ÖØ-öø-ÿ])",
        new_name,
        text,
    )


def infer_replacement(original_name: str, variant_text: str, original_names: set[str]) -> str | None:
    words = word_counts(variant_text)
    for word in words:
        suffix = shared_suffix_artifact(original_name, word)
        if suffix:
            return word[:-len(suffix)]

    for name in extract_proper_nouns(variant_text):
        if name not in original_names:
            return name
    return None


def repair_variant(item: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    repaired = copy.deepcopy(variant)
    text = repaired.get("question", "")
    original_names = extract_proper_nouns(item.get("question", ""))
    original_name_set = set(original_names)

    for old_name in original_names:
        original_count = whole_word_count(item.get("question", ""), old_name)
        current_count = whole_word_count(text, old_name)
        replacement = infer_replacement(old_name, text, original_name_set)

        if replacement:
            for word in list(word_counts(text)):
                if shared_suffix_artifact(old_name, word):
                    text = regex_replace_name(text, word, replacement)
            if current_count and current_count < original_count:
                text = regex_replace_name(text, old_name, replacement)

    repaired["question"] = text
    repaired["entity_repaired_v3"] = repaired.get("question") != variant.get("question")
    return repaired


def failing_variant_ids(report: dict[str, Any]) -> set[str]:
    return {failure["variant_id"] for failure in report.get("failures", [])}


def main() -> None:
    v2_path = DATA_DIR / "eval_verified_v2.json"
    audit_path = DATA_DIR / "entity_audit_results.json"
    v3_path = DATA_DIR / "eval_verified_v3.json"

    v2_data = load_json(v2_path)
    audit_report = load_json(audit_path) if audit_path.exists() else audit_dataset(copy.deepcopy(v2_data))
    failed_variants = failing_variant_ids(audit_report)

    kept_items: list[dict[str, Any]] = []
    kept_intact = 0
    trimmed = 0
    regenerated = 0
    quarantined: list[str] = []

    for item in v2_data["items"]:
        item_copy = copy.deepcopy(item)
        variants = item_copy.get("variants", [])
        item_failed = item_copy["item_id"] in set(audit_report.get("items_with_failures", []))

        if not item_failed:
            kept_intact += 1
            kept_items.append(item_copy)
            continue

        passing_variants = [v for v in variants if v["item_id"] not in failed_variants]
        if len(passing_variants) >= 2:
            item_copy["variants"] = passing_variants
            item_copy["entity_trimmed_v3"] = True
            trimmed += 1
            kept_items.append(item_copy)
            continue

        repaired_item = copy.deepcopy(item)
        repaired_item["variants"] = [repair_variant(item, v) for v in variants]
        repaired_report = audit_dataset({"items": [copy.deepcopy(repaired_item)]})
        repaired_failed = failing_variant_ids(repaired_report)
        repaired_passing = [
            v for v in repaired_item["variants"]
            if v["item_id"] not in repaired_failed
        ]

        if len(repaired_passing) >= 2:
            repaired_item["variants"] = repaired_passing
            repaired_item["entity_regenerated_v3"] = True
            regenerated += 1
            kept_items.append(repaired_item)
        else:
            quarantined.append(item["item_id"])

    v3_data = copy.deepcopy(v2_data)
    v3_data["items"] = kept_items
    metadata = v3_data.setdefault("metadata", {})
    metadata.update({
        "description": "Entity-clean verified dataset after post-hoc entity consistency audit",
        "source": "eval_verified_v2.json + entity audit",
        "n_items": len(kept_items),
        "total_variants": sum(len(item.get("variants", [])) for item in kept_items),
        "entity_audit": {
            "kept_intact": kept_intact,
            "trimmed_variants": trimmed,
            "regenerated": regenerated,
            "quarantined": len(quarantined),
            "quarantined_item_ids": quarantined,
        },
    })
    save_json(v3_path, v3_data)

    print(f"v2 items: {len(v2_data['items'])}, v3 items: {len(kept_items)}")
    print(
        f"Kept intact: {kept_intact}, Trimmed variants: {trimmed}, "
        f"Regenerated: {regenerated}, Quarantined: {len(quarantined)}"
    )
    print(f"Quarantined item IDs: {quarantined}")
    print(f"Saved to: {v3_path}")


if __name__ == "__main__":
    main()
