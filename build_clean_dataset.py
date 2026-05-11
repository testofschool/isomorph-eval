#!/usr/bin/env python3
"""
Task 3: Build verified-clean evaluation dataset from audit results.
Only includes items where ALL variants passed verification.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    with open(DATA_DIR / "eval_subset_100.json") as f:
        dataset = json.load(f)

    with open(DATA_DIR / "audit_results.json") as f:
        audit = json.load(f)

    clean_ids = set()
    for item in audit["audit_results"]:
        if item["status"] == "clean":
            clean_ids.add(item["item_id"])

    clean_items = []
    for item in dataset["items"]:
        if item.get("is_original", False) and item["item_id"] in clean_ids:
            clean_items.append(item)

    clean_dataset = {
        "items": clean_items,
        "metadata": {
            **dataset.get("metadata", {}),
            "description": "Verified-clean subset: only items with ALL variants passing independent verification",
            "source": "eval_subset_100.json filtered by audit_results.json",
            "n_items": len(clean_items),
            "n_variants_per_item": 5 if clean_items else 0,
            "total_entries": len(clean_items) * 6 if clean_items else 0,
        }
    }

    output_path = DATA_DIR / "eval_verified_clean.json"
    with open(output_path, "w") as f:
        json.dump(clean_dataset, f, indent=2)

    print(f"Clean items: {len(clean_items)} / {len([i for i in dataset['items'] if i.get('is_original')])}")
    print(f"Total entries (originals + variants): {len(clean_items) * 6}")
    print(f"Saved to: {output_path}")

    # Print the clean item IDs for reference
    print(f"\nClean item IDs:")
    for item in sorted(clean_items, key=lambda x: x["item_id"]):
        print(f"  {item['item_id']} (answer={item['answer']})")


if __name__ == "__main__":
    main()
