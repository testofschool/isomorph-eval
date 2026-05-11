# Isomorph-Eval GSM8K Dataset

## Overview

This dataset contains **667 original GSM8K problems** with **3,199 verified
tau-isomorphic variants** (3,866 evaluation entries total, ~4.8 variants per
item).

## Source

- **Original benchmark**: GSM8K test set (Cobbe et al., 2021)
- **License**: MIT (derived from GSM8K, which is MIT-licensed)

## Generation Method

1. **Rule-based annotation parsing**: GSM8K solution annotations
   (`<<expression=result>>`) are parsed into reasoning DAGs
2. **Conflation filter**: Problems where the same numeric value serves as
   both an input and an implicit constant are excluded
3. **Deterministic mutation**: Leaf values are replaced with same-digit-class
   random integers; the DAG is forward-executed to compute the new answer
4. **Strict validation pipeline**:
   - Integer answers only (non-integer variants rejected)
   - Positive intermediates (no negative intermediate values)
   - Reasonable range (answers < 10^8)
   - Text replacement verification (variant question text is checked
     for coherent number substitution)
   - Structural fingerprint preservation (tau-signature match)

## Rejection Statistics

The pipeline rejected 19,826 candidate variants:
- 63.5% non-integer answers
- 12.1% text-replacement mismatches
- 9.6% negative values
- 1.2% negative intermediates
- 0.7% overflow (answer > 10^8)

## Format

```json
{
  "items": [
    {
      "item_id": "gsm8k_0000",
      "question": "Janet's ducks lay 16 eggs per day...",
      "answer": 18.0,
      "fingerprint": "a1b2c3d4e5f6",
      "n_nodes": 5,
      "tau_signature": ["assign", "assign", "assign", "subtract", "multiply"],
      "variants": [
        {
          "item_id": "gsm8k_0000_v00",
          "question": "Tomas's ducks lay 65 eggs per day...",
          "answer": 464.0,
          "fingerprint": "a1b2c3d4e5f6",
          "leaf_values": [65.0, 3.0, 4.0, 2.0]
        }
      ]
    }
  ],
  "metadata": {
    "source": "GSM8K test set (openai/gsm8k)",
    "n_originals": 667,
    "total_items": 3866,
    "generation_method": "rule-based parse + conflation filter + strict validation",
    "verification": "annotation chain replay, integer filter, positive intermediates, text replacement verification"
  }
}
```

## Fields

- `item_id`: Unique identifier (format: `gsm8k_NNNN` for originals, `gsm8k_NNNN_vNN` for variants)
- `question`: Natural language math problem
- `answer`: Numeric answer (always a positive integer stored as float)
- `fingerprint`: SHA-256 hash of the tau-signature (structural fingerprint)
- `tau_signature`: List of operation types in the reasoning DAG
- `n_nodes`: Number of nodes in the reasoning DAG
- `leaf_values`: (variants only) The mutated input values

## Verification

All 3,199 variants have been verified:
- 100% positive integer answers
- 100% structural fingerprint match with parent
- 100% forward-execution verified (answer recomputed from DAG)
- 100% reasonable range (0 < answer < 10^8)
