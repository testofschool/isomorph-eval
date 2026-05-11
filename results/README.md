# Evaluation Results

This directory contains raw evaluation outputs from running LLMs on the
Isomorph-Eval GSM8K dataset.

## File Format

Each result file is a JSON object with:

```json
{
  "model": "model-name",
  "provider": "groq",
  "dataset": "isomorph_gsm8k_eval_v2",
  "n_selected": 50,
  "temperature": 0.0,
  "trials": 1,
  "report": {
    "n_items": 50,
    "acc_original": 0.94,
    "acc_isomorphic": 0.653,
    "delta_raw": 0.287,
    "se_delta": 0.065,
    "ci_95": [0.160, 0.413],
    "p_value": 8.51e-05,
    "delta_irt": 0.766,
    "n_flagged": 15,
    "archetype": "PURE_MEMORIZER",
    "per_item": [...]
  },
  "raw_results": { ... }
}
```

## Reproducing Results

```bash
export GROQ_API_KEY=your-key
python run_eval.py --provider groq --model llama-3.1-8b-instant \
  --dataset data/isomorph_gsm8k_eval.json --output results/llama3_8b.json
```
