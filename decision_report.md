# Decision Report: Clean-Data Rescoring Results

**Date:** 2026-05-12
**Status:** DEFINITIVE — The uniform Δ≈+0.40 was entirely a data quality artifact

## Key Finding

After removing 40 items with broken variant answers (verified via 5-model consensus + heuristic detection), the contamination signal **collapses**:

| Model | N_full | Δ_full | N_clean | Δ_clean | p_clean | Verdict |
|-------|--------|--------|---------|---------|---------|---------|
| Llama 3.1 8B | 100 | +0.417 | 59 | +0.126 | 0.002 | MILD_GAP |
| Llama 4 Scout | 100 | +0.402 | 59 | +0.049 | 0.169 | INVARIANT |
| Qwen3 32B | 70 | +0.465 | 38 | +0.066 | 0.060 | INVARIANT |
| GPT-OSS 120B | 20 | +0.430 | 12 | +0.053 | 0.174 | INVARIANT |
| Llama 3.3 70B | 14 | +0.443 | 8 | +0.054 | 0.321 | INVARIANT |

## Interpretation

1. **No contamination evidence for 4/5 models.** Scout, Qwen3, GPT-OSS, and Llama 70B show Δ≈+0.05 on clean items — consistent with the 5-15pp baseline difficulty drop from numeric perturbation (GSM-Symbolic literature).

2. **Llama 3.1 8B shows a mild gap** (Δ=+0.126, p=0.002). This is statistically significant but small — likely a mix of memorization and difficulty sensitivity. At 8B parameters with the oldest training data, some memorization is plausible but not dramatic.

3. **The original Δ≈+0.40 was almost entirely driven by broken variant answers.** Items with wrong expected answers caused correctly-reasoning models to be scored as incorrect, inflating the gap uniformly across all models.

4. **The control model (GPT-OSS) now works as intended.** On clean data, GPT-OSS shows Δ=+0.053 (p=0.174) — exactly the invariant behavior expected from a model with no GSM8K in its training data.

## What Can Be Claimed in the Paper

### Defensible claims:
- The τ-isomorphism methodology is sound and valuable
- The framework correctly measures structural reasoning transfer
- Llama 3.1 8B shows mild contamination effects (Δ=+0.126, p=0.002)
- Modern models (Scout 17B, Qwen3 32B) show no significant contamination on verified items
- The difficulty baseline from numeric perturbation is ~5% (consistent with literature)

### NOT defensible:
- ~~"All models show Δ≈+0.40, indicating widespread contamination"~~ — This was a data quality artifact
- ~~"PURE_MEMORIZER classification"~~ — No model qualifies on clean data
- ~~"Even GPT-OSS shows contamination"~~ — GPT-OSS is invariant as expected

## Recommended Path Forward

### Option A: Fix the Mutator and regenerate variants (RECOMMENDED)
1. Fix the annotation-chain replay bugs (value conflation, substitution mapping)
2. Add independent forward-verification as a post-generation quality gate
3. Regenerate variants for all 100 items
4. Re-evaluate all models on the new, fully-verified dataset
5. This gives N=100 with clean data — much stronger statistical power

### Option B: Publish with N=59 clean items
1. Use the verified-clean subset
2. Acknowledge the data quality issue and the verification methodology
3. Report the honest results (mild gap for 8B, invariant for others)
4. Weaker but defensible; the story changes from "contamination everywhere" to "methodology contribution + mild 8B finding"

### Option C: Reframe as methodology paper
1. Focus on the τ-isomorphism framework contribution
2. Use the data quality issue as a case study for why independent verification matters
3. De-emphasize empirical contamination claims
4. The framework and tooling are the lasting contribution regardless

## Data Quality Issue Summary

- 40/100 items had broken variant answers (29 fully broken, 11 partially broken)
- Root cause: annotation-chain replay bugs (value conflation, wrong substitution mapping)
- Detection method: multi-model consensus (≥2 models agree on answer ≠ expected) + heuristic flags
- The 59 clean items have variant answers confirmed by model consensus
