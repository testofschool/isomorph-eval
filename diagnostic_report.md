# Isomorph-Eval Diagnostic Report: Difficulty vs. Contamination

**Date:** 2026-05-12
**Author:** Automated diagnostic (Claude)
**Status:** CRITICAL — Paper claims require revision before submission

## Executive Summary

The uniform Δ ≈ +0.40-0.47 observed across all 5 models (including the intended control) is **not primarily contamination**. The dominant signal is **broken variant answers** caused by annotation-chain replay bugs. A secondary confound is **systematic answer magnitude inflation** in variants.

**Classification: SCENARIO C (both effects) with data quality as the primary confound.**

---

## Task 1: Structural Difficulty Audit

| Metric | Mean_orig | Mean_var | Diff | p-value | Cohen's d | Verdict |
|--------|-----------|----------|------|---------|-----------|---------|
| word_count | 44.71 | 44.71 | 0.000 | 1.00 | 0.000 | OK |
| answer_magnitude | 1.65 | 2.03 | -0.379 | <0.001 | **-0.946** | **SIGNIFICANT** |
| max_number | 115.80 | 229.57 | -113.77 | <0.001 | **-0.362** | **SIGNIFICANT** |
| digit_complexity | 5.20 | 5.20 | 0.000 | 1.00 | 0.000 | OK |
| n_entities | 1.31 | 1.34 | -0.030 | 1.00 | -0.175 | OK |
| count_numbers | 3.13 | 3.13 | 0.000 | 1.00 | 0.000 | OK |
| sum_of_numbers | 157.93 | 292.72 | -134.79 | <0.001 | **-0.427** | **SIGNIFICANT** |

**Finding:** Structural templates are perfectly preserved (same word count, number count, digit complexity), but **number magnitudes are systematically inflated**. Variant answers are ~2.4x larger (median 98 vs 46). Cohen's d = -0.946 for answer magnitude is a large effect.

This alone would cause an expected 5-15pp accuracy drop (consistent with GSM-Symbolic literature), but cannot explain the full 40+pp drop we observe.

---

## Task 2: GPT-OSS Reversal Diagnosis

- V2 (N=17 originals, eval_subset_50): Acc_orig=88.2%, Acc_iso=84.6%, Δ≈+0.035
- V3 (N=20 originals, eval_subset_100): Acc_orig=100.0%, Acc_iso=58.7%, Δ=+0.430
- **Zero item overlap** between v2 and v3 evaluations
- V2 original IDs: gsm8k_0040, 0063, 0111, ... (17 items)
- V3 original IDs: gsm8k_0125, 0260, 0263, ... (20 items, completely disjoint)

**Diagnosis:** Reversal caused entirely by **different item composition**, not pipeline change. The v3 subset contains items with broken variant answers that inflate Δ.

---

## Task 3: Baseline Calibration

No uncontaminated small model (<3B) available on Groq. However, the cross-model evidence (Task 5) makes a baseline model unnecessary for diagnosis:

All 5 models from 3 different organizations show the **exact same items** as smoking guns. This rules out model-specific contamination as the primary driver, since the pattern is determined by item properties, not model training data.

The expected baseline difficulty drop from numeric perturbation (literature consensus): 5-15pp.

---

## Task 4: Per-Item Distribution Analysis

All 5 models show **strongly bimodal** distributions:

| Model | N | Clean (|Δ|<0.1) | Smoking Gun (Δ>0.8) | Bimodal Ratio |
|-------|---|-----------------|---------------------|---------------|
| Llama 3.1 8B | 100 | 45 (45%) | 31 (31%) | 76% |
| Llama 4 Scout | 100 | 51 (51%) | 36 (36%) | 87% |
| Qwen3 32B | 70 | 34 (49%) | 26 (37%) | 86% |
| GPT-OSS 120B | 20 | 10 (50%) | 8 (40%) | 90% |
| Llama 3.3 70B | 14 | 7 (50%) | 5 (36%) | 86% |

**Critical test:** The answer magnitude shift between smoking-gun and clean items is **statistically identical** (p=0.727):
- Smoking guns: variant magnitude shift = +0.411
- Clean items: variant magnitude shift = +0.365

This rules out magnitude-based difficulty as the differentiator between these groups.

---

## Task 5: Cross-Model Correlation

| Pair | ρ (Spearman) | N shared | Smoking Gun Jaccard |
|------|-------------|----------|---------------------|
| Llama 8B vs Scout | 0.768 | 100 | 0.86 |
| Llama 8B vs Qwen3 | 0.772 | 70 | 0.82 |
| Llama 8B vs GPT-OSS | 0.942 | 20 | 0.88 |
| Scout vs Qwen3 | 0.930 | 70 | 0.90 |
| Scout vs GPT-OSS | 0.942 | 20 | **1.00** |
| GPT-OSS vs Llama 70B | 0.942 | 14 | 0.83 |

**All correlations exceed 0.77; all Jaccard similarities exceed 0.80.**

The SAME items are smoking guns for ALL models across ALL organizations. This is not model-specific contamination — it is an item-level property.

---

## CRITICAL FINDING: Variant Answer Verification Failures

### Proven broken items:

**gsm8k_0328** (smoking gun in ALL 5 models):
- Original: "1 pp per 3 min, 60% charged" → 2 hours (correct)
- Bug: Value conflation — "60" in "60% charged" conflated with "60" in minutes-per-hour conversion
- Variant 0: "9pp/6min, 10% charged" → stated answer: 54 (WRONG, correct is 1.0 hour)
- Proof: (100-10)*6/10 = 54 matches the buggy chain; (100-10)/9*6/60 = 1.0 is correct
- All 5 variant answers verified as WRONG via this mechanism

**gsm8k_0271** (smoking gun in ALL 5 models):
- Original: "2 houses, 3 bedrooms, 2 windows, 4 additional" → 20 (correct)
- Variant 0: "9 houses, 6 bedrooms, 6 windows, 3 additional" → stated: 234, correct: 351
- Pattern: annotation chain uses wrong substitution mapping for house count
- Model (Llama 8B) answered 351 — the CORRECT answer — and was scored WRONG

**gsm8k_0033** (smoking gun in all models with data):
- Original: "110 coins, 30 more gold" → 70 (correct: (110+30)/2)
- Variant 0: "239 coins, 74 more gold" → stated: 114, correct: 156.5 (NON-INTEGER!)
- The variant should have been filtered by the non-integer check but wasn't

### Additional suspicious items:
- **gsm8k_1139, gsm8k_1200**: ALL variant answers identical to original answer despite changed numbers
- **gsm8k_0725**: Variant 0 has answer=25 (same as original) despite different numbers

### Clean item verification:
Clean items (|Δ|<0.1) have CORRECT variant answers. Examples:
- gsm8k_0006: model says 880, expected 880 ✓
- gsm8k_0064: model says 2536, expected 2535 ✓ (within tolerance)
- gsm8k_0072: model says 270, expected 270 ✓

---

## Final Classification

### **SCENARIO C: Both effects, with DATA QUALITY as the dominant confound**

The observed Δ ≈ +0.40 decomposes as:

1. **Δ_data_quality ≈ +0.30-0.35**: Items with broken variant answers where models that reason correctly are penalized. These create the bimodal distribution and account for the vast majority of "smoking guns."

2. **Δ_difficulty ≈ +0.05-0.10**: Systematic answer magnitude inflation in variants (Cohen's d = -0.946). This contributes a small uniform accuracy drop consistent with GSM-Symbolic literature.

3. **Δ_contamination ≈ unknown**: Cannot be isolated until data quality issues are resolved. Any real contamination signal is masked by the dominant data quality confound.

### Evidence summary:
- Bimodal distribution → **item-level effect** (not uniform difficulty)
- Same items affected across ALL models → **item property, not model property**
- Equal magnitude shift in both groups → **not difficulty-driven separation**
- Provably wrong variant answers → **data quality is the separator**
- Clean items verified correct → **clean items genuinely measure reasoning**

---

## Recommended Paper Changes

### Option A (Recommended): Fix the data, then re-evaluate
1. Implement independent variant answer verification (solve each variant problem independently, don't rely on annotation-chain replay)
2. Remove or flag items with value conflation, non-integer intermediates, or answer-invariant variants
3. Re-run all 5 models on the cleaned dataset
4. Then the contamination claim can be properly evaluated

### Option B: Reframe the paper
1. Acknowledge the data quality limitation prominently
2. Report results ONLY on verified-clean items (the ~50% with |Δ|<0.1)
3. Frame the contribution as methodology + tooling, not the specific empirical claims
4. The τ-isomorphism framework is still valuable; the specific variant generation needs improvement

### Option C: Use N=50 results with caveats
1. The v2 results (eval_subset_50) may have fewer broken items (GPT-OSS showed Δ≈0.025)
2. But this hasn't been verified either and has lower power
3. Not recommended without verification

### NOT recommended: Publishing current results as contamination evidence
The provably broken variant answers make the current contamination claims indefensible under peer review.

---

## Whether N=50 or N=100 Results Should Be in the Paper

**Neither**, without variant answer verification. The N=50 results appear cleaner (GPT-OSS as control) but haven't been verified. The N=100 results are demonstrably contaminated by data quality issues.

The paper should use whichever dataset passes an independent verification step (solving each variant problem from scratch using a symbolic solver or manual verification).
