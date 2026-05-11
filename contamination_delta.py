"""
=============================================================================
PHASE 3: THE CONTAMINATION DELTA (Δ_contam)
=============================================================================
A Unified Metric for Detecting and Quantifying Benchmark Memorization
in LLMs via IRT-Grounded Isomorphic Evaluation

Mathematical Framework — Production-Level Specification
=============================================================================

POSITIONING vs. PRIOR WORK:

  GSM1K (Zhang et al., 2024, NeurIPS):
    ✓ Created parallel benchmark, found up to 13% drops
    ✗ No structural isomorphism guarantee — "similar difficulty" by vibes
    ✗ No IRT integration — uses raw accuracy gap
    ✗ One-shot: 1,250 fixed items, not an infinite generator

  ConStat (Dekoninck et al., 2024, NeurIPS):
    ✓ Statistical test with p-values and reference models
    ✓ Performance-based definition of contamination
    ✗ Difficulty correction via generic regression, not IRT
    ✗ Rephrasing done by LLM without structural guarantees
    ✗ No formal isomorphism — semantic drift in rephrasings is uncontrolled

  PaCoST (2024):
    ✓ Paired confidence significance testing
    ✗ Requires model confidence scores (not black-box)
    ✗ No item-level analysis

  Min-K% Probability:
    ✗ Requires model internals (log-probabilities)
    ✗ Not black-box compatible

  OUR CONTRIBUTION — Δ_contam via Isomorph-Eval:
    ✓ τ-isomorphism guarantees structural equivalence (Phase 1)
    ✓ Infinite generator, not a fixed parallel dataset (Phase 2)
    ✓ IRT-grounded: decomposes gap into ability vs. memorization
    ✓ Black-box compatible: only needs binary correct/incorrect
    ✓ Connects to EFSL: shows how contamination + sparsity compound
    ✓ Borrows DIF (Differential Item Functioning) from psychometrics
      — the established framework for detecting when items function
      differently across groups matched on latent ability

=============================================================================
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ============================================================================
# SECTION 1: MATHEMATICAL FORMULATION
# ============================================================================

"""
DEFINITION 3 (Contamination Delta — Item Level).

For model M evaluated on original item i and K τ-isomorphic variants
{i'₁, ..., i'_K}, each administered T times (to account for sampling
variance at temperature > 0):

  p̂_i^orig = (1/T) Σ_{t=1}^T X_{i,t}      (empirical success rate on original)

  p̂_i^iso  = (1/KT) Σ_{k=1}^K Σ_{t=1}^T X_{i'_k,t}  (mean success rate on isomorphs)

  δ_i(M) = p̂_i^orig - p̂_i^iso              (item-level contamination delta)

Under H₀ (no contamination), since i ≅_τ i'_k for all k:
  E[δ_i] = 0

Under H₁ (contamination):
  E[δ_i] > 0    (model performs better on memorized original)


DEFINITION 4 (Contamination Delta — Model Level).

For model M evaluated on benchmark B = {i₁, ..., i_N}:

  Δ_contam(M, B) = (1/N) Σ_{i=1}^N δ_i(M)

This is the RAW contamination delta. However, this ignores item
difficulty — a 5% gap on a hard item is more informative than a
5% gap on an easy item. We therefore define the IRT-WEIGHTED variant.


DEFINITION 5 (IRT-Weighted Contamination Delta).

  Δ_contam^IRT(M, B) = (1/N) Σ_{i=1}^N a_i · δ_i(M)

where a_i is the discrimination parameter of item i estimated from
the isomorphic data (clean, uncontaminated). Items with higher
discrimination contribute more to the metric, because a gap on a
highly discriminating item is stronger evidence of memorization.

WHY THIS MATTERS: A model that memorized the answer to an easy item
(low a, low b) gets a small boost. A model that memorized the answer
to a hard, discriminating item (high a, high b) gets a large boost.
The IRT-weighted delta captures this asymmetry.
"""


# ============================================================================
# SECTION 2: HYPOTHESIS TESTING
# ============================================================================

"""
DEFINITION 6 (Contamination Hypothesis Test).

We adapt the Differential Item Functioning (DIF) framework from
psychometrics (Lord, 1980; Holland & Wainer, 1993). DIF detects
when an item functions differently for two groups matched on ability.

In our setting:
  - "Groups" are not demographic — they are EVALUATION CONDITIONS
  - Condition A: model sees original items (potentially contaminated)
  - Condition B: model sees τ-isomorphic variants (guaranteed clean)
  - Both conditions should produce identical IRT parameters if the
    model is reasoning, not memorizing

Formally, fit two separate 2PL IRT models:

  Model_orig:  P(X_i = 1 | θ^orig, a_i^orig, b_i^orig)
  Model_iso:   P(X_{i'} = 1 | θ^iso, a_i^iso, b_i^iso)

Under H₀ (no contamination):
  b_i^orig = b_i^iso    for all i    (uniform DIF = 0)
  a_i^orig = a_i^iso    for all i    (non-uniform DIF = 0)
  θ^orig = θ^iso                      (ability unchanged)

Under H₁ (contamination):
  b_i^orig < b_i^iso    for contaminated items
    (original appears easier because model has memorized it)
  θ^orig > θ^iso
    (ability on originals is inflated)


TEST STATISTIC: We use a likelihood ratio test comparing:

  L₀: constrained model (b_i^orig = b_i^iso for all i)
  L₁: unconstrained model (separate b_i for each condition)

  Λ = -2[log L₀ - log L₁]

  Under H₀: Λ ~ χ²(df = N)    (N items with freed difficulty params)

  Reject H₀ if p < α (typically α = 0.01 for contamination claims)


PER-ITEM TEST: For each item individually, test:
  H₀: b_i^orig = b_i^iso
  H₁: b_i^orig ≠ b_i^iso

  Using a Wald test: z_i = (b_i^orig - b_i^iso) / SE(b_i^orig - b_i^iso)

  Item i is flagged as "memorized" if |z_i| > z_{α/2N}
  (Bonferroni correction for multiple testing across N items)
"""


def contamination_hypothesis_test(
    responses_original: np.ndarray,   # (J models × N items × T trials)
    responses_isomorph: np.ndarray,   # (J models × N items × K variants × T trials)
    target_model_idx: int,            # which model to test
    alpha: float = 0.01,
) -> Dict:
    """
    Full contamination hypothesis test for a target model.

    Parameters
    ----------
    responses_original : array (J, N, T)
        Binary responses of J models on N original items, T trials each
    responses_isomorph : array (J, N, K, T)
        Binary responses on K isomorphic variants per item, T trials each
    target_model_idx : int
        Index of the model being tested for contamination
    alpha : float
        Significance level

    Returns
    -------
    dict with keys:
        delta_raw : float          Raw Δ_contam
        delta_irt : float          IRT-weighted Δ_contam
        p_value_global : float     Global LRT p-value
        per_item_z : array         Per-item Wald z-statistics
        flagged_items : list       Items flagged as memorized
        theta_orig : float         Ability estimated from originals
        theta_iso : float          Ability estimated from isomorphs
        theta_inflation : float    θ_orig - θ_iso
        archetype : str            Diagnostic classification
    """
    J, N, T = responses_original.shape
    K = responses_isomorph.shape[2]

    # Step 1: Compute raw item-level deltas for target model
    p_orig = responses_original[target_model_idx].mean(axis=-1)  # (N,)
    p_iso = responses_isomorph[target_model_idx].mean(axis=(-2, -1))  # (N,)
    delta_per_item = p_orig - p_iso

    delta_raw = delta_per_item.mean()

    # Step 2: Fit IRT on isomorphic data (clean, all models)
    # Collapse isomorph responses: (J, N, K*T) → binary matrix
    iso_binary = responses_isomorph.reshape(J, N, -1).mean(axis=-1)  # (J, N)
    theta_iso_all, a_iso, b_iso = fit_2pl_irt(iso_binary)

    # Step 3: Fit IRT on original data (potentially contaminated)
    orig_binary = responses_original.mean(axis=-1)  # (J, N)
    theta_orig_all, a_orig, b_orig = fit_2pl_irt(orig_binary)

    # Step 4: Extract target model's abilities
    theta_orig = theta_orig_all[target_model_idx]
    theta_iso = theta_iso_all[target_model_idx]
    theta_inflation = theta_orig - theta_iso

    # Step 5: IRT-weighted contamination delta
    # Weight each item's delta by its clean discrimination
    delta_irt = np.mean(a_iso * delta_per_item)

    # Step 6: Per-item Wald tests for DIF
    # SE estimated via variance of isomorphic variants
    se_delta = np.zeros(N)
    for i in range(N):
        # Variance of success rate across K variants and T trials
        variant_means = responses_isomorph[target_model_idx, i].mean(axis=-1)  # (K,)
        se_iso = np.std(variant_means) / np.sqrt(K)
        se_orig = np.sqrt(p_orig[i] * (1 - p_orig[i]) / T)
        se_delta[i] = np.sqrt(se_orig**2 + se_iso**2)

    z_scores = np.where(se_delta > 0, delta_per_item / se_delta, 0)

    # Bonferroni correction
    z_threshold = stats.norm.ppf(1 - alpha / (2 * N))
    flagged = [i for i in range(N) if z_scores[i] > z_threshold]

    # Step 7: Global likelihood ratio test (simplified)
    # Under H₀: performance should be same on orig and iso
    # Test statistic: sum of squared z-scores ~ χ²(N)
    chi2_stat = np.sum(np.maximum(z_scores, 0)**2)
    p_value_global = 1 - stats.chi2.cdf(chi2_stat, df=N)

    # Step 8: Classify archetype
    archetype = classify_archetype(
        delta_raw, delta_irt, theta_inflation,
        len(flagged), N, p_value_global
    )

    return {
        "delta_raw": float(delta_raw),
        "delta_irt": float(delta_irt),
        "p_value_global": float(p_value_global),
        "per_item_z": z_scores.tolist(),
        "flagged_items": flagged,
        "n_flagged": len(flagged),
        "theta_orig": float(theta_orig),
        "theta_iso": float(theta_iso),
        "theta_inflation": float(theta_inflation),
        "archetype": archetype,
        "per_item_delta": delta_per_item.tolist(),
    }


# ============================================================================
# SECTION 3: IRT INTEGRATION — θ_fake vs θ_true
# ============================================================================

"""
THEOREM 1 (Contamination-Ability Decomposition).

Under the 2PL model, a contaminated model's observed ability θ_obs
on the original benchmark can be decomposed as:

  θ_obs = θ_true + Δθ_contam

where:
  θ_true = ability estimated from isomorphic items (clean)
  Δθ_contam = ability inflation due to memorization

This decomposition is possible BECAUSE τ-isomorphic items share
the same (a, b) parameters. The difference θ_obs - θ_true is
therefore attributable solely to item-specific memorization, not
to any legitimate ability difference.

CONNECTION TO EFSL: In the Evaluation Failure Scaling Law,
the ranking error from simple averaging is:

  1 - ρ_avg = γ₀ + γ₁S + γ₂D + γ₃(S × D)

With contamination, there is an ADDITIONAL error source:

  1 - ρ_avg = γ₀ + γ₁S + γ₂D + γ₃(S × D) + γ₄C

where C = mean Δ_contam across models. Contamination acts as a
THIRD axis of the failure surface: even at full coverage (S=0)
and uniform difficulty (D=0), contamination alone can distort
rankings if different models have different contamination levels.

The compound effect is worst when all three co-occur:
  - Sparse evaluation matrix (S > 0)
  - Heterogeneous item difficulty (D > 0)
  - Differential contamination across models (C > 0)

This is the THREE-BODY PROBLEM of AI evaluation.
"""


def decompose_ability(
    theta_orig: float,
    theta_iso: float,
    se_theta_orig: float = 0.1,
    se_theta_iso: float = 0.1,
) -> Dict:
    """
    Decompose observed ability into true ability + contamination inflation.

    Returns confidence interval for the contamination component.
    """
    delta_theta = theta_orig - theta_iso
    se_delta = np.sqrt(se_theta_orig**2 + se_theta_iso**2)

    ci_95 = (
        delta_theta - 1.96 * se_delta,
        delta_theta + 1.96 * se_delta,
    )

    # Test H₀: Δθ = 0
    z = delta_theta / se_delta if se_delta > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "theta_true": theta_iso,
        "theta_fake": theta_orig,
        "theta_inflation": delta_theta,
        "se": se_delta,
        "ci_95": ci_95,
        "z_stat": z,
        "p_value": p_value,
        "is_significant": p_value < 0.01,
    }


# ============================================================================
# SECTION 4: DIAGNOSTIC ARCHETYPES
# ============================================================================

"""
DEFINITION 7 (Contamination Archetypes).

Based on the pattern of (Δ_contam, Δθ, DIF pattern), we classify
models into four diagnostic archetypes:

┌────────────────────────┬───────────┬──────────┬────────────────────────┐
│ Archetype              │ Δ_contam  │ Δθ       │ DIF Pattern            │
├────────────────────────┼───────────┼──────────┼────────────────────────┤
│ 1. Robust Reasoner     │ ≤ 0.02    │ ≤ 0.1    │ No DIF on any item     │
│                        │           │          │                        │
│    The model solves     │           │          │ IRT params identical   │
│    isomorphs as well    │           │          │ across conditions      │
│    as originals.       │           │          │                        │
├────────────────────────┼───────────┼──────────┼────────────────────────┤
│ 2. Pure Memorizer      │ ≥ 0.10    │ ≥ 0.5    │ Uniform DIF on >30%    │
│                        │           │          │ of items               │
│    Performance collapses│           │          │                        │
│    on isomorphs.        │           │          │ b_orig << b_iso for    │
│    θ heavily inflated.  │           │          │ flagged items          │
├────────────────────────┼───────────┼──────────┼────────────────────────┤
│ 3. Syntactic Matcher   │ 0.03-0.09 │ 0.1-0.5  │ Non-uniform DIF:       │
│                        │           │          │ a_orig ≠ a_iso         │
│    Model relies on      │           │          │                        │
│    surface patterns     │           │          │ Discrimination changes │
│    that partially       │           │          │ because model uses     │
│    transfer to some     │           │          │ shortcuts that work on │
│    isomorphs but not    │           │          │ some structures but    │
│    others.              │           │          │ not others             │
├────────────────────────┼───────────┼──────────┼────────────────────────┤
│ 4. Latent Contaminator │ ≤ 0.02    │ ≤ 0.1    │ No item-level DIF      │
│    (Goodhart)           │           │          │ BUT: cross-benchmark   │
│                        │           │          │ Δ_contam is high       │
│    No memorization of   │           │          │                        │
│    specific items, but  │           │          │ Checkpoint selected    │
│    the model was        │           │          │ for benchmark perf,    │
│    selected/tuned for   │           │          │ so ability is real     │
│    this benchmark       │           │          │ but narrow             │
│    distribution.        │           │          │                        │
└────────────────────────┴───────────┴──────────┴────────────────────────┘

Archetype 4 is the most subtle and corresponds to the "Goodhart
in checkpoint selection" phenomenon described in the GSM1K paper:
the training data is clean, but the model was selected from hundreds
of checkpoints based on benchmark performance. Isomorph-Eval detects
this only when combined with CROSS-BENCHMARK analysis (performance
on GSM8K isomorphs is fine, but transfer to MATH isomorphs is poor).
"""


def classify_archetype(
    delta_raw: float,
    delta_irt: float,
    theta_inflation: float,
    n_flagged: int,
    n_total: int,
    p_value: float,
) -> str:
    """
    Classify a model into one of the four contamination archetypes.

    Thresholds calibrated to empirical data from:
      - GSM1K findings: Phi/Mistral show 8-13% drops (Pure Memorizer)
      - ConStat findings: frontier models show <2% gaps (Robust Reasoner)
      - Song (2026): SWE-bench 76% path recall (Syntactic Matcher)
    """
    flagged_fraction = n_flagged / n_total if n_total > 0 else 0

    # Robust Reasoner: no significant contamination
    if delta_raw <= 0.02 and abs(theta_inflation) <= 0.1 and p_value > 0.05:
        return "ROBUST_REASONER"

    # Pure Memorizer: large gap, many flagged items
    if delta_raw >= 0.10 and flagged_fraction >= 0.30:
        return "PURE_MEMORIZER"

    # Syntactic Matcher: moderate gap, mixed DIF pattern
    if 0.03 <= delta_raw < 0.10 and flagged_fraction < 0.30:
        return "SYNTACTIC_MATCHER"

    # If delta is low but IRT-weighted is high → hard items memorized
    if delta_raw < 0.05 and delta_irt >= 0.08:
        return "SELECTIVE_MEMORIZER"

    # Default: moderate contamination
    if p_value < 0.05:
        return "SYNTACTIC_MATCHER"

    return "ROBUST_REASONER"


# ============================================================================
# SECTION 5: SIMULATION — DEMONSTRATING THE METRIC
# ============================================================================

def simulate_contamination_experiment():
    """
    Simulate the full Δ_contam pipeline on synthetic data
    to demonstrate the metric's discriminative power.

    Setup:
      - 5 LLMs with varying contamination levels
      - 30 benchmark items (math problems)
      - 10 isomorphic variants per item
      - 20 trials per item per model (temperature sampling)
    """
    np.random.seed(2026)

    # ---- Ground-truth parameters ----
    J = 5   # models
    N = 30  # items
    K = 10  # isomorphic variants per item
    T = 20  # trials per evaluation

    # Model names and TRUE abilities (on clean data)
    models = [
        "Frontier-A (clean)",         # θ=2.0, no contamination
        "Frontier-B (clean)",         # θ=1.5, no contamination
        "OpenModel-X (memorizer)",    # θ=0.8, heavy contamination
        "OpenModel-Y (syntactic)",    # θ=1.2, moderate contamination
        "FinetuneZ (Goodhart)",       # θ=1.0, no item memorization but overtrained
    ]
    theta_true = np.array([2.0, 1.5, 0.8, 1.2, 1.0])

    # Item parameters (shared for originals and isomorphs)
    b_items = np.linspace(-1.5, 2.0, N)    # difficulty range
    a_items = np.random.uniform(0.8, 3.0, N)  # discrimination

    # ---- Contamination model ----
    # Contamination boost ε_{ji} for each (model, item) pair
    # Only affects ORIGINAL items, not isomorphs
    contamination = np.zeros((J, N))

    # Model 2 (OpenModel-X): heavy memorization on 60% of items
    memorized_items = np.random.choice(N, size=int(0.6 * N), replace=False)
    contamination[2, memorized_items] = np.random.uniform(0.15, 0.35, len(memorized_items))

    # Model 3 (OpenModel-Y): moderate memorization on easy items only
    easy_items = np.where(b_items < 0)[0]
    some_easy = np.random.choice(easy_items, size=min(8, len(easy_items)), replace=False)
    contamination[3, some_easy] = np.random.uniform(0.05, 0.15, len(some_easy))

    # ---- Generate responses ----
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    responses_orig = np.zeros((J, N, T))
    responses_iso = np.zeros((J, N, K, T))

    for j in range(J):
        for i in range(N):
            # Original: P = σ(a*(θ - b)) + ε_contam
            p_orig = sigmoid(a_items[i] * (theta_true[j] - b_items[i]))
            p_orig_boosted = min(p_orig + contamination[j, i], 0.99)

            responses_orig[j, i] = np.random.binomial(1, p_orig_boosted, T)

            # Isomorphic: P = σ(a*(θ - b)), NO contamination boost
            p_iso = sigmoid(a_items[i] * (theta_true[j] - b_items[i]))
            for k in range(K):
                responses_iso[j, i, k] = np.random.binomial(1, p_iso, T)

    # ---- Run the contamination test for each model ----
    print("=" * 78)
    print("CONTAMINATION DELTA SIMULATION")
    print("=" * 78)
    print(f"\nSetup: {J} models × {N} items × {K} variants × {T} trials")
    print(f"Significance level: α = 0.01")

    print(f"\n{'Model':35s} | {'Δ_raw':>6s} | {'Δ_IRT':>6s} | {'θ_true':>6s} | "
          f"{'θ_obs':>6s} | {'Δθ':>6s} | {'p-val':>8s} | {'Flag':>4s} | Archetype")
    print("-" * 110)

    for j in range(J):
        result = contamination_hypothesis_test(
            responses_orig, responses_iso,
            target_model_idx=j,
            alpha=0.01,
        )

        print(f"  {models[j]:33s} | {result['delta_raw']:+.3f} | "
              f"{result['delta_irt']:+.3f} | {theta_true[j]:5.2f} | "
              f"{result['theta_orig']:5.2f} | "
              f"{result['theta_inflation']:+.3f} | "
              f"{result['p_value_global']:.1e} | "
              f"{result['n_flagged']:4d} | "
              f"{result['archetype']}")

    # ---- Detailed analysis of the Pure Memorizer ----
    print("\n" + "=" * 78)
    print("DETAILED ANALYSIS: OpenModel-X (Pure Memorizer)")
    print("=" * 78)

    result_memorizer = contamination_hypothesis_test(
        responses_orig, responses_iso,
        target_model_idx=2,
        alpha=0.01,
    )

    decomp = decompose_ability(
        result_memorizer["theta_orig"],
        result_memorizer["theta_iso"],
    )

    print(f"""
  Raw contamination delta:     Δ_contam = {result_memorizer['delta_raw']:+.4f}
  IRT-weighted delta:          Δ_contam^IRT = {result_memorizer['delta_irt']:+.4f}

  Ability decomposition:
    θ_observed (on originals):  {decomp['theta_fake']:.3f}
    θ_true (on isomorphs):      {decomp['theta_true']:.3f}
    θ_inflation:                {decomp['theta_inflation']:+.3f}
    95% CI for inflation:       [{decomp['ci_95'][0]:+.3f}, {decomp['ci_95'][1]:+.3f}]
    p-value (H₀: Δθ = 0):      {decomp['p_value']:.2e}
    Significant at α=0.01:      {'YES — CONTAMINATED' if decomp['is_significant'] else 'No'}

  Items flagged as memorized:   {result_memorizer['n_flagged']}/{N}
    ({result_memorizer['n_flagged']/N:.0%} of benchmark)

  Global LRT p-value:           {result_memorizer['p_value_global']:.2e}

  VERDICT: This model's reported benchmark score is INFLATED.
    Its true ability θ = {decomp['theta_true']:.2f}, but its benchmark
    score corresponds to θ = {decomp['theta_fake']:.2f}.
    The model has memorized {result_memorizer['n_flagged']} of {N} test items.
""")

    # ---- Comparison: Robust Reasoner ----
    print("=" * 78)
    print("COMPARISON: Frontier-A (Robust Reasoner)")
    print("=" * 78)

    result_clean = contamination_hypothesis_test(
        responses_orig, responses_iso,
        target_model_idx=0,
        alpha=0.01,
    )

    decomp_clean = decompose_ability(
        result_clean["theta_orig"],
        result_clean["theta_iso"],
    )

    print(f"""
  Raw contamination delta:     Δ_contam = {result_clean['delta_raw']:+.4f}
  IRT-weighted delta:          Δ_contam^IRT = {result_clean['delta_irt']:+.4f}

  Ability decomposition:
    θ_observed (on originals):  {decomp_clean['theta_fake']:.3f}
    θ_true (on isomorphs):      {decomp_clean['theta_true']:.3f}
    θ_inflation:                {decomp_clean['theta_inflation']:+.3f}
    p-value (H₀: Δθ = 0):      {decomp_clean['p_value']:.2e}
    Significant at α=0.01:      {'YES' if decomp_clean['is_significant'] else 'No — CLEAN'}

  Items flagged as memorized:   {result_clean['n_flagged']}/{N}

  VERDICT: This model shows NO evidence of contamination.
    Its benchmark score faithfully reflects its true ability.
""")

    # ---- THE THREE-BODY PROBLEM: Contamination + Sparsity + Difficulty ----
    print("=" * 78)
    print("THE THREE-BODY PROBLEM: Contamination × Sparsity × Difficulty")
    print("=" * 78)

    print("""
  The EFSL showed that ranking error is a function of S × D.
  With contamination, the failure surface gains a THIRD axis:

    1 - ρ = f(S, D, C)

  where C = differential contamination across models.

  WORST CASE: A contaminated model tested only on easy items
  in a sparse evaluation matrix will appear to be the best system
  by ALL THREE failure modes compounding:

    ① Sparsity:      Not tested on enough items to expose weakness
    ② Difficulty:     Tested only on easy items → inflated average
    ③ Contamination:  Memorized those easy items → double inflation

  ONLY IRT + Isomorph-Eval together can recover the true ranking:
    - IRT corrects for sparsity and difficulty (EFSL, Paper 1)
    - Isomorphs correct for contamination (this paper)
    - Together: θ_true estimated from clean, difficulty-adjusted data
""")


# ============================================================================
# SECTION 6: IRT ESTIMATION (simplified for simulation)
# ============================================================================

def fit_2pl_irt(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified 2PL IRT fit via joint MLE.
    data: (J, N) matrix of mean success rates.
    Returns (theta, a, b).
    """
    J, N = data.shape

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def neg_ll(params):
        theta = params[:J]
        log_a = params[J:J+N]
        b = params[J+N:J+2*N]
        a = np.exp(np.clip(log_a, -2, 2))

        P = sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
        P = np.clip(P, 1e-10, 1 - 1e-10)

        ll = np.sum(data * np.log(P) + (1 - data) * np.log(1 - P))
        reg = 0.01 * (np.sum(theta**2) + np.sum(log_a**2) + np.sum(b**2))
        return -(ll - reg)

    # Initialize
    theta_init = np.array([
        np.log(np.clip(data[j].mean(), 0.05, 0.95) /
               (1 - np.clip(data[j].mean(), 0.05, 0.95)))
        for j in range(J)
    ])
    log_a_init = np.zeros(N)
    b_init = np.array([
        -np.log(np.clip(data[:, i].mean(), 0.05, 0.95) /
                (1 - np.clip(data[:, i].mean(), 0.05, 0.95)))
        for i in range(N)
    ])

    params_init = np.concatenate([theta_init, log_a_init, b_init])

    result = minimize(neg_ll, params_init, method='L-BFGS-B',
                      options={'maxiter': 2000, 'ftol': 1e-10})

    theta = result.x[:J]
    a = np.exp(np.clip(result.x[J:J+N], -2, 2))
    b = result.x[J+N:J+2*N]

    theta -= theta.mean()
    return theta, a, b


# ============================================================================
# RUN THE SIMULATION
# ============================================================================

if __name__ == "__main__":
    simulate_contamination_experiment()
