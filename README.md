<div align="center">

# Isomorph-Eval

### Structurally Equivalent Benchmarks Reveal That<br>Annotation-Chain Replay Produces False Contamination Signals

**A framework for generating verified benchmark variants via tau-isomorphism**

[![arXiv](https://img.shields.io/badge/arXiv-2606.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2606.XXXXX)
[![Paper 1: EFSL](https://img.shields.io/badge/Paper_1-EFSL_(Kang_2026)-blue.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg)](https://www.python.org/)
[![No GPU Required](https://img.shields.io/badge/GPU-not_required-brightgreen.svg)]()

[Paper](#the-science) •
[The Cautionary Finding](#the-cautionary-finding) •
[How It Works](#how-it-works) •
[Results](#results) •
[Quickstart](#quickstart) •
[Citation](#citation)

</div>

---

## The Cautionary Finding

We built a framework to detect benchmark contamination. Instead, we discovered that **the standard method for computing variant answers produces systematically wrong answers in 41% of items** — creating false contamination signals that are indistinguishable from genuine memorization.

| | Before Verification | After Verification |
|---|---|---|
| **Apparent signal** | All 5 models show Delta ~ +0.42 | 3/5 invariant, 2/5 mild gap (Delta 0.04-0.11) |
| **Interpretation** | "Universal contamination" | Perturbation sensitivity, not contamination |
| **Negative control** | GPT-OSS 120B also "contaminated" | GPT-OSS 120B invariant (p = 0.174) |
| **Root cause** | 41 of 100 items had wrong variant answers | Verified answers via forward graph execution |

The lesson: **any benchmark variant generation pipeline must independently verify its answers**. Annotation-chain replay — substituting new values into the original solution chain — is unreliable.

### Seven Bug Classes

We identified seven distinct failure modes in annotation-chain replay:

1. **Value conflation** — Same numeric value serves multiple semantic roles; mutation changes both uses in the graph but only one in the text
2. **Phantom leaf values** — Solution chain introduces intermediate results as leaf nodes not present in the question
3. **Positional replacement collisions** — Sequential text replacement causes new values to shadow old values of other nodes
4. **Magnitude inflation** — Same-digit-class sampling produces systematically larger variant answers
5. **Answer invariance** — Some graph structures produce the same answer regardless of leaf mutations
6. **Partial name replacement** — Word-boundary-unaware replacement causes "Beth" to match inside "Bethany"
7. **Node reuse** — Safety fallback promotes values appearing multiple times, reintroducing conflation

---

## The Science

### tau-Isomorphism

We formalize structural equivalence as **tau-isomorphism**: two benchmark items are tau-isomorphic if there exists a graph isomorphism between their reasoning graphs that preserves operation types, dependency structure, and complexity weights. Unlike semantic similarity heuristics, tau-isomorphism is a deterministic, verifiable structural condition.

### The Isomorphic Engine

A four-stage pipeline that produces verified tau-isomorphic variants:

```
Parser  -->  Mutator  -->  Generator  -->  Verifier
S -> G,V     G -> G'       G' -> S'        S' -> G'' ≅ G'
```

**Parser**: Decomposes a benchmark item into its reasoning graph and verification function.
**Mutator**: Swaps entities and values while locking graph topology and operation types. Forward-executes the graph to compute answers automatically.
**Generator**: Produces fluent text from the mutated graph.
**Verifier**: Re-parses output and compares structural fingerprints. Rejects corrupted items.

### Connection to EFSL

In prior work ([EFSL](https://arxiv.org/abs/XXXX.XXXXX)), we showed that evaluation accuracy degrades as a function of data sparsity (S) and item difficulty heterogeneity (D). Contamination (C) represents a third axis: `1 - rho = f(S, D, C)`. IRT on verified isomorphic data addresses all three axes simultaneously.

---

## The Metric: Delta_iso

```
Delta_iso^IRT(M, B) = (1/N) Sum_i a_i * [P(X_i=1) - P(X_i'=1)]
```

Weighted by IRT item discrimination `a_i`. Grounded in the psychometric Differential Item Functioning (DIF) framework. A positive delta may indicate memorization *or* perturbation sensitivity — the negative control disambiguates.

### Diagnostic Archetypes

| Archetype | Delta_iso | Interpretation |
|-----------|-----------|----------------|
| **Invariant** | Not significant (p > 0.05) | Robust reasoning transfer to novel variants |
| **Mild Gap** | Significant, Delta < 0.15 | Perturbation sensitivity or mild memorization |
| **Partial Memorizer** | Significant, 0.15-0.30 | Mix of reasoning and recall (theoretical) |
| **Pure Memorizer** | Significant, Delta > 0.30 | Performance collapses on isomorphs (theoretical) |

---

## Results

### Verified Results (N=69)

| Model | N | Acc_orig | Acc_iso | Delta_iso [95% CI] | p | Archetype |
|-------|---|----------|---------|-------------------|---|-----------|
| Llama 3.1 8B | 69 | 95.7% | 84.4% | +0.113 [+0.049, +0.180] | 0.001** | Mild |
| Llama 4 Scout 17B | 69 | 94.2% | 90.1% | +0.041 [-0.011, +0.099] | 0.169 | Inv |
| Qwen3 32B | 45 | 100.0% | 94.0% | +0.060 [+0.009, +0.120] | 0.044* | Mild |
| GPT-OSS 120B* | 13 | 100.0% | 95.2% | +0.048 [+0.000, +0.123] | 0.174 | Inv |
| Llama 3.3 70B | 9 | 100.0% | 95.2% | +0.048 [+0.000, +0.133] | 0.320 | Inv |

*GPT-OSS 120B serves as a negative control (no GSM8K in training data).

Three of five models show no significant performance gap between originals and variants. Llama 3.1 8B and Qwen3 32B show mild gaps consistent with known numeric perturbation sensitivity (GSM-Symbolic reports up to 15pp baseline).

![Pre/Post Verification Comparison](figures/fig1_pre_post_comparison.png)
*Figure 1: The effect of answer verification. Left: all models appear uniformly contaminated (Delta ~ +0.42). Right: after verification, the signal collapses.*

![Delta Distribution](figures/fig2_delta_distribution.png)
*Figure 2: Per-item delta distributions. Left: bimodal (broken items inflate mean). Right: concentrated near zero on verified data.*

---

## Quickstart

### Installation

```bash
pip install numpy scipy pydantic openai matplotlib
git clone https://github.com/testofschool/isomorph-eval.git
cd isomorph-eval
```

### Run the Evaluation

```bash
# Against any OpenAI-compatible endpoint
python api_runner.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --base-url https://api.together.xyz/v1 \
  --api-key $TOGETHER_API_KEY \
  --dataset data/eval_verified_v2.json \
  --trials 5 --concurrency 20 --output results.json

# Against local vLLM
python api_runner.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --dataset data/eval_verified_v2.json
```

### Generate Figures

```bash
python plot_results.py --output figures/
```

---

## Repository Structure

```
isomorph-eval/
├── api_runner.py              # Async evaluation runner
├── plot_results.py            # Publication figure generator
├── generate_eval_dataset_v2.py # Variant generation pipeline
├── rescore_v2.py              # Result rescoring on verified data
├── core/
│   ├── data_structures.py     # ReasoningGraph, tau-isomorphism types
│   ├── gsm8k_parser.py        # GSM8K annotation parser
│   └── pipeline.py            # Mutator, Generator, Verifier
├── data/
│   ├── eval_verified_v2.json  # Verified dataset (69 items)
│   └── eval_subset_100.json   # Full 100-item subset (pre-verification)
├── paper/
│   ├── main.tex               # Paper source
│   ├── paper_outline_and_intro.tex
│   ├── section4_methodology.tex
│   └── section5_6_empirical.tex
├── figures/
│   ├── fig1_pre_post_comparison.pdf
│   ├── fig2_delta_distribution.pdf
│   └── fig3_three_body_surface.pdf
└── results/
    └── rescored_clean.json    # Verified results
```

## Comparison with Prior Work

| Feature | GSM1K | ConStat | GSM-Symbolic | **Isomorph-Eval** |
|---------|-------|---------|--------------|-------------------|
| Structural isomorphism | No | No | No | tau-isomorphism |
| Verified variant answers | No | No | No | Forward graph execution |
| IRT difficulty correction | No | No | No | 2PL DIF |
| Scalable generation | Fixed 1,250 | No | Templates | Arbitrary scale |
| Bug taxonomy | No | No | No | 7 classes documented |
| Negative control | No | No | No | GPT-OSS 120B |
| Black-box compatible | Yes | Yes | Yes | Yes |

---

## Contributions

1. **tau-isomorphism**: A verifiable graph-theoretic condition for benchmark item equivalence
2. **Isomorphic Engine**: Four-stage pipeline producing unlimited verified variants with automatically computed answers
3. **Seven bug classes**: Taxonomy of annotation-chain replay failures that produce false contamination signals
4. **Empirical evidence**: On verified data, current LLMs show robust reasoning transfer to novel numeric contexts
5. **Delta_iso metric**: DIF-based perturbation sensitivity metric with IRT discrimination weighting

---

## Citation

```bibtex
@article{kang2026isomorpheval,
  title={Isomorph-Eval: Structurally Equivalent Benchmarks Reveal
         That Annotation-Chain Replay Produces False Contamination
         Signals},
  author={Kang, Jung Min},
  journal={arXiv preprint arXiv:2606.XXXXX},
  year={2026}
}

@article{kang2026efsl,
  title={The Scaling Law of Evaluation Failure: Why Simple Averaging
         Collapses Under Data Sparsity and Item Difficulty Gaps,
         and How Item Response Theory Recovers Ground Truth
         Across Domains},
  author={Kang, Jung Min},
  journal={arXiv preprint arXiv:2505.XXXXX},
  year={2026}
}
```

## License

MIT
