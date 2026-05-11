<div align="center">

# 🔬 Isomorph-Eval

### Is your model a reasoner — or a memorizer?

**The first unified framework for contamination-proof AI evaluation**<br>
**combining structurally equivalent benchmarks with Item Response Theory**

[![arXiv](https://img.shields.io/badge/arXiv-2606.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2606.XXXXX)
[![Paper 1: EFSL](https://img.shields.io/badge/Paper_1-EFSL_(Kang_2026)-blue.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg)](https://www.python.org/)
[![No GPU Required](https://img.shields.io/badge/GPU-not_required-brightgreen.svg)]()

[Paper](#the-science) •
[Quickstart](#quickstart) •
[How It Works](#how-it-works) •
[Results](#results) •
[Contributing](#contributing) •
[Citation](#citation)

---

**Your model scores 92% on GSM8K. But can it solve this?**

> *Tomás keeps a busy apiary. His bees produce 23 jars of honey every day. Each morning, he sets aside 5 jars for breakfast recipes, and every afternoon he processes another 7 jars into beeswax candles for the village cooperative. He takes whatever jars remain to the harbor market, where each one sells for €3. How many euros does Tomás earn at the harbor market each day?*

Same reasoning graph. Same difficulty. Same IRT discrimination parameter.<br>
Completely novel surface. **Never existed on the internet.**

If performance drops — that 92% was memory, not math.

</div>

---

## The Problem: Three Simultaneous Crises

AI benchmark scores are broken in three ways at once, and nobody has addressed the combination.

| Crisis | Evidence | Prior Fix | Gap |
|--------|----------|-----------|-----|
| **Contamination** | Phi/Mistral drop 8-13% on GSM1K vs GSM8K (Zhang et al., NeurIPS 2024) | GSM1K, ConStat | No IRT, no structural guarantees |
| **Aggregation failure** | Simple averaging collapses to ρ=0.24 under sparsity × difficulty gaps (Kang, 2026) | IRT (PSN-IRT, MEDIRT) | No contamination correction |
| **Compound interaction** | A contaminated model tested on easy items in a sparse matrix gets *triple-inflated* scores | **Nobody** | **Isomorph-Eval** |

We call this the **Three-Body Problem of AI Evaluation**: sparsity (S) × difficulty gap (D) × contamination (C). Each axis distorts rankings independently. Together, they compound.

## The Science: Two Papers, One Unified Theory

### Paper 1: The Evaluation Failure Scaling Law (EFSL)

*When does simple averaging fail?*

We proved through a 150-condition grid sweep that ranking accuracy degrades monotonically as S × D increases. The S × D interaction is strong and statistically significant (γ₃ = +0.199, t = 13.05). 2PL IRT maintains ρ ≥ 0.993 across all conditions.

**EFSL fixes HOW we aggregate.** → [Paper](https://arxiv.org/abs/XXXX.XXXXX) • [Code](https://github.com/testofschool/evaluation-failure-scaling-law)

### Paper 2: Isomorph-Eval (this repo)

*How do we evaluate when benchmarks are contaminated?*

We formalize **τ-isomorphism** — a graph-theoretic definition of structural equivalence — and build an engine that generates unlimited verified variants of any benchmark item. Then we measure the gap.

**Isomorph-Eval fixes WHAT we test.** → [Paper](https://arxiv.org/abs/2606.XXXXX)

### Together: The Complete Solution

```
1 − ρ = f(S, D, C)

IRT corrects S × D → Paper 1
Isomorphs correct C → Paper 2
IRT on isomorphic data corrects all three → This framework
```

---

## How It Works

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Parser  │────▶│ Mutator  │────▶│Generator │────▶│ Verifier │
│  S → G,V │     │ G → G'   │     │ G' → S'  │     │ S'→ G''≅G'│
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                        │
                                              ┌─────────┴──────────┐
                                              │  Round-Trip         │
                                              │  Verification       │
                                              │  (the safety net)   │
                                              └────────────────────┘
```

**Stage 1 — Parser**: Decomposes a benchmark item into its reasoning graph (operation DAG + complexity weights) and verification function. Rule-based extraction covers 57% of GSM8K; LLM structured extraction handles the rest.

**Stage 2 — Mutator**: Swaps entities, numbers, units, and context theme while **locking** the graph topology, operation types, and digit classes. Forward-executes the new graph to compute the correct answer automatically.

**Stage 3 — Generator**: Produces fluent text from the mutated graph. The LLM is a *surface realizer*, never a reasoner. Template-first architecture ensures mathematical content is preserved.

**Stage 4 — Verifier**: Re-parses the output and compares structural fingerprints. If the LLM corrupted the logic, the item is rejected. Seven static checks + round-trip verification.

---

## The Metric: Δ<sub>contam</sub><sup>IRT</sup>

```
Δ_contam^IRT(M, B) = (1/N) Σᵢ aᵢ · [P(Xᵢ=1) − P(Xᵢ'=1)]
```

Weighted by IRT item discrimination `aᵢ` estimated from clean isomorphic data. Grounded in the psychometric Differential Item Functioning (DIF) framework — the gold-standard test for measurement fairness, now applied to LLMs.

### Diagnostic Archetypes

| Archetype | Δ<sub>contam</sub> | θ inflation | What it means |
|-----------|----------|-------------|---------------|
| 🟢 **Robust Reasoner** | ≤ 2% | ≤ 0.1 | Generalizes to novel variants |
| 🟡 **Syntactic Matcher** | 3–9% | 0.1–0.5 | Relies on surface patterns |
| 🔴 **Pure Memorizer** | ≥ 10% | ≥ 0.5 | Performance collapses on isomorphs |
| ⚫ **Latent Contaminator** | ≤ 2%* | ≤ 0.1* | Goodharted checkpoint selection |

---

## Results

![Diagnostic Chart](figures/fig1_diagnostic_chart.png)
*Figure 1: The IRT-weighted delta amplifies contamination signals — 23× more discriminative than raw accuracy gaps on hard, diagnostic items.*

![Three-Body Surface](figures/fig2_three_body_surface.png)
*Figure 2: The Three-Body Problem of AI Evaluation. (A) Simple averaging under biased missingness. (B) MCAR control. (C) Contamination as the third axis. (D) Only IRT + Isomorphs corrects all three simultaneously.*

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
# Against Together AI (or any OpenAI-compatible endpoint)
python api_runner.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --base-url https://api.together.xyz/v1 \
  --api-key $TOGETHER_API_KEY \
  --dataset isomorph_gsm8k.json \
  --trials 5 --concurrency 20 --output results.json

# Against local vLLM
python api_runner.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --dataset isomorph_gsm8k.json

# Quick demo with synthetic data (no API key needed for pipeline test)
python api_runner.py --model gpt-4o-mini --dataset demo --trials 2
```

### Generate Figures

```bash
python plot_results.py --results results.json --output figures/
```

### Parse GSM8K and Generate Isomorphic Variants

```python
from core.gsm8k_parser import GSM8KParser
from core.pipeline import IsomorphicEngine, SemanticMutator

# Parse a problem
parser = GSM8KParser()
graph, verification = parser.parse(
    question="Janet's ducks lay 16 eggs per day...",
    solution="She eats 3 + 4 = <<3+4=7>>7 eggs...\n#### 18",
    answer=18,
)

# Generate 10 isomorphic variants
engine = IsomorphicEngine(parser=parser)
original_item = IsomorphicItem(
    item_id="gsm8k_0000",
    surface_text="Janet's ducks...",
    reasoning_graph=graph,
    verification=verification,
)
variants = engine.generate_variants(original_item, n_variants=10)
```

---

## Repository Structure

```
isomorph-eval/
├── api_runner.py              # Async evaluation runner
├── plot_results.py            # Publication figure generator
├── core/
│   ├── data_structures.py     # ReasoningGraph, τ-isomorphism types
│   ├── gsm8k_parser.py        # Production GSM8K parser
│   └── pipeline.py            # Mutator, Generator, Verifier
├── contamination_delta.py     # Δ_contam metric + DIF test
├── paper/
│   ├── paper_outline_and_intro.tex
│   └── section4_methodology.tex
├── figures/
│   ├── fig1_diagnostic_chart.pdf
│   └── fig2_three_body_surface.pdf
└── validate_gsm8k.py          # Parser validation on full GSM8K
```

## Comparison with Prior Work

| Feature | GSM1K | ConStat | LiveBench | PSN-IRT | **Isomorph-Eval** |
|---------|-------|---------|-----------|---------|-------------------|
| Structural isomorphism | ❌ | ❌ | ❌ | ❌ | ✅ τ-isomorphism |
| Infinite generation | ❌ Fixed 1,250 | ❌ | ❌ Monthly | ❌ | ✅ Unlimited |
| IRT difficulty correction | ❌ | ❌ Regression | ❌ | ✅ | ✅ 2PL DIF |
| Contamination detection | ✅ Gap | ✅ p-value | ❌ Prevention | ❌ | ✅ Δ<sub>contam</sub><sup>IRT</sup> |
| Sparsity correction | ❌ | ❌ | ❌ | Partial | ✅ via EFSL |
| Round-trip verification | ❌ | ❌ | ❌ | ❌ | ✅ S' → G'' ≅ G' |
| Black-box compatible | ✅ | ✅ | ✅ | ✅ | ✅ |
| Per-item significance test | ❌ | ✅ | ❌ | ❌ | ✅ Wald + Bonferroni |
| Backward compatible | ❌ New benchmark | ❌ | ❌ New benchmark | ❌ | ✅ Existing benchmarks |

---

## Supported Domains

| Domain | Parser | Status |
|--------|--------|--------|
| Mathematical Reasoning (GSM8K) | `GSM8KParser` | ✅ Released |
| Logic (LogiQA, ReClor) | `LogicParser` | 🚧 In progress |
| Code Generation (HumanEval) | `CodeParser` | 🚧 In progress |
| Cybersecurity (ATT&CK scenarios) | `CyberParser` | 📋 Planned |
| Medical (USMLE-style) | `MedicalParser` | 📋 Planned |

---

## Citation

```bibtex
@article{kang2026isomorpheval,
  title={Isomorph-Eval: Separating Reasoning from Recall in {LLM}
         Evaluation via Structurally Equivalent Benchmarks and
         Item Response Theory},
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

MIT. Use it. Break your benchmarks. Find the truth.
