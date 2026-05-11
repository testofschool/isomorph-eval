#!/usr/bin/env python3
"""
Isomorph-Eval: Empirical Evaluation Runner
==========================================
Evaluates real LLMs on GSM8K originals vs. isomorphic variants.
Computes Δ_contam and classifies models into archetypes.

QUICKSTART (takes ~15 min per model on free tier):

  # Option 1: Groq (FREE, no credit card, fastest)
  export GROQ_API_KEY=gsk_...  # get from console.groq.com
  python run_eval.py --provider groq --model llama-3.3-70b-versatile

  # Option 2: OpenRouter (FREE, 30+ models)
  export OPENROUTER_API_KEY=sk-or-...  # get from openrouter.ai
  python run_eval.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct:free

  # Option 3: Mistral (FREE, 1B tokens/month)
  export MISTRAL_API_KEY=...  # get from console.mistral.ai
  python run_eval.py --provider mistral --model open-mistral-nemo

  # Option 4: Any OpenAI-compatible endpoint
  python run_eval.py --base-url http://localhost:8000/v1 --model my-model --api-key none

Requirements: pip install openai numpy
"""

import argparse, asyncio, json, math, os, re, sys, time
import numpy as np

try:
    from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError
except ImportError:
    print("Install: pip install openai numpy"); sys.exit(1)

# ── Provider configs ─────────────────────────────────────────────

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "rpm": 25,  # conservative for free tier
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768", "gemma2-9b-it"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "rpm": 15,
        "models": ["meta-llama/llama-3.3-70b-instruct:free",
                    "qwen/qwen3-32b:free",
                    "deepseek/deepseek-r1:free"],
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "rpm": 2,  # very restricted free tier
        "models": ["open-mistral-nemo", "mistral-small-latest"],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "rpm": 50,
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    },
    "custom": {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "rpm": 30,
        "models": [],
    },
}

# ── Prompt & Extraction ─────────────────────────────────────────

SYSTEM = r"""Solve this math problem step by step. Show your work.
Put your FINAL numeric answer inside \boxed{}.
Example: The answer is \boxed{42}"""

BOXED = re.compile(r"\\boxed\{([^}]*)\}")
HASH = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
LAST_NUM = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

def extract_answer(text):
    if not text: return None
    for pattern in [BOXED, HASH]:
        m = pattern.findall(text)
        if m:
            s = m[-1].replace(",","").replace("$","").replace("€","").replace("\\","").replace("{","").replace("}","").rstrip(".")
            try:
                v = float(s)
                return v if not (math.isnan(v) or math.isinf(v)) else None
            except: pass
    # Last number fallback (from tail)
    m = LAST_NUM.findall(text[-200:])
    if m:
        try: return float(m[-1].replace(",",""))
        except: pass
    return None

def is_correct(extracted, expected, tol=0.01):
    if extracted is None: return False
    if expected == 0: return abs(extracted) < tol
    return abs(extracted - expected) / max(abs(expected), 1e-10) < tol

# ── Async evaluation ─────────────────────────────────────────────

async def evaluate_item(client, model, question, semaphore, delay):
    async with semaphore:
        await asyncio.sleep(delay)  # rate limiting
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError:
                await asyncio.sleep(2 ** attempt + 1)
            except (APITimeoutError, APIError):
                await asyncio.sleep(2)
            except Exception as e:
                return ""
        return ""

async def run_evaluation(client, model, items, rpm):
    """Evaluate all items and return results."""
    semaphore = asyncio.Semaphore(min(rpm, 10))
    delay = 60.0 / rpm  # seconds between requests

    results = {}
    total = len(items)
    done = 0
    correct = 0
    t0 = time.time()

    # Process in batches
    batch_size = min(rpm, 20)
    for i in range(0, total, batch_size):
        batch = items[i:i+batch_size]
        tasks = [
            evaluate_item(client, model, item["question"], semaphore, delay * (j % batch_size))
            for j, item in enumerate(batch)
        ]
        responses = await asyncio.gather(*tasks)

        for item, response in zip(batch, responses):
            extracted = extract_answer(response)
            corr = is_correct(extracted, item["answer"])
            results[item["item_id"]] = {
                "extracted": extracted,
                "expected": item["answer"],
                "correct": corr,
                "response_len": len(response),
            }
            done += 1
            if corr: correct += 1

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        acc = correct / done if done > 0 else 0
        print(f"\r  [{done:3d}/{total}] acc={acc:.1%} | {rate:.1f} items/s | ETA {eta:.0f}s", end="", flush=True)

    print()
    return results

# ── Δ_contam computation ────────────────────────────────────────

def compute_delta(results, dataset):
    """Compute contamination delta from evaluation results."""
    originals = []
    per_item = []

    for item in dataset["items"]:
        oid = item["item_id"]
        if oid not in results:
            continue

        p_orig = 1.0 if results[oid]["correct"] else 0.0

        # Variant accuracy
        variant_correct = []
        for v in item["variants"]:
            vid = v["item_id"]
            if vid in results:
                variant_correct.append(1.0 if results[vid]["correct"] else 0.0)

        if not variant_correct:
            continue

        p_iso = np.mean(variant_correct)
        delta_i = p_orig - p_iso

        per_item.append({
            "item_id": oid,
            "p_orig": p_orig,
            "p_iso": p_iso,
            "delta": delta_i,
            "n_variants": len(variant_correct),
        })

        originals.append({
            "p_orig": p_orig,
            "p_iso": p_iso,
        })

    if not originals:
        return None

    acc_orig = np.mean([o["p_orig"] for o in originals])
    acc_iso = np.mean([o["p_iso"] for o in originals])
    delta_raw = acc_orig - acc_iso
    deltas = [p["delta"] for p in per_item]

    # Variance-weighted delta (proxy for IRT discrimination weighting)
    variances = np.array([
        np.var([p["p_orig"]] + [p["p_iso"]])
        for p in per_item
    ])
    weights = variances / (variances.mean() + 1e-10)
    delta_irt = float(np.mean(weights * np.array(deltas)))

    # Count items where original >> isomorphic
    n_flagged = sum(1 for d in deltas if d > 0.5)
    flagged_frac = n_flagged / len(deltas)

    # Archetype
    if abs(delta_raw) <= 0.02 and n_flagged == 0:
        archetype = "ROBUST_REASONER"
    elif delta_raw >= 0.10 and flagged_frac >= 0.30:
        archetype = "PURE_MEMORIZER"
    elif 0.03 <= delta_raw < 0.10:
        archetype = "SYNTACTIC_MATCHER"
    elif delta_raw >= 0.10:
        archetype = "PURE_MEMORIZER"
    else:
        archetype = "ROBUST_REASONER"

    return {
        "n_items": len(originals),
        "acc_original": float(acc_orig),
        "acc_isomorphic": float(acc_iso),
        "delta_raw": float(delta_raw),
        "delta_irt": delta_irt,
        "n_flagged": n_flagged,
        "flagged_fraction": flagged_frac,
        "archetype": archetype,
        "per_item": per_item,
    }

# ── Report ───────────────────────────────────────────────────────

def print_report(model, report):
    arch_emoji = {
        "ROBUST_REASONER": "🟢", "PURE_MEMORIZER": "🔴",
        "SYNTACTIC_MATCHER": "🟡",
    }
    emoji = arch_emoji.get(report["archetype"], "⚪")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     ISOMORPH-EVAL: CONTAMINATION DIAGNOSTIC REPORT      ║
╠══════════════════════════════════════════════════════════╣
║  Model: {model[:48]:48s}  ║
║  Items: {report['n_items']:4d} originals + variants{' '*28}║
╠══════════════════════════════════════════════════════════╣
║  Accuracy (originals):    {report['acc_original']:6.1%}{' '*28}║
║  Accuracy (isomorphs):    {report['acc_isomorphic']:6.1%}{' '*28}║
║  ──────────────────────────────────{' '*22}║
║  Δ_contam (raw):          {report['delta_raw']:+6.3f}{' '*28}║
║  Δ_contam (IRT-weighted): {report['delta_irt']:+6.3f}{' '*28}║
║  Items flagged:           {report['n_flagged']:4d} / {report['n_items']:4d}{' '*27}║
╠══════════════════════════════════════════════════════════╣
║  {emoji} VERDICT: {report['archetype']:20s}{' '*27}║
╚══════════════════════════════════════════════════════════╝""")

# ── Main ─────────────────────────────────────────────────────────

async def async_main(args):
    # Load dataset
    dataset_path = args.dataset or os.path.join(
        os.path.dirname(__file__), "isomorph_gsm8k_eval.json"
    )
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Flatten items
    all_items = []
    for item in dataset["items"]:
        all_items.append({
            "item_id": item["item_id"],
            "question": item["question"],
            "answer": item["answer"],
            "is_original": True,
        })
        for v in item["variants"]:
            all_items.append({
                "item_id": v["item_id"],
                "question": v["question"],
                "answer": v["answer"],
                "is_original": False,
            })

    print(f"Total items to evaluate: {len(all_items)} "
          f"({len(dataset['items'])} originals + variants)")

    # Setup client
    provider = PROVIDERS.get(args.provider, PROVIDERS["custom"])
    base_url = args.base_url or provider["base_url"]
    api_key = args.api_key or os.environ.get(provider["env_key"]) or os.environ.get("OPENAI_API_KEY", "not-needed")
    rpm = args.rpm or provider["rpm"]

    if not api_key or api_key == "not-needed":
        env_var = provider["env_key"]
        print(f"\n  ERROR: No API key found. Set {env_var}:")
        print(f"    export {env_var}=your-key-here")
        if args.provider == "groq":
            print(f"    Get free key: https://console.groq.com")
        elif args.provider == "openrouter":
            print(f"    Get free key: https://openrouter.ai")
        elif args.provider == "mistral":
            print(f"    Get free key: https://console.mistral.ai")
        return

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=60.0,
        max_retries=0,
    )

    print(f"Provider: {args.provider} | Model: {args.model} | RPM: {rpm}")
    print(f"Estimated time: ~{len(all_items) * 60 / rpm / 60:.0f} minutes\n")

    # Run evaluation
    print(f"Evaluating {args.model}...")
    results = await run_evaluation(client, args.model, all_items, rpm)

    # Compute delta
    report = compute_delta(results, dataset)
    if report is None:
        print("ERROR: Could not compute delta (no results)")
        return

    # Print report
    print_report(args.model, report)

    # Save results
    output = {
        "model": args.model,
        "provider": args.provider,
        "report": report,
        "raw_results": {k: {kk: vv for kk, vv in v.items() if kk != "response"}
                        for k, v in results.items()},
    }
    outpath = args.output or f"results_{args.model.replace('/','_')}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {outpath}")

    # Print LaTeX table row
    print(f"\n  LaTeX table row:")
    print(f"  {args.model.split('/')[-1]:30s} & "
          f"{report['acc_original']:.1%} & "
          f"{report['acc_isomorphic']:.1%} & "
          f"{report['delta_raw']:+.3f} & "
          f"{report['delta_irt']:+.3f} & "
          f"{report['n_flagged']}/{report['n_items']} & "
          f"{report['archetype']} \\\\")

    await client.close()


def main():
    p = argparse.ArgumentParser(
        description="Isomorph-Eval: Empirical LLM Evaluation",
        epilog="""
Examples:
  # Groq (FREE, no credit card, ~10 min)
  export GROQ_API_KEY=gsk_...
  python run_eval.py --provider groq --model llama-3.3-70b-versatile

  # OpenRouter (FREE, many models)
  export OPENROUTER_API_KEY=sk-or-...
  python run_eval.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct:free

  # Multiple models (run sequentially)
  for model in llama-3.3-70b-versatile llama-3.1-8b-instant mixtral-8x7b-32768; do
    python run_eval.py --provider groq --model $model
  done
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--provider", default="groq", choices=list(PROVIDERS.keys()),
                   help="API provider (default: groq)")
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument("--dataset", default=None, help="Path to eval dataset JSON")
    p.add_argument("--base-url", default=None, help="Custom API base URL")
    p.add_argument("--api-key", default=None, help="API key")
    p.add_argument("--rpm", type=int, default=None, help="Requests per minute")
    p.add_argument("--output", default=None, help="Output JSON path")
    args = p.parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
