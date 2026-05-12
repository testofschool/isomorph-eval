#!/usr/bin/env python3
"""
Isomorph-Eval v2: Optimized evaluation runner for Groq free tier.
Evaluates one model at a time on a stratified subset of the v2 dataset.

Usage:
  python3 run_eval_v2.py --model llama-3.3-70b-versatile
  python3 run_eval_v2.py --model llama-3.1-8b-instant
  python3 run_eval_v2.py --model meta-llama/llama-4-scout-17b-16e-instruct
  python3 run_eval_v2.py --model openai/gpt-oss-120b
"""
import argparse, asyncio, json, math, os, re, sys, time, random
import numpy as np
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError

API_KEY = os.environ.get("GROQ_API_KEY", "")
BASE_URL = "https://api.groq.com/openai/v1"

SYSTEM = r"""Solve this math problem step by step. Show your work.
Put your FINAL numeric answer inside \boxed{}.
Example: The answer is \boxed{42}"""

BOXED = re.compile(r"\\boxed\{([^}]*)\}")
HASH_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
LAST_NUM = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

def extract_answer(text):
    if not text: return None
    for pattern in [BOXED, HASH_RE]:
        m = pattern.findall(text)
        if m:
            s = m[-1].replace(",","").replace("$","").replace("€","").replace("\\","").replace("{","").replace("}","").rstrip(".")
            try:
                v = float(s)
                return v if not (math.isnan(v) or math.isinf(v)) else None
            except: pass
    m = LAST_NUM.findall(text[-200:])
    if m:
        try: return float(m[-1].replace(",",""))
        except: pass
    return None

def is_correct(extracted, expected, tol=0.01):
    if extracted is None: return False
    if expected == 0: return abs(extracted) < tol
    return abs(extracted - expected) / max(abs(expected), 1e-10) < tol

async def evaluate_item(client, model, question, semaphore, temperature=0.0):
    async with semaphore:
        await asyncio.sleep(2.5)  # ~24 RPM
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": question},
                    ],
                    temperature=temperature,
                    max_tokens=768,
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                wait = min(60, 3 ** (attempt + 1))
                err_msg = str(e)
                if "tokens per day" in err_msg:
                    print(f"\n  *** DAILY TOKEN LIMIT HIT ***")
                    return "__TPD_LIMIT__"
                print(f"\n  Rate limited, waiting {wait}s...", end="", flush=True)
                await asyncio.sleep(wait)
            except (APITimeoutError, APIError) as e:
                await asyncio.sleep(3)
            except Exception as e:
                print(f"\n  Error: {type(e).__name__}: {e}")
                return ""
        return ""

async def run_model(model, dataset_path, n_items=50, temperature=0.0, trials=1):
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Stratified sample: spread across complexity levels
    items = dataset["items"]
    items_sorted = sorted(items, key=lambda x: x.get("n_nodes", 0))
    step = max(1, len(items_sorted) // n_items)
    selected = items_sorted[::step][:n_items]
    random.seed(2026)
    random.shuffle(selected)

    # Flatten
    all_items = []
    for item in selected:
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

    n_orig = sum(1 for x in all_items if x["is_original"])
    n_var = sum(1 for x in all_items if not x["is_original"])
    total = len(all_items) * trials
    
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Items: {n_orig} originals + {n_var} variants = {len(all_items)}")
    print(f"Trials: {trials}, Temperature: {temperature}")
    print(f"Total API calls: {total}")
    print(f"Estimated tokens: ~{total * 400:,}")
    print(f"Estimated time: ~{total / 24:.0f} minutes")
    print(f"{'='*60}\n")

    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=60.0,
        max_retries=0,
    )

    semaphore = asyncio.Semaphore(5)
    results = {}
    done = 0
    correct = 0
    t0 = time.time()
    tpd_hit = False

    for trial in range(trials):
        if tpd_hit:
            break
        trial_label = f" [trial {trial+1}/{trials}]" if trials > 1 else ""
        
        for i in range(0, len(all_items), 6):
            batch = all_items[i:i+6]
            tasks = [evaluate_item(client, model, item["question"], semaphore, temperature) for item in batch]
            responses = await asyncio.gather(*tasks)

            for item, response in zip(batch, responses):
                if response == "__TPD_LIMIT__":
                    tpd_hit = True
                    break
                    
                extracted = extract_answer(response)
                corr = is_correct(extracted, item["answer"])
                
                key = f"{item['item_id']}_t{trial}"
                results[key] = {
                    "item_id": item["item_id"],
                    "trial": trial,
                    "extracted": extracted,
                    "expected": item["answer"],
                    "correct": corr,
                    "response_len": len(response),
                    "is_original": item["is_original"],
                }
                done += 1
                if corr: correct += 1

            if tpd_hit:
                break

            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            acc = correct / done if done > 0 else 0
            remaining = total - done
            eta = remaining / rate if rate > 0 else 0
            print(f"\r  {trial_label} [{done:4d}/{total}] acc={acc:.1%} | {rate:.1f}/s | ETA {eta:.0f}s", end="", flush=True)
    
    print(f"\n  Done! {done} items evaluated in {time.time()-t0:.0f}s")
    
    if tpd_hit:
        print(f"  ⚠ Daily token limit hit after {done} items")

    # Compute contamination delta
    report = compute_delta_v2(results, selected)
    if report:
        print_report(model, report)
    
    # Save
    safe_name = model.replace("/", "_")
    output = {
        "model": model,
        "dataset": "isomorph_gsm8k_eval_v2",
        "n_selected": len(selected),
        "temperature": temperature,
        "trials": trials,
        "tpd_limit_hit": tpd_hit,
        "report": report,
        "raw_results": results,
    }
    outpath = os.path.join(os.path.dirname(dataset_path), f"results_v2_{safe_name}.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {outpath}")
    
    await client.close()
    return report

def compute_delta_v2(results, selected_items):
    """Compute contamination delta from results."""
    per_item = []
    
    for item in selected_items:
        oid = item["item_id"]
        
        # Collect original results across trials
        orig_scores = []
        for key, r in results.items():
            if r["item_id"] == oid and r["is_original"]:
                orig_scores.append(1.0 if r["correct"] else 0.0)
        
        if not orig_scores:
            continue
        
        p_orig = np.mean(orig_scores)
        
        # Collect variant results
        variant_scores = []
        for v in item["variants"]:
            vid = v["item_id"]
            for key, r in results.items():
                if r["item_id"] == vid:
                    variant_scores.append(1.0 if r["correct"] else 0.0)
        
        if not variant_scores:
            continue
        
        p_iso = np.mean(variant_scores)
        delta_i = p_orig - p_iso
        
        per_item.append({
            "item_id": oid,
            "p_orig": p_orig,
            "p_iso": p_iso,
            "delta": delta_i,
            "n_variants_scored": len(variant_scores),
        })
    
    if not per_item:
        return None
    
    acc_orig = np.mean([p["p_orig"] for p in per_item])
    acc_iso = np.mean([p["p_iso"] for p in per_item])
    delta_raw = float(acc_orig - acc_iso)
    deltas = np.array([p["delta"] for p in per_item])
    
    # IRT-weighted delta using variance as discrimination proxy
    variances = np.array([
        np.var([p["p_orig"], p["p_iso"]])
        for p in per_item
    ])
    weights = variances / (variances.mean() + 1e-10)
    delta_irt = float(np.mean(weights * deltas))
    
    # Flagged items (original correct, most/all variants wrong)
    n_flagged = sum(1 for p in per_item if p["delta"] > 0.5)
    
    # SE and CI for delta_raw (bootstrap)
    n = len(deltas)
    se_delta = float(np.std(deltas) / np.sqrt(n))
    ci_low = float(delta_raw - 1.96 * se_delta)
    ci_high = float(delta_raw + 1.96 * se_delta)
    
    # Wilcoxon signed-rank test
    from scipy import stats
    try:
        stat, p_value = stats.wilcoxon(deltas, alternative='greater')
        p_value = float(p_value)
    except Exception:
        p_value = None
    
    # Archetype
    if abs(delta_raw) <= 0.02:
        archetype = "ROBUST_REASONER"
    elif delta_raw >= 0.10:
        archetype = "PURE_MEMORIZER"
    elif 0.02 < delta_raw < 0.10:
        archetype = "SYNTACTIC_MATCHER"
    else:
        archetype = "ROBUST_REASONER"
    
    return {
        "n_items": len(per_item),
        "acc_original": float(acc_orig),
        "acc_isomorphic": float(acc_iso),
        "delta_raw": delta_raw,
        "se_delta": se_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_value,
        "delta_irt": delta_irt,
        "n_flagged": n_flagged,
        "archetype": archetype,
        "per_item": per_item,
    }

def print_report(model, r):
    p_str = f"p={r['p_value']:.2e}" if r['p_value'] is not None else "p=N/A"
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║     ISOMORPH-EVAL v2: CONTAMINATION DIAGNOSTIC            ║
╠═══════════════════════════════════════════════════════════╣
║  Model: {model[:50]:50s} ║
║  Items: {r['n_items']:4d}                                             ║
╠═══════════════════════════════════════════════════════════╣
║  Acc (originals):    {r['acc_original']:6.1%}                              ║
║  Acc (isomorphs):    {r['acc_isomorphic']:6.1%}                              ║
║  ─────────────────────────────────────                    ║
║  Δ_raw:              {r['delta_raw']:+.3f}  SE={r['se_delta']:.3f}                   ║
║  95% CI:             [{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]                   ║
║  {p_str:48s}   ║
║  Δ_IRT:              {r['delta_irt']:+.3f}                                ║
║  Flagged:            {r['n_flagged']:4d} / {r['n_items']:4d}                            ║
╠═══════════════════════════════════════════════════════════╣
║  VERDICT: {r['archetype']:48s}║
╚═══════════════════════════════════════════════════════════╝""")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--dataset", default=None)
    args = p.parse_args()
    
    dataset_path = args.dataset or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "isomorph_gsm8k_eval_v2.json"
    )
    asyncio.run(run_model(args.model, dataset_path, args.n_items, args.temperature, args.trials))
