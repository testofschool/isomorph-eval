#!/usr/bin/env python3
"""
Isomorph-Eval v3: Production evaluation runner with resume support.

Features:
  - Multi-trial evaluation (T=1 or T=3) with configurable temperature
  - Resume from partial results (saves after each batch)
  - Uses pre-built dataset subsets (no internal stratification)
  - Proper per-trial result storage for bootstrap CIs
  - Wilcoxon signed-rank test and clustered bootstrap CIs

Usage:
  python3 run_eval_v3.py --model llama-3.1-8b-instant --dataset data/eval_subset_100.json
  python3 run_eval_v3.py --model llama-3.1-8b-instant --dataset data/eval_subset_100.json --trials 3 --temperature 0.7
  python3 run_eval_v3.py --model llama-3.1-8b-instant --dataset data/eval_subset_100.json --resume  # resume partial run
"""
import argparse, asyncio, json, math, os, re, sys, time, random
import numpy as np
from scipy import stats
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
        await asyncio.sleep(2.5)
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
                if "tokens per day" in str(e):
                    return "__TPD_LIMIT__"
                wait = min(60, 3 ** (attempt + 1))
                print(f"\n  Rate limited, waiting {wait}s...", end="", flush=True)
                await asyncio.sleep(wait)
            except (APITimeoutError, APIError):
                await asyncio.sleep(3)
            except Exception as e:
                print(f"\n  Error: {type(e).__name__}: {e}")
                return ""
        return ""


async def run_evaluation(model, dataset_path, output_path, trials=1,
                         temperature=0.0, resume=False):
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Flatten all items
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

    n_orig = sum(1 for x in all_items if x["is_original"])
    n_var = sum(1 for x in all_items if not x["is_original"])

    # Load existing results for resume
    existing_results = {}
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        existing_results = prev.get("raw_results", {})
        print(f"  Resuming: {len(existing_results)} existing results loaded")

    # Build evaluation schedule: (item, trial_idx) pairs
    schedule = []
    for trial in range(trials):
        for item in all_items:
            key = f"{item['item_id']}_t{trial}"
            if key not in existing_results:
                schedule.append((item, trial, key))

    total = len(schedule)
    if total == 0:
        print("  All items already evaluated!")
        results = existing_results
    else:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"Items: {n_orig} originals + {n_var} variants = {len(all_items)}")
        print(f"Trials: {trials}, Temperature: {temperature}")
        print(f"Already done: {len(existing_results)}, Remaining: {total}")
        print(f"Estimated time: ~{total / 20:.0f} minutes")
        print(f"{'='*60}\n")

        client = AsyncOpenAI(
            base_url=BASE_URL, api_key=API_KEY,
            timeout=60.0, max_retries=0,
        )
        semaphore = asyncio.Semaphore(5)
        results = dict(existing_results)
        done = 0
        correct = 0
        t0 = time.time()
        tpd_hit = False

        for i in range(0, total, 6):
            if tpd_hit:
                break
            batch = schedule[i:i+6]
            tasks = [
                evaluate_item(client, model, item["question"], semaphore, temperature)
                for item, trial, key in batch
            ]
            responses = await asyncio.gather(*tasks)

            for (item, trial, key), response in zip(batch, responses):
                if response == "__TPD_LIMIT__":
                    tpd_hit = True
                    break
                extracted = extract_answer(response)
                corr = is_correct(extracted, item["answer"])
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
            print(f"\r  [{done:4d}/{total}] acc={acc:.1%} | {rate:.1f}/s | ETA {eta:.0f}s",
                  end="", flush=True)

            # Save partial results every 60 items
            if done % 60 == 0:
                _save_partial(output_path, model, dataset, results, trials,
                              temperature, dataset_path)

        print(f"\n  Done! {done} new items evaluated")
        if tpd_hit:
            print(f"  WARNING: Daily token limit hit after {done} items. Use --resume to continue tomorrow.")
        await client.close()

    # Compute report
    report = compute_report(results, dataset)
    if report:
        print_report(model, report)

    # Save final
    _save_partial(output_path, model, dataset, results, trials,
                  temperature, dataset_path)
    print(f"\n  Saved: {output_path}")
    return report


def _save_partial(output_path, model, dataset, results, trials, temperature, dataset_path):
    report = compute_report(results, dataset)
    output = {
        "model": model,
        "dataset": os.path.basename(dataset_path),
        "temperature": temperature,
        "trials": trials,
        "n_results": len(results),
        "report": report,
        "raw_results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def compute_report(results, dataset):
    per_item = []
    for item in dataset["items"]:
        oid = item["item_id"]
        orig_scores = [
            1.0 if r["correct"] else 0.0
            for r in results.values()
            if r["item_id"] == oid and r["is_original"]
        ]
        if not orig_scores:
            continue

        p_orig = np.mean(orig_scores)
        variant_scores = []
        for v in item["variants"]:
            vid = v["item_id"]
            for r in results.values():
                if r["item_id"] == vid:
                    variant_scores.append(1.0 if r["correct"] else 0.0)

        if not variant_scores:
            continue
        p_iso = np.mean(variant_scores)
        per_item.append({
            "item_id": oid,
            "p_orig": p_orig,
            "p_iso": p_iso,
            "delta": p_orig - p_iso,
            "n_orig_trials": len(orig_scores),
            "n_var_scores": len(variant_scores),
        })

    if not per_item:
        return None

    acc_orig = np.mean([p["p_orig"] for p in per_item])
    acc_iso = np.mean([p["p_iso"] for p in per_item])
    delta_raw = float(acc_orig - acc_iso)
    deltas = np.array([p["delta"] for p in per_item])
    n = len(deltas)
    se_delta = float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci_low = float(delta_raw - 1.96 * se_delta)
    ci_high = float(delta_raw + 1.96 * se_delta)

    # Wilcoxon signed-rank test
    try:
        nonzero = deltas[deltas != 0]
        if len(nonzero) >= 5:
            stat, p_value = stats.wilcoxon(nonzero, alternative='greater')
            p_value = float(p_value)
        else:
            p_value = None
    except Exception:
        p_value = None

    # IRT proxy (variance-weighted)
    variances = np.array([np.var([p["p_orig"], p["p_iso"]]) for p in per_item])
    weights = variances / (variances.mean() + 1e-10)
    delta_irt = float(np.mean(weights * deltas))

    n_flagged = sum(1 for d in deltas if d > 0.5)

    # Clustered bootstrap CI
    rng = np.random.default_rng(2026)
    boot_means = [np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(5000)]
    boot_ci = [float(np.percentile(boot_means, 2.5)),
               float(np.percentile(boot_means, 97.5))]

    # Extraction rates
    orig_results = [r for r in results.values() if r["is_original"]]
    var_results = [r for r in results.values() if not r["is_original"]]
    orig_ext = sum(1 for r in orig_results if r["extracted"] is not None) / max(len(orig_results), 1)
    var_ext = sum(1 for r in var_results if r["extracted"] is not None) / max(len(var_results), 1)

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
        "n_items": n,
        "acc_original": float(acc_orig),
        "acc_isomorphic": float(acc_iso),
        "delta_raw": delta_raw,
        "se_delta": se_delta,
        "ci_95": [ci_low, ci_high],
        "boot_ci_95": boot_ci,
        "p_value": p_value,
        "delta_irt": delta_irt,
        "n_flagged": n_flagged,
        "extraction_rate_orig": orig_ext,
        "extraction_rate_var": var_ext,
        "archetype": archetype,
        "per_item": per_item,
    }


def print_report(model, r):
    p_str = f"p={r['p_value']:.2e}" if r['p_value'] is not None else "p=N/A"
    print(f"""
{'='*60}
  Model:     {model}
  Items:     {r['n_items']}
  Acc_orig:  {r['acc_original']:.1%}    Acc_iso: {r['acc_isomorphic']:.1%}
  Delta_raw: {r['delta_raw']:+.3f}   SE: {r['se_delta']:.3f}
  95% CI:    [{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]
  Boot CI:   [{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}]
  {p_str}
  Delta_IRT: {r['delta_irt']:+.3f}
  Flagged:   {r['n_flagged']}/{r['n_items']}
  Extr%:     orig={r['extraction_rate_orig']:.1%}  var={r['extraction_rate_var']:.1%}
  Verdict:   {r['archetype']}
{'='*60}""")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Isomorph-Eval v3: Production runner")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.output is None:
        safe = args.model.replace("/", "_")
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", f"results_{safe}_t{args.trials}.json"
        )

    asyncio.run(run_evaluation(
        args.model, args.dataset, args.output,
        args.trials, args.temperature, args.resume,
    ))
