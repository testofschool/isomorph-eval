#!/usr/bin/env python3
"""
Evaluate models on regenerated v2 variants.

Only runs evaluation on regenerated items' variants (clean items
already have complete evaluation data from the v1 run).

Usage:
    python3 run_eval_v2_variants.py --model llama-3.1-8b-instant
    python3 run_eval_v2_variants.py --model llama-3.1-8b-instant --resume
"""

import json
import asyncio
import argparse
import re
import time
from pathlib import Path
from openai import AsyncOpenAI

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a math problem solver. Solve the problem step by step.
Put your final numeric answer inside \\boxed{}.
If no boxed answer, end with #### followed by the numeric answer."""

def extract_answer(response: str):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    if boxed:
        try:
            return float(boxed[-1].replace(',', '').replace('$', ''))
        except ValueError:
            pass
    hashes = re.findall(r'####\s*([\d,.\-]+)', response)
    if hashes:
        try:
            return float(hashes[-1].replace(',', ''))
        except ValueError:
            pass
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', response)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def answers_match(extracted, expected, tol=0.01):
    if extracted is None:
        return False
    if expected == 0:
        return abs(extracted) < 0.01
    return abs(extracted - expected) / max(abs(expected), 1e-9) < tol


async def evaluate_item(client, model, question, temperature, extra_kwargs=None):
    if extra_kwargs is None:
        extra_kwargs = {}
    for retry in range(5):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=temperature,
                max_tokens=768,
                **extra_kwargs,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                wait = 3 * (2 ** retry)
                print(f"  Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            elif "daily" in err.lower() or "limit" in err.lower():
                return None
            else:
                await asyncio.sleep(2)
    return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    with open(DATA_DIR / "eval_verified_v2.json") as f:
        dataset = json.load(f)

    regen_items = [i for i in dataset["items"] if i.get("regenerated_v2")]
    print(f"Regenerated items to evaluate: {len(regen_items)}")

    # Build eval queue: only variant questions from regenerated items
    eval_queue = []
    for item in regen_items:
        for var in item["variants"]:
            eval_queue.append({
                "item_id": var["item_id"],
                "question": var["question"],
                "expected": float(var["answer"]),
                "is_original": False,
            })
        # Also add the original (we have existing results but may want fresh ones)
        eval_queue.append({
            "item_id": item["item_id"],
            "question": item["question"],
            "expected": float(item["answer"]),
            "is_original": True,
        })

    print(f"Total eval items: {len(eval_queue)}")

    # Load existing results for resume
    safe_model = args.model.replace("/", "_")
    output_path = RESULTS_DIR / f"v2_results_{safe_model}.json"
    existing = {}
    if args.resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f).get("raw_results", {})
        print(f"Resuming: {len(existing)} existing results")

    client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    extra_kwargs = {}
    if "qwen3" in args.model.lower():
        extra_kwargs["extra_body"] = {"reasoning_effort": "none"}

    results = dict(existing)
    done = 0
    total = len(eval_queue)

    for entry in eval_queue:
        key = f"{entry['item_id']}_t0"
        if key in results:
            continue

        response = await evaluate_item(
            client, args.model, entry["question"], args.temperature, extra_kwargs,
        )

        if response is None:
            print(f"\n  Daily limit hit after {done} items. Use --resume to continue.")
            break

        extracted = extract_answer(response)
        correct = answers_match(extracted, entry["expected"])

        results[key] = {
            "item_id": entry["item_id"],
            "trial": 0,
            "extracted": extracted,
            "expected": entry["expected"],
            "correct": correct,
            "response_len": len(response),
            "is_original": entry["is_original"],
        }

        done += 1

        if done % 3 == 0:
            n_correct = sum(1 for r in results.values() if r["correct"])
            acc = n_correct / len(results) * 100
            print(f"  [{done:>4}/{total}] acc={acc:.1f}%", end="", flush=True)

        if done % 30 == 0:
            save_data = {
                "model": args.model,
                "dataset": "eval_verified_v2.json",
                "n_results": len(results),
                "raw_results": results,
            }
            with open(output_path, "w") as f:
                json.dump(save_data, f, indent=2)

        await asyncio.sleep(args.delay)

    # Final save
    save_data = {
        "model": args.model,
        "dataset": "eval_verified_v2.json",
        "n_results": len(results),
        "raw_results": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
