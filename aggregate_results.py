#!/usr/bin/env python3
"""
Aggregate results from multiple model evaluations into
the paper's Table 2 (main results).

Usage:
  python aggregate_results.py results_*.json
"""
import json, sys, glob

def main():
    files = sys.argv[1:] or sorted(glob.glob("results_*.json"))
    if not files:
        print("Usage: python aggregate_results.py results_*.json")
        print("No result files found.")
        return

    print(f"\n{'='*90}")
    print(f"ISOMORPH-EVAL: EMPIRICAL RESULTS — Table 2 (Main Results)")
    print(f"{'='*90}")
    print(f"\n{'Model':35s} | {'Acc_orig':>8s} | {'Acc_iso':>7s} | {'Δ_raw':>6s} | "
          f"{'Δ_IRT':>6s} | {'Flagged':>7s} | {'Archetype':>20s}")
    print("-" * 100)

    rows = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        r = data["report"]
        model = data["model"].split("/")[-1]
        print(f"  {model:33s} | {r['acc_original']:7.1%} | {r['acc_isomorphic']:6.1%} | "
              f"{r['delta_raw']:+.3f} | {r['delta_irt']:+.3f} | "
              f"{r['n_flagged']:3d}/{r['n_items']:3d} | {r['archetype']:>20s}")
        rows.append(data)

    print(f"\n{'='*90}")
    print(f"\nLaTeX table rows (copy into paper):\n")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Model & Acc$_{\mathrm{orig}}$ & Acc$_{\mathrm{iso}}$ & $\Delta_{\mathrm{raw}}$ & $\Dcont$ & Flagged & Archetype \\")
    print(r"\midrule")
    for data in rows:
        r = data["report"]
        m = data["model"].split("/")[-1].replace("_",r"\_")
        arch = r["archetype"].replace("_"," ").title()
        print(f"{m} & {r['acc_original']:.1%} & {r['acc_isomorphic']:.1%} & "
              f"{r['delta_raw']:+.3f} & {r['delta_irt']:+.3f} & "
              f"{r['n_flagged']}/{r['n_items']} & {arch} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main()
