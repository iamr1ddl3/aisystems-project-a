"""
Regression Checker — Compare current eval results against baseline.

Loads baseline_scores.json and eval_results.json, compares key metrics,
and flags regressions that exceed the threshold (default: 5 percentage points).

Exit code 0 = no regression, exit code 1 = regression detected (CI-friendly).

Usage:
  python scripts/check_regression.py
  python scripts/check_regression.py --threshold 0.10
"""
import os
import sys
import json

SCRIPT_DIR = os.path.dirname(__file__)
THRESHOLD = 0.05  # 5 percentage points

# Metrics to check: (key, display_name, scale)
# scale=1 means metric is 0-1 (hit_rate, mrr), scale=5 means metric is 0-5 (faithfulness, correctness)
METRICS = [
    ("hit_rate", "Hit Rate", 1),
    ("mean_mrr", "Mean MRR", 1),
    ("avg_faithfulness", "Avg Faithfulness", 5),
    ("avg_correctness", "Avg Correctness", 5),
]


def load_baseline():
    """Load baseline_scores.json. Returns the aggregate summary dict."""
    path = os.path.join(SCRIPT_DIR, "baseline_scores.json")
    if not os.path.exists(path):
        print("No baseline_scores.json found. Run eval_harness.py --save-baseline first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return data["aggregate"]


def load_current():
    """Load eval_results.json. Returns the summary dict."""
    path = os.path.join(SCRIPT_DIR, "eval_results.json")
    if not os.path.exists(path):
        print("No eval_results.json found. Run eval_harness.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return data["summary"]


def check_regression(baseline, current, threshold=THRESHOLD):
    """
    Compare current metrics against baseline.
    A regression is flagged when a metric drops by more than `threshold`
    (normalized to 0-1 scale).

    Returns a list of dicts: {metric, baseline, current, delta, regressed}
    """
    results = []
    for key, name, scale in METRICS:
        b_val = baseline.get(key, 0)
        c_val = current.get(key, 0)

        # Normalize to 0-1 scale for comparison
        b_norm = b_val / scale
        c_norm = c_val / scale
        delta = c_norm - b_norm

        regressed = delta < -threshold

        results.append({
            "metric": name,
            "key": key,
            "baseline": b_val,
            "current": c_val,
            "delta": round(delta, 4),
            "delta_pct": round(delta * 100, 1),
            "regressed": regressed,
        })

    return results


def display_results(results):
    """Print a formatted comparison table and overall verdict."""
    any_regression = any(r["regressed"] for r in results)

    print()
    print("=" * 65)
    print("              REGRESSION CHECK")
    print("=" * 65)
    print(f"  {'Metric':<20} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>10}")
    print("  " + "-" * 58)

    for r in results:
        sign = "+" if r["delta_pct"] >= 0 else ""
        status = "REGRESSION" if r["regressed"] else "OK"
        marker = "X" if r["regressed"] else " "
        print(f"  {r['metric']:<20} {r['baseline']:>10.4f} {r['current']:>10.4f} "
              f"{sign}{r['delta_pct']:>8.1f}%  [{marker}] {status}")

    print("=" * 65)

    if any_regression:
        print("  REGRESSION DETECTED")
        regressed = [r for r in results if r["regressed"]]
        for r in regressed:
            print(f"    - {r['metric']}: dropped {abs(r['delta_pct']):.1f}pp "
                  f"(threshold: {THRESHOLD*100:.0f}pp)")
    else:
        print("  NO REGRESSION")

    print("=" * 65)

    return any_regression


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check for eval metric regressions")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Regression threshold in 0-1 scale (default: {THRESHOLD})")
    args = parser.parse_args()

    baseline = load_baseline()
    current = load_current()
    results = check_regression(baseline, current, threshold=args.threshold)
    has_regression = display_results(results)

    sys.exit(1 if has_regression else 0)


if __name__ == "__main__":
    main()
