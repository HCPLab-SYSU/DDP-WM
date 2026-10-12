"""Collect the final success rates of planning runs into a table.

Usage: python collect_results.py [--dir plan_outputs] [--out eval_summary.csv]
"""
import os
import json
import argparse

import pandas as pd


def parse_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return None
    try:
        # a single JSON object
        return json.loads(content)
    except json.JSONDecodeError:
        # JSON lines: the last line holds the final evaluation
        try:
            return json.loads(content.split("\n")[-1])
        except json.JSONDecodeError:
            return None


def find_seed(run_dir, max_levels=3):
    """Read `seed:` from the run config, searching up to max_levels parent directories."""
    current_dir = run_dir
    for _ in range(max_levels):
        for cand in ("config.yaml", "hydra.yaml"):
            for path in (os.path.join(current_dir, ".hydra", cand), os.path.join(current_dir, cand)):
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("seed:"):
                            return line.split("seed:")[-1].strip()
        current_dir = os.path.dirname(current_dir)
    return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Collect planning results")
    parser.add_argument("--dir", type=str, default="plan_outputs", help="root directory to search")
    parser.add_argument("--out", type=str, default="eval_summary.csv", help="output CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Directory '{args.dir}' does not exist; set it with --dir.")
        return

    results = []
    for root, _, files in os.walk(args.dir):
        if "logs.json" not in files:
            continue
        data = parse_json_file(os.path.join(root, "logs.json"))
        success_rate = data.get("final_eval/success_rate") if data else None
        if success_rate is None:  # unfinished run
            continue
        rel_dir = os.path.relpath(root, args.dir)
        results.append({
            # run directories are named <timestamp>_<model_name>_gH<goal_H>
            "Directory": "".join(rel_dir.split("_")[1:-1]),
            "Seed": find_seed(root),
            "Success": f"{success_rate:.0%}",
        })

    if not results:
        print("No logs.json with a final success rate was found.")
        return

    df = pd.DataFrame(results).sort_values(by=["Seed"])
    print(df.to_string(index=False))
    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
