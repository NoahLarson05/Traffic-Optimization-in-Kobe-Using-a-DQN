import csv
import argparse

import exp_common as ec


def pick_days(which):
    train_files, val_files = ec.get_train_val_split()
    if which == "train":
        return train_files
    if which == "all":
        return sorted(train_files + val_files)
    return val_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", choices=["val", "train", "all"], default="val")
    args = parser.parse_args()

    route_files = pick_days(args.days)
    print(f"Baseline evaluation on {len(route_files)} days ({args.days})")

    controller = ec.BaseController()
    results = []

    for route_file in route_files:
        print("Running baseline with", route_file)
        result = ec.run_eval_episode(controller, route_file)
        results.append(result)

    with open("baseline_all_days_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = ec.average_results(results)
    with open("baseline_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + ec.METRIC_KEYS + ["num_days"])
        writer.writerow(["baseline"] + [summary[k] for k in ec.METRIC_KEYS] + [summary["num_days"]])

    print("\nBaseline all-days finished. Days evaluated:", summary["num_days"])
    for key in ec.METRIC_KEYS:
        print(f"  {key}: {summary[key]:.4f}")
    print("Saved: baseline_all_days_results.csv, baseline_summary.csv")


if __name__ == "__main__":
    main()
