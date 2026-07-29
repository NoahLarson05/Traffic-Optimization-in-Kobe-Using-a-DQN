import os
import sys
import csv
import json
import math
import time
import random
import shutil
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIALS_DIR = os.path.join(PROJECT_DIR, "opt_trials")
BASELINE_CACHE = os.path.join(PROJECT_DIR, "baselines_val_cache.json")
LEADERBOARD = os.path.join(PROJECT_DIR, "optimization_results.csv")
BEST_MODEL = os.path.join(PROJECT_DIR, "dqn_traffic_model_best.pth")
BEST_CONFIG = os.path.join(PROJECT_DIR, "dqn_best_config.json")

METRIC_KEYS = [
    "average_halting_vehicles",
    "average_waiting_time",
    "episode_mean_total_delay",
    "average_traffic_density_veh_per_km",
    "throughput_vehicles",
]

LOWER_IS_BETTER = METRIC_KEYS[:4]


def get_reference_metrics():
    if os.path.exists(BASELINE_CACHE):
        with open(BASELINE_CACHE, encoding="utf-8") as f:
            print("Loaded cached reference metrics from", BASELINE_CACHE)
            return json.load(f)

    import exp_common as ec
    import run_mp_sotl_all_metrics as mp_sotl

    _, val_files = ec.get_train_val_split()
    reference = {}

    for controller in [ec.BaseController(),
                       mp_sotl.MaxPressureController(),
                       mp_sotl.SotlController()]:
        results = []
        for route_file in val_files:
            print(f"[reference] {controller.name} on {route_file}")
            results.append(ec.run_eval_episode(controller, route_file))
        reference[controller.name] = ec.average_results(results)

    with open(BASELINE_CACHE, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)
    print("Reference metrics cached to", BASELINE_CACHE)
    return reference


def best_reference(reference):
    best = {}
    for key in METRIC_KEYS:
        values = [reference[c][key] for c in reference]
        best[key] = min(values) if key in LOWER_IS_BETTER else max(values)
    return best


def score_summary(summary, best_ref):
    ratios = []
    for key in LOWER_IS_BETTER:
        ratios.append(summary[key] / max(best_ref[key], 1e-9))
    ratios.append(best_ref["throughput_vehicles"] / max(summary["throughput_vehicles"], 1e-9))
    return sum(ratios) / len(ratios)


def loguniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_config(rng):
    cfg = {
        key: round(loguniform(rng, lo, hi), 6 if key == "lr" else 5)
        for key, (lo, hi) in BOUNDS.items()
    }
    cfg["gamma"] = rng.choice(GAMMA_CHOICES)
    cfg["action_interval"] = rng.choice(INTERVAL_LADDER)
    cfg["target_update"] = rng.choice(TARGET_CHOICES)
    cfg["batch_size"] = rng.choice(BATCH_LADDER)
    return cfg


DEFAULT_CONFIG = {
    "w_delay": 1.0, "w_halt": 0.3, "w_wait": 0.005, "w_arr": 0.3,
    "lr": 0.0005, "gamma": 0.99, "action_interval": 30, "target_update": 10,
    "batch_size": 64,
}

BOUNDS = {
    "w_delay": (0.1, 5.0), "w_halt": (0.03, 3.0), "w_wait": (0.0003, 0.05),
    "w_arr": (0.05, 5.0), "lr": (5e-5, 3e-3),
}
INTERVAL_LADDER = [10, 15, 20, 30, 45, 60]
BATCH_LADDER = [32, 64, 128]
GAMMA_CHOICES = [0.9, 0.95, 0.99, 0.995]
TARGET_CHOICES = [3, 5, 10, 20]


def sample_refined(rng, center, jitter=1.5):
    cfg = {}
    for key, (lo, hi) in BOUNDS.items():
        value = center[key] * math.exp(rng.uniform(-math.log(jitter), math.log(jitter)))
        cfg[key] = round(min(max(value, lo), hi), 6 if key == "lr" else 5)

    ladder_index = min(range(len(INTERVAL_LADDER)),
                       key=lambda i: abs(INTERVAL_LADDER[i] - center["action_interval"]))
    neighbor = ladder_index + rng.choice([-1, 0, 0, 1])
    cfg["action_interval"] = INTERVAL_LADDER[min(max(neighbor, 0), len(INTERVAL_LADDER) - 1)]

    cfg["gamma"] = center["gamma"] if rng.random() < 0.85 else rng.choice(GAMMA_CHOICES)
    cfg["target_update"] = rng.choice(TARGET_CHOICES)

    batch_index = BATCH_LADDER.index(int(center.get("batch_size", 64)))
    batch_neighbor = batch_index + rng.choice([-1, 0, 0, 1])
    cfg["batch_size"] = BATCH_LADDER[min(max(batch_neighbor, 0), len(BATCH_LADDER) - 1)]
    return cfg


def load_leaderboard():
    if not os.path.exists(LEADERBOARD):
        return []
    rows = []
    with open(LEADERBOARD, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            typed = {"trial": int(row["trial"]), "score": float(row["score"])}
            for key in METRIC_KEYS + list(DEFAULT_CONFIG.keys()):
                typed[key] = float(row.get(key) or DEFAULT_CONFIG.get(key))
            for key in ("action_interval", "target_update", "batch_size"):
                typed[key] = int(typed[key])
            rows.append(typed)
    return rows


def trial_prefix(index):
    return os.path.join(TRIALS_DIR, f"trial_{index:02d}_")


def run_trial(index, config, episodes, timeout_s):
    prefix = trial_prefix(index)
    summary_csv = prefix + "dqn_validation_summary.csv"

    if os.path.exists(summary_csv):
        print(f"Trial {index:02d}: already done, skipping training.")
        return read_summary(summary_csv)

    cmd = [
        sys.executable, os.path.join(PROJECT_DIR, "dqn_sumo_tls_all_metrics.py"),
        "--episodes", str(episodes),
        "--prefix", prefix,
        "--w-delay", str(config["w_delay"]),
        "--w-halt", str(config["w_halt"]),
        "--w-wait", str(config["w_wait"]),
        "--w-arr", str(config["w_arr"]),
        "--lr", str(config["lr"]),
        "--gamma", str(config["gamma"]),
        "--action-interval", str(config["action_interval"]),
        "--target-update", str(config["target_update"]),
        "--batch-size", str(config.get("batch_size", 64)),
    ]

    log_path = prefix + "train.log"
    print(f"Trial {index:02d}: training {episodes} episodes "
          f"(log: {os.path.basename(log_path)})")
    print(f"  config: {config}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=log, env=env,
                                stderr=subprocess.STDOUT, timeout=timeout_s)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Trial {index:02d} FAILED (exit {result.returncode}), see {log_path}")
        return None

    print(f"Trial {index:02d}: finished in {elapsed/60:.1f} min")
    return read_summary(summary_csv)


def read_summary(summary_csv):
    with open(summary_csv, encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {key: float(row[key]) for key in METRIC_KEYS}


def write_leaderboard(rows):
    fields = (["trial", "score"] + METRIC_KEYS + list(DEFAULT_CONFIG.keys()))
    rows_sorted = sorted(rows, key=lambda r: r["score"])
    with open(LEADERBOARD, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_sorted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=60,
                        help="training episodes per trial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-timeout-min", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4,
                        help="trials to run in parallel")
    parser.add_argument("--refine", action="store_true",
                        help="sample around the top-2 configs on the existing "
                             "leaderboard instead of the full space")
    parser.add_argument("--jitter", type=float, default=1.5,
                        help="multiplicative jitter range for --refine")
    args = parser.parse_args()

    os.makedirs(TRIALS_DIR, exist_ok=True)
    rng = random.Random(args.seed)

    reference = get_reference_metrics()
    best_ref = best_reference(reference)

    print("\nReference (validation-day averages):")
    for name, summary in reference.items():
        print(f"  {name}: " + ", ".join(f"{k}={summary[k]:.2f}" for k in METRIC_KEYS))
    print("  best-per-metric:", {k: round(v, 2) for k, v in best_ref.items()})
    print()

    rows = load_leaderboard()
    best_score = min((r["score"] for r in rows), default=float("inf"))
    best_index = min(rows, key=lambda r: r["score"])["trial"] if rows else None
    start_index = max((r["trial"] for r in rows), default=-1) + 1

    if args.refine:
        if not rows:
            sys.exit("--refine needs an existing optimization_results.csv")
        centers = [
            {k: r[k] for k in DEFAULT_CONFIG}
            for r in sorted(rows, key=lambda r: r["score"])[:2]
        ]
        print(f"Refining around top-{len(centers)} configs "
              f"(trials {[r['trial'] for r in sorted(rows, key=lambda r: r['score'])[:2]]}), "
              f"jitter x{args.jitter}, starting at trial {start_index}\n")
        configs = {
            start_index + i: sample_refined(rng, centers[i % len(centers)], args.jitter)
            for i in range(args.trials)
        }
    else:
        configs = {
            start_index + i: (DEFAULT_CONFIG.copy() if start_index + i == 0
                              else sample_config(rng))
            for i in range(args.trials)
        }

    def worker(index):
        try:
            return index, run_trial(index, configs[index], args.episodes,
                                    args.trial_timeout_min * 60)
        except subprocess.TimeoutExpired:
            print(f"Trial {index:02d} TIMED OUT, skipping.")
            return index, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, index) for index in sorted(configs)]

        for future in as_completed(futures):
            index, summary = future.result()
            if summary is None:
                continue

            config = configs[index]
            score = score_summary(summary, best_ref)
            rows.append({"trial": index, "score": round(score, 4),
                         **{k: round(summary[k], 4) for k in METRIC_KEYS},
                         **config})
            write_leaderboard(rows)

            marker = ""
            if score < best_score:
                best_score = score
                best_index = index
                shutil.copyfile(trial_prefix(index) + "dqn_traffic_model.pth", BEST_MODEL)
                with open(BEST_CONFIG, "w", encoding="utf-8") as f:
                    json.dump({"trial": index, "score": score,
                               "episodes": args.episodes, **config}, f, indent=2)
                marker = "  <-- new best"

            print(f"Trial {index:02d}: score={score:.4f} "
                  f"(1.0 = matches best reference controller){marker}\n")

    print("=" * 60)
    if best_index is None:
        print("No trial finished successfully.")
        return

    print(f"Best trial: {best_index:02d}  score={best_score:.4f}")
    print("Best model:  ", BEST_MODEL)
    print("Best config: ", BEST_CONFIG)
    print("Leaderboard: ", LEADERBOARD)
    if best_score < 1.0:
        print("The best DQN beats the best reference controller on average.")
    else:
        print("The best DQN does NOT yet beat the best reference controller "
              "(score >= 1.0). Consider more trials or more episodes per trial.")


if __name__ == "__main__":
    main()
