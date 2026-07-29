import os
import csv
import argparse

SEEDS = [11, 22, 33, 44, 55]
CONTROLLER_NAMES = ["baseline", "max_pressure", "sotl", "dqn"]

METRIC_KEYS = [
    "average_halting_vehicles",
    "average_waiting_time",
    "episode_mean_total_delay",
    "average_traffic_density_veh_per_km",
    "throughput_vehicles",
]


def make_controller(name):
    import exp_common as ec
    if name == "baseline":
        return ec.BaseController()
    if name in ("max_pressure", "sotl"):
        import run_mp_sotl_all_metrics as mp_sotl
        return (mp_sotl.MaxPressureController() if name == "max_pressure"
                else mp_sotl.SotlController())
    if name == "dqn":
        import json
        import torch
        from dqn_sumo_tls_all_metrics import DQN, DQNController, STATE_SIZE, ACTION_SIZE
        policy = DQN(STATE_SIZE, ACTION_SIZE)
        policy.load_state_dict(torch.load("dqn_traffic_model_best.pth",
                                          weights_only=True))
        policy.eval()
        with open("dqn_best_config.json", encoding="utf-8") as f:
            action_interval = json.load(f)["action_interval"]
        return DQNController(policy, action_interval)
    raise ValueError(name)


def run(controller_name):
    import exp_common as ec
    controller = make_controller(controller_name)
    _, val_files = ec.get_train_val_split(write_json=False)

    out_csv = f"robustness_{controller_name}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["controller", "route_file", "sumo_seed"] + METRIC_KEYS)

        for route_file in val_files:
            for seed in SEEDS:
                result = ec.run_eval_episode(controller, route_file,
                                             sumo_seed=seed)
                writer.writerow([controller_name, route_file, seed]
                                + [result[k] for k in METRIC_KEYS])
                f.flush()
                print(f"{controller_name} {route_file} seed={seed} "
                      f"wait={result['average_waiting_time']:.1f}")

    print("Saved", out_csv)


def aggregate():
    import pandas as pd

    frames = []
    for name in CONTROLLER_NAMES:
        path = f"robustness_{name}.csv"
        if not os.path.exists(path):
            print("Missing", path, "- run that controller first")
            return
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for name in CONTROLLER_NAMES:
        sub = df[df["controller"] == name]
        per_seed = sub.groupby("sumo_seed")[METRIC_KEYS].mean()
        row = {"controller": name}
        for key in METRIC_KEYS:
            row[f"{key}_mean"] = round(per_seed[key].mean(), 4)
            row[f"{key}_std"] = round(per_seed[key].std(), 4)
        rows.append(row)

    with open("robustness_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nRobustness over", len(SEEDS), "SUMO seeds (mean +/- std across seeds,")
    print("each seed averaged over the 9 validation days):\n")
    for row in rows:
        print(row["controller"])
        for key in METRIC_KEYS:
            print(f"  {key}: {row[f'{key}_mean']} +/- {row[f'{key}_std']}")
    print("\nSaved robustness_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLER_NAMES)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    elif args.controller:
        run(args.controller)
    else:
        parser.error("pass --controller <name> or --aggregate")
