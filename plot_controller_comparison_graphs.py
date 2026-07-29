import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Plot comparison graphs for traffic signal control experiments
# ============================================================
# Expected CSV files:
#   baseline_summary.csv
#   controller_comparison_results.csv
#   dqn_episode_results.csv
#
# These are produced by:
#   baseline_test_all_metrics.py
#   run_mp_sotl_all_metrics.py
#   dqn_sumo_tls_all_metrics.py
#
# Output folder:
#   comparison_graphs/
# ============================================================

OUTPUT_DIR = "comparison_graphs"
DQN_LAST_N_EPISODES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_method_name(name):
    name = str(name).strip().lower()
    mapping = {
        "baseline": "Baseline",
        "max_pressure": "Max Pressure",
        "sotl": "SOTL",
        "dqn": "DQN",
    }
    return mapping.get(name, name)


def read_baseline():
    path = "baseline_summary.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "baseline_summary.csv not found. Run baseline_test_all_metrics.py first."
        )

    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    row["method"] = "Baseline"
    return row


def read_mp_sotl():
    path = "controller_comparison_results.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "controller_comparison_results.csv not found. Run run_mp_sotl_all_metrics.py first."
        )

    df = pd.read_csv(path)
    df["method"] = df["controller"].apply(normalize_method_name)
    return df.to_dict("records")


def read_dqn():
    path = "dqn_episode_results.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "dqn_episode_results.csv not found. Run dqn_sumo_tls_all_metrics.py first."
        )

    df = pd.read_csv(path)

    # Use the average of the last N episodes for final DQN performance.
    # This is more stable than using only the final episode.
    recent = df.tail(DQN_LAST_N_EPISODES)

    return {
        "method": "DQN",
        "average_halting_vehicles": recent["avg_halting_vehicles"].mean(),
        "average_waiting_time": recent["avg_waiting_time"].mean(),
        "episode_mean_total_delay": recent["episode_mean_total_delay"].mean(),
        "average_traffic_density_veh_per_km": recent["avg_traffic_density_veh_per_km"].mean(),
        "throughput_vehicles": recent["throughput_vehicles"].mean(),
    }


def build_comparison_table():
    rows = []
    rows.append(read_baseline())
    rows.extend(read_mp_sotl())
    rows.append(read_dqn())

    df = pd.DataFrame(rows)

    # Keep a clean method order
    order = ["Baseline", "Max Pressure", "SOTL", "DQN"]
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method")

    df.to_csv(os.path.join(OUTPUT_DIR, "final_controller_comparison.csv"), index=False)
    return df


def save_bar_chart(df, metric, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(df["method"].astype(str), df[metric])
    plt.title(title)
    plt.xlabel("Controller")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def save_learning_curve():
    path = "dqn_episode_results.csv"
    df = pd.read_csv(path)

    plt.figure(figsize=(9, 5))
    plt.plot(df["episode"], df["total_reward"], label="Total reward")

    if len(df) >= 10:
        df["rolling_reward"] = df["total_reward"].rolling(10).mean()
        plt.plot(df["episode"], df["rolling_reward"], label="10-episode moving average")

    plt.title("DQN learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_dqn_learning_curve.png"), dpi=300)
    plt.close()


def save_density_over_time():
    # Optional but useful: compare density over simulation time.
    # This uses the per-step CSV files if they exist.
    files = {
        "Baseline": "baseline_results.csv",
        "Max Pressure": "max_pressure_results.csv",
        "SOTL": "sotl_results.csv",
    }

    available = {name: path for name, path in files.items() if os.path.exists(path)}

    if not available:
        return

    plt.figure(figsize=(10, 5))

    for name, path in available.items():
        df = pd.read_csv(path)

        if "traffic_density_veh_per_km" not in df.columns:
            continue

        # Smooth slightly to make the line easier to read in slides.
        y = df["traffic_density_veh_per_km"].rolling(30, min_periods=1).mean()
        plt.plot(df["step"], y, label=name)

    plt.title("Traffic density over time")
    plt.xlabel("Simulation step")
    plt.ylabel("Traffic density (vehicles/km)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_traffic_density_over_time.png"), dpi=300)
    plt.close()


def main():
    comparison = build_comparison_table()

    # Graph 1: DQN learning curve
    save_learning_curve()

    # Graph 2: EMTD comparison
    save_bar_chart(
        comparison,
        metric="episode_mean_total_delay",
        title="Episode mean total delay by controller",
        ylabel="Episode mean total delay",
        filename="02_emtd_comparison.png"
    )

    # Graph 3: Average halting vehicles comparison
    save_bar_chart(
        comparison,
        metric="average_halting_vehicles",
        title="Average halting vehicles by controller",
        ylabel="Average halting vehicles",
        filename="03_average_halting_comparison.png"
    )

    # Graph 4: Traffic density comparison
    save_bar_chart(
        comparison,
        metric="average_traffic_density_veh_per_km",
        title="Average traffic density by controller",
        ylabel="Vehicles per km",
        filename="04_traffic_density_comparison.png"
    )

    # Optional: time-series density graph
    save_density_over_time()

    print("Graphs created in:", OUTPUT_DIR)
    print("Generated files:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        print(" -", os.path.join(OUTPUT_DIR, file))


if __name__ == "__main__":
    main()
