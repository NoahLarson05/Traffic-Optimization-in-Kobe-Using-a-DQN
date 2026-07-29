import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "final_comparison_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = {
    "Controller": ["Baseline", "SOTL", "Max Pressure", "DQN"],
    "Average Halting Vehicles": [6.9026, 6.9477, 8.3254, 6.1249],
    "Average Waiting Time": [158.1404, 168.9972, 199.1434, 122.0873],
    "Episode Mean Total Delay": [9.1644, 9.2084, 11.5696, 8.1967],
    "Traffic Density": [12.4062, 12.4328, 14.5063, 11.4969],
    "Throughput": [1412.6667, 1435.5556, 1431.2222, 1412.6667],
}

df = pd.DataFrame(data)

def save_bar(metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Controller"], df[metric])
    plt.xlabel("Controller")
    plt.ylabel(ylabel)
    plt.title(f"{metric} Comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

save_bar("Average Halting Vehicles", "Average Halting Vehicles", "01_halting_comparison.png")
save_bar("Average Waiting Time", "Average Waiting Time", "02_waiting_time_comparison.png")
save_bar("Episode Mean Total Delay", "Episode Mean Total Delay", "03_emtd_comparison.png")
save_bar("Traffic Density", "Traffic Density (veh/km)", "04_density_comparison.png")
save_bar("Throughput", "Throughput Vehicles", "05_throughput_comparison.png")

dqn_days = pd.read_csv("opt_trials/trial_105_dqn_validation_all_days_results.csv")
base_days = pd.read_csv("baseline_all_days_results.csv")

for d in (dqn_days, base_days):
    d["day"] = (d["route_file"].str.replace("routes_", "", regex=False)
                .str.replace(".rou.xml", "", regex=False))

merged = base_days.merge(dqn_days, on="day", suffixes=("_base", "_dqn")).sort_values("day")

plt.figure(figsize=(10, 5))
x = range(len(merged))
width = 0.38
plt.bar([i - width / 2 for i in x], merged["average_waiting_time_base"],
        width=width, label="Baseline")
plt.bar([i + width / 2 for i in x], merged["average_waiting_time_dqn"],
        width=width, label="DQN")
plt.xticks(list(x), merged["day"], rotation=45, ha="right")
plt.xlabel("Validation Day")
plt.ylabel("Average Waiting Time")
plt.title("Average Waiting Time per Validation Day")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_waiting_by_day_dqn_vs_baseline.png"), dpi=300)
plt.close()

df.to_csv(os.path.join(OUTPUT_DIR, "final_controller_comparison_table.csv"), index=False)

print("Saved graphs to:", OUTPUT_DIR)
