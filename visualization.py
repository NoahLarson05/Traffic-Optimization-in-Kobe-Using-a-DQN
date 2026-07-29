import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# Output folder
# =========================
OUTPUT_DIR = "dqn_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Training graphs
# =========================
train_df = pd.read_csv("dqn_episode_results.csv")

def save_line_plot(df, x_col, y_col, title, y_label, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col])
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

save_line_plot(
    train_df,
    "episode",
    "total_reward",
    "DQN Training Reward Over Episodes",
    "Total Reward",
    "01_training_reward.png"
)

save_line_plot(
    train_df,
    "episode",
    "avg_waiting_time",
    "DQN Training Average Waiting Time",
    "Average Waiting Time",
    "02_training_waiting_time.png"
)

save_line_plot(
    train_df,
    "episode",
    "episode_mean_total_delay",
    "DQN Training EMTD Over Episodes",
    "Episode Mean Total Delay",
    "03_training_emtd.png"
)

save_line_plot(
    train_df,
    "episode",
    "avg_halting_vehicles",
    "DQN Training Average Halting Vehicles",
    "Average Halting Vehicles",
    "04_training_halting.png"
)

save_line_plot(
    train_df,
    "episode",
    "avg_traffic_density_veh_per_km",
    "DQN Training Traffic Density",
    "Traffic Density (veh/km)",
    "05_training_density.png"
)

# =========================
# Validation graphs
# =========================
val_df = pd.read_csv("dqn_validation_all_days_results.csv")

# Shorter route file labels
val_df["day"] = val_df["route_file"].str.replace("routes_", "", regex=False)
val_df["day"] = val_df["day"].str.replace(".rou.xml", "", regex=False)

def save_bar_plot(df, x_col, y_col, title, y_label, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col], df[y_col])
    plt.xlabel("Validation Day")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

save_bar_plot(
    val_df,
    "day",
    "episode_mean_total_delay",
    "DQN Validation EMTD by Day",
    "Episode Mean Total Delay",
    "06_validation_emtd_by_day.png"
)

save_bar_plot(
    val_df,
    "day",
    "average_waiting_time",
    "DQN Validation Waiting Time by Day",
    "Average Waiting Time",
    "07_validation_waiting_by_day.png"
)

save_bar_plot(
    val_df,
    "day",
    "average_halting_vehicles",
    "DQN Validation Halting Vehicles by Day",
    "Average Halting Vehicles",
    "08_validation_halting_by_day.png"
)

save_bar_plot(
    val_df,
    "day",
    "average_traffic_density_veh_per_km",
    "DQN Validation Density by Day",
    "Traffic Density (veh/km)",
    "09_validation_density_by_day.png"
)

print("Graphs saved in:", OUTPUT_DIR)