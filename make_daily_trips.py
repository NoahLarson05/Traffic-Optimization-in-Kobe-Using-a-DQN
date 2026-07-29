import os
import subprocess
import pandas as pd
import numpy as np
import random
from xml.sax.saxutils import escape

# ============================================================
# make_daily_trips.py
# Creates one trips file and one route file per date.
#
# Example output:
#   trips_2026-05-01.xml
#   routes_2026-05-01.rou.xml
#
# Use these route files for DQN random episode training.
# ============================================================

CSV_FILE = "suma_traffic_max.csv"
NET_FILE = "osm.net.xml.gz"

# Time window used from every available date
START_TIME = "06:45"
END_TIME = "07:30"

# Main road origins/destinations
UP_FROM = "-1363015444#0"
DOWN_FROM = "32639957#0"

UP_TO_OPTIONS = [
    "32639957#0",
    "32639957#0",
    "32639957#0",
    "32639957#0",
    "-326399028#0",   # Tsukimiyama
    "-81816144#2",    # Suma North
]

DOWN_TO_OPTIONS = [
    "-1363015444#0",
    "-1363015444#0",
    "-1363015444#0",
    "-1363015444#0",
    "-326399028#0",   # Tsukimiyama
    "-81816144#2",    # Suma North
]

# Side road origins
TSUKIMI_FROM = "326399028#0"
KITA_FROM = "81816144#0"

# Side road vehicles go to Route 2 east/west
SIDE_TO_OPTIONS = [
    "32639957#0",
    "-1363015444#0",
]

# Demand settings
MAIN_SCALE = 0.5
SIDE_CARS_PER_5MIN = 50

# Optional limit for number of days; set None to use all days
MAX_DAYS = None


def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def generate_depart_times(count, begin, scale=1.0):
    count = int(count * scale)
    if count <= 0:
        return []

    interval = 300
    lam = count / interval

    depart_times = []
    t = begin

    while True:
        t += np.random.exponential(1 / lam)
        if t >= begin + interval:
            break
        depart_times.append(t)

    return depart_times


def add_trips(trips, prefix, veh_type, count, begin, from_edge, to_edge_options, scale=1.0):
    depart_times = generate_depart_times(count, begin, scale)

    for j, depart in enumerate(depart_times):
        if isinstance(to_edge_options, list):
            selected_to = random.choice(to_edge_options)
        else:
            selected_to = to_edge_options

        trips.append({
            "id": f"{prefix}_{j}",
            "type": veh_type,
            "depart": depart,
            "from": from_edge,
            "to": selected_to,
        })


def write_trips_file(output_file, trips):
    # Important: sort by depart time so duarouter/SUMO does not ignore vehicles
    trips = sorted(trips, key=lambda x: x["depart"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("<routes>\n")
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/>\n')
        f.write('    <vType id="truck" accel="1.3" decel="4.0" sigma="0.5" length="12" maxSpeed="11.1"/>\n\n')

        for trip in trips:
            f.write(
                f'    <trip id="{escape(trip["id"])}" '
                f'type="{escape(trip["type"])}" '
                f'depart="{trip["depart"]:.2f}" '
                f'from="{escape(trip["from"])}" '
                f'to="{escape(trip["to"])}"/>\n'
            )

        f.write("</routes>\n")


def run_duarouter(trips_file, routes_file):
    cmd = [
        "duarouter",
        "--ignore-errors",
        "--unsorted-input",
        "-n", NET_FILE,
        "-r", trips_file,
        "-o", routes_file,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print("WARNING: duarouter returned non-zero status for", trips_file)


def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"{CSV_FILE} not found.")

    df = pd.read_csv(CSV_FILE)
    df = df.fillna(0)
    df["datetime"] = pd.to_datetime(df["datetime"])

    start_time = pd.to_datetime(START_TIME).time()
    end_time = pd.to_datetime(END_TIME).time()

    # Keep same time window across all dates
    df = df[
        (df["datetime"].dt.time >= start_time) &
        (df["datetime"].dt.time <= end_time)
    ].copy()

    df["date"] = df["datetime"].dt.date

    grouped = list(df.groupby("date"))
    grouped = sorted(grouped, key=lambda x: x[0])

    if MAX_DAYS is not None:
        grouped = grouped[:MAX_DAYS]

    print("Number of dates found:", len(grouped))

    created_routes = []

    for date, day_df in grouped:
        date_str = str(date)
        trips_file = f"trips_{date_str}.xml"
        routes_file = f"routes_{date_str}.rou.xml"

        day_df = day_df.sort_values("datetime").reset_index(drop=True)

        trips = []

        for i, row in day_df.iterrows():
            begin = i * 300

            up_small = safe_int(row["up_small"])
            up_large = safe_int(row["up_large"])
            down_small = safe_int(row["down_small"])
            down_large = safe_int(row["down_large"])

            add_trips(trips, f"up_car_{date_str}_{i}", "car", up_small, begin, UP_FROM, UP_TO_OPTIONS, MAIN_SCALE)
            add_trips(trips, f"up_truck_{date_str}_{i}", "truck", up_large, begin, UP_FROM, UP_TO_OPTIONS, MAIN_SCALE)
            add_trips(trips, f"down_car_{date_str}_{i}", "car", down_small, begin, DOWN_FROM, DOWN_TO_OPTIONS, MAIN_SCALE)
            add_trips(trips, f"down_truck_{date_str}_{i}", "truck", down_large, begin, DOWN_FROM, DOWN_TO_OPTIONS, MAIN_SCALE)
            add_trips(trips, f"tsukimi_car_{date_str}_{i}", "car", SIDE_CARS_PER_5MIN, begin, TSUKIMI_FROM, SIDE_TO_OPTIONS, 1.0)
            add_trips(trips, f"kita_car_{date_str}_{i}", "car", SIDE_CARS_PER_5MIN, begin, KITA_FROM, SIDE_TO_OPTIONS, 1.0)

        if len(trips) == 0:
            print("Skipping", date_str, "because no trips were generated.")
            continue

        write_trips_file(trips_file, trips)
        print("Created:", trips_file, "vehicles:", len(trips))

        run_duarouter(trips_file, routes_file)
        created_routes.append(routes_file)

    with open("daily_route_files.txt", "w", encoding="utf-8") as f:
        for route_file in created_routes:
            f.write(route_file + "\n")

    print("\nDone.")
    print("Created route files:", len(created_routes))
    print("Saved list to daily_route_files.txt")
    print("Use glob pattern in DQN: routes_*.rou.xml")


if __name__ == "__main__":
    main()
