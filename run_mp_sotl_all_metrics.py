import csv
import argparse

import exp_common as ec
from exp_common import traci, TLS_ID, MAIN_PHASES

ACTION_INTERVAL = 30

SOTL_THRESHOLD = 50
SOTL_PLATOON_THRESHOLD = 3


# =========================
# Phase helpers
# =========================
def is_green(ch):
    return ch in ("G", "g")


def get_phase_lanes(tls_id, phase_index):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phase_state = logic.phases[phase_index].state
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    incoming = set()
    outgoing = set()

    for signal_index, links in enumerate(controlled_links):
        if signal_index >= len(phase_state):
            continue
        if not is_green(phase_state[signal_index]):
            continue
        for link in links:
            if len(link) >= 2:
                if link[0]:
                    incoming.add(link[0])
                if link[1]:
                    outgoing.add(link[1])

    return incoming, outgoing


def get_all_main_incoming_lanes(tls_id):
    incoming = set()
    for phase in MAIN_PHASES:
        phase_incoming, _ = get_phase_lanes(tls_id, phase)
        incoming.update(phase_incoming)
    return incoming


def lane_vehicle_count(lanes):
    return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)


def lane_halting_count(lanes):
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)


def apply_phase(tls_id, phase):
    traci.trafficlight.setPhase(tls_id, phase)


# =========================
# Max Pressure controller
# =========================
def choose_max_pressure_phase(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)

    best_phase = MAIN_PHASES[0]
    best_pressure = float("-inf")

    for phase in MAIN_PHASES:
        phase_state = logic.phases[phase].state
        phase_pressure = 0.0
        used_movements = set()

        for signal_index, links in enumerate(controlled_links):
            if signal_index >= len(phase_state):
                continue
            if phase_state[signal_index] not in ("G", "g"):
                continue

            for link in links:
                if len(link) < 2:
                    continue
                in_lane, out_lane = link[0], link[1]
                if not in_lane or not out_lane:
                    continue

                movement = (in_lane, out_lane)
                if movement in used_movements:
                    continue
                used_movements.add(movement)

                incoming_queue = traci.lane.getLastStepHaltingNumber(in_lane)
                outgoing_occupancy = traci.lane.getLastStepVehicleNumber(out_lane)
                phase_pressure += incoming_queue - outgoing_occupancy

        if phase_pressure > best_pressure:
            best_pressure = phase_pressure
            best_phase = phase

    return best_phase


class MaxPressureController(ec.BaseController):
    name = "max_pressure"

    def reset(self):
        self.current_main_index = 0
        apply_phase(TLS_ID, MAIN_PHASES[self.current_main_index])

    def before_step(self, step):
        if step % ACTION_INTERVAL != 0:
            return
        selected_phase = choose_max_pressure_phase(TLS_ID)
        if selected_phase != MAIN_PHASES[self.current_main_index]:
            self.current_main_index = (self.current_main_index + 1) % len(MAIN_PHASES)
            apply_phase(TLS_ID, MAIN_PHASES[self.current_main_index])


# =========================
# SOTL controller
# =========================
class SotlController(ec.BaseController):
    name = "sotl"

    def reset(self):
        self.current_main_index = 0
        self.red_queue_integral = 0
        apply_phase(TLS_ID, MAIN_PHASES[self.current_main_index])

    def after_step(self, step):
        current_phase = MAIN_PHASES[self.current_main_index]
        current_green_lanes, _ = get_phase_lanes(TLS_ID, current_phase)

        all_incoming = get_all_main_incoming_lanes(TLS_ID)
        red_lanes = all_incoming - current_green_lanes

        red_queue = lane_halting_count(red_lanes)
        green_vehicles = lane_vehicle_count(current_green_lanes)

        self.red_queue_integral += red_queue

        if (self.red_queue_integral >= SOTL_THRESHOLD
                and green_vehicles <= SOTL_PLATOON_THRESHOLD):
            self.current_main_index = (self.current_main_index + 1) % len(MAIN_PHASES)
            apply_phase(TLS_ID, MAIN_PHASES[self.current_main_index])
            self.red_queue_integral = 0


# =========================
# Main
# =========================
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
    print(f"MP/SOTL evaluation on {len(route_files)} days ({args.days})")

    all_results = []
    summary_rows = []

    for controller in [MaxPressureController(), SotlController()]:
        controller_results = []

        for route_file in route_files:
            print(f"Running {controller.name} with {route_file}")
            result = ec.run_eval_episode(controller, route_file)
            controller_results.append(result)
            all_results.append(result)

        summary = ec.average_results(controller_results)
        summary_rows.append({"controller": controller.name, **summary})

        print(f"\n{controller.name} all-days average ({summary['num_days']} days)")
        for key in ec.METRIC_KEYS:
            print(f"  {key}: {summary[key]:.4f}")
        print()

    with open("controller_all_days_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    with open("controller_comparison_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Saved: controller_all_days_results.csv, controller_comparison_results.csv")


if __name__ == "__main__":
    main()
