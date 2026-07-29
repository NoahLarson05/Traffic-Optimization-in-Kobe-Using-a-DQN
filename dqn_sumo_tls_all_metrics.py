import os
import json
import csv
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import exp_common as ec
from exp_common import traci


# =========================
# DQN Network
# =========================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, *transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# =========================
# State
# =========================
STATE_SIZE = 11
ACTION_SIZE = 2


def get_state():
    state = np.array([
        ec.get_queue(ec.WEST_TO_EAST_LANES) / 20.0,
        ec.get_queue(ec.EAST_TO_WEST_LANES) / 20.0,
        ec.get_queue(ec.TSUKIMI_LANES) / 20.0,
        ec.get_queue(ec.KITA_LANES) / 20.0,
        ec.get_queue(ec.LOCAL_LANES) / 20.0,

        ec.get_waiting(ec.WEST_TO_EAST_LANES) / 500.0,
        ec.get_waiting(ec.EAST_TO_WEST_LANES) / 500.0,
        ec.get_waiting(ec.TSUKIMI_LANES) / 500.0,
        ec.get_waiting(ec.KITA_LANES) / 500.0,
        ec.get_waiting(ec.LOCAL_LANES) / 500.0,

        traci.trafficlight.getPhase(ec.TLS_ID) / 10.0,
    ], dtype=np.float32)
    return state


def apply_action(action):
    """action 0 = keep current phase, action 1 = advance to next phase."""
    if action == 0:
        return
    current = traci.trafficlight.getPhase(ec.TLS_ID)
    total = len(traci.trafficlight.getAllProgramLogics(ec.TLS_ID)[0].phases)
    traci.trafficlight.setPhase(ec.TLS_ID, (current + 1) % total)


class DQNController(ec.BaseController):
    name = "dqn"

    def __init__(self, policy_net, action_interval):
        self.policy_net = policy_net
        self.action_interval = action_interval

    def before_step(self, step):
        if step % self.action_interval == 0:
            state = get_state()
            with torch.no_grad():
                q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(q).item()
            apply_action(action)


# =========================
# Training
# =========================
def train(cfg):
    train_files, val_files = ec.get_train_val_split()
    print(f"Train days: {len(train_files)}  Validation days: {len(val_files)}")
    print("Validation days:", val_files)

    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    policy_net = DQN(STATE_SIZE, ACTION_SIZE)
    target_net = DQN(STATE_SIZE, ACTION_SIZE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["lr"])
    memory = ReplayMemory()

    epsilon = 1.0
    if cfg["epsilon_decay"] == "auto":
        horizon = max(1, int(cfg["episodes"] * 0.8))
        epsilon_decay = cfg["epsilon_min"] ** (1.0 / horizon)
    else:
        epsilon_decay = float(cfg["epsilon_decay"])

    interval = cfg["action_interval"]
    prefix = cfg["prefix"]

    episode_csv = prefix + "dqn_episode_results.csv"
    with open(episode_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "route_file", "total_reward", "epsilon",
            "avg_halting_vehicles", "avg_waiting_time",
            "episode_mean_total_delay", "avg_traffic_density_veh_per_km",
            "throughput_vehicles", "simulation_time",
        ])

        for episode in range(cfg["episodes"]):
            route_file = rng.choice(train_files)

            traci.start(ec.sumo_cmd(route_file))
            lanes = ec.get_controlled_lanes()
            lane_km = ec.get_total_lane_length_km(lanes)

            for _ in range(cfg["warmup_steps"]):
                traci.simulationStep()

            state = get_state()
            total_reward = 0.0

            ep_sums = {"halting": 0.0, "waiting": 0.0, "delay": 0.0, "density": 0.0}
            throughput = 0
            steps = 0
            done = False

            while not done and steps < cfg["max_steps"]:
                if rng.random() < epsilon:
                    action = rng.randint(0, ACTION_SIZE - 1)
                else:
                    with torch.no_grad():
                        q = policy_net(torch.FloatTensor(state).unsqueeze(0))
                        action = torch.argmax(q).item()

                apply_action(action)

                w_halt_sum = w_wait_sum = w_delay_sum = 0.0
                w_arrived = 0
                w_steps = 0

                for _ in range(interval):
                    traci.simulationStep()
                    steps += 1

                    halting, waiting, delay, density = ec.compute_metrics(lanes, lane_km)
                    arrived = traci.simulation.getArrivedNumber()

                    w_halt_sum += halting
                    w_wait_sum += waiting
                    w_delay_sum += delay
                    w_arrived += arrived
                    w_steps += 1

                    ep_sums["halting"] += halting
                    ep_sums["waiting"] += waiting
                    ep_sums["delay"] += delay
                    ep_sums["density"] += density
                    throughput += arrived

                    if traci.simulation.getMinExpectedNumber() == 0:
                        done = True
                        break

                n = max(w_steps, 1)
                reward = (
                    - cfg["w_delay"] * (w_delay_sum / n)
                    - cfg["w_halt"] * (w_halt_sum / n)
                    - cfg["w_wait"] * (w_wait_sum / n)
                    + cfg["w_arr"] * w_arrived
                )

                next_state = get_state()
                memory.push(state, action, reward, next_state, float(done))
                state = next_state
                total_reward += reward

                if len(memory) >= cfg["batch_size"]:
                    batch = memory.sample(cfg["batch_size"])
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.FloatTensor(np.array(states))
                    actions = torch.LongTensor(actions).unsqueeze(1)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(np.array(next_states))
                    dones = torch.FloatTensor(dones).unsqueeze(1)

                    current_q = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q = rewards + cfg["gamma"] * max_next_q * (1 - dones)

                    loss = nn.MSELoss()(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            sim_time = traci.simulation.getTime()
            traci.close()

            n = max(steps, 1)
            epsilon = max(cfg["epsilon_min"], epsilon * epsilon_decay)

            if (episode + 1) % cfg["target_update"] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            writer.writerow([
                episode + 1, route_file, total_reward, epsilon,
                ep_sums["halting"] / n, ep_sums["waiting"] / n,
                ep_sums["delay"] / n, ep_sums["density"] / n,
                throughput, sim_time,
            ])
            f.flush()

            print(
                f"Episode {episode+1}/{cfg['episodes']} [{route_file}] "
                f"reward={total_reward:.1f} eps={epsilon:.3f} "
                f"halt={ep_sums['halting']/n:.2f} wait={ep_sums['waiting']/n:.1f} "
                f"delay={ep_sums['delay']/n:.2f} thru={throughput}"
            )

    model_path = prefix + "dqn_traffic_model.pth"
    torch.save(policy_net.state_dict(), model_path)
    with open(prefix + "dqn_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Model saved:", model_path)

    return policy_net, val_files


# =========================
# Validation
# =========================
def validate(policy_net, val_files, cfg):
    controller = DQNController(policy_net, cfg["action_interval"])
    prefix = cfg["prefix"]

    results = []
    for route_file in val_files:
        print("Validating on:", route_file)
        result = ec.run_eval_episode(controller, route_file)
        results.append(result)
        print(
            f"  halt={result['average_halting_vehicles']:.2f} "
            f"wait={result['average_waiting_time']:.1f} "
            f"delay={result['episode_mean_total_delay']:.2f} "
            f"density={result['average_traffic_density_veh_per_km']:.2f} "
            f"thru={result['throughput_vehicles']}"
        )

    with open(prefix + "dqn_validation_all_days_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = ec.average_results(results)
    with open(prefix + "dqn_validation_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + ec.METRIC_KEYS + ["num_validation_days"])
        writer.writerow(["dqn"] + [summary[k] for k in ec.METRIC_KEYS] + [summary["num_days"]])

    print("\nDQN validation average over", summary["num_days"], "days")
    for key in ec.METRIC_KEYS:
        print(f"  {key}: {summary[key]:.4f}")

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="DQN SUMO traffic light trainer")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=ec.WARMUP_STEPS)
    p.add_argument("--max-steps", type=int, default=ec.MAX_STEPS)
    p.add_argument("--action-interval", type=int, default=30)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", default="auto",
                   help="'auto' = reach epsilon-min at 80%% of episodes, or a float like 0.985")
    p.add_argument("--target-update", type=int, default=10,
                   help="episodes between target network updates")

    p.add_argument("--w-delay", type=float, default=1.0)
    p.add_argument("--w-halt", type=float, default=0.3)
    p.add_argument("--w-wait", type=float, default=0.005)
    p.add_argument("--w-arr", type=float, default=0.3)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="",
                   help="prefix for all output files (e.g. opt_trials/trial_00_)")
    p.add_argument("--validate-only", action="store_true",
                   help="skip training, load --model-in and run validation")
    p.add_argument("--model-in", default="dqn_traffic_model.pth")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "episodes": args.episodes,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "action_interval": args.action_interval,
        "gamma": args.gamma,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "target_update": args.target_update,
        "w_delay": args.w_delay,
        "w_halt": args.w_halt,
        "w_wait": args.w_wait,
        "w_arr": args.w_arr,
        "seed": args.seed,
        "prefix": args.prefix,
    }

    if cfg["prefix"]:
        prefix_dir = os.path.dirname(cfg["prefix"])
        if prefix_dir:
            os.makedirs(prefix_dir, exist_ok=True)

    if args.validate_only:
        policy_net = DQN(STATE_SIZE, ACTION_SIZE)
        policy_net.load_state_dict(torch.load(args.model_in, weights_only=True))
        policy_net.eval()
        _, val_files = ec.get_train_val_split()
        validate(policy_net, val_files, cfg)
        return

    print("DQN training started with config:")
    print(json.dumps(cfg, indent=2))
    policy_net, val_files = train(cfg)
    validate(policy_net, val_files, cfg)


if __name__ == "__main__":
    main()
