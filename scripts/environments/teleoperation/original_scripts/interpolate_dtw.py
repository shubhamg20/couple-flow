#!/usr/bin/env python3
"""
interpolate_dtw.py

Align GR1 episodes to Franka with DTW and interpolate GR1 EE pos + gripper.

Run:
  python scripts/environments/teleoperation/interpolate_dtw.py --task banana --episode 4
  python scripts/environments/teleoperation/interpolate_dtw.py --task banana

Outputs:
  Traj:  source/recorded_runs/gr1_wrapped/<task>/episode<N>.pkl
  Plots: source/recorded_runs/gr1_wrapped/<task>/visuals/episode<N>_comparison.png
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # no

def load_pkl(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {pkl_path}: {len(data['trajectory'])} timesteps")
    return data

def extract_positions(trajectory):
    return np.array([step["franka_eef"]["pos"] for step in trajectory])

def extract_gripper(trajectory):
    return np.array([step["franka_eef"]["gripper"] for step in trajectory])

def find_gripper_close_index(trajectory, threshold: float = 0.033):
    """First index where gripper < threshold, else midpoint."""
    gripper_states = extract_gripper(trajectory)
    for i, state in enumerate(gripper_states):
        if state < threshold:
            return i
    return len(trajectory) // 2

def compute_dtw_with_task_cost(gr1_positions, franka_positions, reference_point):
    """
    DTW on EE positions with extra cost on distance-to-reference.
    Returns list of (i, j) pairs (GR1 idx, Franka idx).
    """
    n, m = len(gr1_positions), len(franka_positions)

    gr1_dists = np.linalg.norm(gr1_positions - reference_point, axis=1)
    franka_dists = np.linalg.norm(franka_positions - reference_point, axis=1)

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # First column: GR1 moves, Franka at 0
    for i in range(1, n + 1):
        pos_cost = np.linalg.norm(gr1_positions[i - 1] - franka_positions[0])
        task_cost = abs(gr1_dists[i - 1] - franka_dists[0])
        dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + pos_cost + task_cost

    # First row: Franka moves, GR1 at 0
    for j in range(1, m + 1):
        pos_cost = np.linalg.norm(gr1_positions[0] - franka_positions[j - 1])
        task_cost = abs(gr1_dists[0] - franka_dists[j - 1])
        dtw_matrix[0, j] = dtw_matrix[0, j - 1] + pos_cost + task_cost

    # Core DP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pos_cost = np.linalg.norm(gr1_positions[i - 1] - franka_positions[j - 1])
            task_cost = abs(gr1_dists[i - 1] - franka_dists[j - 1])
            cost = pos_cost + task_cost
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],     # GR1 moves
                dtw_matrix[i, j - 1],     # Franka moves
                dtw_matrix[i - 1, j - 1]  # both move
            )

    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1)) #yiu dont want to append when its moving vertical
        candidates = [
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
            (dtw_matrix[i - 1, j], i - 1, j),
            (dtw_matrix[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    return path


def path_to_fractional_times(path, n_gr1, m_franka):
    """
    For each Franka index j, compute τ_j as mean GR1 index i along the path.
    Returns np.array of shape (m_franka,).
    """
    buckets = [[] for _ in range(m_franka)]
    for i, j in path:
        if 0 <= j < m_franka:
            buckets[j].append(i)

    taus = np.zeros(m_franka, dtype=float)
    last_tau = 0.0
    for j in range(m_franka):
        if buckets[j]:
            tau = float(np.mean(buckets[j]))
            last_tau = tau
        else:
            tau = last_tau  # fallback if path somehow skipped j
        taus[j] = np.clip(tau, 0.0, max(0.0, n_gr1 - 1.0))
    return taus


# Interpolated resampling  
def resample_trajectory_interpolated(gr1_trajectory, taus, timestep_offset=0):
    """
    Build a new traj with len = len(taus) using GR1 interpolation.
    Interpolates franka_eef.pos and franka_eef.gripper; copies other fields.
    """
    n = len(gr1_trajectory)
    if n == 0:
        return []

    pos = extract_positions(gr1_trajectory)
    grip = extract_gripper(gr1_trajectory)

    normalized = []
    for new_t, tau in enumerate(taus):
        tau = float(np.clip(tau, 0.0, max(0.0, n - 1.0)))
        i0 = int(np.floor(tau))
        i1 = min(i0 + 1, n - 1)
        alpha = tau - i0

        pos_interp = (1.0 - alpha) * pos[i0] + alpha * pos[i1]
        grip_interp = (1.0 - alpha) * grip[i0] + alpha * grip[i1]

        base_step = gr1_trajectory[i0].copy()
        # Copy nested dicts defensively
        base_step["franka_eef"] = base_step["franka_eef"].copy()
        base_step["franka_eef"]["pos"] = pos_interp
        base_step["franka_eef"]["gripper"] = float(grip_interp)
        base_step["timestep"] = timestep_offset + new_t

        normalized.append(base_step)

    return normalized


# Episode normalization

def normalize_gr1_to_franka(task_name, episode_num):
    base = Path("source/recorded_runs")
    gr1_pkl = base / f"gr1t2demos/{task_name}/episode{episode_num}.pkl"
    franka_pkl = base / f"franka_set_2/{task_name}/episode{episode_num}.pkl"
    output_pkl = base / f"gr1t2_wrapped_set_2/{task_name}/episode{episode_num}.pkl"

    print("=" * 60)
    print(f"Task: {task_name}, Episode: {episode_num}")
    print("=" * 60)

    gr1_data = load_pkl(gr1_pkl)
    franka_data = load_pkl(franka_pkl)

    gr1_traj = gr1_data["trajectory"]
    franka_traj = franka_data["trajectory"]

    object_pose = franka_traj[0]["objects"][task_name]["pos"]
    goal_pose = np.array([0.41, 0.42, 1.0])

    gr1_close_idx = find_gripper_close_index(gr1_traj)
    franka_close_idx = find_gripper_close_index(franka_traj)

    print(f"Gripper close index - GR1: {gr1_close_idx}, Franka: {franka_close_idx}")

    gr1_pick = gr1_traj[: gr1_close_idx + 1]
    gr1_place = gr1_traj[gr1_close_idx + 1 :]
    franka_pick = franka_traj[: franka_close_idx + 1]
    franka_place = franka_traj[franka_close_idx + 1 :]

    gr1_pick_pos = extract_positions(gr1_pick)
    franka_pick_pos = extract_positions(franka_pick)
    gr1_place_pos = extract_positions(gr1_place)
    franka_place_pos = extract_positions(franka_place)

    # Pick phase
    print(f"\nPick phase: GR1={len(gr1_pick_pos)}, Franka={len(franka_pick_pos)}")
    pick_path = compute_dtw_with_task_cost(gr1_pick_pos, franka_pick_pos, object_pose)
    pick_taus = path_to_fractional_times(pick_path, len(gr1_pick_pos), len(franka_pick_pos))
    normalized_pick = resample_trajectory_interpolated(gr1_pick, pick_taus, timestep_offset=0)

    # Place phase
    if len(gr1_place) > 0 and len(franka_place) > 0:
        print(f"Place phase: GR1={len(gr1_place_pos)}, Franka={len(franka_place_pos)}")
        place_path = compute_dtw_with_task_cost(gr1_place_pos, franka_place_pos, goal_pose)
        place_taus = path_to_fractional_times(place_path, len(gr1_place_pos), len(franka_place_pos))
        normalized_place = resample_trajectory_interpolated(
            gr1_place, place_taus, timestep_offset=len(normalized_pick)
        )
    else:
        normalized_place = []

    normalized_traj = normalized_pick + normalized_place
    wrapped_pos = extract_positions(normalized_traj)

    gr1_pos = extract_positions(gr1_traj)
    franka_pos = extract_positions(franka_traj)

    visualize_trajectories_3d(
        gr1_pos,
        franka_pos,
        wrapped_pos,
        output_pkl,
        gr1_pick_idx=gr1_close_idx,
        wrapped_pick_idx=len(normalized_pick) - 1,
    )

    normalized_data = {
        "episode": gr1_data["episode"],
        "trajectory": normalized_traj,
        "initial_objects": gr1_data["initial_objects"],
    }

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(normalized_data, f)

    print(f"\nSaved: {output_pkl}")
    print(
        f"Lengths: GR1={len(gr1_traj)} -> Wrapped={len(normalized_traj)} "
        f"(Franka={len(franka_traj)})"
    )
    print("=" * 60)


# Episode discovery for task

def find_available_episodes(task_name):
    """
    Find episodes present for both GR1 and Franka for a task.
    Looks for files named episode<N>.pkl.
    """
    base = Path("source/recorded_runs")
    gr1_dir = base / f"gr1t2demos/{task_name}"
    franka_dir = base / f"franka_set_2/{task_name}"


    if not gr1_dir.exists() or not franka_dir.exists():
        print(f"[WARN] Missing dirs for task '{task_name}': {gr1_dir}, {franka_dir}")
        return []

    episode_nums = set()
    for p in franka_dir.glob("episode*.pkl"):
        m = re.match(r"episode(\d+)\.pkl", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if (gr1_dir / p.name).exists():
            episode_nums.add(ep)

    return sorted(episode_nums)


# Visualization

def visualize_trajectories_3d(
    gr1_pos,
    franka_pos,
    wrapped_pos,
    output_path,
    gr1_pick_idx=None,
    wrapped_pick_idx=None,
):
    """3D plot of GR1, Franka, and wrapped GR1, saved next to output_path."""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gr1_pos[:, 0], gr1_pos[:, 1], gr1_pos[:, 2], "b-", alpha=0.3, linewidth=1)
    ax.plot(franka_pos[:, 0], franka_pos[:, 1], franka_pos[:, 2], "g-", alpha=0.3, linewidth=1)
    ax.plot(wrapped_pos[:, 0], wrapped_pos[:, 1], wrapped_pos[:, 2], "m-", alpha=0.3, linewidth=1)

    ax.scatter(
        gr1_pos[:, 0],
        gr1_pos[:, 1],
        gr1_pos[:, 2],
        c="blue",
        s=30,
        alpha=0.6,
        marker="o",
        label=f"GR1 ({len(gr1_pos)} pts)",
    )
    ax.scatter(
        franka_pos[:, 0],
        franka_pos[:, 1],
        franka_pos[:, 2],
        c="green",
        s=30,
        alpha=0.6,
        marker="^",
        label=f"Franka ({len(franka_pos)} pts)",
    )
    ax.scatter(
        wrapped_pos[:, 0],
        wrapped_pos[:, 1],
        wrapped_pos[:, 2],
        c="magenta",
        s=40,
        alpha=0.7,
        marker="s",
        label=f"Wrapped ({len(wrapped_pos)} pts)",
    )

    # Start points
    ax.scatter(
        *gr1_pos[0],
        color="darkblue",
        s=250,
        marker="o",
        edgecolors="black",
        linewidths=2.5,
        zorder=10,
    )
    ax.scatter(
        *franka_pos[0],
        color="darkgreen",
        s=250,
        marker="o",
        edgecolors="black",
        linewidths=2.5,
        zorder=10,
    )

    # End points
    ax.scatter(
        *gr1_pos[-1],
        color="darkblue",
        s=250,
        marker="X",
        edgecolors="black",
        linewidths=2.5,
        zorder=10,
    )
    ax.scatter(
        *franka_pos[-1],
        color="darkgreen",
        s=250,
        marker="X",
        edgecolors="black",
        linewidths=2.5,
        zorder=10,
    )

    # Pick markers (uses same index for GR1/Franka as in original script)
    if gr1_pick_idx is not None and 0 <= gr1_pick_idx < len(gr1_pos):
        ax.scatter(
            *gr1_pos[gr1_pick_idx],
            color="orange",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=2.5,
            zorder=10,
        )
        ax.scatter(
            *franka_pos[gr1_pick_idx],
            color="orange",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=2.5,
            zorder=10,
        )
    if wrapped_pick_idx is not None and 0 <= wrapped_pick_idx < len(wrapped_pos):
        ax.scatter(
            *wrapped_pos[wrapped_pick_idx],
            color="darkorange",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=2.5,
            zorder=10,
        )

    # Legend markers
    ax.scatter([], [], color="gray", s=200, marker="o", edgecolors="black", linewidths=2.5, label="Start")
    ax.scatter([], [], color="gray", s=200, marker="X", edgecolors="black", linewidths=2.5, label="End")
    ax.scatter([], [], color="orange", s=200, marker="*",
               edgecolors="black", linewidths=2.5, label="Pick")

    text_str = "GR1 → Wrapped Timesteps:\n"
    text_str += "─" * 30 + "\n"
    if gr1_pick_idx is not None and wrapped_pick_idx is not None:
        text_str += f"PICK:  0→{gr1_pick_idx} → 0→{wrapped_pick_idx}\n"
        text_str += f"       ({gr1_pick_idx + 1} → {wrapped_pick_idx + 1} steps)\n\n"
        text_str += (
            f"PLACE: {gr1_pick_idx + 1}→{len(gr1_pos) - 1} → "
            f"{wrapped_pick_idx + 1}→{len(wrapped_pos) - 1}\n"
        )
        text_str += (
            f"       ({len(gr1_pos) - gr1_pick_idx - 1} → "
            f"{len(wrapped_pos) - wrapped_pick_idx - 1} steps)\n\n"
        )
    text_str += f"TOTAL: {len(gr1_pos)} → {len(wrapped_pos)} steps"

    ax.text2D(
        0.02,
        0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )

    ax.set_xlabel("X (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (m)", fontsize=11, fontweight="bold")
    ax.set_zlabel("Z (m)", fontsize=11, fontweight="bold")
    ax.set_title("DTW Trajectory Alignment", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2,
              columnspacing=1.0, handletextpad=0.5, borderpad=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path.parent / "visuals" / f"{output_path.stem}_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task name (e.g., banana)")
    parser.add_argument(
        "--episode",
        type=int,
        help="Episode number (e.g., 4). If omitted, process all episodes.",
    )
    args = parser.parse_args()

    if args.episode is not None:
        normalize_gr1_to_franka(args.task, args.episode)
    else:
        episodes = find_available_episodes(args.task)
        if not episodes:
            print(f"No matching episodes found for task '{args.task}'.")
            return
        print(f"Processing episodes for task '{args.task}': {episodes}")
        for ep in episodes:
            try:
                normalize_gr1_to_franka(args.task, ep)
            except Exception as e:
                print(f"[ERROR] Failed to normalize task={args.task}, episode={ep}: {e}")


if __name__ == "__main__":
    main()
