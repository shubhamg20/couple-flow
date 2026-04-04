#python scripts/environments/teleoperation/normalize_gr1_to_franka.py --task banana --episode 4
#visualiations will be saved in source/recorded_runs/gr1_wrapped/<task>/visuals/

#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from dtaidistance import dtw
    DTW_LIBRARY = "dtaidistance"
except ImportError:
    from fastdtw import fastdtw
    DTW_LIBRARY = "fastdtw"

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {pkl_path}: {len(data['trajectory'])} timesteps")
    return data

def extract_positions(trajectory):
    return np.array([step['franka_eef']['pos'] for step in trajectory])

def extract_gripper(trajectory):
    return np.array([step['franka_eef']['gripper'] for step in trajectory])

def find_gripper_close_index(trajectory):
    gripper_states = extract_gripper(trajectory)
    for i, state in enumerate(gripper_states):
        if state >= 0.9:
            return i
    return len(trajectory) // 2

def compute_dtw_with_task_cost(gr1_positions, franka_positions, reference_point):
    n, m = len(gr1_positions), len(franka_positions)
    gr1_dists = np.linalg.norm(gr1_positions - reference_point, axis=1)
    franka_dists = np.linalg.norm(franka_positions - reference_point, axis=1)
    
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Initialize first column (GR1 advances, Franka at start)
    for i in range(1, n + 1):
        pos_cost = np.linalg.norm(gr1_positions[i-1] - franka_positions[0])
        task_cost = abs(gr1_dists[i-1] - franka_dists[0])
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + pos_cost + task_cost
    
    # Initialize first row (Franka advances, GR1 at start)
    for j in range(1, m + 1):
        pos_cost = np.linalg.norm(gr1_positions[0] - franka_positions[j-1])
        task_cost = abs(gr1_dists[0] - franka_dists[j-1])
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + pos_cost + task_cost
    
    # Fill the rest of the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pos_cost = np.linalg.norm(gr1_positions[i-1] - franka_positions[j-1])
            task_cost = abs(gr1_dists[i-1] - franka_dists[j-1])
            cost = pos_cost + task_cost
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [
            (dtw_matrix[i-1, j-1], i-1, j-1),
            (dtw_matrix[i-1, j], i-1, j),
            (dtw_matrix[i, j-1], i, j-1)
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    path.reverse()
    
    mapping = np.zeros(m, dtype=int)
    for gr1_idx, franka_idx in path:
        mapping[franka_idx] = gr1_idx
    
    return mapping

def resample_trajectory(gr1_trajectory, dtw_mapping, timestep_offset=0):
    normalized = []
    for new_t, gr1_idx in enumerate(dtw_mapping):
        gr1_idx = max(0, min(int(gr1_idx), len(gr1_trajectory) - 1))
        step = gr1_trajectory[gr1_idx].copy()
        step['timestep'] = timestep_offset + new_t
        normalized.append(step)
    return normalized

def normalize_gr1_to_franka(task_name, episode_num):
    base = Path("source/recorded_runs")
    gr1_pkl = base / f"gr1t2/{task_name}/episode{episode_num}.pkl"
    franka_pkl = base / f"franka/{task_name}/episode{episode_num}.pkl"
    output_pkl = base / f"gr1_wrapped/{task_name}/episode{episode_num}.pkl"
    
    print("="*60)
    print(f"Task: {task_name}, Episode: {episode_num}")
    print("="*60)
    
    gr1_data = load_pkl(gr1_pkl)
    franka_data = load_pkl(franka_pkl)

    object_pose = franka_data['trajectory'][0]['objects'][task_name]["pos"]
    goal_pose = np.array([.41, .42, 1.0])

    gr1_close_idx = find_gripper_close_index(gr1_data['trajectory'])
    franka_close_idx = find_gripper_close_index(franka_data['trajectory'])
    
    print(f"Gripper close - GR1: {gr1_close_idx}, Franka: {franka_close_idx}")

    gr1_pick = gr1_data['trajectory'][:gr1_close_idx+1]
    gr1_place = gr1_data['trajectory'][gr1_close_idx+1:]
    franka_pick = franka_data['trajectory'][:franka_close_idx+1]
    franka_place = franka_data['trajectory'][franka_close_idx+1:]

    gr1_pick_pos = extract_positions(gr1_pick)
    franka_pick_pos = extract_positions(franka_pick)
    gr1_place_pos = extract_positions(gr1_place)
    franka_place_pos = extract_positions(franka_place)

    print(f"\nPick phase: GR1={len(gr1_pick_pos)}, Franka={len(franka_pick_pos)}")
    dtw_pick_mapping = compute_dtw_with_task_cost(gr1_pick_pos, franka_pick_pos, object_pose)
    normalized_pick = resample_trajectory(gr1_pick, dtw_pick_mapping, timestep_offset=0)

    if len(gr1_place) > 0 and len(franka_place) > 0:
        print(f"Place phase: GR1={len(gr1_place_pos)}, Franka={len(franka_place_pos)}")
        dtw_place_mapping = compute_dtw_with_task_cost(gr1_place_pos, franka_place_pos, goal_pose)
        normalized_place = resample_trajectory(gr1_place, dtw_place_mapping, timestep_offset=len(normalized_pick))
    else:
        normalized_place = []

    normalized_traj = normalized_pick + normalized_place
    wrapped_pos = extract_positions(normalized_traj)
    
    gr1_pos = extract_positions(gr1_data['trajectory'])
    franka_pos = extract_positions(franka_data['trajectory'])
    visualize_trajectories_3d(gr1_pos, franka_pos, wrapped_pos, output_pkl, gr1_pick_idx=gr1_close_idx, wrapped_pick_idx=len(normalized_pick)-1)
    
    normalized_data = {
        'episode': gr1_data['episode'],
        'trajectory': normalized_traj,
        'initial_objects': gr1_data['initial_objects']
    }
    
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(normalized_data, f)
    
    print(f"\nSaved: {output_pkl}")
    print(f"Lengths: GR1={len(gr1_data['trajectory'])} -> Wrapped={len(normalized_traj)} (Franka={len(franka_data['trajectory'])})")
    print("="*60)

def visualize_trajectories_3d(gr1_pos, franka_pos, wrapped_pos, output_path,
                              gr1_pick_idx=None, wrapped_pick_idx=None):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories as thin lines
    ax.plot(gr1_pos[:, 0], gr1_pos[:, 1], gr1_pos[:, 2], 
            'b-', alpha=0.3, linewidth=1)
    ax.plot(franka_pos[:, 0], franka_pos[:, 1], franka_pos[:, 2], 
            'g-', alpha=0.3, linewidth=1)
    ax.plot(wrapped_pos[:, 0], wrapped_pos[:, 1], wrapped_pos[:, 2], 
            'm-', alpha=0.3, linewidth=1)
    
    # Plot all points
    ax.scatter(gr1_pos[:, 0], gr1_pos[:, 1], gr1_pos[:, 2], 
               c='blue', s=30, alpha=0.6, marker='o', 
               label=f'GR1 ({len(gr1_pos)} pts)')
    ax.scatter(franka_pos[:, 0], franka_pos[:, 1], franka_pos[:, 2], 
               c='green', s=30, alpha=0.6, marker='^', 
               label=f'Franka ({len(franka_pos)} pts)')
    ax.scatter(wrapped_pos[:, 0], wrapped_pos[:, 1], wrapped_pos[:, 2], 
               c='magenta', s=40, alpha=0.7, marker='s', 
               label=f'Wrapped ({len(wrapped_pos)} pts)')
    
    # Mark start points
    ax.scatter(*gr1_pos[0], color='darkblue', s=250, marker='o', 
               edgecolors='black', linewidths=2.5, zorder=10)
    ax.scatter(*franka_pos[0], color='darkgreen', s=250, marker='o', 
               edgecolors='black', linewidths=2.5, zorder=10)
    
    # Mark end points
    ax.scatter(*gr1_pos[-1], color='darkblue', s=250, marker='X', 
               edgecolors='black', linewidths=2.5, zorder=10)
    ax.scatter(*franka_pos[-1], color='darkgreen', s=250, marker='X', 
               edgecolors='black', linewidths=2.5, zorder=10)
    
    # Mark picking points
    if gr1_pick_idx is not None:
        ax.scatter(*gr1_pos[gr1_pick_idx], color='orange', s=250, marker='*', 
                   edgecolors='black', linewidths=2.5, zorder=10)
        ax.scatter(*franka_pos[gr1_pick_idx], color='orange', s=250, marker='*', 
                   edgecolors='black', linewidths=2.5, zorder=10)
    if wrapped_pick_idx is not None:
        ax.scatter(*wrapped_pos[wrapped_pick_idx], color='darkorange', s=250, marker='*', 
                   edgecolors='black', linewidths=2.5, zorder=10)
    
    # Add dummy legend entries for markers
    ax.scatter([], [], color='gray', s=200, marker='o', 
               edgecolors='black', linewidths=2.5, label='Start')
    ax.scatter([], [], color='gray', s=200, marker='X', 
               edgecolors='black', linewidths=2.5, label='End')
    ax.scatter([], [], color='orange', s=200, marker='*', 
               edgecolors='black', linewidths=2.5, label='Pick')
    
    # Timestep mapping text box
    text_str = "GR1 → Wrapped Timesteps:\n"
    text_str += "─" * 30 + "\n"
    if gr1_pick_idx is not None and wrapped_pick_idx is not None:
        text_str += f"PICK:  0→{gr1_pick_idx} → 0→{wrapped_pick_idx}\n"
        text_str += f"       ({gr1_pick_idx+1} → {wrapped_pick_idx+1} steps)\n\n"
        text_str += f"PLACE: {gr1_pick_idx+1}→{len(gr1_pos)-1} → {wrapped_pick_idx+1}→{len(wrapped_pos)-1}\n"
        text_str += f"       ({len(gr1_pos)-gr1_pick_idx-1} → {len(wrapped_pos)-wrapped_pick_idx-1} steps)\n\n"
    text_str += f"TOTAL: {len(gr1_pos)} → {len(wrapped_pos)} steps"
    
    ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title('DTW Trajectory Alignment', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2, 
              columnspacing=1.0, handletextpad=0.5, borderpad=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path.parent / "visuals" / f"{output_path.stem}_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"Plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task name (e.g., banana)")
    parser.add_argument("--episode", type=int, required=True, help="Episode number (e.g., 0)")
    args = parser.parse_args()
    normalize_gr1_to_franka(args.task, args.episode)

if __name__ == "__main__":
    main()