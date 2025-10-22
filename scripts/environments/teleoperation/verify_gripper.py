import numpy as np
from pathlib import Path
import pickle

episodes_dir = Path("source/recorded_runs/gr1t2/sushi/")
episode_files = sorted(episodes_dir.glob("episode*.pkl"))

for episode_file in episode_files:

    with open(episode_file, 'rb') as f:
        episode_data = pickle.load(f)

    gripper_values = [int(step['franka_eef']['gripper']) for step in episode_data['trajectory']]
    
    print(f"{episode_file.name}: {gripper_values}")


# import numpy as np
# from pathlib import Path
# import pickle
# import time
# import sys

# # Get one episode file to analyze
# episode_file = Path("source/recorded_runs/gr1t2/sushi/episode0.pkl")

# # Check file size
# file_size_mb = episode_file.stat().st_size / (1024 * 1024)
# print(f"File size: {file_size_mb:.2f} MB")

# # Time the loading
# start = time.time()
# with open(episode_file, 'rb') as f:
#     episode_data = pickle.load(f)
# load_time = time.time() - start
# print(f"Load time: {load_time:.2f} seconds")

# # Analyze the data structure
# print(f"\nTop-level keys: {episode_data.keys()}")
# print(f"Number of steps: {len(episode_data['trajectory'])}")

# # Check size of each component in first step
# if len(episode_data['trajectory']) > 0:
#     first_step = episode_data['trajectory'][0]
#     print(f"\nFirst step keys: {first_step.keys()}")
    
#     print("\nMemory usage per step component:")
#     for key, value in first_step.items():
#         if isinstance(value, dict):
#             for subkey, subvalue in value.items():
#                 if isinstance(subvalue, np.ndarray):
#                     size_mb = subvalue.nbytes / (1024 * 1024)
#                     print(f"  {key}['{subkey}']: shape={subvalue.shape}, dtype={subvalue.dtype}, size={size_mb:.2f} MB")
#                 else:
#                     print(f"  {key}['{subkey}']: {type(subvalue).__name__}, size={sys.getsizeof(subvalue)} bytes")
#         elif isinstance(value, np.ndarray):
#             size_mb = value.nbytes / (1024 * 1024)
#             print(f"  {key}: shape={value.shape}, dtype={value.dtype}, size={size_mb:.2f} MB")
#         else:
#             print(f"  {key}: {type(value).__name__}")

# # Calculate total memory per episode
# total_size = 0
# for step in episode_data['trajectory']:
#     for key, value in step.items():
#         if isinstance(value, dict):
#             for subvalue in value.values():
#                 if isinstance(subvalue, np.ndarray):
#                     total_size += subvalue.nbytes
#         elif isinstance(value, np.ndarray):
#             total_size += value.nbytes

# print(f"\nTotal array data size: {total_size / (1024 * 1024):.2f} MB")