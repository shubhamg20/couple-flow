"""
Detailed walkthrough of dataset sampling with concrete example
"""
import numpy as np

# ============================================================================
# EXAMPLE SETUP: One complete episode (50 timesteps)
# ============================================================================

# Raw episode data (T=50 timesteps)
T = 50
state_data = np.array([[1.0 + i*0.1, 2.0 + i*0.1, 3.0 + i*0.1] for i in range(T)], dtype=np.float32)
action_data = np.array([[10.0 + i*0.5, 20.0 + i*0.5] for i in range(T)], dtype=np.float32)

raw_episode = {
    'state': state_data,
    'action': action_data,
}

print("=" * 80)
print(f"RAW EPISODE DATA (T={T} timesteps)")
print("=" * 80)
print("\nState shape:", raw_episode['state'].shape)
print("State (first 5 and last 5 timesteps):")
print("First 5:\n", raw_episode['state'][:5])
print("Last 5:\n", raw_episode['state'][-5:])
print("\nAction shape:", raw_episode['action'].shape)
print("Action (first 5 and last 5 timesteps):")
print("First 5:\n", raw_episode['action'][:5])
print("Last 5:\n", raw_episode['action'][-5:])

# ============================================================================
# STEP 1: create_sample_indices
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: CREATE_SAMPLE_INDICES")
print("=" * 80)

sequence_length = 8  # pred_horizon = 4
obs_horizon = 3
action_horizon = 4
pad_before = obs_horizon - 1  # = 1
pad_after = action_horizon - 1  # = 0

episode_ends = np.array([50])  # Episode 0 ends at index 50
episode_start_idx = 0
episode_end_idx = 50
episode_length = 50

print(f"\nParameters:")
print(f"  sequence_length (pred_horizon) = {sequence_length}")
print(f"  obs_horizon = {obs_horizon}")
print(f"  action_horizon = {action_horizon}")
print(f"  pad_before = {pad_before}")
print(f"  pad_after = {pad_after}")
print(f"  episode_length = {episode_length}")

min_start = -pad_before  # = -1
max_start = episode_length - sequence_length + pad_after  # = 50 - 8 + 3 = 45

print(f"\nIndex ranges:")
print(f"  min_start = {min_start}")
print(f"  max_start = {max_start}")
print(f"  Iterate idx from {min_start} to {max_start} (inclusive)")
print(f"  Total samples from this episode: {max_start - min_start + 1}")

indices = []
for idx in range(min_start, max_start + 1):
    buffer_start_idx = max(idx, 0) + episode_start_idx
    buffer_end_idx = min(idx + sequence_length, episode_length) + episode_start_idx
    start_offset = buffer_start_idx - (idx + episode_start_idx)
    end_offset = (idx + sequence_length + episode_start_idx) - buffer_end_idx
    sample_start_idx = 0 + start_offset
    sample_end_idx = sequence_length - end_offset
    
    indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    
    print(f"\n  idx={idx}:")
    print(f"    buffer_start_idx={buffer_start_idx}, buffer_end_idx={buffer_end_idx}")
    print(f"    sample_start_idx={sample_start_idx}, sample_end_idx={sample_end_idx}")
    print(f"    Data will be placed at positions [{sample_start_idx}:{sample_end_idx}] in output")

# ============================================================================
# STEP 2: sample_sequence
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: SAMPLE_SEQUENCE - Process each index")
print("=" * 80)

def sample_sequence_debug(train_data, sequence_length, buffer_start_idx, buffer_end_idx, 
                          sample_start_idx, sample_end_idx, verbose=True):
    """Sample with debug output"""
    result = dict()
    for key, input_arr in train_data.items():
        # Extract buffer
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        
        if verbose:
            print(f"\n  Key: '{key}'")
            print(f"    Extracted sample shape: {sample.shape}")
            print(f"    Sample:\n{sample}")
        
        # Check if padding needed
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            if verbose:
                print(f"    PADDING NEEDED: sample_start_idx={sample_start_idx}, sample_end_idx={sample_end_idx}")
            
            # Create padded output
            data = np.zeros((sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            
            # Pad start
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
                if verbose:
                    print(f"    Pad start [0:{sample_start_idx}] = sample[0] = {sample[0]}")
            
            # Pad end
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
                if verbose:
                    print(f"    Pad end [{sample_end_idx}:{sequence_length}] = sample[-1] = {sample[-1]}")
            
            # Place real data
            data[sample_start_idx:sample_end_idx] = sample
            if verbose:
                print(f"    Place data [{sample_start_idx}:{sample_end_idx}]")
        else:
            if verbose:
                print(f"    NO PADDING: data stays as extracted sample")
        
        result[key] = data
        if verbose:
            print(f"    Final output shape: {data.shape}")
            print(f"    Final output:\n{data}")
    
    return result

# Pick a random index from the middle of episode
np.random.seed(42)
random_idx = np.random.randint(len(indices)//3, 2*len(indices)//3)
print(f"\nProcessing random sample from MIDDLE of episode: sample_index={random_idx}")
print(f"(out of {len(indices)} total samples)")

buf_start, buf_end, samp_start, samp_end = indices[random_idx]
print(f"\nIndices for sample {random_idx}:")
print(f"  buffer_start_idx={buf_start}, buffer_end_idx={buf_end}")
print(f"  sample_start_idx={samp_start}, sample_end_idx={samp_end}")
print(f"  This extracts data from raw episode positions [{buf_start}:{buf_end}]")

print(f"\n{'='*60}")
print(f"SAMPLE {random_idx} (FROM MIDDLE OF EPISODE - idx range ~{len(indices)//3}-{2*len(indices)//3}):")
print(f"{'='*60}")
nsample = sample_sequence_debug(raw_episode, sequence_length, buf_start, buf_end, 
                                samp_start, samp_end, verbose=True)

# ============================================================================
# STEP 3: __getitem__
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: __getitem__ - What the model receives")
print("=" * 80)

buf_start, buf_end, samp_start, samp_end = indices[0]  # First sample
nsample = sample_sequence_debug(raw_episode, sequence_length, buf_start, buf_end, 
                                samp_start, samp_end, verbose=False)

print(f"\nAfter sample_sequence:")
print(f"  state shape: {nsample['state'].shape}")
print(f"  action shape: {nsample['action'].shape}")

# Trim to obs_horizon
nsample['state'] = nsample['state'][:obs_horizon, :]
print(f"\nAfter trimming to obs_horizon={obs_horizon}:")
print(f"  state shape: {nsample['state'].shape}")
print(f"  state:\n{nsample['state']}")

print(f"\nFull sequence available (before trimming):")
nsample_full = sample_sequence_debug(raw_episode, sequence_length, buf_start, buf_end, 
                                     samp_start, samp_end, verbose=False)
print(f"  All {sequence_length} timesteps:")
print(f"  state:\n{nsample_full['state']}")
print(f"  action:\n{nsample_full['action']}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
For this episode:
- Raw data: {T} timesteps
- With sequence_length={sequence_length}, obs_horizon={obs_horizon}, action_horizon={action_horizon}
- Generated {len(indices)} training samples
- Sampled from sample #{random_idx} (middle of episode)
- Each sample:
  - Provides {obs_horizon} observation timesteps (used by model)
  - Contains {action_horizon} action timesteps (targets)
  - Has {sequence_length - obs_horizon - action_horizon} future timesteps (for prediction)
  
Data characteristics:
- Extracted from positions [{buf_start}:{buf_end}] in raw episode
- Placed at positions [{samp_start}:{samp_end}] in output (with padding if needed)
- When sample_start_idx={samp_start} > 0: padding at start
- When sample_end_idx={samp_end} < {sequence_length}: padding at end
""")
