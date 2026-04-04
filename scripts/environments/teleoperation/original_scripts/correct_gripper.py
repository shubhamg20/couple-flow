#!/usr/bin/env python3
"""Interactive gripper event annotator for pick-and-place episodes.

Usage: python gripper_annotator.py --task_path source/recorded_runs/franka/sushi

Controls:
  Right Arrow / D / ]: Next frame / Skip 10 frames forward
  Left Arrow / A / [: Previous frame / Skip 10 frames backward
  C: Mark CLOSE event (gripper closes)
  O: Mark OPEN event (gripper opens)
  S: Save episode and move to next
  Q: Quit current episode without saving
  Space: Play/Pause (auto-play at 30fps)
"""

import pickle
import argparse
import cv2
import numpy as np
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Annotate gripper events")
parser.add_argument("--task_path", type=str, required=True, help="Path to task folder")
args = parser.parse_args()

task_path = Path(args.task_path)
task_name = task_path.name
output_path = task_path.parent / f"edited_{task_name}"
output_path.mkdir(parents=True, exist_ok=True)

# Get all episodes
episode_files = sorted(task_path.glob("episode*.pkl"))
print(f"\nFound {len(episode_files)} episodes in {task_path}")
print(f"Output directory: {output_path}\n")

# Process each episode
for ep_idx, pkl_file in enumerate(episode_files):
    print(f"\n{'='*60}")
    print(f"Episode {ep_idx+1}/{len(episode_files)}: {pkl_file.name}")
    print(f"{'='*60}")
    
    # Check if already processed
    output_file = output_path / pkl_file.name
    if output_file.exists():
        print(f"✓ Already processed, skipping...")
        continue
    
    # Load episode
    with open(pkl_file, 'rb') as f:
        episode_data = pickle.load(f)
    
    trajectory = episode_data['trajectory']
    total_frames = len(trajectory)
    
    # Decode all frames
    frames = []
    print(f"Loading {total_frames} frames...")
    for step in trajectory:
        if 'image' in step and step['image'] is not None:
            img_data = step['image']
            if isinstance(img_data, (bytes, bytearray)):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = np.array(img_data, dtype=np.uint8)[..., ::-1].copy()
            frames.append(img)
        else:
            frames.append(None)
    
    
    # Interactive annotation
    current_frame = 0
    close_frame = None
    open_frame = None
    playing = False
    
    print("\nControls:")
    print("  Arrow Keys / A,D / [,]: Navigate")
    print("  C: Mark Close | O: Mark Open")
    print("  S: Save & Next | Q: Skip episode")
    print("  Space: Play/Pause\n")
    
    while True:
        if frames[current_frame] is None:
            current_frame = (current_frame + 1) % total_frames
            continue
        
        # Display frame
        display = frames[current_frame].copy()
        h, w = display.shape[:2]
        
        # Determine current gripper state based on marks
        if close_frame is not None and open_frame is not None:
            if current_frame < close_frame:
                gripper_state = 0
                state_color = (255, 100, 0)  # Blue
            elif current_frame < open_frame:
                gripper_state = 1
                state_color = (0, 0, 255)  # Red
            else:
                gripper_state = 0
                state_color = (255, 100, 0)  # Blue
        else:
            gripper_state = -1
            state_color = (128, 128, 128)  # Gray
        
        # Draw status overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (10, h-80), (w-20, h-10), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

        # Status text
        cv2.putText(display, f"Frame: {current_frame}/{total_frames-1}", 
                (20, h-85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if close_frame is not None:
            cv2.putText(display, f"CLOSE: Frame {close_frame}", 
                    (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(display, "CLOSE: Not marked (press C)", 
                    (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        if open_frame is not None:
            cv2.putText(display, f"OPEN: Frame {open_frame}", 
                    (20, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        else:
            cv2.putText(display, "OPEN: Not marked (press O)", 
                    (20, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        if gripper_state != -1:
            gripper_text = f"Gripper: {'CLOSED' if gripper_state == 1 else 'OPEN'}"
            cv2.putText(display, gripper_text, 
                    (20, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
            
        # Draw gripper state indicator bar
        bar_height = 8
        bar_y = h - 30
        if close_frame is not None and open_frame is not None:
            # Draw state segments
            seg1_width = int(w * close_frame / total_frames)
            seg2_width = int(w * (open_frame - close_frame) / total_frames)
            
            cv2.rectangle(display, (0, bar_y), (seg1_width, bar_y + bar_height), (255, 100, 0), -1)
            cv2.rectangle(display, (seg1_width, bar_y), (seg1_width + seg2_width, bar_y + bar_height), (0, 0, 255), -1)
            cv2.rectangle(display, (seg1_width + seg2_width, bar_y), (w, bar_y + bar_height), (255, 100, 0), -1)
        
        # Current position indicator
        pos_x = int(w * current_frame / total_frames)
        cv2.line(display, (pos_x, bar_y), (pos_x, bar_y + bar_height), (0, 255, 0), 2)
        
        cv2.imshow('Gripper Annotator', display)
        
        # Handle keyboard
        key = cv2.waitKey(33 if playing else 0) & 0xFF
        
        if key == ord('q'):  # Quit without saving
            print("Skipped episode (not saved)")
            break
        
        elif key == ord('s'):  # Save and next
            if close_frame is None or open_frame is None:
                print("\nWarning: Both CLOSE and OPEN must be marked!")
                continue
            
            # Update gripper values
            print(f"\nUpdating gripper values...")
            print(f"  Frames 0-{close_frame-1}: gripper = 0 (open)")
            print(f"  Frames {close_frame}-{open_frame-1}: gripper = 1 (closed)")
            print(f"  Frames {open_frame}-{total_frames-1}: gripper = 0 (open)")
            
            for i, step in enumerate(trajectory):
                if 'franka_eef' in step:
                    if i < close_frame:
                        step['franka_eef']['gripper'] = 0
                    elif i < open_frame:
                        step['franka_eef']['gripper'] = 1
                    else:
                        step['franka_eef']['gripper'] = 0
            
            # Save edited episode
            output_file = output_path / pkl_file.name
            with open(output_file, 'wb') as f:
                pickle.dump(episode_data, f)
            print(f"✓ Saved to: {output_file}")
            break
        
        elif key == ord('c'):  # Mark close
            close_frame = current_frame
            print(f"✓ Marked CLOSE at frame {current_frame}")
        
        elif key == ord('o'):  # Mark open
            open_frame = current_frame
            print(f"✓ Marked OPEN at frame {current_frame}")
        
        elif key == ord(' '):  # Toggle play/pause
            playing = not playing
            print(f"{'Playing' if playing else 'Paused'}")
        
        elif key == 83 or key == ord('d') or key == ord(']'):  # Right arrow / D / ]
            if key == ord('d') or key == ord(']'):
                current_frame = min(current_frame + 10, total_frames - 1)
            else:
                current_frame = min(current_frame + 1, total_frames - 1)
        
        elif key == 81 or key == ord('a') or key == ord('['):  # Left arrow / A / [
            if key == ord('a') or key == ord('['):
                current_frame = max(current_frame - 10, 0)
            else:
                current_frame = max(current_frame - 1, 0)
        
        # Auto-advance if playing
        if playing:
            current_frame = (current_frame + 1) % total_frames
    
    cv2.destroyAllWindows()

print(f"\n{'='*60}")
print("Annotation complete!")
print(f"Edited episodes saved to: {output_path}")
print(f"{'='*60}\n")