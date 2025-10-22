# Usage:
# python /workspace/isaaclab/scripts/environments/teleoperation/pkl_viewer.py --robot franka --task_name sushi --episode 0
# python /workspace/isaaclab/scripts/environments/teleoperation/pkl_viewer.py --robot franka --task_name sushi --episode 0 --output_video
# python /workspace/isaaclab/scripts/environments/teleoperation/pkl_viewer.py --robot franka --task_name sushi --episode 0 --show_preview

"""Load and inspect pkl episode files, extract camera frames into video with EE trajectory overlay."""

import pickle
import argparse
import cv2
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="View and verify pkl episode files.")
parser.add_argument("--robot", type=str, default="franka", help="Robot name")
parser.add_argument("--task_name", type=str, required=True, help="Task name folder")
parser.add_argument("--episode", type=int, required=True, help="Episode number to inspect")
parser.add_argument("--output_video", action="store_true", help="Save frames as video")
parser.add_argument("--show_preview", action="store_true", help="Show live preview of frames")
parser.add_argument("--no_overlay", action="store_true", help="Disable EE trajectory overlay")

args = parser.parse_args()

pkl_file = Path(f"source/recorded_runs/{args.robot}/{args.task_name}/episode{args.episode}.pkl")

if not pkl_file.exists():
    print(f"Error: {pkl_file} not found")
    exit(1)

# Load pkl file
with open(pkl_file, 'rb') as f:
    episode_data = pickle.load(f)

print(f"\n{'='*60}")
print(f"Episode Data Inspector")
print(f"{'='*60}\n")

# Check structure
print(f"Episode: {episode_data['episode']}")
print(f"Timesteps in trajectory: {len(episode_data['trajectory'])}")
print(f"Initial objects: {list(episode_data['initial_objects'].keys())}")

# Inspect first timestep
first_step = episode_data['trajectory'][0]
print(f"\nFirst timestep keys: {list(first_step.keys())}")
if 'action' in first_step:
    print(f"  - action shape: {np.array(first_step['action']).shape}")
print(f"  - EEF pos: {first_step['franka_eef']['pos']}")
print(f"  - gripper: {first_step['franka_eef']['gripper']}")

for step in episode_data['trajectory']:
    print(step['franka_eef']['gripper'])
    # print(step['objects']['sushi'])

# Check for camera data
has_camera_data = 'camera' in first_step
if has_camera_data:
    print(f"  - Camera data: Available")
else:
    print(f"  - Camera data: Not available (overlay disabled)")

# Check for image data (new format uses 'image' key with compressed bytes)
has_images = 'image' in first_step
has_legacy_frames = 'camera_frame' in first_step

print(f"\n{'='*60}")
print(f"Image Data Status")
print(f"{'='*60}")


def project_pose_to_image(ee_pos, camera_data, img_shape):
    """Project 3D EE position to 2D image coordinates."""
    K = np.array(camera_data['intrinsics'])
    R_cam = np.array(camera_data['rotation'])
    t = np.array(camera_data['translation'])
    
    # Transform point from world to camera frame
    pt_cam = R_cam.T @ (ee_pos - t)
    
    # Project to image plane
    if pt_cam[0] > 0:  # Check if point is in front of camera (+X is forward)
        px = K @ np.array([-pt_cam[1], -pt_cam[2], pt_cam[0]])
        u = int(px[0] / px[2])
        v = int(px[1] / px[2])
        
        # Check if projection is within image bounds
        if 0 <= u < img_shape[1] and 0 <= v < img_shape[0]:
            return (u, v)
    
    return None


def draw_trajectory_on_frame(img, all_projections, current_idx):
    """Draw all EE positions up to current frame as green dots."""
    # Draw all previous points
    for i in range(current_idx + 1):
        if all_projections[i] is not None:
            u, v = all_projections[i]
            # Draw green dot
            cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
            
            # Optional: draw line connecting points
            if i > 0 and all_projections[i-1] is not None:
                u_prev, v_prev = all_projections[i-1]
                cv2.line(img, (u_prev, v_prev), (u, v), (0, 255, 0), 1)
    
    # Highlight current position with larger circle
    if all_projections[current_idx] is not None:
        u, v = all_projections[current_idx]
        cv2.circle(img, (u, v), 1, (0, 255, 255), 2)  # Yellow outline
    
    return img


if has_images:
    # New format: compressed JPEG bytes
    img_data = first_step['image']
    print(f"Image format: Compressed JPEG bytes")
    print(f"Image size (bytes): {len(img_data)}")
    
    # Decode first frame to check dimensions
    if isinstance(img_data, (bytes, bytearray)):
        img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    else:
        img_data = np.array(img_data, dtype=np.uint8)
        img_array = img_data[..., ::-1].copy()
    if img_array is not None:
        print(f"Decoded frame shape: {img_array.shape}")
        print(f"Decoded frame dtype: {img_array.dtype}")
        
        # Count frames with data
        frames_with_data = sum(1 for step in episode_data['trajectory'] if 'image' in step and step['image'] is not None)
        frames_missing = len(episode_data['trajectory']) - frames_with_data
        print(f"Frames with data: {frames_with_data}/{len(episode_data['trajectory'])}")
        if frames_missing > 0:
            print(f"Frames missing: {frames_missing}")
        
        # Pre-compute all EE projections if camera data available
        all_projections = []
        all_object_projections = []
        if has_camera_data and not args.no_overlay:
            print(f"\nComputing EE trajectory projections...")
            for step in episode_data['trajectory']:
                if 'camera' in step and 'franka_eef' in step:
                    ee_pos = np.array(step['franka_eef']['pos'])
                    object_pos = step["objects"][args.task_name]['pos']
                    projection = project_pose_to_image(ee_pos, step['camera'], img_array.shape)
                    all_projections.append(projection)
                    obj_projection = project_pose_to_image(object_pos, step['camera'], img_array.shape)
                    all_object_projections.append(obj_projection)
                else:
                    all_projections.append(None)
            
            valid_projections = sum(1 for p in all_projections if p is not None)
            print(f"Valid projections: {valid_projections}/{len(all_projections)}")
        
        # Save video or show preview if requested
        if args.output_video or args.show_preview:
            frames_decoded = []
            print(f"\nDecoding {frames_with_data} frames...")
            
            frame_idx = 0
            for i, step in enumerate(episode_data['trajectory']):
                if 'image' in step and step['image'] is not None:
                    img_data = step['image']

                    if isinstance(img_data, (bytes, bytearray)):
                        img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    else:
                        img_data = np.array(img_data, dtype=np.uint8)
                        img_array = img_data[..., ::-1].copy()
                    
                    if img_array is not None:
                        # Draw trajectory overlay if available
                        if all_projections and not args.no_overlay:
                            img_array = draw_trajectory_on_frame(img_array, all_projections, i)
                            img_array = draw_trajectory_on_frame(img_array, all_object_projections, i)
                        
                        frames_decoded.append(img_array)
                        
                        # Show preview if requested
                        if args.show_preview:
                            cv2.imshow('Episode Replay', img_array)
                            if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 fps
                                break
                    
                    frame_idx += 1
            
            if args.show_preview:
                cv2.destroyAllWindows()
            
            if args.output_video and frames_decoded:
                output_dir = Path(f"source/recorded_runs/{args.robot}/{args.task_name}/videos")
                output_dir.mkdir(parents=True, exist_ok=True)
                video_file = output_dir / f"episode{args.episode}_replay.mp4"
                
                height, width = frames_decoded[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_file), fourcc, 30, (width, height))
                
                print(f"Writing video with {len(frames_decoded)} frames...")
                for frame in frames_decoded:
                    writer.write(frame)
                writer.release()
                
                print(f"\nVideo saved: {video_file}")
                print(f"Video resolution: {width}x{height}")
                print(f"Video duration: {len(frames_decoded)/30:.2f} seconds")
                if all_projections and not args.no_overlay:
                    print(f"EE trajectory overlay: Enabled")
            elif args.output_video:
                print("\nNo frames to save!")
    else:
        raise ValueError("ERROR: Failed to decode first frame!")

elif has_legacy_frames:
    # Legacy format: numpy arrays
    print(f"Image format: Legacy numpy arrays (camera_frame)")
    frame = first_step['camera_frame']
    if frame is not None:
        print(f"Frame type: {type(frame)}")
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        
        # Count frames with data
        frames_with_data = sum(1 for step in episode_data['trajectory'] if step.get('camera_frame') is not None)
        frames_missing = len(episode_data['trajectory']) - frames_with_data
        print(f"Frames with data: {frames_with_data}/{len(episode_data['trajectory'])}")
        if frames_missing > 0:
            print(f"Frames missing: {frames_missing}")
        
        # Save video if requested (no overlay for legacy format)
        if args.output_video:
            output_dir = Path(f"source/recorded_runs/{args.task_name}")
            video_file = output_dir / f"episode{args.episode}_replay.mp4"
            
            frames_to_save = [step['camera_frame'] for step in episode_data['trajectory'] 
                             if step.get('camera_frame') is not None]
            
            if frames_to_save:
                height, width = frames_to_save[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_file), fourcc, 30, (width, height))
                
                for frame in frames_to_save:
                    writer.write(frame)
                writer.release()
                
                print(f"\nVideo saved: {video_file}")
            else:
                print("\nNo frames to save!")
    else:
        print("All camera frames are None")
else:
    print("No image data found in this episode!")
    print(f"Available keys: {list(first_step.keys())}")

print(f"\n{'='*60}\n")