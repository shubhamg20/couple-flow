import torch
from isaaclab.app import AppLauncher
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import imageio
import pathlib
import cv2

# Initialize app launcher first
app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

# Now import isaaclab_tasks to register environments
import isaaclab_tasks
# pick_place is blacklisted in isaaclab_tasks so it's never auto-imported; import it so Isaac-PickPlace-Franka-custom gets registered
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg



def get_image(env, env_idx=None) -> torch.Tensor:
    """Get RGB image from camera."""
    rgb_data = env.scene["tiled_camera"].data.output["rgb"]
    if env_idx is None:
        env_idx = 0
    return rgb_data[env_idx].clone()

class IsaacLabGymEnv(gym.Env):
    def __init__(self, task_name="Isaac-PickPlace-Franka-custom", device="cuda", num_envs=1, cfg=None):
        self.simulation_app = simulation_app
        
        self.device = device
        self.task_name = task_name
        self.env_cfg = parse_env_cfg('Isaac-PickPlace-Franka-custom', device=device, num_envs=num_envs)
        # Headless + RTX sensors: avoid hanging in reset() due to wait_for_textures loop
        self.env_cfg.wait_for_textures = False
        # Force a render after reset so we capture lit frames (otherwise camera buffer is unlit)
        self.env_cfg.rerender_on_reset = True
        # Higher-res camera for video (default is 256x256)
        self.env_cfg.scene.tiled_camera.width = 512
        self.env_cfg.scene.tiled_camera.height = 512
        self.env = gym.make('Isaac-PickPlace-Franka-custom', cfg=self.env_cfg, render_mode="rgb_array").unwrapped
        self.env.reset()

    def reset(self, env_ids=None):
        obs= self.env.reset(env_ids=env_ids)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
        self.simulation_app.close()

def main():
    # Initialize environment
    env = IsaacLabGymEnv(num_envs=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup video output
    video_dir = pathlib.Path("source/online_videos/")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "reset_test_1000.mp4"
    
    frames = []
    num_resets = 100

    print(f"Starting reset test with {num_resets} resets...")
    for i in tqdm(range(num_resets), desc="Recording resets"):
        # Reset the environment
        obs = env.reset()

        # Capture frame
        img = get_image(env.env, 0)
        frame = img.cpu().numpy() if torch.is_tensor(img) else img
        for _ in range(30): frames.append(frame)

    # Save video at 30 fps
    print(f"Saving video to {video_path}...")
    fps = 30
    imageio.mimsave(str(video_path), [frame.astype(np.uint8) for frame in frames], fps=fps)
    print(f"Video saved successfully to {video_path}")
    print(f"Total frames: {len(frames)}, Duration: {len(frames)/fps:.2f} seconds")

    # Cleanup
    env.close()

if __name__ == "__main__":
    main()
