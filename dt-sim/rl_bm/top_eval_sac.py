import os 
import torch
import wandb
import numpy as np
import gymnasium as gym
import argparse
from sac_continuous_action import Actor
from utils.env_lunch import EnvLunch
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TD3 Agent in Duckietown")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the .cleanrl_model file")
    parser.add_argument("--env-id", type=str, default=None,
                        help="The name of the Duckietown map")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Whether to render the environment")
    parser.add_argument("--capture-video", type=bool, default=True,
                        help="Capture video of the evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum number of steps for each episode" )
    parser.add_argument("--local", type=bool, default=False,
                        help="Whether the model path is the wandb artifact or local")
    return parser.parse_args()


def world_to_pixel(x, z, h,w , scale=200):

    px = int(x * scale + w / 2)-300
    py = int(z * scale + h / 2)-175

    return px, py

def overlay_trajectory_on_video(video_path, trajectory, output_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    pixel_traj = [world_to_pixel(x, z, h, w) for x, z in trajectory]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw trajectory up to current frame
        for i in range(1, min(frame_idx, len(pixel_traj))):
            cv2.line(frame,
                     pixel_traj[i-1],
                     pixel_traj[i],
                     (0, 0, 255),
                     2)
        if pixel_traj:
            cv2.circle(frame, pixel_traj[0], 5, (0, 255, 0), -1)
        if frame_idx < len(pixel_traj):
            cv2.circle(frame, pixel_traj[frame_idx], 5, (255, 0, 0), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.local: 
        model_path = args.model_path
    else:
        print("Downloading Artifact")
        api = wandb.Api()
        artifact = api.artifact(args.model_path)
        artifact_dir = artifact.download()
        model_path = f"{artifact_dir}/td3_Final.cleanrl_model"
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    #Check for rgb or grayscale
    state_dict = checkpoint['actor_state_dict']
    first_layer = state_dict['encoder.convnet.0.weight']
    grayscale = True if first_layer.shape[1] == 4 else False

    env_id = args.env_id or checkpoint.get("env_id", "oval_loop")

    # Handle randomization toggles (env_params)
    sim_params = {
            "domain_rand": checkpoint.get("domain_rand", False),
            "distortion": checkpoint.get("distortion", False),
            "dynamics_rand": checkpoint.get("dynamics_rand", False),
            "camera_rand": checkpoint.get("camera_rand", False),
        }

    print(f"--- Metadata Extracted ---")
    print(f"Map: {env_id} | Grayscale: {grayscale}")
    print(f"Randomizations: {sim_params}")

    env_luncher = EnvLunch(
        run_name="eval",
        max_steps=4000,
        grayscale=grayscale,
        **sim_params
    )
    env_func = env_luncher.make_env_fn(
        seed=2, 
        idx=0,
        capture_video="rgb_array")

    env = env_func()

    print("env.render_mode:", getattr(env, "render_mode", None))
    print("env.metadata:", getattr(env, "metadata", None))

    path_parts = args.model_path.split('/')
    run_name_short = path_parts[-1].split(':')[0] if not args.local else os.path.basename(args.model_path)
    if args.capture_video:
        video_folder = f"videos/eval/"
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)


        def top_down_render():
            return env.unwrapped.render("top_down")

        env.render = top_down_render

        env = gym.wrappers.RecordVideo(
            env, 
            video_folder, 
            # it tells the wrapper to record if episode_id >= 0
            episode_trigger=lambda x: True, 
            name_prefix="Top-view"
        )
        print(f"Recording videos to {video_folder}")

    class DummyEnv:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    dummy = DummyEnv(env)
    actor = Actor(dummy, grayscale=grayscale).to(device)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    all_rewards = []
    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=2)
        done = False
        episodic_reward = 0

        trajectory = []

        while not done:

             # Prepare observation: (C, H, W) -> (1, C, H, W)
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                # Use mean_action for deterministic evaluation
                _, _, action = actor.get_action(obs_tensor)
            
            action = action.cpu().numpy().reshape(-1)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episodic_reward += reward

            #Store position 
            x, _, z = env.unwrapped.cur_pos
            trajectory.append((x, z))

            if args.render:
                env.unwrapped.render("top_down")
            



        all_rewards.append(episodic_reward)
        print(f"Episode {episode + 1}: Reward = {episodic_reward:.2f}")


        # Get map image
        top_down = env.unwrapped.render("top_down")
        top_down = cv2.cvtColor(top_down, cv2.COLOR_RGB2BGR)

        h, w = top_down.shape[:2]

        pixel_traj = []
        for (x, z) in trajectory:
            px, py = world_to_pixel(x, z, h, w)
            pixel_traj.append((px, py))

        # Draw trajectory
        for i in range(1, len(pixel_traj)):
            cv2.line(
                top_down,
                pixel_traj[i - 1],
                pixel_traj[i],
                (0, 0, 255),
                2
            )

        # Start / End markers
        if len(pixel_traj) > 0:
            cv2.circle(top_down, pixel_traj[0], 6, (0, 255, 0), -1)
            cv2.circle(top_down, pixel_traj[-1], 6, (255, 0, 0), -1)

        # Save
        save_path = f"{video_folder}/episode_{episode}_overlay.png"
        cv2.imwrite(save_path, top_down)

        print("Saved:", save_path)

    env.close()

    print("starting video")
    video_in = f"{video_folder}/Top-view-episode-0.mp4"
    video_out = f"{video_folder}/Top-view-episode-0_overlay.mp4"
    overlay_trajectory_on_video(video_in, trajectory, video_out)


# =================================================
if __name__ == "__main__":
    evaluate()