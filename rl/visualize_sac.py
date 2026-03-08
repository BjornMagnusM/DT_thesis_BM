import torch
import numpy as np
import gymnasium as gym
import argparse
import time
from sac_continuous_action import Actor, make_env

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SAC Agent")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .cleanrl_model")
    parser.add_argument("--map-name", type=str, default="oval_loop", help="Map name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "top_down"], 
                        help="Render view: 'human' (POV) or 'top_down'")
    return parser.parse_args()

def visualize():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Environment with exact training wrappers
    # We use make_env from your training script to ensure frame stacking (12 channels) is identical
    env_thunk = make_env(seed=1, idx=0, capture_video=False, run_name="visualize")
    env = env_thunk()
    
    # 2. Reconstruct Actor
    # Using a dummy class to satisfy the Actor's init which expects env.single_observation_space
    class DummyEnv:
        def __init__(self, e):
            self.single_observation_space = e.observation_space
            self.single_action_space = e.action_space
    
    actor = Actor(DummyEnv(env)).to(device)
    
    # 3. Load Weights
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\n--- Starting Episode {ep+1} ---")
        
        while not done:
            # Prepare observation for CNN: (12, 120, 160) -> (1, 12, 120, 160)
            obs_t = torch.as_tensor(obs, device=device).float() / 255.0
    
            # 2. Add batch dimension: (12, 120, 160) -> (1, 12, 120, 160)
            obs_t = obs_t.unsqueeze(0)
            
            with torch.no_grad():
                # Use mean_action for deterministic evaluation
                _, _, action = actor.get_action(obs_t)
            
            action = action.cpu().numpy().reshape(-1)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            # Render using the specified mode
            env.unwrapped.render(mode=args.render_mode)
            
            # Print periodic status
            if step % 50 == 0:
                # Extract lane info if available (thanks to full_transparency=True)
                sim_info = info.get("Simulator", {})
                lp = sim_info.get("lane_position", {})
                dist = lp.get("dist", "N/A")
                print(f"Step {step} | Reward: {total_reward:.2f} | Dist from Center: {dist}")

        # Final episode summary
        sim_msg = info.get("Simulator", {}).get("msg", "Unknown")
        print(f"Finished Episode {ep+1} | Total Reward: {total_reward:.2f} | Steps: {step}")
        print(f"Reason for termination: {sim_msg}")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    visualize()