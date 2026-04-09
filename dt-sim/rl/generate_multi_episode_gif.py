import os
import torch
import numpy as np
import gymnasium as gym
import imageio
from rl.sac_continuous_action import Actor, make_env

def generate_multi_episode_gif():
    # --- CONFIG ---
    MODEL_PATH = "runs/sac_step_step_450000.cleanrl_model"
    OUTPUT_NAME = "duckiebot_multi_eval.gif"
    NUM_EPISODES = 3      # Number of episodes to capture
    MAX_STEPS_PER_EP = 500 # Maximum length of a single episode
    SEED = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- ENV SETUP ---
    # make_env() returns a thunk that creates the env with all wrappers
    env_func = make_env(seed=SEED, idx=0, capture_video=False, run_name="multi_gif_gen")
    env = env_func()
    
    # Manually add attributes for the Actor class initialization
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    
    actor = Actor(env).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # --- RECORDING ---
    all_frames = []
    
    for episode in range(NUM_EPISODES):
        print(f"Starting Episode {episode + 1}/{NUM_EPISODES}...")
        # Reset the environment for each new episode
        obs, _ = env.reset()
        
        for step in range(MAX_STEPS_PER_EP):
            # Capture the raw RGB array from the simulator
            # Mode 'rgb_array' returns a numpy array of the current state
            frame = env.unwrapped.render() 
            frame = np.flipud(frame)
            all_frames.append(frame)

            # Get action from model using deterministic mean_action
            with torch.no_grad():
                obs_t = torch.Tensor(obs).unsqueeze(0).to(device)
                _, _, actions = actor.get_action(obs_t)
                action = actions.cpu().numpy()[0]

            # Step through the environment logic
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode {episode + 1} ended at step {step}.")
                break

    # --- SAVE ---
    if all_frames:
        print(f"Compiling GIF with {len(all_frames)} frames: {OUTPUT_NAME}...")
        imageio.mimsave(OUTPUT_NAME, all_frames, fps=30)
        print("Done!")
    else:
        print("No frames captured.")
        
    env.close()

if __name__ == "__main__":
    generate_multi_episode_gif()