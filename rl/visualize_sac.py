import argparse
import torch
import numpy as np
import gymnasium as gym
from gym_duckietown.envs import DuckietownEnv
from utils.wrappers import NormalizeWrapper, ImgWrapper, ActionWrapper, ResizeWrapper, DtRewardWrapper
from rl.sac_continuous_action import Actor

def enjoy():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-name", default="oval_loop")
    parser.add_argument("--checkpoint", required=True, help="Path to .cleanrl_model file")
    parser.add_argument("--seed", default=123, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Recreate the EXACT environment stack from training
    def make_eval_env():
        # 1. Initializing the Duckietown env
        #env = launch_env() # Output: (480, 640, 3)
        env = DuckietownEnv(
            seed=123,  # random seed
            map_name="oval_loop",
            max_steps=1000,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=160,
            camera_height=120,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=False,
        )
        print("Initialized environment")



        # 4. Wrappers
        #env = ResizeWrapper(env)
        
        print(f"Observation space before ImgWrapper: {env.observation_space.shape}")
        env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
        print(f"Observation space after ImgWrapper: {env.observation_space.shape}")

        # We don't need nomailiztion since we are saving them in uint8 and these will probably make them zero
        # env = NormalizeWrapper(env)

        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        print("Initialized Wrappers")

        # 5. Stack 4 frames
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        print(f"Observation space after stacking: {env.observation_space.shape}")

        #Flatten the 4x3 channels into 12 for the DQNEncoder
        new_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(12, 120, 160), dtype=np.uint8)
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: obs.reshape(12, 120, 160),
            observation_space=new_obs_space  
        )   

        return env
    env = make_eval_env()
    
    # 2. Load the Actor Model
    # We pass the env to Actor so it knows the action/obs shapes
    # Note: Wrap in a dummy object to mimic SyncVectorEnv's 'single_action_space'
    class DummyEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
    
    actor = Actor(DummyEnvs(env)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    print(f"Loaded model from {args.checkpoint}")

    # 3. Main Loop
    obs, _ = env.reset(seed=args.seed)
    
    try:
        while True:
            # Prepare observation for torch (Batch, C, H, W)
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Get the mean action (deterministic) for evaluation
                _, _, action = actor.get_action(obs_tensor)
                action = action.cpu().numpy().reshape(-1)

            obs, reward, done, truncated, info = env.step(action)
            
            # Render for human viewing
            # We use 'top_down' or 'human' (first-person)
            env.render()

            if done or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("Closing...")
    finally:
        env.close()

if __name__ == "__main__":
    enjoy()