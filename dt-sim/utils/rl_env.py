import os
import gymnasium as gym
import numpy as np
from gym_duckietown.simulator import Simulator
from gymnasium.wrappers import NormalizeReward
from utils.wrappers import (
    KinematicActionWrapper, ActionWrapper, ResizeWrapper, 
    CropResizeWrapper, ImgWrapper, CustomRewardWrapper, DtRewardWrapper,VideoOverlayWrapper,
    LapTerminationWrapperV2,LapTerminationWrapperV3,LapTerminationWrapperV4,
    TimeOptimalRewardV2,TimeOptimalRewardV3
)

class DuckieOvalEnv(Simulator):
    """
    A specialized Duckietown environment for Oval navigation.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('map_name', "oval_loop")
        kwargs.setdefault('camera_width', 160)
        kwargs.setdefault('camera_height', 120)
        kwargs.setdefault('accept_start_angle_deg', 4)
        kwargs.setdefault('full_transparency', True)
        kwargs.setdefault('max_steps', 5000)
        
        kwargs.setdefault('frame_skip', 1) 
        
        super().__init__(**kwargs)
        
        self.wheel_dist = 0.102 
        self.robot_radius = 0.0318
        self.motor_k = 27.0

    def reset(self, **kwargs):
        self.start_pose =  (np.array([1.7, 0, 0.3]),0.0)
        obs, info = super().reset(**kwargs)

        return obs, info


    @classmethod
    def create_wrapped(cls, run_name, capture_video=False, motion_blur=False, grayscale=False, frame_stack=4,max_lap_reward=2000,lap_termination=False, 
                       time_optimal_reward=False , cap_reward=False, norm_reward=False,  **kwargs):
        """
        Static method to build the fully wrapped stack.
        """
        env = cls(**kwargs)

        # Kinematics (v, w -> wl, wr)
        env = KinematicActionWrapper(env, wheel_dist=0.102, radius=0.0318, k=27.0)
        env = ActionWrapper(env)
        

        if capture_video:
            video_folder = f"videos/{run_name}"
            os.makedirs(video_folder, exist_ok=True)
            env = VideoOverlayWrapper(env)
            env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)

        # Vision Pipeline (Sim2Real Insurance)
        env = ResizeWrapper(env, shape=(120, 160, 3)) # Ensure 120x160 base
        env = CropResizeWrapper(env, shape=(84, 84))  # Crop sky, resize to 84x84
        
        if grayscale:
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        
        env = ImgWrapper(env) # Transpose to CHW
        

        #  Reward System
        if time_optimal_reward:
            print("using time optimal reward")
            env = TimeOptimalRewardV3(env)
        

        ##BM added a termination criteria after finishing a lap 
        if lap_termination:
            print("using lap termination")
            env = LapTerminationWrapperV4(env,max_lap_reward=max_lap_reward)



        if cap_reward:
            print("using reward cap")
            env = DtRewardWrapper(env)
        

        if norm_reward:
            print("using Normalize reward wrapper")
            env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)



        #  Temporal Stacking
        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            c = 1 if grayscale else 3
            final_channels = c * frame_stack
            new_obs_space = gym.spaces.Box(
                low=0, 
                high=255, 
                shape=(final_channels , 84, 84), 
                dtype=np.uint8
            )   
            env = gym.wrappers.TransformObservation(
                env, 
                lambda obs: np.array(obs).reshape(final_channels, 84, 84),
                observation_space=new_obs_space
            )
        


        return gym.wrappers.RecordEpisodeStatistics(env)