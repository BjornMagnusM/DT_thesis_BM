import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image

from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import get_dir_vec

from gym_duckietown.exceptions import InvalidMapException, NotInLane
from typing import Tuple, Optional
import math
from collections import namedtuple
import cv2
import wandb

class TemporalWrapper(gym.Wrapper):
    def __init__(self, env=None, frame_skip=3, motion_blur=True):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.motion_blur = motion_blur
        self.unwrapped.delta_time = self.unwrapped.delta_time / (self.frame_skip + 1)
        
        self.weights = [0.01, 0.04, 0.15, 0.8]  
        
    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        motion_blur_window = []
        processed_action = self.env.action(action)
        if hasattr(self.env, 'action'):
            processed_action = self.env.action(action)

        for _ in range(self.frame_skip + 1):
            obs = self.unwrapped.render_obs()
            motion_blur_window.append(obs)

            self.unwrapped.update_physics(processed_action)
            
        if not self.motion_blur:
            processed_obs = motion_blur_window[-1]
        else:
            current_weights = self.weights[:len(motion_blur_window)]
            if np.sum(current_weights) == 0:
                processed_obs = motion_blur_window[-1]
            else:
                processed_obs = np.average(
                    motion_blur_window, 
                    axis=0, 
                    weights=current_weights
                ).astype(np.uint8)

        d_info = self.unwrapped._compute_done_reward(processed_action)
        self.unwrapped.step_count += 1


        return processed_obs, d_info.reward, d_info.done, False, self.unwrapped.get_agent_info()


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=shape, 
            dtype=np.uint8
        )
        self.shape = shape # (120, 160, 3)
    def observation(self, observation):
        #from scipy.misc import imresize

        #return imresize(observation, self.shape)
        
        resized = Image.fromarray(observation).resize((self.shape[1], self.shape[0]))
        return np.array(resized)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)



# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_

class CropResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        # Update the observation space to the new dimensions
        # Assuming RGB (3 channels)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.shape[0], self.shape[1], 3), 
            dtype=np.uint8
        )

    def observation(self, obs):
        img = Image.fromarray(obs)
        
        width, height = img.size
        
        # PIL crop box is (left, top, right, bottom)
        top_boundary = int(height * (1/4))
        img = img.crop((0, top_boundary, width, height))
        
        # target shape (84x84)
        img = img.resize((self.shape[1], self.shape[0]), Image.BILINEAR)
        
        return np.array(img)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -50

        return reward

    
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)

    def reward(self, reward):

        if reward <= -15.0:
            return reward

        # Get internal simulator state for custom math
        sim = self.env.unwrapped 
        pos = sim.cur_pos
        angle = sim.cur_angle
        speed = sim.speed
        current_action = sim.last_action
        
        try:
            lp = sim.get_lane_pos2(pos, angle)
        except NotInLane:
            return -10.0 
            
        # Asymmetric Logic
        coords = sim.get_grid_coords(pos) #
        tile = sim._get_tile(*coords) #
        tile_kind = tile["kind"] if tile else ""
        direction = sim.episode_dir

        # Lookahead Logic
        lookahead_dist = 0.25 
        dir_vec = np.array([np.cos(angle), 0, -np.sin(angle)]) # Based on get_dir_vec
        lookahead_pos = pos + dir_vec * lookahead_dist
        
        look_coords = sim.get_grid_coords(lookahead_pos)
        look_tile = sim._get_tile(*look_coords)
        look_kind = look_tile["kind"] if look_tile else ""

        in_curve = "curve" in tile_kind
        approaching_curve = "curve" in look_kind
        in_danger_zone = (direction == "CW") and approaching_curve



        if in_danger_zone:
            # Special "Stabilization" Values
            speed_coeff = 1.0
            dist_coeff = -15.0
            jerk_coeff = -1.2
            target_offset = 0.05
            alignment_k = 5.0
        else:
            # "Race Mode" for straights
            speed_coeff = 2.5
            dist_coeff = -10.0
            jerk_coeff = -0.5
            target_offset = 0.0
            alignment_k = 2.0
        
        reward_speed = speed_coeff * speed * lp.dot_dir
        reward_alignment = np.exp(alignment_k * (lp.dot_dir - 1.0)) # tanh like behaviour to add a higher gradint near 1
        reward_distance = dist_coeff * (lp.dist - target_offset)**2
        reward_angle = -0.03 * np.abs(lp.angle_deg)
        
        action_diff = np.linalg.norm(current_action - self.prev_action) 
        reward_jerk = jerk_coeff * action_diff

        self.prev_action = current_action.copy()

        return reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk
    

#BM wrapper 
class TimeOptimalReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)
    
    def reset(self, **kwargs):
        self.prev_action = np.zeros(2)
        return self.env.reset(**kwargs)

    def reward (self, reward):
        # Get internal simulator state for custom math
        sim = self.env.unwrapped
        reward_const = 2.4 
        speed = sim.speed
        #Lane logig 
        pos = sim.cur_pos
        angle = sim.cur_angle
        current_action = sim.last_action
        try:
            lp = get_road_pos2(sim, pos, angle)
        except NotInLane:
            return -10.0  
        
        reward_speed = 2.0 * speed
        reward_alignment = 2.0 * (lp.dot_dir ** 2) if lp.dot_dir > 0 else 4.0 * lp.dot_dir # tanh like behaviour to add a higher gradint near 1
        reward_distance = -10.0 * np.abs(lp.dist)
        reward_angle = -0.1 * np.abs(lp.angle_deg)
        # Jerk Penalty: Penalize sudden changes in angle
        # self.last_action stores the [v, omega] from the PREVIOUS step
        action_diff = np.linalg.norm(current_action - self.prev_action)
        reward_jerk = -0.5 * action_diff  # Start with -0.5 and tune if needed
        self.prev_action = current_action.copy()
        reward = reward_const+reward_speed + reward_alignment + reward_distance + reward_angle + reward_jerk
        return reward


class TimeOptimalRewardV2(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_action = np.zeros(2)
    
    def reset(self, **kwargs):
        self.prev_action = np.zeros(2)
        return self.env.reset(**kwargs)

    def reward (self, reward):
        # Get internal simulator state for custom math
        sim = self.env.unwrapped
        reward_const = -1
        speed = sim.speed / 0.83 
        #Lane logig 
        pos = sim.cur_pos
        angle = sim.cur_angle  #This is in Radians where max is 2pi 
        current_action = sim.last_action 
        try:
            lp = get_road_pos2(sim, pos, angle)
        except NotInLane:
            return -10.0  
        
        reward_speed_align = 2.5 * speed*lp.dot_dir
        
        reward_distance = -1.0 * (np.abs(lp.dist) / 0.23)**3  #Max would be 0.23
        reward_angle = -10 * np.abs(lp.angle_deg) / 90  ##where max would be +-90deg 

        # Jerk Penalty: Penalize sudden changes in angle
        action_diff = np.linalg.norm(current_action - self.prev_action)
        reward_jerk = -0.5 * action_diff / 2.2  # Start with -0.5 and tune if needed, and max would be 2.2
        self.prev_action = current_action.copy()
        reward += reward_const + reward_speed_align + reward_distance + reward_jerk + reward_angle
        return reward


class LapTerminationWrapperV4(gym.Wrapper):
    def __init__(self, env, max_lap_reward):
        super().__init__(env)
        self.start_tile = None 
        self.finish_tile = None
        self.prev_lenght=1
        self.visited_tiles = set()
        self.step_counter = 0
        self.step_tile = 0
        self.max_lap_reward = max_lap_reward


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sim = self.env.unwrapped

        self.start_tile = None
        self.finish_tile = None
        self.prev_lenght = 1
        self.visited_tiles = set()
        self.step_counter = 0
        self.step_tile = 0

        return obs, info


    def step(self, action): 
        obs, reward, done, truncated, misc = self.env.step(action)
        self.step_counter += 1
        self.step_tile += 1

        sim = self.env.unwrapped 
        current_tile = sim.get_grid_coords(sim.cur_pos)

        #Define the starting tile once
        if self.start_tile is None: 
            self.start_tile = current_tile
        
        #Define the finishing tile as the 2nd tile so the agent will atleast do one full lap 
        if self.finish_tile is None and current_tile != self.start_tile: 
            self.finish_tile = current_tile  
        

        #Add current tile of tile is not in the set 
        self.visited_tiles.add(current_tile)
        
        if len(self.visited_tiles)>self.prev_lenght: 
            tile_reward = max(300-self.step_tile,0.0)
            reward += tile_reward
            self.step_tile = 0 

        misc["progress_ratio"] = len(self.visited_tiles) / 12
        
         #Mark the episode as done if the agent have completed a whole lap  
        if len(self.visited_tiles) == 12 and current_tile == self.finish_tile: 
            done = True
            lap_reward = max(self.max_lap_reward-self.step_counter,0.0)
            misc["lap_step"] =  self.step_counter
            reward += lap_reward
            print(f"completed a lap within {self.step_counter} steps")
        self.prev_lenght = len(self.visited_tiles)
        

        return obs, reward, done, truncated, misc


class LapTerminationWrapperV3(gym.Wrapper):
    def __init__(self, env, max_lap_reward):
        super().__init__(env)
        self.start_tile = None 
        self.finish_tile = None
        self.prev_lenght=1
        self.visited_tiles = set()
        self.step_counter = 0
        self.max_lap_reward = max_lap_reward


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sim = self.env.unwrapped

        self.start_tile = None
        self.finish_tile = None
        self.prev_lenght = 1
        self.visited_tiles = set()
        self.step_counter = 0

        return obs, info


    def step(self, action): 
        obs, reward, done, truncated, misc = self.env.step(action)
        self.step_counter += 1

        sim = self.env.unwrapped 
        current_tile = sim.get_grid_coords(sim.cur_pos)

        #Define the starting tile once
        if self.start_tile is None: 
            self.start_tile = current_tile
        
        #Define the finishing tile as the 2nd tile so the agent will atleast do one full lap 
        if self.finish_tile is None and current_tile != self.start_tile: 
            self.finish_tile = current_tile  
        

        #Add current tile of tile is not in the set 
        self.visited_tiles.add(current_tile)
        
        if len(self.visited_tiles)>self.prev_lenght: 
            reward += 100
        
        misc["progress_ratio"] = len(self.visited_tiles) / 12
        
         #Mark the episode as done if the agent have completed a whole lap  
        if len(self.visited_tiles) == 12 and current_tile == self.finish_tile: 
            done = True
            lap_reward = max(self.max_lap_reward-2*self.step_counter,0.0)
            misc["lap_step"] =  self.step_counter
            reward += lap_reward
            print(f"completed a lap within {self.step_counter} steps")
        self.prev_lenght = len(self.visited_tiles)
        

        return obs, reward, done, truncated, misc


class LapTerminationWrapperV2(gym.Wrapper):
    def __init__(self, env, max_lap_reward):
        super().__init__(env)
        self.start_tile = None 
        self.finish_tile = None
        self.visited_tiles = set()
        self.step_counter = 0
        self.max_lap_reward = max_lap_reward


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sim = self.env.unwrapped

        self.start_tile = None
        self.finish_tile = None
        self.visited_tiles = set()
        self.step_counter = 0

        return obs, info


    def step(self, action): 
        obs, reward, done, truncated, misc = self.env.step(action)
        self.step_counter += 1

        sim = self.env.unwrapped 
        current_tile = sim.get_grid_coords(sim.cur_pos)

        #Define the starting tile once
        if self.start_tile is None: 
            self.start_tile = current_tile
        
        #Define the finishing tile as the 2nd tile so the agent will atleast do one full lap 
        if self.finish_tile is None and current_tile != self.start_tile: 
            self.finish_tile = current_tile  
            
        #Add current tile of tile is not in the set 
        self.visited_tiles.add(current_tile)
        
         #Mark the episode as done if the agent have completed a whole lap  
        if len(self.visited_tiles) == 12 and current_tile == self.finish_tile: 
            done = True
            lap_reward = max(self.max_lap_reward-self.step_counter,0.0)
            reward += lap_reward
            print(f"completed a lap within {self.step_counter} steps")
        

        return obs, reward, done, truncated, misc




class KinematicActionWrapper(gym.ActionWrapper):
    def __init__(self, env, gain=1.0, trim=0.0, wheel_dist=0.102, radius=0.0318, k=27.0, limit=1.0):
        super().__init__(env)
        self.gain = gain
        self.trim = trim
        self.radius = radius
        self.k = k
        self.limit = limit
        self.wheel_dist = wheel_dist

    def action(self, action):
        # Action is [v, omega] from the RL Agent
        vel, angle = action

        # Adjust motor constants by gain and trim
        k_r_inv = (self.gain + self.trim) / self.k
        k_l_inv = (self.gain - self.trim) / self.k

        # Calculate angular velocities for wheels
        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # Convert to duty cycle (PWM)
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # Apply physical limits (max motor power)
        u_r_limited = np.clip(u_r, -self.limit, self.limit)
        u_l_limited = np.clip(u_l, -self.limit, self.limit)

        return np.array([u_l_limited, u_r_limited], dtype=np.float32)


def get_road_pos2(sim ,pos, angle):
    """
    Get the position of the agent relative to the center of the right lane

    Raises NotInLane if the Duckiebot is not in a lane.
    """

    # Get the closest point along the right lane's Bezier curve,
    # and the tangent at that point
    point, tangent = sim.closest_curve_point(pos, angle)
    if point is None or tangent is None:
        msg = f"Point not in lane: {pos}"
        raise NotInLane(msg)

    assert point is not None and tangent is not None

    # Compute the alignment of the agent direction with the curve tangent
    dirVec = get_dir_vec(angle)
    dotDir = np.dot(dirVec, tangent)
    dotDir = np.clip(dotDir, -1.0, +1.0)

    # Compute the signed distance to the curve
    # Right of the curve is negative, left is positive
   
    upVec = np.array([0, 1, 0])
    rightVec = np.cross(tangent, upVec)
    
    #Recompute the point so its in the middle of the road and not the lane
    lane_width = 0.23  #Where width is 0.23m 
    road_center_point = point - rightVec * (lane_width / 2)
    posVec = pos - road_center_point
    signedDist = np.dot(posVec, rightVec)
  
    # Compute the signed angle between the direction and curve tangent
    # Right of the tangent is negative, left is positive
    angle_rad = math.acos(dotDir)

    if np.dot(dirVec, rightVec) < 0:
        angle_rad *= -1

    angle_deg = np.rad2deg(angle_rad)
    # return signedDist, dotDir, angle_deg

 
    return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg, angle_rad=angle_rad)

LanePosition0 = namedtuple("LanePosition", "dist dot_dir angle_deg angle_rad")

class LanePosition(LanePosition0):
    def as_json_dict(self):
        """Serialization-friendly format."""
        return dict(dist=self.dist, dot_dir=self.dot_dir, angle_deg=self.angle_deg, angle_rad=self.angle_rad)


class VideoOverlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None

    def step(self, action):
        self.last_action = action
        return self.env.step(action)

    def render(self, *args, **kwargs):
        frame = self.env.render(*args, **kwargs)

        if frame is None or self.last_action is None:
            return frame

        v, omega = self.last_action

        text = f"v={v:.2f}, omega={omega:.2f}, speed={self.env.unwrapped.speed:.2f}"

        # Draw text on frame
        frame = cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame
