# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
#from dm_env import specs dont use dm but duckitown env 

import utils
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

#BM import
import gymnasium as gym

#Ali Duckietown Specific
from gym_duckietown.envs import DuckietownEnv
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper, CropResizeWrapper

#Ali Target the specific logger used in the simulator
import logging
duckietown_logger = logging.getLogger("gym-duckietown")
duckietown_logger.setLevel(logging.WARNING)

#Ali Disable error checking for maximum training throughput
import pyglet
pyglet.options['debug_gl'] = False



#BM note: helps speeds up training 
torch.backends.cudnn.benchmark = True


#BM spec-> shape 
def make_agent(obs_shape, action_shape, cfg): 
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_shape.shape
    return hydra.utils.instantiate(cfg)

def make_env(cfg, seed, idx, capture_video, run_name): #BM added cgf for use of config file
    def thunk():
        render_mode = "rgb_array" if (capture_video and idx == 0) else None
        #BM changed to work with config.yaml file 
        # 1. Initializing the Duckietown env 
        env = DuckietownEnv(
            seed = cfg.seed,  
            map_name = cfg.map_name,
            max_steps = cfg.max_steps, 
            domain_rand = cfg.domain_rand,
            camera_width = cfg.camera_width,
            camera_height = cfg.camera_height,
            accept_start_angle_deg = cfg.accept_start_angle_deg,  # start close to straight
            full_transparency = cfg.full_transparency,
            distortion = cfg.distortion,
            render_mode = render_mode,
            frame_skip = cfg.frame_skip
        )
        print("Initialized environment")

        # 2. Record video if requested (CleanRL standard)   
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # 3. Crop and Resize first (from 120x160 to 84x84)
        env = CropResizeWrapper(env, shape=(84, 84))

        # 4. To make the images from W*H*C into C*H*W 
        env = ImgWrapper(env)


        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        print("Initialized Wrappers")

        # 5. Stack 4 frames
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        #Flatten the 4x3 channels into 12 for the DQNEncoder
        new_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(12, 84, 84), dtype=np.uint8)
        
        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: obs.reshape(12, 84, 84),
            observation_space=new_obs_space  
        )   
        # 6. Basic RL statistics for Tensorboard
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk




class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

       
        self.agent = make_agent(self.train_env.observation_space.shape, #BM note changed spec() -> space.shape
                                self.train_env.action_space.shape, #BM note changed spec() -> space.shape
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        #BM note changed to work with DT to use make_env instead of dmc and use of config file
        # create envs
        self.train_env = make_env(self.cfg, self.cfg.seed, 0, self.cfg.capture_video, self.cfg.exp_name)()
        self.eval_env = make_env(self.cfg, self.cfg.seed, 0, False, self.cfg.exp_name)()
        # create replay buffer
        data_specs = (self.train_env.observation_space, #BM note changed spec() -> space
                      self.train_env.action_space, #BM note changed spec() -> space
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    #BM note, my want to remove this 
    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        #Ali reset the environment
        obs, _ = envs.reset(seed=args.seed) #With this change 
                                            # time_step.observation -> obs
                                            # time_step.reward -> reward 
                                            # time_step.last() -> done 

        while eval_until_episode(episode):
            #Ali TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(obs)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(obs)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(obs,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(obs)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()