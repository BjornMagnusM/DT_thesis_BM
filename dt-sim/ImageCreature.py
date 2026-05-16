#!/usr/bin/env python
"""
Duckietown manual control + full observation pipeline visualization
Press ENTER to save preprocessing figure (RGB + grayscale)
"""

from PIL import Image
import argparse
import sys
import os

import gymnasium as gym
import numpy as np
import pyglet
from pyglet.window import key

import cv2
import matplotlib.pyplot as plt

from src.gym_duckietown.envs import DuckietownEnv
from utils.wrappers import (
    CropResizeWrapper,
    TimeOptimalRewardV2,
    LapTerminationWrapperV4,
    TimeOptimalRewardV3,
    LapTerminationWrapperV3,
)

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown")
parser.add_argument("--map-name", default="oval_loop_backround")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true")
parser.add_argument("--draw-bbox", action="store_true")
parser.add_argument("--domain-rand", action="store_true")
parser.add_argument("--dynamics_rand", action="store_true")
parser.add_argument("--frame-skip", default=1, type=int)
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

# -------------------------
# Environment
# -------------------------
env = DuckietownEnv(
    seed=args.seed,
    map_name=args.map_name,
    draw_curve=args.draw_curve,
    draw_bbox=args.draw_bbox,
    domain_rand=args.domain_rand,
    frame_skip=args.frame_skip,
    distortion=args.distortion,
    camera_rand=args.camera_rand,
    dynamics_rand=args.dynamics_rand,
    camera_width=160,
    camera_height=120,
    accept_start_angle_deg=20,
)

env = CropResizeWrapper(env, shape=(84, 84))
env = LapTerminationWrapperV3(env, 2000)
env = TimeOptimalRewardV3(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

render_modes = ["human", "top_down", "free_cam", "rgb_array"]
view = render_modes[2]

env.reset(seed=args.seed)
env.unwrapped.render(view)

print("Camera shape:", env.unwrapped.render_obs().shape)

# -------------------------
# Helpers
# -------------------------
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def resize(img, size=(84, 84)):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def crop(img):
    # remove top 1/3 (sky)
    h = img.shape[0]
    return img[h // 3 :, :]

def full_pipeline(img):
    """
    returns:
    original, resized, cropped, final
    """
    resized = resize(img)
    cropped = crop(resized)
    return img, resized, cropped, cropped  # final = cropped here

def full_pipeline_gray(img):
    gray = to_gray(img)
    resized = resize(gray)
    cropped = crop(resized)
    return gray, resized, cropped, cropped

# -------------------------
# Keyboard
# -------------------------
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.BACKSPACE:
        env.reset(seed=args.seed)
        env.unwrapped.render(view)

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

# -------------------------
# Update loop
# -------------------------
def update(dt):

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0.0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, truncated, info = env.step(action)

    print(f"step={env.unwrapped.step_count}, reward={reward:.3f}")

    # -------------------------
    # SAVE FIGURE ON ENTER
    # -------------------------
    if key_handler[key.RETURN]:

        rgb_o, rgb_r, rgb_c, rgb_f = full_pipeline(obs)
        g_o, g_r, g_c, g_f = full_pipeline_gray(obs)

        fig, axs = plt.subplots(2, 4, figsize=(14, 6))

        # ---------------- RGB row ----------------
        axs[0, 0].imshow(rgb_o)
        axs[0, 0].set_title("RGB Original")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(rgb_r)
        axs[0, 1].set_title("RGB Resized")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(rgb_c)
        axs[0, 2].set_title("RGB Cropped")
        axs[0, 2].axis("off")

        axs[0, 3].imshow(rgb_f)
        axs[0, 3].set_title("RGB Final")
        axs[0, 3].axis("off")

        # ---------------- Gray row ----------------
        axs[1, 0].imshow(g_o, cmap="gray")
        axs[1, 0].set_title("Gray Original")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(g_r, cmap="gray")
        axs[1, 1].set_title("Gray Resized")
        axs[1, 1].axis("off")

        axs[1, 2].imshow(g_c, cmap="gray")
        axs[1, 2].set_title("Gray Cropped")
        axs[1, 2].axis("off")

        axs[1, 3].imshow(g_f, cmap="gray")
        axs[1, 3].set_title("Gray Final")
        axs[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig("observation_pipeline_full.png", dpi=250)
        plt.close()

        print("Saved: observation_pipeline_full.png")

    # -------------------------
    # reset episode
    # -------------------------
    if done:
        print("Episode done:", info.get("episode", {}))
        env.reset(seed=args.seed)
        env.unwrapped.render(view)

    env.unwrapped.render(view)

# -------------------------
# Run
# -------------------------
pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
pyglet.app.run()

env.close()