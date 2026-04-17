import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Any, Final, SupportsFloat
from collections import deque


# Modified DT wrappers 
class CropResizeWrapperROS:
    def __init__(self, shape=(84, 84)):
        self.shape = shape

    def observation(self, obs):
        # 1. Convert to PIL for easy manipulation
        img = Image.fromarray(obs)
        
        width, height = img.size
        
        # 2. Crop: Keep the bottom 2/3
        # PIL crop box is (left, top, right, bottom)
        top_boundary = int(height * (1/3))
        img = img.crop((0, top_boundary, width, height))
        
        # 3. Resize to target shape (84x84)
        # Note: Image.resize takes (width, height)
        img = img.resize((self.shape[1], self.shape[0]), Image.BILINEAR)
        
        return np.array(img)

class ImgWrapperROS:
    def __init__(self):
        pass

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class FrameStackObservationROS:
    def __init__(self, stack_size: int, padding_type="reset"):
        self.stack_size = stack_size
        self.padding_type = padding_type
        self.obs_queue = deque(maxlen=stack_size)
        self.padding_value = None

    def reset(self, obs):
        self.obs_queue.clear()

        if self.padding_type == "reset":
            self.padding_value = obs.copy()
        elif self.padding_type == "zero":
            self.padding_value = np.zeros_like(obs)
        else:
            self.padding_value = self.padding_type

        for _ in range(self.stack_size - 1):
            self.obs_queue.append(self.padding_value)

        self.obs_queue.append(obs)
        return self._get()

    def append(self, obs):
        self.obs_queue.append(obs)
        return self._get()

    def _get(self):
        return np.concatenate(list(self.obs_queue), axis=0)


# --- Initialization Helpers ---

def weight_init(m):
    """Orthogonal initialization for stable gradients in RL."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data') and m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data') and m.bias is not None:
            m.bias.data.fill_(0.0)

def impala_init(module, weight_init, bias_init, gain=1):
    """Custom initialization utility for Impala-style architectures."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class DrQEncoderV2(nn.Module):  
    def __init__(self, obs_shape=9, feature_dim=50, pretrained=False):
        super().__init__()

        #assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Flatten())
        self.linear = nn.Linear(self.repr_dim, feature_dim)
        self.apply(weight_init)

        #if pretrained:
        #    pretrained_agent = torch.load(pretrained)
        #    self.load_state_dict(pretrained_agent.encoder.state_dict())

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")

    def forward(self, obs):
        obs = obs.float()/ 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        
        h = self.linear(h)
        return F.layer_norm(h, h.size())


class ImpalaCNNLN(nn.Module):
    """A more efficient, high-stride CNN for faster feature extraction."""
    def __init__(self, in_channels=9, feature_dim=50, default_init=True):

        super().__init__()

        if default_init:
            init_ = lambda m: impala_init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        else:
            init_ = lambda m:m # no init, use pytorch default

        self.main = nn.Sequential(
            init_(nn.Conv2d(in_channels, 16, 8, stride=4)), LayerNormalization(), nn.LeakyReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2)), LayerNormalization(), nn.LeakyReLU(), nn.Flatten(),
            init_(nn.Linear(32 * 81, feature_dim)))
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")
        
    def forward(self, obs):
        x = self.main(torch.unsqueeze(obs, dim=0)/255 - 0.5)
        return torch.squeeze(F.layer_norm(x, x.size()))





LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_shape,action_dim):
        super().__init__()

        #Testings ALi's model 
        self.encoder = ImpalaCNNLN(
            in_channels=4,
            obs_shape=(4, 84, 84),
            feature_dim=256
        )
        
        

        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling BM: changed for a manual scaling 
        self.register_buffer(
            "action_scale",
            torch.tensor([1.0, 1.0], dtype=torch.float32)
        )

        self.register_buffer(
            "action_bias",
            torch.tensor([0.0, 0.0], dtype=torch.float32)
        )

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.encoder(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        v_t = torch.sigmoid(x_t[:,0:1])
        omega_t = torch.tanh(x_t[:,1:2])
        y_t = torch.cat([v_t, omega_t], dim=-1)

        #y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # --- LOG PROB CORRECTION (Jacobian) ---
        log_prob = normal.log_prob(x_t)

        # Sigmoid correction: log(d/dx sigmoid) = log(sigmoid * (1 - sigmoid))
        log_prob[:, 0:1] -= torch.log(v_t * (1.0 - v_t) + 1e-6)
        
        # Tanh correction: log(d/dx tanh) = log(1 - tanh^2)
        log_prob[:, 1:2] -= torch.log(1.0 - omega_t.pow(2) + 1e-6)
        
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Mean for evaluation (Deterministic mode)
        mean_v = torch.sigmoid(mean[:, 0:1])
        mean_omega = torch.tanh(mean[:, 1:2])
        mean_action = torch.cat([mean_v, mean_omega], dim=-1) * self.action_scale + self.action_bias

        #reducing the actions by 50% for test 
        mean_action = mean_action*3/4

        # Enforcing Action Bound
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        #log_prob = log_prob.sum(1, keepdim=True)
        #mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action
