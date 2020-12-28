import gym
import collections
import numpy as np
import cv2
import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._buffer = collections.deque(maxlen=skip)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._buffer.clear()
        obs = self.env.reset()
        self._buffer.append(obs)
        return obs


class ProcessFrames84(gym.ObservationWrapper):
    def __init__(self, env, image_size=(84, 84)):
        gym.ObservationWrapper.__init__(self, env)
        self.image_size = image_size
        # self.observation_space = gym.spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.image_size[0], self.image_size[1], 1),
        #     dtype=np.uint8
        # )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.image_size)
        return frame


class ImageToPyTorch(gym.Wrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, frame):
        frame = torch.from_numpy(frame)
        return frame


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps=4):
        gym.ObservationWrapper.__init__(self, env)
        W, H, _ = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(n_steps, W, H), dtype=np.float32)
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        self.n_steps = n_steps

    def reset(self):
        return self.observation(self.env.reset())

    def observation(self, frame):
        n = self.n_steps - 1
        self.buffer[:n, :, :] = self.buffer[1:, :, :]
        self.buffer[n, :, :] = frame
        return self.buffer


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, frame):
        frame = frame/255.0
        return frame


def make_env(env):
    env = MaxAndSkipEnv(env)
    env = ProcessFrames84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, RIGHT_ONLY)