import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch


class MaxConsecutiveFames(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.prev_frame = None

    def observation(self, frame):
        if self.prev_frame is not None:
            frame = np.maximum(frame, self.prev_frame)
        self.prev_frame = frame
        return frame


class CustomMarioReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.status2int = {"small": 0, "tall": 1, "fireball": 2}
        self.reset_reward_state()

    def reset_reward_state(self):
        self.prev_score = 0
        self.prev_status = self.status2int["small"]
        self.prev_x = 0
        self.prev_time = 400

    def step(self, action):
        state, _, done, info = self.env.step(action)
        reward = self._get_reward(done, info)
        return (state, reward, done, info)

    def _get_reward(self, done, info):
        reward = 0

        # update reward from time
        reward += (self.prev_time - info["time"])*-0.1
        self.prev_time = info["time"]

        # update reward from status
        current_status = self.status2int[info["status"]]
        reward += (current_status - self.prev_status)*5
        self.prev_status = current_status

        # update reward from distance
        reward += min(max(info["x_pos"] - self.prev_x, -1), 3)
        self.prev_x = info["x_pos"]

        # update reward from score
        reward += (info["score"] - self.prev_score)*0.0025
        self.prev_score = info["score"]

        # update done score and reset vars
        if done:
            reward += 50 if info["flag_get"] else -50
            self.reset_reward_state()
        return reward


class SkipKFrames(gym.Wrapper):
    def __init__(self, env, do_render=True, k=4):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.do_render = do_render

    def step(self, action):
        state, total_reward, done, info = self.env.step(action)
        if not done:
            for _ in range(self.k - 1):
                if self.do_render:
                    self.env.render() # last action rendered by caller
                _, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
        return state, total_reward, done, info


class Grayscale(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, frame):
        gray = np.dot(frame, [0.299, 0.587, 0.114])
        return gray


class Resize(gym.ObservationWrapper):
    def __init__(self, env, new_size=(84, 84)):
        self.size = (new_size[0], new_size[1], 1)
        env.observation_space = gym.spaces.Box(0, 255, self.size, np.uint8)
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        resized = cv2.resize(observation, self.size[:2], interpolation=cv2.INTER_AREA)
        return resized


class CropMario(gym.ObservationWrapper):
    def __init__(self, env):
        env.observation_space = gym.spaces.Box(0, 255, (64, 84, 1), np.uint8)
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, frame):
        return frame[14:78, :]


class StackMFrames(gym.ObservationWrapper):
    def __init__(self, env, m=4):
        w, h, _ = env.observation_space.shape
        self.buffer = np.zeros((w, h, m))
        self.m = m
        env.observation_space = gym.spaces.Box(0, 255, self.buffer.shape, np.uint8)
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        self.buffer[:, :, 1:] = self.buffer[:, :, :-1]
        self.buffer[:, :, 0] = observation
        return self.buffer


class ToTorchTensor(gym.ObservationWrapper):
    def __init__(self, env, device="cpu"):
        gym.ObservationWrapper.__init__(self, env)
        self.device = device

    def observation(self, frame):
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1).div_(np.max(frame))
        #tensor = torch.from_numpy(frame).float().unsqueeze(0)/np.max(frame)
        return tensor.to(self.device)


def list_games():
    return [
        "SuperMarioBros-1-1-v0",
    ]

def make(name, device, do_render=False):
    env = gym_super_mario_bros.make(name)
    env = CustomMarioReward(env)
    env = MaxConsecutiveFames(env)
    env = SkipKFrames(env, do_render, k=3)
    env = Resize(env)
    #env = Grayscale(env)
    #env = CropMario(env)
    env = ToTorchTensor(env, device)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    return env

