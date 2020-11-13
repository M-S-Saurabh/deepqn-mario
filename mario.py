'''
blog: https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
source code: https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
'''
import collections
import random


import attr
import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


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
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size[0], self.image_size[1], 1),
            dtype=np.uint8
        )

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


class DQN(nn.Module):
    def __init__(self, input_channels, batch_size, output_size):
        nn.Module.__init__(self)
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.output_size = output_size

        self.c1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.rc1 = nn.ReLU()
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.rc2 = nn.ReLU()
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.rc3 = nn.ReLU()

        self.h1 = nn.Linear(3136, 512)
        self.rh1 = nn.ReLU()
        self.h2 = nn.Linear(512, output_size)

    def forward(self, x):
        i1 = self.c1(x)
        i2 = self.rc1(i1)
        i3 = self.c2(i2)
        i4 = self.rc2(i3)
        i5 = self.c3(i4)
        i6 = self.rc3(i5)

        intermediate = i6.reshape(self.batch_size, 1, -1)
        i7 = self.h1(intermediate)
        i8 = self.rh1(i7)
        i9 = self.h2(i8)
        return i9


class Memory:
    def __init__(self, memory_sample_size, memory_size, state_space):
        self.i = 0
        self.memory_sample_size = memory_sample_size
        self.memory_size = memory_size
        self.num_in_queue = 0

        self.actions = torch.zeros(memory_size, 1)
        self.dones = torch.zeros(memory_size, 1)
        self.next_states = torch.zeros(memory_size, *state_space)
        self.rewards = torch.zeros(memory_size, 1)
        self.states = torch.zeros(memory_size, *state_space)

    def remember(self, action, done, next_state, reward, state):
        i = self.i
        self.actions[i] = action.float()
        self.dones[i] = done.float()
        self.next_states[i] = next_state.float()
        self.rewards[i] = reward.float()
        self.states[i] = state.float()

        self.i = (i+1)%self.memory_size
        self.num_in_queue = min(self.num_in_queue+1, self.memory_size)

    def recall(self, device):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        action = self.actions[idx].to(device)
        done = self.dones[idx].to(device)
        next_state = self.next_states[idx].to(device)
        reward = self.rewards[idx].to(device)
        state = self.states[idx].to(device)

        return action, done, next_state, reward, state


class MarioAgent:
    def __init__(self, dqn, gamma, lr, exploration_max, exploration_min, exploration_decay, memory, device="cpu"):
        self.Q = dqn
        self.Q.to(device)
        self.device = device
        self.exploration_decay = exploration_decay
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_rate = exploration_max
        self.gamma = gamma
        self.loss_func = nn.MSELoss()
        self.memory = memory
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

    def act(self, state):
        if random.random() < self.exploration_rate:
            action = random.randrange(self.Q.output_size)
        else:
            state = state.squeeze(0)
            state = torch.stack(self.Q.batch_size*[state], dim=0)
            q = self.Q(state.to(self.device))
            action_tensor = torch.argmax(q[0, :])
            action = action_tensor.item()
        return action

    def q_update(self, _state, _action, _reward, _next_state, _done):
        _action = torch.tensor(_action)
        _reward = torch.tensor(_reward)
        _done = torch.tensor(1.0 if _done else 0)
        self.memory.remember(_action, _done, _next_state, _reward, _state)
        action, done, next_state, reward, state = self.memory.recall(self.device)

        self.optimizer.zero_grad()
        temp = self.Q_target(next_state).max(1)[0]
        target = reward + torch.mul(self.gamma*temp.max(1).values.unsqueeze(1), 1-done)
        current = self.Q(state).squeeze(1).gather(1, action.long())
        loss = self.loss_func(current, target)
        loss.backward()
        self.optimizer.step()

    def copy(self):
        self.Q_target = DQN(self.Q.input_channels, self.Q.batch_size, self.Q.output_size)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.to(self.device)

    def save(self):
        torch.save(self.Q.state_dict(), "Q.pt")
        torch.save(self.Q_target.state_dict(), "Q_target.pt")

    def load(self):
        self.Q.load_state_dict(torch.load("Q.pt"))
        self.copy()
        self.Q_target.load_state_dict(torch.load("Q.pt"))

    def update_exploration_rate(self):
        r = self.exploration_decay*self.exploration_rate
        r = min(r, self.exploration_max)
        r = max(r, self.exploration_min)
        self.exploration_rate = r


@attr.s
class Config:
    batch_size = attr.ib(4)
    copy_step = attr.ib(4)
    device = attr.ib("cuda")
    do_load_model = attr.ib(False)
    exploration_decay = attr.ib(0.99)
    exploration_max = attr.ib(1.0)
    exploration_min = attr.ib(0.02)
    gamma = attr.ib(0.90)
    is_training = attr.ib(True)
    learning_rate = attr.ib(0.00025)
    memory_size = attr.ib(10)
    number_of_episodes = attr.ib(5000)


def run():
    config = Config()
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = make_env(env)

    state_shape = env.observation_space.shape
    output_size = env.action_space.n
    dqn = DQN(state_shape[0], config.batch_size, output_size)
    memory = Memory(config.batch_size, config.memory_size, state_shape)

    agent = MarioAgent(dqn,
                       gamma=config.gamma,
                       lr=config.learning_rate,
                       exploration_max=config.exploration_max,
                       exploration_min=config.exploration_min,
                       exploration_decay=config.exploration_decay,
                       memory=memory,
                       device=config.device)
    do_load = config.do_load_model or not config.is_training
    if do_load:
        agent.load()

    max_reward = 0
    rewards = list()
    for ep_num in range(config.number_of_episodes):
        state = env.reset()
        state = torch.from_numpy(state).unsqueeze(0)
        episode_reward = 0
        step = 0
        done = False
        while not done:
            if step%config.copy_step == 0:
                agent.copy()

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(next_state).unsqueeze(0)
            agent.q_update(state, action, reward, next_state, done)

            episode_reward += reward
            step += 1
            env.render()
            agent.update_exploration_rate()

        print(f"{ep_num}: reward = {episode_reward}")
        rewards.append(episode_reward)
        if episode_reward > max_reward:
            max_reward = episode_reward
            print(f"saving: max reward = {max_reward}")
            agent.save()

    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    run()
