'''
blog: https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
source code: https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
'''

import random
import attr
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import gym_super_mario_bros

# User libraries
from wrappers import make_env
from networks import DQN
from agents import Memory, MarioAgent

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
    for ep_num in tqdm(range(config.number_of_episodes)):
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

        # print(f"{ep_num}: reward = {episode_reward}")
        rewards.append(episode_reward)
        if episode_reward > max_reward:
            max_reward = episode_reward
            print(f"saving: max reward = {max_reward}")
            agent.save()

    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    run()
