'''
blog: https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
source code: https://console.paperspace.com/gcn-team/notebook/pr5ddt1g9
'''

import random
import attr
from tqdm import tqdm
import pickle

import torch
import torchviz
import gym_super_mario_bros

# User libraries
from wrappers import make_env
from networks import DQN
from agents import Memory, MarioAgent
from rewards import plot_rewards, save_rewards, load_rewards

@attr.s
class Config:
    batch_size = attr.ib(32)
    copy_step = attr.ib(5000)
    device = attr.ib("cuda")
    do_load_model = attr.ib(True)
    exploration_decay = attr.ib(0.99)
    exploration_max = attr.ib(1.0)
    exploration_min = attr.ib(0.02)
    gamma = attr.ib(0.90)
    is_training = attr.ib(False)
    learning_rate = attr.ib(0.00025)
    memory_size = attr.ib(30000)
    number_of_episodes = attr.ib(5000)

def run():
    config = Config()
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = make_env(env)

    state_shape = env.observation_space.shape
    output_size = env.action_space.n
    dqn = DQN(state_shape[0], config.batch_size, output_size)
    dqn_target = DQN(state_shape[0], config.batch_size, output_size)
    if config.is_training:
        memory = Memory(config.batch_size, config.memory_size, state_shape)
    else:
        memory = None

    agent = MarioAgent(dqn,
                       double_dqn = dqn_target,
                       copy_step=config.copy_step,
                       gamma=config.gamma,
                       lr=config.learning_rate,
                       exploration_max=config.exploration_max,
                       exploration_min=config.exploration_min,
                       exploration_decay=config.exploration_decay,
                       memory=memory,
                       device=config.device)

    max_reward = 0
    rewards = list()
    do_load = config.do_load_model or not config.is_training
    if do_load:
        agent.load()
        if config.is_training:
            agent.memory.load()
            rewards = load_rewards()

    if not config.is_training:
        agent.exploration_rate = 0.05

    for ep_num in tqdm(range(config.number_of_episodes)):
        state = env.reset()
        state = torch.from_numpy(state).unsqueeze(0)
        episode_reward = 0
        step = 0
        done = False
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if config.is_training:
                agent.q_update(state, action, reward, next_state, done)
                agent.update_exploration_rate()
            episode_reward += reward
            step += 1
            state = torch.from_numpy(next_state).unsqueeze(0)
            if done:
                break

        rewards.append(episode_reward)

        if config.is_training and (ep_num+1)%1000 == 0:
            agent.save()
            agent.memory.save()
            save_rewards(rewards)

    if config.is_training:
        agent.save()
        # agent.memory.save()
        save_rewards(rewards)
        plot_rewards(rewards)


if __name__ == "__main__":
    run()

