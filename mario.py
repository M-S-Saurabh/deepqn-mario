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
from plot_utils import plot_rewards, save_rewards, load_rewards, VisdomLinePlotter

@attr.s
class Config:
    batch_size = attr.ib(32)
    copy_step = attr.ib(20000)
    device = attr.ib("cuda")
    do_dynamic_plot = attr.ib(True)
    do_load_model = attr.ib(False)
    exploration_decay = attr.ib(0.99)
    exploration_max = attr.ib(1.0)
    exploration_min = attr.ib(0.08)
    gamma = attr.ib(0.90)
    is_training = attr.ib(True)
    iterative_loss_threshold = attr.ib(.01)
    learning_rate = attr.ib(1e-4)
    memory_size = attr.ib(30000)
    number_of_episodes = attr.ib(10000)


def get_env_and_agent(config):
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
                       double_dqn=dqn_target,
                       copy_step=config.copy_step,
                       gamma=config.gamma,
                       lr=config.learning_rate,
                       exploration_max=config.exploration_max,
                       exploration_min=config.exploration_min,
                       exploration_decay=config.exploration_decay,
                       memory=memory,
                       device=config.device,
                       iterative_loss_threshold=config.iterative_loss_threshold)
    return (env, agent)


def run():
    config = Config()
    plotter = VisdomLinePlotter() if config.do_dynamic_plot else None
    split_name = "train" if config.is_training else "test"
    env, agent = get_env_and_agent(config)

    rewards = list()
    do_load = config.do_load_model or not config.is_training
    if do_load:
        agent.load()
        if config.is_training:
            agent.memory.load()
            rewards = load_rewards()

    if not config.is_training:
        agent.exploration_rate = 0.05
        agent.eval()

    wins = 0
    for episode in tqdm(range(config.number_of_episodes)):
        state = env.reset()
        state = torch.from_numpy(state).unsqueeze(0)
        episode_reward = 0
        step = 0
        done = False
        losses = list()
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if config.is_training:
                loss, has_copy = agent.q_update(state, action, reward, next_state, done)
                if has_copy:
                    print(f"{episode+1}: NEW Q_TARGET")
                losses.append(loss)
                agent.update_exploration_rate()
            episode_reward += reward
            step += 1
            state = torch.from_numpy(next_state).unsqueeze(0)
            if info["flag_get"]:
                wins += 1
        average_loss = sum(losses)/len(losses)
        #print(f"{episode_reward}, {average_loss}")
        if plotter:
            plotter.plot(var_name="reward", split_name=split_name,
                         title_name="Episode Reward",
                         xlabel="episode",
                         x=episode, y=episode_reward)
            plotter.plot(var_name="loss", split_name=split_name,
                         title_name="Episode Average Loss",
                         xlabel="episode",
                         x=episode, y=average_loss)

        rewards.append(episode_reward)

        if config.is_training and (episode+1)%1000 == 0:
            agent.save()
            agent.memory.save()
            save_rewards(rewards)

        if episode%100 == 0:
            print(f"win rate: {wins}%")
            if plotter:
                plotter.plot(var_name="wins", split_name=split_name,
                             title_name="Win rate (per 100 episodes)",
                             xlabel="100 episodes",
                             x=episode, y=wins)
            wins = 0
    if config.is_training:
        agent.save()
        # agent.memory.save()
        save_rewards(rewards)
        plot_rewards(rewards)


if __name__ == "__main__":
    run()

