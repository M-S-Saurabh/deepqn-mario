import numpy as np
import random
import pickle
import torch
from torch import nn
from plot_utils import VisdomLinePlotter

from networks import DQN

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
        rewards = self.rewards[:self.num_in_queue] - torch.min(self.rewards) # enforce r >= 0
        s = torch.sum(rewards)
        p = (rewards[:self.num_in_queue]/s).reshape(-1).numpy() if s > 0 else None
        replace = s == 0 or len([x for x in p if x > 0]) < self.memory_sample_size
        idx = np.random.choice(self.num_in_queue, self.memory_sample_size, replace=replace, p=p)

        action = self.actions[idx].to(device)
        done = self.dones[idx].to(device)
        next_state = self.next_states[idx].to(device)
        reward = self.rewards[idx].to(device)
        state = self.states[idx].to(device)

        return action, done, next_state, reward, state
    
    def save(self):
        # Save class attributes
        with open('memory_dump.pkl', 'wb+') as memfile:
            pickle.dump( self.__dict__, memfile)

    def load(self):
        # Load class attributes
        with open('memory_dump.pkl', 'rb') as memfile:
            vars = pickle.load(memfile)
            for k,v in vars.items():
                setattr(self, k, v)
        

class MarioAgent:
    def __init__(self, dqn, gamma, lr, exploration_max, exploration_min,
                 exploration_decay, iterative_loss_threshold,
                 memory, device="cpu", double_dqn=None,
                 copy_step=None, plotter=None):
        self.Q = dqn
        self.Q.to(device)
        if double_dqn is not None:
            self.Q_target = double_dqn.to(device)
            self.copy_step = copy_step
        else:
            self.Q_target = None
        self.step = 0
        self.device = device
        self.exploration_decay = exploration_decay
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_rate = exploration_max
        self.gamma = gamma
        self.iterative_loss_threshold = iterative_loss_threshold
        self.loss_func = nn.SmoothL1Loss().to(self.device)#nn.MSELoss()
        self.memory = memory
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.plotter = plotter

    def eval(self):
        self.Q.eval()
        if self.Q_target is not None:
            self.Q_target.eval()

    def act(self, state):
        if random.random() < self.exploration_rate:
            action = random.randrange(self.Q.output_size)
        else:
            q = self.Q(state.to(self.device))
            action_tensor = torch.argmax(q)
            action = action_tensor.item()
        self.step += 1
        return action

    def q_update(self, _state, _action, _reward, _next_state, _done):
        if self.step % self.copy_step == 0:
            self.copy()
        _action = torch.tensor(_action)
        _reward = torch.tensor(_reward)
        _done = torch.tensor(1.0 if _done else 0)
        _next_state = torch.from_numpy(_next_state).unsqueeze(0)
        self.memory.remember(_action, _done, _next_state, _reward, _state)

        prev_loss = 0
        loss_diff = 1
        while loss_diff > self.iterative_loss_threshold:
            action, done, next_state, reward, state = self.memory.recall(self.device)
            self.optimizer.zero_grad()
            target_actions = self.Q_target(next_state).max(1).values.unsqueeze(1)
            target = reward + self.gamma*torch.mul(target_actions, 1 - done)
            current = self.Q(state).gather(1, action.long())
            loss = self.loss_func(current, target)
            loss.backward()
            loss_diff = abs(prev_loss - loss.item())
            prev_loss = loss.item()
        if self.plotter:
            self.plotter.plot(var_name="loss", split_name="q_update",
                              title_name="Update Loss",
                              x=self.step, y=loss.item())
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1) # https://stackoverflow.com/a/58752096/4646773
        self.optimizer.step()

    def copy(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()

    def save(self):
        torch.save(self.Q.state_dict(), "Q.pt")
        torch.save(self.Q_target.state_dict(), "Q_target.pt")

    def load(self):
        self.Q.load_state_dict(torch.load("Q.pt"))
        self.copy()

    def update_exploration_rate(self):
        r = self.exploration_decay*self.exploration_rate
        r = min(r, self.exploration_max)
        r = max(r, self.exploration_min)
        self.exploration_rate = r