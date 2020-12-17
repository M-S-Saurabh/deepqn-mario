import random
import pickle
import torch
from torch import nn
import torchviz

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
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

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
        
class DQNAgent:
    def __init__(self, dqn, gamma, lr, exploration_max, exploration_min, exploration_decay, memory, device="cpu", double_dqn=None, copy_step=None):
        self.Q = dqn
        self.Q.to(device)
        if double_dqn is not None:
            self.Q_target = double_dqn.to(device)
            self.copy_step = copy_step
        self.step = 0
        self.device = device
        self.exploration_decay = exploration_decay
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_rate = exploration_max
        self.gamma = gamma
        self.loss_func = nn.SmoothL1Loss().to(self.device)#nn.MSELoss()
        self.memory = memory
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

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
        
        action, done, next_state, reward, state = self.memory.recall(self.device)
        self.optimizer.zero_grad()
        target_actions = self.Q_target(next_state).max(1).values.unsqueeze(1)
        target = reward + self.gamma*torch.mul(target_actions, 1 - done)
        current = self.Q(state).gather(1, action.long())
        loss = self.loss_func(current, target)
        loss.backward()
        self.optimizer.step()

    def copy(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

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


from torch.distributions.categorical import Categorical
import numpy as np
from networks import ActorCriticNet

class ActorCriticAgent:
    def __init__(self, nn_model, learning_rate, gamma=0.9, max_steps=5000, device='cpu'):
        super().__init__()
        self.model = nn_model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_steps = max_steps
        self.gamma = gamma
        self.device = device
        self.entropy_term = 0

    def train_step(self, env):
        # Run a full episode
        log_probs, values, rewards, last_Qval, num_steps = self.run_episode(env)

        # Compute loss
        total_loss = self.compute_loss(log_probs, values, rewards, last_Qval)

        # Back-propagate
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return sum(rewards), num_steps

    def run_episode(self, env):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for step in range(self.max_steps):
            # Run model and get action probs and critic values
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            value, policy_dist = self.model.forward(state)
            value = value.detach().cpu().numpy()[0,0]
            dist = policy_dist.detach().cpu().numpy()
            state = state.cpu()

            # Sample an action with these probs
            num_actions = dist.size
            action = np.random.choice(num_actions, p=np.squeeze(dist))

            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = - np.sum(np.mean(dist) * np.log(dist + 1e-10))

            # Run step with this action
            state, reward, done, _ = env.step(action)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            self.entropy_term += entropy

            if done or step == self.max_steps-1:
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
                Qval, _ = self.model.forward(state)
                Qval = Qval.detach().cpu().numpy()[0,0]
                break

        return log_probs, values, rewards, Qval, step

    def compute_loss(self, log_probs, values, rewards, Q_val):
        Q_vals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Q_val = rewards[t] + self.gamma * Q_val
            Q_vals[t] = Q_val

        values = torch.FloatTensor(values).to(self.device)
        Q_vals = torch.FloatTensor(Q_vals).to(self.device)
        log_probs = torch.stack(log_probs)

        advantage = Q_vals - values

        actor_loss = - (log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        total_loss = actor_loss + critic_loss + 0.001 * self.entropy_term
        return total_loss

    def act(self, state):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        value, action_probs = self.model(state)
        # Sample an action with these probs
        action = Categorical(action_probs).sample().item()
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path+"model.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path+"model.pt"))