import random
import torch
from torch import nn

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