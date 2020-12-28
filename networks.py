import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class DQN(nn.Module):
    def __init__(self, input_channels, batch_size, output_size):
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.output_size = output_size
        self.batch_size = batch_size
        # 4 x 84 x 84
        self.c1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        # 32 x 20 x 20
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 64 x 9 x 9
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 64 x 7 x 7 = 3136
        self.h1 = nn.Linear(3136, 512)
        self.h2 = nn.Linear(512, output_size)

    def forward(self, x):
        i1 = F.relu(self.c1(x))
        i2 = F.relu(self.c2(i1))
        i3 = F.relu(self.c3(i2))
        # intermediate = i3.reshape(self.batch_size, 1, -1)
        intermediate = i3.view(-1, 3136)
        i4 = F.relu(self.h1(intermediate))
        i5 = self.h2(i4)
        return i5

class ActorCriticNet(nn.Module):
    def __init__(self, input_channels, output_size):
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.output_size = output_size
        # 4 x 84 x 84
        self.c1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        # 32 x 20 x 20
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 64 x 9 x 9
        self.c3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        # 32 x 7 x 7

        num_hidden = 512
        # 32 x 7 x 7 = 1568
        self.flat_size = 32 * 7 * 7
        self.h1 = nn.Linear(self.flat_size, num_hidden)
        self.actor = nn.Linear(num_hidden, output_size)
        self.critic = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu( self.c1(x))
        x = F.relu( self.c2(x))
        x = F.relu( self.c3(x))
        intermediate = x.view(-1, self.flat_size)
        x = F.relu( self.h1(intermediate))

        value = self.critic(x)
        policy_dist = F.softmax(self.actor(x), dim=1)
        return value, policy_dist

    def act(self, inputs, deterministic=False):
        value, actor_features = self(inputs)
        dist = Categorical(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self(inputs)
        dist = Categorical(actor_features)

        action_log_probs = dist.log_prob(torch.stack(action))
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy