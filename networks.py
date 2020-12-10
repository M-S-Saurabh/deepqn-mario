from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels, batch_size, output_size, do_initialization=True):
        nn.Module.__init__(self)
        self.input_channels = input_channels
        self.output_size = output_size
        self.batch_size = batch_size

        self.convolution_layer = nn.Sequential(
            # 4 x 84 x 84
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 32 x 20 x 20
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 64 x 9 x 9
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            # 64 x 7 x 7 = 3136
            nn.Linear(3136, 512),
            nn.ReLU(),
            # output linear layer
            nn.Linear(512, output_size),
        )
        # initialize network using glorot initialization
        if do_initialization:
            for i in range(0, len(self.convolution_layer), 2):
                nn.init.xavier_normal_(self.convolution_layer[i].weight, gain=i//2 + 1)
            offset = len(self.convolution_layer)//2
            for i in range(0, len(self.mlp), 2):
                nn.init.xavier_normal_(self.mlp[i].weight, gain=i//2 + offset + 1)

    def forward(self, x):
        features = self.convolution_layer(x)
        intermediate = features.view(-1, 3136)
        output = self.mlp(intermediate)
        return output