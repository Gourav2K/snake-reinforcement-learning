import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, grid_size):
            super(DQN, self).__init__()
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(64 * (grid_size-4) * (grid_size-4), 128)  # Adjust the size based on input and conv layers
            self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x