import torch

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, channel_in=64):
        super(CNN, self).__init__()
        self.kernel_size = 3
        self.conv1 = nn.Conv1d(in_channels=channel_in, out_channels=channel_in * 2, kernel_size=21, stride=10)
        self.conv2 = nn.Conv1d(in_channels=channel_in * 2, out_channels=channel_in * 4, kernel_size=21, stride=10)

        self.linear1 = nn.LazyLinear(1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = torch.nn.functional.layer_norm(x, normalized_shape=x.shape[-1:])

        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    cnn = CNN()
    cnn(torch.zeros((1, 256, 500)))
