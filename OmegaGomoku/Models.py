import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class GomokuAI1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GomokuAI1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.model_name = "GomokuAI-1"

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GomokuAI2(nn.Sequential):
    def __init__(self, board_size):
        # super(GomokuAI, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size).to('cuda' if cuda else 'cpu')
        # self.fc2 = nn.Linear(hidden_size, output_size).to('cuda' if cuda else 'cpu')
        ks = (2, 2)
        super(GomokuAI2, self).__init__(
            nn.Conv2d(2, 64, ks, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),

            nn.Conv2d(64, 128, ks, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),

            nn.Conv2d(128, 256, ks, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),
            nn.Flatten(),
            nn.Linear(256 * (1 if board_size <= 3 else (board_size - 3)) ** 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, board_size ** 2))
        self.model_name = "GomokuAI-2"

    # def forward(self, x):
    # x = x.view(x.size(0), -1)
    # x = torch.relu(self.fc1(x))
    # x = self.fc2(x)
    # return x


class GomokuAI3(nn.Module):
    def __init__(self, board_size=8):
        super(GomokuAI3, self).__init__()
        out_size = [64, 256, 1024]
        self.conv1 = nn.Conv2d(2, out_size[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size[0])
        self.conv2 = nn.Conv2d(out_size[0], out_size[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size[1])
        self.conv3 = nn.Conv2d(out_size[1], out_size[2], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_size[2])

        def conv2dSizeOut(size, kernelSize=3, stride=1, padding=1):
            return (size - kernelSize + 2 * padding) // stride + 1

        conv_size = conv2dSizeOut(conv2dSizeOut(conv2dSizeOut(board_size)))

        self.output = nn.Linear(conv_size ** 2 * out_size[2], board_size ** 2)
        self.model_name = "GomokuAI-3"

    def forward(self, input):
        input = nn.functional.relu(self.bn1(self.conv1(input)))
        input = nn.functional.relu(self.bn2(self.conv2(input)))
        input = nn.functional.relu(self.bn3(self.conv3(input)))
        return self.output(input.view(input.size(0), -1))


class GomokuAI4(nn.Module):
    def __init__(self, board_size=8):
        super(GomokuAI4, self).__init__()
        self.input_size = board_size ** 2 * 2
        self.action_size = board_size ** 2
        __hidden_size = 722
        __hidden_size2 = 540
        # __hidden_size = 360
        # __hidden_size2 = 200
        self.fc1 = nn.Linear(self.input_size, __hidden_size)
        self.fc2 = nn.Linear(__hidden_size, __hidden_size)
        self.fc3 = nn.Linear(__hidden_size, __hidden_size2)
        self.fc4 = nn.Linear(__hidden_size2, self.action_size)
        self.model_name = "GomokuAI-4"

    def forward(self, input):
        x = input.view(-1, self.input_size)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


GomokuAI = GomokuAI3
