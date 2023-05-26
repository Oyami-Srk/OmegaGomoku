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


class GomokuAI3(nn.Module):
    def __init__(self, board_size=8):
        super(GomokuAI3, self).__init__()
        # conv_setup = [2, 64, 128, 256]
        conv_setup = [(2, 0, 0, 0), (64, 5, 1, 2), (128, 3, 1, 1), (128, 3, 1, 1)]
        for i in range(len(conv_setup) - 1):
            in_channel = conv_setup[i][0]
            out_channel, kernel_size, stride, padding = conv_setup[i + 1]
            self.__setattr__(f"conv{i + 1}", nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
            self.__setattr__(f"bn{i + 1}", nn.BatchNorm2d(out_channel))
        self.conv_size = conv_setup

        def conv2dSizeOut(size, layer=1):
            size = size if layer == len(conv_setup) - 1 else conv2dSizeOut(size, layer + 1)
            _, kernel_size, stride, padding = conv_setup[layer]
            return (size - kernel_size + 2 * padding) // stride + 1

        out_ = conv2dSizeOut(board_size)

        self.fc1 = nn.Linear(out_ ** 2 * conv_setup[-1][0], board_size ** 2)
        self.model_name = "GomokuAI-3"

    def forward(self, input):
        for i in range(len(self.conv_size) - 1):
            input = nn.functional.relu(
                self.__getattr__(f"bn{i + 1}")(
                    self.__getattr__(f"conv{i + 1}")(input)))
        return self.fc1(input.view(input.size(0), -1))
        # input = nn.functional.dropout(input, p=0.8, training=True)
        # return self.fc2(input)


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
