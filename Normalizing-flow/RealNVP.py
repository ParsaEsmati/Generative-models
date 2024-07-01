import os
import time
from os import listdir
from os.path import join
import argparse
import gc
from PIL import Image
import matplotlib.pyplot as plt
import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CheckerboardMask(nn.Module):
    def __init__(self, height, width, channels, requires_grad=False):
        super(CheckerboardMask, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels

        self.mask = nn.Parameter(torch.zeros(height, width, channels))
        self.checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]

        mask = torch.tensor(self.checkerboard, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(1, channels, 1, 1)
        self.mask = nn.Parameter(mask, requires_grad=requires_grad)


    def forward(self, x, invert_switch=False):
        if invert_switch:
            mask = 1 - self.mask
        else:
            mask = self.mask
        return x * mask


class RealNVP(nn.Module):
    def __init__(self, input_channels, height, width, n_layers, hidden_channels):
        super(RealNVP, self).__init__()
        self.input_channels = input_channels
        self.n_layers = n_layers

        self.s = nn.ModuleList([self.init_net(input_channels, hidden_channels) for _ in range(n_layers)])
        self.t = nn.ModuleList([self.init_net(input_channels, hidden_channels) for _ in range(n_layers)])
        self.mask = CheckerboardMask(height, width, input_channels, requires_grad=False)

    def init_net(self, input_channels, hidden_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, reverse=False):
        log_det_jacobian = 0
        invert_switch = False

        if not reverse:
            for i in range(self.n_layers):
                x_static = self.mask(x, invert_switch=invert_switch)
                x_dynamic = self.mask(x, invert_switch=not invert_switch)
                
                s = self.s[i](x_static)
                t = self.t[i](x_static)

                x_dynamic = x_dynamic * torch.exp(s) + t
                log_det_jacobian += s.sum(dim=[1, 2, 3])

                x = x_static + x_dynamic
                invert_switch = not invert_switch
        else:
            for i in reversed(range(self.n_layers)):
                x_static = self.mask(x, invert_switch=invert_switch)
                x_dynamic = self.mask(x, invert_switch=not invert_switch)
                
                s = self.s[i](x_static)
                t = self.t[i](x_static)

                x_dynamic = (x_dynamic - t) * torch.exp(-s)
                log_det_jacobian -= s.sum(dim=[1, 2, 3])

                x = x_static + x_dynamic
                invert_switch = not invert_switch

        return x, log_det_jacobian

    def loss(self, x, prior):
        z, log_det_jacobian = self.forward(x)
        log_prob_z = prior.log_prob(z).sum()
        loss = -log_prob_z - log_det_jacobian.sum()
        return loss

