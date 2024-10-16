# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchrl
import torch.nn as nn


class Sender(nn.Module):
    def __init__(self, n_features, n_hidden, images):
        super(Sender, self).__init__()
        if images:
            self.images = True
            self.conv = torchrl.modules.ConvNet(in_features=3,
                                                depth=6,
                                                num_cells=32,
                                                strides=[1, 2, 1, 2, 1, 2],
                                                kernel_sizes=3,
                                                activation_class=torch.nn.ReLU,
                                                norm_class=torch.nn.LazyBatchNorm2d)
            self.fc1 = nn.Linear(5*5*32, n_hidden)
        else:
            self.images = False
            self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _aux_input=None):
        if self.images:
            batch_size = x.shape[0]
            image_size = 64
            x = x.reshape((-1, 3, image_size, image_size))
            x = self.conv(x)
            x = x.reshape((batch_size, -1))
        x = self.fc1(x).tanh()
        return x


class Receiver(nn.Module):
    def __init__(self, n_features, linear_units, images):
        super(Receiver, self).__init__()
        if images:
            self.images = True
            self.conv = torchrl.modules.ConvNet(in_features=3,
                                                depth=6,
                                                num_cells=32,
                                                strides=[1, 2, 1, 2, 1, 2],
                                                kernel_sizes=3,
                                                activation_class=torch.nn.ReLU,
                                                norm_class=torch.nn.LazyBatchNorm2d)
            self.fc1 = nn.Linear(5*5*32, linear_units)
        else:
            self.images = False
            self.fc1 = nn.Linear(n_features, linear_units)

    def forward(self, x, _input, _aux_input=None):
        if self.images:
            batch_size = _input.shape[0]
            image_size = 64
            _input = _input.reshape((-1, 3, image_size, image_size))
            _input = self.conv(_input)
            _input = _input.reshape((batch_size, 5, -1))
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))

        return energies.squeeze()
