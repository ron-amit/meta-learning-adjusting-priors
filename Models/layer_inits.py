
from __future__ import absolute_import, division, print_function

import math


import torch.nn as nn
from Models.stochastic_inits import init_stochastic_conv2d, init_stochastic_linear
from Models.stochastic_layers import StochasticLinear, StochasticConv2d, StochasticLayer

'''   Xavier initialization
Like in PyTorch's default initializer'''


def init_layers(model, log_var_init=None):

    for m in model.modules():
        init_module(m, log_var_init)


def init_module(m, log_var_init):
    # Conv2d standard
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        stdv = 1. / math.sqrt(n)
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, +stdv)

    # Linear standard
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        stdv = 1. / math.sqrt(n)
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, +stdv)

    # BatchNorm2d
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    # Conv2d stochastic
    elif isinstance(m, StochasticConv2d):
        init_stochastic_conv2d(m, log_var_init)

    # Linear stochastic
    elif isinstance(m, StochasticLinear):
        init_stochastic_linear(m, log_var_init)


