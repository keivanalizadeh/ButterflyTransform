from torch import nn
import numpy as np
import math
from torch.nn.parameter import Parameter
import torch

from os import path
import os
import torch.nn.functional as F


# K'th digit of 2^m
def get_kth_digit_2pow_m(n, k, m):
    kth_pow = (1 << (m*(k+1))) - (1 << (m*k))
    return n&kth_pow, n&kth_pow >> (m*k)


class Butterfly_general_matrix(nn.Conv2d):
    '''
    This class implements a butterfly transform as a matrix multiplication.
    '''
    # Base directory for caching graph computation
    
    _BASE_DIR = "cplex"
    if not os.path.exists(_BASE_DIR):
        os.makedirs(_BASE_DIR)

    def __init__(self, in_channels, out_channels, butterfly_K=4, residual_method="no_residual", fan_degree=0):
        '''
        :param in_channels: number of input channels
        :param out_channels:  number of output channels.
        :param butterfly_K: base of butterfly transform. This implementation assumes K is a power of 2.
        :param residual_method: using a residual connection for butterfly transform,
                                by default there is no residual connection. We can add
                                residual from start to end by "residual_stoe"
        '''
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                         bias=False)

        self.residual_method = residual_method
        n = max(in_channels, out_channels)
        power = int(math.ceil(math.log(butterfly_K)))
        assert (1 << power) == butterfly_K
        log_n = int(np.ceil(math.log(n, 1 << power)))
        self.weight1 = Parameter(torch.Tensor((1 << power) * n * log_n))

        if fan_degree == 0:
            fan_degree = in_channels
        stdv_all = math.sqrt(2. / fan_degree)
        stdv_mul = math.pow(2., ((log_n - 1) / log_n)) * math.pow(stdv_all, 1. / log_n)

        self.weight1.data.uniform_(-stdv_mul, stdv_mul)

        file_path = path.join(
            self._BASE_DIR,
            "inc{}_outc{}_pow{}".format(in_channels, out_channels, power),
        )
        print(file_path)

        if path.isfile(file_path):
            print(
                "Loading the graph structure for Butterfly Transform..."
            )
            self.cir_cplex = torch.load(file_path)
        else:
            # cir_cplex[idx][i][j] = idx'th edge on path from input channel i to output channel j
            self.cir_cplex = torch.LongTensor(log_n, out_channels, in_channels)
            for idx in range(log_n):
                for i in range(self.in_channels):
                    for j in range(self.out_channels):
                        _, j_idx_digit = get_kth_digit_2pow_m(j, idx, power)
                        part_1 = (1 << (log_n - idx) * power) - 1
                        part_2 = (1 << log_n * power) - (1 << (log_n - idx) * power)
                        ji_mixture = (part_1 & i) + (part_2 & j)
                        self.cir_cplex[idx, j, i] = idx * (1 << power) * n + j_idx_digit * n + (ji_mixture)
            torch.save(self.cir_cplex, file_path)

    def prod_from_edges(self, w):
        return torch.prod(w[self.cir_cplex], dim=0).view(self.out_channels, self.in_channels, 1, 1)

    def forward(self, input):
        y = F.conv2d(input, self.prod_from_edges(self.weight1), self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        if self.residual_method == "no_residual":
            return y
        if self.residual_method == "residual_stoe":
            if self.out_channels == self.in_channels:
                return y + input
            else:
                raise RuntimeError("input and output should have the same size")


class ButterflyTransform(nn.Conv2d):
    """
    This class is a wrapper around Butterfly_general_matrix. It breaks input or output channels to a set
    of equally-sized parts, and does BFT over each part.
    """
    def __init__(self, in_channels, out_channels, butterfly_K=4):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                         bias=False)
        if self.in_channels <= self.out_channels:
            # self.conv1 = Butterfly_general_matrix(in_channels, in_channels, butterfly_K=butterfly_K, method="residual_stoe")
            self.num_balanced_bfts = int(np.ceil(self.out_channels/self.in_channels))
            self.balanced_bfts = nn.ModuleList([Butterfly_general_matrix(in_channels, in_channels, butterfly_K=butterfly_K,
                                                                         residual_method="residual_stoe",
                                                                         fan_degree=self.out_channels)
                                                for i in range(self.num_balanced_bfts)])
        else:
            # self.conv1 = Butterfly_general_matrix(out_channels, out_channels, butterfly_K=butterfly_K, method="residual_stoe")
            self.num_balanced_bfts = int(np.ceil(self.in_channels/self.out_channels))
            self.balanced_bfts = nn.ModuleList([Butterfly_general_matrix(out_channels, out_channels, butterfly_K=butterfly_K,
                                                                         residual_method="residual_stoe",
                                                                         fan_degree=self.out_channels) for i in
                                                range(self.num_balanced_bfts)])

    def forward(self, x):
        if self.in_channels <= self.out_channels:
            outputs = []
            for i, bft in enumerate(self.balanced_bfts):
                outputs.append(bft(x))
            return torch.cat(outputs, 1)[:, :self.out_channels, :, :]
        else:
            N, C, H, W = x.shape
            output = torch.zeros(N, self.out_channels, H, W).to(x.device)
            for i, bft in enumerate(self.balanced_bfts):
                current_input = x[:, i*self.out_channels:(i+1)*self.out_channels, :, :]
                if current_input.shape[1] < self.out_channels:
                    zero_channels = self.out_channels-current_input.shape[1]
                    current_input = torch.cat([current_input,
                                              torch.zeros(N, zero_channels, H, W).to(x.device)], dim=1)
                output += bft(current_input)
            return output


class Fusion(nn.Module):

    def __init__(self, in_channels, out_channels, fusion_method='conv2d'):
        super().__init__()
        self.fusion_method = fusion_method
        print(self.fusion_method)
        if self.fusion_method == "conv2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        elif self.fusion_method == "butterfly":
            self.conv = ButterflyTransform(in_channels, out_channels)


    def forward(self, x):
        return self.conv(x)

