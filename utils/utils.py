import math
import os.path as osp
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib.pyplot as plt

def recursive_to(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict):
        for name in input:
            if isinstance(input[name], torch.Tensor):
                input[name] = input[name].to(device)
        return input
    if isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = recursive_to(item, device)
        return input
    assert False