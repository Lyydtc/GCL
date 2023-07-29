import torch
import torch.nn.functional as F
from torch import nn


def info_nce(query, key, t=0.1):

    # Normalize to unit vectors
    query, key = normalize(query, key)

    sim_matrix = query @ transpose(key)
    sim_matrix /= t
    sim_matrix = torch.exp(sim_matrix)

    return -torch.log(torch.diagonal(sim_matrix).sum() / sim_matrix.sum())


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
