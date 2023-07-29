import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class AlignLayer(nn.Module):

    def __init__(self, input_dim, out_nodes):
        super().__init__()
        self.conv1 = GCNConv(input_dim, input_dim)
        self.conv2 = GCNConv(input_dim, input_dim)
        self.assign = nn.Linear(input_dim, out_nodes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        s = F.softmax(self.assign(x), dim=1)
        return s
