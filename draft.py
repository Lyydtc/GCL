import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_node, to_dense_batch, to_dense_adj
import random
from my_parser import parsed_args
from my_dataset import MyDataset
import dgl
from dgl import DropNode
import torch
from dgl import DropEdge
from model import MultiHeadCrossAttentionLayer
from model import MyModel, MLP_Decoder, GTCNet


args = parsed_args
model = GTCNet(args)

input = torch.randn((64, 128))

