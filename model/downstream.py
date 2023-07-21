import torch
from torch import nn
from .reference import SimCNN


# 下游分类任务解码器 Sim_CNN
class GTCNet(torch.nn.Module):
    def __init__(self, args):
        super(GTCNet, self).__init__()
        self.args = args
        self.sim_CNN = SimCNN(args).to(args.device)

    def forward(self, sub_embeddings_0, sub_embeddings_1):
        sim_mat = torch.cat([sub_embeddings_0, sub_embeddings_1], dim=1)
        score = self.sim_CNN(sim_mat)
        return score


# 下游分类任务解码器 MLP
class MLP_Decoder(nn.Module):
    def __init__(self, args):
        super(MLP_Decoder, self).__init__()
        self.args = args
        self.line_1 = nn.Linear(2 * self.args.embedding_size, self.args.embedding_size)
        self.line_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size // 2)
        self.line_3 = nn.Linear(self.args.embedding_size // 2, self.args.num_classes)

        self.active = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x1 = self.dropout(self.active(self.line_1(x)))
        x2 = self.dropout(self.active(self.line_2(x1)))
        x3 = self.active(self.line_3(x2))
        return x3

