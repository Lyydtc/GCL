import torch
from torch import nn


# 下游分类任务解码器 MLP
class MLP_Decoder(nn.Module):
    def __init__(self, args):
        super(MLP_Decoder, self).__init__()
        self.args = args
        self.line_1 = nn.Linear(2 * self.args.n_heads * self.args.embedding_size, self.args.embedding_size)
        self.line_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size // 4)
        self.line_3 = nn.Linear(self.args.embedding_size // 4, self.args.embedding_size // 8)

        if self.args.task == 'cls':
            self.line_4 = nn.Linear(self.args.embedding_size // 8, self.args.num_classes)
        elif self.args.task == 'gsl':
            self.line_4 = nn.Linear(self.args.embedding_size // 8, 1)

        self.active = nn.ReLU()
        self.dropout = nn.Dropout(self.args.ds_dropout)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x1 = self.dropout(self.active(self.line_1(x)))
        x2 = self.dropout(self.active(self.line_2(x1)))
        x3 = self.dropout(self.active(self.line_3(x2)))

        if self.args.task == 'cls':
            x4 = self.out_act(self.line_4(x3))
        elif self.args.task == 'gsl':
            x4 = self.line_4(x3)

        return x4

class MLP_Decoder5(nn.Module):
    def __init__(self, args):
        super(MLP_Decoder5, self).__init__()
        self.args = args
        self.line_1 = nn.Linear(2 * self.args.n_heads * self.args.embedding_size,
                                8 * self.args.embedding_size)
        self.line_2 = nn.Linear(8 * self.args.embedding_size, 4 * self.args.embedding_size)
        self.line_3 = nn.Linear(4 * self.args.embedding_size, self.args.embedding_size // 2)
        self.line_4 = nn.Linear(self.args.embedding_size // 2, self.args.embedding_size // 4)

        if self.args.task == 'cls':
            self.line_5 = nn.Linear(self.args.embedding_size // 4, self.args.num_classes)
        elif self.args.task == 'gsl':
            self.line_5 = nn.Linear(self.args.embedding_size // 4, 1)

        self.active = nn.ReLU()
        self.dropout = nn.Dropout(self.args.ds_dropout)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x1 = self.dropout(self.active(self.line_1(x)))
        x2 = self.dropout(self.active(self.line_2(x1)))
        x3 = self.dropout(self.active(self.line_3(x2)))
        x4 = self.dropout(self.active(self.line_4(x3)))

        if self.args.task == 'cls':
            x5 = self.out_act(self.line_5(x4))
        elif self.args.task == 'gsl':
            x5 = self.line_5(x4)

        return x5
