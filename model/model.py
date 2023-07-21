import torch
from torch import nn
from torch_geometric.utils import unbatch, unbatch_edge_index
import dgl
from timeit import default_timer as timer

from .reference import SimpleAlign
from .sub_encoder import SubEncoder
from .cens_sub_encoder import CensSubEncoder
from .cross_attention import MultiHeadCrossAttentionLayer


class MyModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # subgraph
        if args.switch:
            self.encoder1 = CensSubEncoder(self.args)
            self.encoder2 = CensSubEncoder(self.args)
        else:
            self.encoder1 = SubEncoder(self.args)
            self.encoder2 = SubEncoder(self.args)

        # subgraph num_node align
        self.align1 = SimpleAlign(self.args.embedding_size, self.args.align_size, self.args.dropout)
        self.align2 = SimpleAlign(self.args.embedding_size, self.args.align_size, self.args.dropout)

        self.cross_attention = MultiHeadCrossAttentionLayer(args.embedding_size, args.n_heads)

    def forward(self, data1, data2):

        # get subgraph representations
        if self.args.switch:
            subgraphs11, subgraphs12 = self.encoder1(data1)
            subgraphs21, subgraphs22 = self.encoder2(data2)

            sub1_x1 = self.align_unbatched_graph(subgraphs11, self.align1)
            sub1_x2 = self.align_unbatched_graph(subgraphs12, self.align1)

            sub2_x1 = self.align_unbatched_graph(subgraphs21, self.align2)
            sub2_x2 = self.align_unbatched_graph(subgraphs22, self.align2)

            sub1 = sub1_x1 + sub1_x2
            sub2 = sub2_x1 + sub2_x2

        else:
            subgraphs11, subgraphs12, subgraphs13 = self.encoder1(data1)
            subgraphs21, subgraphs22, subgraphs23 = self.encoder2(data2)

            sub1_x1 = self.align_batched_graph(subgraphs11, self.align1)
            sub1_x2 = self.align_batched_graph(subgraphs12, self.align1)
            sub1_x3 = self.align_batched_graph(subgraphs13, self.align1)

            sub2_x1 = self.align_batched_graph(subgraphs21, self.align2)
            sub2_x2 = self.align_batched_graph(subgraphs22, self.align2)
            sub2_x3 = self.align_batched_graph(subgraphs23, self.align2)

            sub1 = sub1_x1 + sub1_x2 + sub1_x3
            sub2 = sub2_x1 + sub2_x2 + sub2_x3

        # CLS 对应相减
        cls1 = torch.sum(sub1, dim=1, keepdim=True) - torch.sum(sub2, dim=1, keepdim=True)
        cls2 = -cls1

        # add cls to sub
        sub_with_cls1 = torch.cat((cls1, sub1), dim=1)
        sub_with_cls2 = torch.cat((cls2, sub2), dim=1)

        # cross attention
        out1, out2 = self.cross_attention(sub_with_cls1, sub_with_cls2)
        sub_x1 = out1[:, 0, :]
        sub_x2 = out2[:, 0, :]

        return sub_x1, sub_x2

    def get_embeddings(self, data1, data2):
        sub1_x1, sub1_x2, sub1_x3 = self.encoder1(data1)
        sub2_x1, sub2_x2, sub2_x3 = self.encoder2(data2)
        sub1 = sub1_x1 + sub1_x2 + sub1_x3
        sub2 = sub2_x1 + sub2_x2 + sub2_x3
        sim1, sim2 = self.crossAttention(sub1, None, sub2, None, None)
        # out=torch.cat([sim1,sim2],dim=1)
        return sim1, sim2

    def align_batched_graph(self, graph, align_layer):
        batched_x = graph['x']
        batched_edge_index = graph['edge_index']
        batch = graph['batch']

        # unbatch graph data
        x_list = unbatch(batched_x, batch)
        edge_index_list = unbatch_edge_index(batched_edge_index.to(self.args.device), batch.to(self.args.device))

        aligned_x_list = []
        for i in range(self.args.batch_size):
            x = x_list[i]
            edge_index = edge_index_list[i]
            dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
            aligned_x = align_layer(dgl_graph, x)
            aligned_x_list.append(aligned_x)
        return torch.stack(aligned_x_list)

    def align_unbatched_graph(self, graphs, align_layer):
        aligned_x_list = []
        for i in range(self.args.batch_size):
            x = graphs[i]['x']
            edge_index = graphs[i]['edge_index']
            dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
            aligned_x = align_layer(dgl_graph, x)
            aligned_x_list.append(aligned_x)
        return torch.stack(aligned_x_list)
