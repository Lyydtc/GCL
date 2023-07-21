from torch import nn
from torch_geometric.nn.pool import TopKPooling

from .reference import GCNLayer, MultiHeadAttentionLayer
from .utils import sort_edge_index


class SubEncoder(nn.Module):
    def __init__(self, args):
        super(SubEncoder, self).__init__()
        self.args = args
        self.gcn_layer = GCNLayer(self.args)
        self.Select_Sub1 = SelectSubAttention(self.args, self.args.embedding_size)
        self.Select_Sub2 = SelectSubAttention(self.args, self.args.embedding_size)
        self.Select_Sub3 = SelectSubAttention(self.args, self.args.embedding_size)

    def forward(self, data):
        edge_index = data['edge_index']
        dense_x = data['dense_x']
        mask = data['mask']
        adj = data['adj']
        dist = None
        batch = data['batch']

        embeddings = self.gcn_layer(dense_x, adj, mask, dist)

        embeddings1, sub_graphs1 = self.Select_Sub1(embeddings, edge_index, mask, dist, batch)
        embeddings2, sub_graphs2 = self.Select_Sub2(embeddings1, edge_index, mask, dist, batch)
        embeddings3, sub_graphs3 = self.Select_Sub3(embeddings2, edge_index, mask, dist, batch)

        return sub_graphs1, sub_graphs2, sub_graphs3


class SelectSubAttention(nn.Module):
    def __init__(self, args, hid_dim):
        super(SelectSubAttention, self).__init__()
        self.args = args
        self.MultiHeadAttention_layer = MultiHeadAttentionLayer(self.args, hid_dim)
        self.topk_pool = TopKPooling(self.args.embedding_size, ratio=self.args.ratio)
        # self.sagPool = SAGPooling(self.args.embedding_size, ratio=self.args.ratio)

    def forward(self, embeddings, edge_index, mask, dist, batch):
        encoder_result = self.MultiHeadAttention_layer(embeddings, mask, dist)

        # dense->spare
        encoder_result_sparse = encoder_result[mask]

        # get batched subgraphs after topk
        # note that topk returns disordered sub_edge_index
        sub_x, sub_edge_index, _, sub_batch, _, _ = self.topk_pool(x=encoder_result_sparse,
                                                                   edge_index=edge_index,
                                                                   batch=batch)
        sub_edge_index = sort_edge_index(sub_edge_index)
        subgraphs = {
            'x': sub_x,
            'edge_index': sub_edge_index,
            'batch': sub_batch
        }

        return encoder_result, subgraphs
