from torch import nn
from torch_geometric.nn.pool import TopKPooling
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import unbatch, unbatch_edge_index

from .censnet import *
from .reference import MultiHeadAttentionLayer
from .utils import combine_graph, sort_edge_index


class CensSubEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.cens_block = CensGCN(args, args.in_features, args.embedding_size, args.nfeat_e, args.dropout)

        self.select_sub1 = SelectSubAttention(self.args, self.args.embedding_size, self.args.topk_ratio)
        self.select_sub2 = SelectSubAttention(self.args, self.args.embedding_size, self.args.topk_ratio)
        self.select_sub3 = SelectSubAttention(self.args, self.args.embedding_size, self.args.topk_ratio)

        self.select_sub4 = SelectSubAttention(self.args, self.args.nfeat_e, self.args.e_topk_ratio)
        self.select_sub5 = SelectSubAttention(self.args, self.args.nfeat_e, self.args.e_topk_ratio)
        self.select_sub6 = SelectSubAttention(self.args, self.args.nfeat_e, self.args.e_topk_ratio)

    def forward(self, data):
        # node-wise data
        v_x = data['sparse_x']
        v_edge_index = data['edge_index']
        v_adj = torch.sparse_coo_tensor(indices=v_edge_index,
                                        values=torch.ones(v_edge_index.shape[1]).to(self.args.device),
                                        size=(v_x.shape[0], v_x.shape[0])).to(self.args.device)
        v_batch = data['batch']
        dist = None

        # edge-wise data
        e_x = data['e_x']
        e_edge_index = data['e_edge_index']
        e_adj = torch.sparse_coo_tensor(indices=e_edge_index,
                                        values=torch.ones(e_edge_index.shape[1]).to(self.args.device),
                                        size=(e_x.shape[0], e_x.shape[0])).to(self.args.device)
        e_batch = data['e_batch']

        # switch gcn, get node repr and edge repr
        # v_adj and e_adj don't have self-loop, model will add that
        T = create_T(v_x.shape[0], v_edge_index).to(self.args.device)
        v_x, e_x = self.cens_block(v_x, v_adj, e_x, e_adj, T)

        # node-wise subgraph
        v_x, v_mask = to_dense_batch(v_x, batch=v_batch)
        v_x1, v_subgraphs1 = self.select_sub1(v_x, v_edge_index, v_mask, dist, v_batch)
        v_x2, v_subgraphs2 = self.select_sub2(v_x1, v_edge_index, v_mask, dist, v_batch)

        # edge-wise subgraph
        e_x, e_mask = to_dense_batch(e_x, batch=e_batch)
        e_x1, e_subgraphs1 = self.select_sub4(e_x, e_edge_index, e_mask, dist, e_batch)
        e_x2, e_subgraphs2 = self.select_sub5(e_x1, e_edge_index, e_mask, dist, e_batch)

        # final subgraph
        v_unbatched_edge_index = unbatch_edge_index(v_edge_index, v_batch)
        subgraphs1 = combine_graph(v_subgraphs1, v_unbatched_edge_index, e_subgraphs1, self.args.batch_size)
        subgraphs2 = combine_graph(v_subgraphs2, v_unbatched_edge_index, e_subgraphs2, self.args.batch_size)

        return subgraphs1, subgraphs2


class SelectSubAttention(nn.Module):
    def __init__(self, args, hid_dim, ratio):
        super(SelectSubAttention, self).__init__()
        self.args = args
        self.MultiHeadAttention_layer = MultiHeadAttentionLayer(self.args, hid_dim)
        self.topk_pool = TopKPooling(hid_dim, ratio=ratio)
        # self.sagPool = SAGPooling(self.args.embedding_size, ratio=self.args.ratio)

    def forward(self, embeddings, edge_index, mask, dist, batch):
        encoder_result = self.MultiHeadAttention_layer(embeddings, mask, dist)

        # dense->spare
        encoder_result_sparse = encoder_result[mask]

        # must unbatch graphs to get usable perm for combining e-v graphs
        # note that topk returns disordered sub_edge_index
        x_list = unbatch(encoder_result_sparse, batch)
        edge_index_list = unbatch_edge_index(edge_index, batch)
        subgraphs = []
        for i in range(self.args.batch_size):
            sub_x, sub_edge_index, _, _, perm, _ = self.topk_pool(x=x_list[i],
                                                                  edge_index=edge_index_list[i])
            sub_edge_index = sort_edge_index(sub_edge_index)
            subgraph = {
                'x': sub_x,
                'edge_index': sub_edge_index,
                'perm': perm
            }
            subgraphs.append(subgraph)
        return encoder_result, subgraphs
