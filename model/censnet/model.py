import torch.nn as nn
import torch.nn.functional as F
from .layer import GraphConvolution


class CensGCN(nn.Module):
    def __init__(self, args, nfeat_v, nhid_v, nfeat_e, dropout):
        super().__init__()

        self.gc1 = GraphConvolution(args, nfeat_v, nhid_v, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(args, nhid_v, nhid_v, nfeat_e, nfeat_e, node_layer=False)
        self.gc3 = GraphConvolution(args, nhid_v, nhid_v, nfeat_e, nfeat_e, node_layer=True)
        self.dropout = dropout

    def forward(self, v_x, v_adj, e_x, e_adj, T):
        """

        Args:
            v_x: (num_nodes, node_feats_dim)
            e_x: (num_edges, edge_feats_dim)
            e_adj: (2, num_e_edges) -> torch sparse tensor
            v_adj: (2, num_v_edges) -> torch sparse tensor
            T: (num_v_nodes, num_v_edges) -> torch sparse tensor

        Returns:

        """
        gc1 = self.gc1(v_x, e_x, e_adj, v_adj, T)
        v_x, e_x = F.relu(gc1[0]), F.relu(gc1[1])

        v_x = F.dropout(v_x, self.dropout, training=self.training)
        e_x = F.dropout(e_x, self.dropout, training=self.training)

        gc2 = self.gc2(v_x, e_x, e_adj, v_adj, T)
        v_x, e_x = F.relu(gc2[0]), F.relu(gc2[1])

        v_x = F.dropout(v_x, self.dropout, training=self.training)
        e_x = F.dropout(e_x, self.dropout, training=self.training)

        v_x, e_x = self.gc3(v_x, e_x, e_adj, v_adj, T)
        return v_x, e_x
