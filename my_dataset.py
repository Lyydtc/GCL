import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree, to_dense_batch, to_dense_adj

from TUDataset import TUDataset
from my_utils import get_line_graph

np.random.seed(1)


class MyDataset(object):
    def __init__(self, args):
        self.args = args
        self.training_graphs = None
        self.training_set = None
        self.val_set = None
        self.testing_set = None
        self.testing_graphs = None
        self.nged_matrix = None
        self.real_data_size = None
        self.number_features = None
        self.normed_dist_mat_all = None
        self.n_max_nodes = 0
        self.n_all_graphs = 0
        self.process_dataset()

    def process_dataset(self):
        print('\nPreparing dataset...')
        print('Dataset path: ', self.args.data_dir + self.args.dataset)
        self.training_graphs = TUDataset(self.args.data_dir, name=self.args.dataset)
        self.testing_graphs = TUDataset(self.args.data_dir, name=self.args.dataset).shuffle()
        max_degree = 0
        for g in self.training_graphs + self.testing_graphs:
            if g.edge_index.size(1) > 0:
                max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
        self.args.node_feature_size = self.training_graphs.num_features
        # train_num = len(self.training_graphs) - len(self.testing_graphs)
        # val_num = len(self.testing_graphs)
        # self.training_set, self.val_set = random_split(self.training_graphs, [train_num, val_num])

    def create_batch(self, graphs):
        # 单个图，做图分类使用
        return DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

    def create_batches(self, graphs):
        # 一对图，图相似性计算
        # graphs = graphs.dataset
        source_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        target_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        return list(zip(source_loader, target_loader))

    def transform_single(self, data):
        # dense
        x_dense, mask = to_dense_batch(data.x, batch=data.batch)
        adj = to_dense_adj(data.edge_index, batch=data.batch)

        g = {
            # sparse
            # don't use sparse_x because it's not sync with dense_x, which is used in the model
            'sparse_x': data.x.to(self.args.device),
            'edge_index': data.edge_index.to(self.args.device),
            'batch': data.batch.to(self.args.device),
            'target': data.y.to(self.args.device),
            # dense
            'dense_x': x_dense.to(self.args.device),
            'adj': adj.to(self.args.device),
            'mask': mask.to(self.args.device),
        }
        return g

    def transform_single_for_switch(self, data):
        # dense
        x_dense, mask = to_dense_batch(data.x, batch=data.batch)
        adj = to_dense_adj(data.edge_index, batch=data.batch)

        # line graph
        # get_line_graph() returns edge_index for edge_graph
        # unbatch graph first to reduce time complexity
        e_graph_list = []
        for i in range(self.args.batch_size):
            n_edge_index = data[i].edge_index
            e_x = torch.ones(data[i].num_edges, self.args.nfeat_e)
            e_edge_index = get_line_graph(n_edge_index)
            e_graph_list.append(Data(x=e_x, edge_index=e_edge_index))
        batched_e_graph = Batch.from_data_list(e_graph_list)

        g = {
            # sparse
            'sparse_x': data.x.to(self.args.device),
            'edge_index': data.edge_index.to(self.args.device),
            'batch': data.batch.to(self.args.device),
            'target': data.y.to(self.args.device),
            # dense
            'dense_x': x_dense.to(self.args.device),
            'adj': adj.to(self.args.device),
            'mask': mask.to(self.args.device),
            # line graph
            'e_x': batched_e_graph.x.to(self.args.device),
            'e_edge_index': batched_e_graph.edge_index.to(self.args.device),
            'e_batch': batched_e_graph.batch.to(self.args.device)
        }
        return g
