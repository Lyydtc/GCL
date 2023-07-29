import numpy as np
import torch
from sklearn.utils import shuffle
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree, to_dense_batch, to_dense_adj
import torch_geometric.transforms as T
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import TUDataset, GEDDataset
from sklearn.model_selection import train_test_split

# from TUDataset import TUDataset
from my_utils import get_line_graph, sample_mask


class MyDataset(object):
    def __init__(self, args):
        self.args = args

        if self.args.task == 'cls':
            self.dataset = TUDataset(self.args.data_dir, name=self.args.dataset)
            dataset_size = len(self.dataset)
            shuffled_idx = shuffle(np.array(range(dataset_size)), random_state=0)  # 已经被随机打乱
            train_idx = shuffled_idx[:int(self.args.split * dataset_size)].tolist()
            test_idx = shuffled_idx[int(self.args.split * dataset_size):].tolist()
            self.train_graphs = self.dataset[sample_mask(train_idx, dataset_size)]
            self.test_graphs = self.dataset[sample_mask(test_idx, dataset_size)]
        elif self.args.task == 'gsl':
            self.train_graphs = GEDDataset(self.args.data_dir, name=self.args.dataset, train=True)
            self.test_graphs = GEDDataset(self.args.data_dir, name=self.args.dataset, train=False)
            self.dataset = self.train_graphs + self.test_graphs

        self.process_dataset()

    def process_dataset(self):
        print('Dataset: ', self.args.dataset)
        print('Preparing dataset...')

        if self.test_graphs.data.x is None:
            max_degree = 0
            degs = []
            for data in self.dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if self.args.init_node_encoding == 'RWPE':
                transform = T.AddRandomWalkPE(walk_length=self.args.rwpe_size, attr_name=None)
            elif self.args.init_node_encoding == 'OneHot':
                transform = T.OneHotDegree(max_degree)
            elif self.args.init_node_encoding == 'LapPE':
                transform = T.AddLaplacianEigenvectorPE(k=8, attr_name=None, is_undirected=True)
            self.train_graphs.transform = transform
            self.test_graphs.transform = transform

            # else:
            #     deg = torch.cat(degs, dim=0).to(torch.float)
            #     mean, std = deg.mean().item(), deg.std().item()
            #     self.dataset.transform = NormalizedDegree(mean, std)

    def create_batches(self, graphs):
        # graph pairs for pre-train
        source_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        target_loader = DataLoader(graphs, batch_size=self.args.batch_size, shuffle=True)
        return list(zip(source_loader, target_loader))

    def transform_single(self, data):
        # dense
        x_dense, mask = to_dense_batch(data.x, batch=data.batch)
        adj = to_dense_adj(data.edge_index, batch=data.batch)

        if self.args.task == 'cls':
            target = data.y.to(self.args.device)
        elif self.args.task == 'gsl':
            target = None

        g = {
            # sparse
            # don't use sparse_x because it's not sync with dense_x, which is used in the model
            'sparse_x': data.x.to(self.args.device),
            'edge_index': data.edge_index.to(self.args.device),
            'batch': data.batch.to(self.args.device),
            'target': target,
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
        for i in range(x_dense.shape[0]):
            n_edge_index = data[i].edge_index
            e_x = torch.ones(data[i].num_edges, self.args.nfeat_e)
            e_edge_index = get_line_graph(n_edge_index)
            e_graph_list.append(Data(x=e_x, edge_index=e_edge_index))
        batched_e_graph = Batch.from_data_list(e_graph_list)

        if self.args.task == 'cls':
            target = data.y.to(self.args.device)
        elif self.args.task == 'gsl':
            target = None

        g = {
            # sparse
            'sparse_x': data.x.to(self.args.device),
            'edge_index': data.edge_index.to(self.args.device),
            'batch': data.batch.to(self.args.device),
            'target': target,
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
