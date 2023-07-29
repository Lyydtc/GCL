import random

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_node, dropout_edge, dropout_path


def augment(data, args):
    # unbatch data then augment
    new_list = []
    for i in range(args.batch_size):
        dice = random.randint(1, 3)
        if dice == 1:
            new_graph = drop_node(data[i], args.p_node)
        elif dice == 2:
            new_graph = drop_edge(data[i], args.p_edge)
        else:
            new_graph = drop_path(data[i], args.p_path, args.path_length)

        new_list.append(new_graph)

    new_data = Batch.from_data_list(new_list)
    return new_data


def drop_node(data: Data, p):
    new_edge_index, edge_mask, node_mask = dropout_node(data.edge_index, p)
    edge_index = reindex(new_edge_index)
    new_data = Data(x=data.x[node_mask], edge_index=edge_index, y=data.y)
    return new_data


def reindex(edge_index):
    # make the node index continuies
    id_list = list(set((edge_index[0].tolist() + edge_index[1].tolist())))
    id_list.sort()
    mapping = {old: new for new, old in enumerate(id_list)}
    for i in range(edge_index.shape[1]):
        edge_index[0][i] = torch.tensor(mapping[int(edge_index[0][i])])
        edge_index[1][i] = torch.tensor(mapping[int(edge_index[1][i])])
    return edge_index


def drop_edge(data: Data, p):
    new_edge_index, edge_mask = dropout_edge(data.edge_index, p)
    new_data = Data(x=data.x, edge_index=new_edge_index, y=data.y)
    return new_data


def drop_path(data: Data, p, length):
    new_edge_index, edge_mask = dropout_path(data.edge_index, p, walk_length=length)
    new_data = Data(x=data.x, edge_index=new_edge_index, y=data.y)
    return new_data
