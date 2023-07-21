import torch


def create_T(v_num_nodes, v_edge_index):
    v_nums_edges = v_edge_index.shape[1]
    T = torch.zeros(v_num_nodes, v_nums_edges)

    v_edge_pairs = [pair for pair in zip(v_edge_index[0], v_edge_index[1])]
    for pair in v_edge_pairs:
        T[pair[0]][pair[1]] = 1
    T = T.to_sparse_coo()
    return T
