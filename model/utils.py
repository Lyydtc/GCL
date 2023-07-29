import torch
import dgl
from torch_geometric.utils import subgraph


def combine_graph(v_graphs, v_unbatched_edge_index, e_graphs, batch_size):
    """
    *_graphs is a list of subgraph

    v_unbatched_edge_index is a list of original edge_index before topk, use
    perm of each graph to know which nodes are selected by e_graph

    return a batch-size list of combined graphs
    """
    subgraphs = []
    for i in range(batch_size):
        v_graph = v_graphs[i]
        v_x = v_graph['x']
        v_edge_index = v_graph['edge_index']
        v_perm = v_graph['perm']
        v_origin_edge_index = v_unbatched_edge_index[i]
        v_origin_edge_pair = [(int(pair[0]), int(pair[1]))
                              for pair in zip(v_origin_edge_index[0], v_origin_edge_index[1])]

        e_graph = e_graphs[i]
        e_perm = e_graph['perm'].tolist()

        # get node that is also selected by e_graph
        mask = torch.full((v_perm.shape[0], ), False)
        node = torch.arange(start=0, end=v_perm.shape[0])
        for index in e_perm:
            (ni, nj) = v_origin_edge_pair[index]
            if ni in v_perm.tolist():
                mask[v_perm.tolist().index(ni)] = True
            if nj in v_perm.tolist():
                mask[v_perm.tolist().index(nj)] = True
        selected_node = node[mask].tolist()

        new_x = v_x[mask]

        # use dgl.node_sugraph() to get new_edge_index, pyg now don't have such api
        g = dgl.graph((v_origin_edge_index[0], v_origin_edge_index[1]))
        sg = dgl.node_subgraph(g, selected_node)
        new_edge_index = sg.edges()
        new_edge_index = torch.tensor([new_edge_index[0].tolist(), new_edge_index[1].tolist()])

        combined_graph = {
            'x': new_x,
            'edge_index': new_edge_index
        }

        subgraphs.append(combined_graph)
    return subgraphs


def sort_edge_index(edge_index):
    edge_pair_list = [pair for pair in zip(edge_index[0], edge_index[1])]
    edge_pair_list.sort(key=lambda pair: pair[0])
    edge_index = torch.tensor([[pair[0] for pair in edge_pair_list], [pair[1] for pair in edge_pair_list]])
    return edge_index


def reindex(edge_index):
    # make the node index continuies
    id_list = list(set((edge_index[0].tolist() + edge_index[1].tolist())))
    id_list.sort()
    mapping = {old: new for new, old in enumerate(id_list)}
    for i in range(edge_index.shape[1]):
        edge_index[0][i] = torch.tensor(mapping[int(edge_index[0][i])])
        edge_index[1][i] = torch.tensor(mapping[int(edge_index[1][i])])
    return edge_index
