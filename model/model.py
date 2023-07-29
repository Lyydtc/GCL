import torch
from torch import nn
from torch_geometric.utils import unbatch, unbatch_edge_index, to_dense_batch
from timeit import default_timer as timer

from .reference import SimpleAlign
from .my_align import AlignLayer
from .sub_encoder import SubEncoder
from .cens_sub_encoder import CensSubEncoder
from .cross_attention import MultiHeadCrossAttentionLayer


class MyModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # subgraph
        if args.switch:
            self.encoder1 = CensSubEncoder(args)
            self.encoder2 = CensSubEncoder(args)
        else:
            self.encoder1 = SubEncoder(args)
            self.encoder2 = SubEncoder(args)

        # subgraph num_node align
        # self.align1 = SimpleAlign(self.args.embedding_size, self.args.align_size, self.args.dropout)
        # self.align2 = SimpleAlign(self.args.embedding_size, self.args.align_size, self.args.dropout)
        self.align1 = AlignLayer(args.embedding_size, args.align_size)
        self.align2 = AlignLayer(args.embedding_size, args.align_size)

        self.cross_attention = MultiHeadCrossAttentionLayer(args, args.embedding_size, args.n_heads)

        self.projection_head = nn.Sequential(nn.Linear(args.embedding_size * args.n_heads, args.embedding_size),
                                             nn.ReLU(),
                                             nn.Dropout(p=self.args.dropout),
                                             nn.Linear(args.embedding_size, args.embedding_size),
                                             nn.ReLU(),
                                             nn.Dropout(p=self.args.dropout),
                                             nn.Linear(args.embedding_size, args.embedding_size))

    def forward(self, data1, data2):

        # get subgraph representations
        if self.args.switch:
            subgraphs11, subgraphs12 = self.encoder1(data1)
            subgraphs21, subgraphs22 = self.encoder2(data2)

            if self.args.align:
                sub1_x1 = self.align_graphs(subgraphs11, self.align1)
                sub1_x2 = self.align_graphs(subgraphs12, self.align1)

                sub2_x1 = self.align_unbatched_graphs(subgraphs21, self.align2)
                sub2_x2 = self.align_unbatched_graphs(subgraphs22, self.align2)
            else:
                print('to be finished...')
                exit()

            sub1 = sub1_x1 + sub1_x2
            sub2 = sub2_x1 + sub2_x2

        else:
            # tic = timer()
            subgraphs11, subgraphs12 = self.encoder1(data1)
            subgraphs21, subgraphs22 = self.encoder2(data2)
            # print('encoder done, ', timer() - tic)
            # tic = timer()

            if self.args.align:
                sub1_x1 = self.align_batched_graph(subgraphs11, self.align1)
                sub1_x2 = self.align_batched_graph(subgraphs12, self.align1)

                sub2_x1 = self.align_batched_graph(subgraphs21, self.align2)
                sub2_x2 = self.align_batched_graph(subgraphs22, self.align2)
                # print('align done, ', timer() - tic)
                # tic = timer()
            else:
                sub1_x1, _ = to_dense_batch(subgraphs11['x'], subgraphs11['batch'])
                sub1_x2, _ = to_dense_batch(subgraphs12['x'], subgraphs12['batch'])

                sub2_x1, _ = to_dense_batch(subgraphs21['x'], subgraphs21['batch'])
                sub2_x2, _ = to_dense_batch(subgraphs22['x'], subgraphs22['batch'])

            sub1 = sub1_x1 + sub1_x2
            sub2 = sub2_x1 + sub2_x2

        # CLS
        cls1 = torch.mean(sub1, dim=1, keepdim=True)
        cls2 = torch.mean(sub2, dim=1, keepdim=True)

        # add cls to sub
        sub_with_cls1 = torch.cat((cls1, sub1), dim=1)
        sub_with_cls2 = torch.cat((cls2, sub2), dim=1)

        # cross attention
        out1, out2 = self.cross_attention(sub_with_cls1, sub_with_cls2)
        sub_x1 = out1[:, 0, :]
        sub_x2 = out2[:, 0, :]

        sub_x1 = self.projection_head(sub_x1)
        sub_x2 = self.projection_head(sub_x2)

        return sub_x1, sub_x2

    def get_embeddings(self, data1, data2):
        # get subgraph representations
        if self.args.switch:
            subgraphs11, subgraphs12 = self.encoder1(data1)
            subgraphs21, subgraphs22 = self.encoder2(data2)

            if self.args.align:
                sub1_x1 = self.align_graphs(subgraphs11, self.align1)
                sub1_x2 = self.align_graphs(subgraphs12, self.align1)

                sub2_x1 = self.align_unbatched_graphs(subgraphs21, self.align2)
                sub2_x2 = self.align_unbatched_graphs(subgraphs22, self.align2)
            else:
                print('to be finished...')
                exit()

            sub1 = sub1_x1 + sub1_x2
            sub2 = sub2_x1 + sub2_x2

        else:
            # tic = timer()
            subgraphs11, subgraphs12 = self.encoder1(data1)
            subgraphs21, subgraphs22 = self.encoder2(data2)
            # print('encoder done, ', timer() - tic)
            # tic = timer()

            if self.args.align:
                sub1_x1 = self.align_batched_graph(subgraphs11, self.align1)
                sub1_x2 = self.align_batched_graph(subgraphs12, self.align1)

                sub2_x1 = self.align_batched_graph(subgraphs21, self.align2)
                sub2_x2 = self.align_batched_graph(subgraphs22, self.align2)
                # print('align done, ', timer() - tic)
                # tic = timer()
            else:
                sub1_x1, _ = to_dense_batch(subgraphs11['x'], subgraphs11['batch'])
                sub1_x2, _ = to_dense_batch(subgraphs12['x'], subgraphs12['batch'])

                sub2_x1, _ = to_dense_batch(subgraphs21['x'], subgraphs21['batch'])
                sub2_x2, _ = to_dense_batch(subgraphs22['x'], subgraphs22['batch'])

            sub1 = sub1_x1 + sub1_x2
            sub2 = sub2_x1 + sub2_x2

        # CLS
        cls1 = torch.mean(sub1, dim=1, keepdim=True)
        cls2 = torch.mean(sub2, dim=1, keepdim=True)

        # add cls to sub
        sub_with_cls1 = torch.cat((cls1, sub1), dim=1)
        sub_with_cls2 = torch.cat((cls2, sub2), dim=1)

        # cross attention
        out1, out2 = self.cross_attention(sub_with_cls1, sub_with_cls2)
        sub_x1 = out1[:, 0, :]
        sub_x2 = out2[:, 0, :]

        return sub_x1, sub_x2

    def align_batched_graph(self, graph, align_layer):
        batched_x = graph['x']
        batched_edge_index = graph['edge_index']
        batch = graph['batch']

        # unbatch graph data
        x_list = unbatch(batched_x, batch)
        edge_index_list = unbatch_edge_index(batched_edge_index.to(self.args.device), batch.to(self.args.device))

        aligned_x_list = []
        for i in range(len(x_list)):
            x = x_list[i]
            # if i >= len(edge_index_list):
            #     aligned_x = x[:self.args.align_size]
            #     aligned_x_list.append(aligned_x)
            #     continue
            edge_index = edge_index_list[i]
            assignment = align_layer(x, edge_index)
            aligned_x = torch.mm(assignment.T, x)
            aligned_x_list.append(aligned_x)
        return torch.stack(aligned_x_list)

    def align_graphs(self, graphs, align_layer):
        aligned_x_list = []
        for i in range(len(graphs)):
            x = graphs[i]['x']
            edge_index = graphs[i]['edge_index']
            assignment = align_layer(x, edge_index)
            aligned_x = torch.mm(assignment.T, x)
            aligned_x_list.append(aligned_x)
        return torch.stack(aligned_x_list)
