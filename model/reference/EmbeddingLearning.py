import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.dense import DenseGCNConv, DenseGINConv, DenseSAGEConv


class GCNLayer(nn.Module):
    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.args = args
        self.GCN_first = DenseGCNConv(args.in_features, args.embedding_size)
        self.GCN_second = DenseGCNConv(args.embedding_size, args.embedding_size)
        self.GCN_third = DenseGCNConv(args.embedding_size, args.embedding_size)

        torch.nn.init.xavier_uniform_(self.GCN_first.lin.weight)
        torch.nn.init.xavier_uniform_(self.GCN_second.lin.weight)
        torch.nn.init.xavier_uniform_(self.GCN_third.lin.weight)

    def forward(self, x, adj, mask, dist=None):
        first_gcn_result = F.relu(self.GCN_first(x, adj, mask))
        second_gcn_result = F.relu(self.GCN_second(first_gcn_result, adj, mask))
        gcn_result = F.relu(self.GCN_third(second_gcn_result, adj, mask))
        # 残差连接消融实验部分
        gcn_result = gcn_result+first_gcn_result + second_gcn_result
        return gcn_result


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, args, hid_dim):
        super(MultiHeadAttentionLayer, self).__init__()
        self.args = args
        self.d_k = hid_dim
        self.self_attention_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(args, hid_dim)
        self.self_attention_dropout = nn.Dropout(args.dropout)

        self.ffn_norm = nn.LayerNorm(hid_dim)
        self.ffn = FeedForwardNetwork(args, hid_dim)
        self.ffn_dropout = nn.Dropout(args.dropout)

    def forward(self, gcn_result, mask, dist=None):
        self_att_result = self.self_attention_norm(gcn_result)
        self_att_result = self.self_attention(self_att_result, dist, mask)
        self_att_result = self.self_attention_dropout(self_att_result)

        self_att_result = gcn_result + self_att_result
        ffn_result = self.ffn_norm(self_att_result)
        ffn_result = self.ffn(ffn_result)
        # batch_size N enbedding_size
        # batch_size N
        # batch_size N enbedding_size
        ffn_result = torch.einsum('bne,bn->bne', ffn_result, mask)
        ffn_result = self.ffn_dropout(ffn_result)
        self_att_result = self_att_result + ffn_result
        encoder_result = gcn_result + self_att_result
        return encoder_result


class FeedForwardNetwork(nn.Module):
    def __init__(self, args, hid_dim):
        super(FeedForwardNetwork, self).__init__()
        self.args = args

        hidden_size = hid_dim
        ffn_size = args.encoder_ffn_size
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, args, hid_dim):
        super(MultiHeadAttention, self).__init__()
        self.args = args

        self.num_heads = num_heads = args.n_heads
        embedding_size = hid_dim

        self.att_size = embedding_size
        self.scale = embedding_size ** -0.5

        self.linear_q = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.linear_k = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.linear_v = nn.Linear(embedding_size, num_heads * embedding_size, bias=args.msa_bias)
        self.att_dropout = nn.Dropout(args.dropout)

        self.output_layer = nn.Linear(num_heads * embedding_size, embedding_size)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        torch.nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(self, x, dist=None, mask=None):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = x.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, d_k).transpose(-2, -3).transpose(-1, -2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, d_v).transpose(-2, -3)
        q = torch.einsum('bhne,bn->bhne', q, mask)
        k = torch.einsum('bhen,bn->bhen', k, mask)
        v = torch.einsum('bhne,bn->bhne', v, mask)

        q = q * self.scale
        a = torch.matmul(q, k)

        # distance消融实验部分
        if dist is not None:
            a += torch.stack([dist] * self.args.n_heads, dim=1).to(self.args.device)

        attention_mask = torch.einsum('ij,ik->ijk', mask, mask)
        a = a.transpose(0, 1).masked_fill(attention_mask == 0, -1e9)
        a = torch.softmax(a, dim=3).masked_fill(attention_mask == 0, 0).transpose(0, 1)

        a = self.att_dropout(a)

        y = a.matmul(v).transpose(-2, -3).contiguous().view(batch_size, -1, self.num_heads * d_v)
        y = self.output_layer(y)

        y = torch.einsum('bne,bn->bne', y, mask)

        return y


class GCNTransformerEncoder(nn.Module):
    def __init__(self, args, q=None, k=None):
        super(GCNTransformerEncoder, self).__init__()
        self.args = args

        self.GCN_first = DenseGCNConv(args.in_features, args.embedding_size)
        self.GCN_second = DenseGCNConv(args.embedding_size, args.embedding_size)
        self.GCN_third = DenseGCNConv(args.embedding_size, args.embedding_size)

        self.d_k = args.embedding_size

        torch.nn.init.xavier_uniform_(self.GCN_first.lin.weight)
        torch.nn.init.xavier_uniform_(self.GCN_second.lin.weight)
        torch.nn.init.xavier_uniform_(self.GCN_third.lin.weight)

        self.self_attention_norm = nn.LayerNorm(args.embedding_size)
        self.self_attention = MultiHeadAttention(args, q, k)
        self.self_attention_dropout = nn.Dropout(args.dropout)

        self.ffn_norm = nn.LayerNorm(args.embedding_size)
        self.ffn = FeedForwardNetwork(args)
        self.ffn_dropout = nn.Dropout(args.dropout)

    def forward(self, x, adj, mask, dist=None):
        first_gcn_result = F.relu(self.GCN_first(x, adj, mask))
        second_gcn_result = F.relu(self.GCN_second(first_gcn_result, adj, mask))
        gcn_result = F.relu(self.GCN_third(second_gcn_result, adj, mask))
        # 残差连接消融实验部分
        gcn_result = gcn_result + first_gcn_result + second_gcn_result
        # GT
        self_att_result = self.self_attention_norm(gcn_result)
        self_att_result = self.self_attention(self_att_result, dist, mask)
        self_att_result = self.self_attention_dropout(self_att_result)

        self_att_result = gcn_result + self_att_result
        ffn_result = self.ffn_norm(self_att_result)
        ffn_result = self.ffn(ffn_result)
        # batch_size N enbedding_size
        # batch_size N
        # batch_size N enbedding_size
        ffn_result = torch.einsum('bne,bn->bne', ffn_result, mask)
        ffn_result = self.ffn_dropout(ffn_result)
        self_att_result = self_att_result + ffn_result
        encoder_result = gcn_result + self_att_result
        return encoder_result
