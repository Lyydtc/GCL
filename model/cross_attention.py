import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttentionLayer(nn.Module):
    def __init__(self, args, input_size, num_heads):
        super(MultiHeadCrossAttentionLayer, self).__init__()
        self.args = args
        self.input_size = input_size
        self.num_heads = num_heads

        self.fc_query = nn.Linear(input_size, input_size * num_heads)
        self.fc_key = nn.Linear(input_size, input_size * num_heads)
        self.fc_value = nn.Linear(input_size, input_size * num_heads)
        self.fc_output = nn.Linear(input_size * num_heads, input_size * num_heads)

        self.scale_factor = 1.0 / (input_size ** 0.5)  # Scaling factor

    def forward(self, input1, input2):
        # query1 = self.fc_query(input1)
        # key1 = self.fc_key(input2)
        # value1 = self.fc_value(input2)
        #
        # query2 = self.fc_query(input2)
        # key2 = self.fc_key(input1)
        # value2 = self.fc_value(input1)
        #
        # batch_size = input1.size(0)
        # query1 = query1.view(batch_size, self.num_heads, -1, self.input_size)
        # key1 = key1.view(batch_size, self.num_heads, -1, self.input_size)
        # value1 = value1.view(batch_size, self.num_heads, -1, self.input_size)
        #
        # query2 = query2.view(batch_size, self.num_heads, -1, self.input_size)
        # key2 = key2.view(batch_size, self.num_heads, -1, self.input_size)
        # value2 = value2.view(batch_size, self.num_heads, -1, self.input_size)
        #
        # scores1 = torch.matmul(query1, key1.transpose(-2, -1)) * self.scale_factor
        # attention_weights1 = F.softmax(scores1, dim=-1)
        #
        # scores2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.scale_factor
        # attention_weights2 = F.softmax(scores2, dim=-1)
        #
        # # Compute the weighted sum of value vectors for output 1
        # output1 = torch.matmul(attention_weights1, value1)
        # output1 = output1.view(batch_size, -1, self.input_size * self.num_heads)
        # output1 = self.fc_output(output1)
        #
        # output2 = torch.matmul(attention_weights2, value2)
        # output2 = output2.view(batch_size, -1, self.input_size * self.num_heads)
        # output2 = self.fc_output(output2)
        
        query1 = self.fc_query(input1)
        key1 = self.fc_key(input2)
        # value1 = self.fc_value(input2)
        value1 = self.fc_value(input1)

        query2 = self.fc_query(input2)
        key2 = self.fc_key(input1)
        # value2 = self.fc_value(input1)
        value2 = self.fc_value(input2)

        batch_size = input1.shape[0]
        query1 = query1.view(batch_size, self.num_heads, -1, self.input_size)
        key1 = key1.view(batch_size, self.num_heads, -1, self.input_size)
        value1 = value1.view(batch_size, self.num_heads, -1, self.input_size)

        query2 = query2.view(batch_size, self.num_heads, -1, self.input_size)
        key2 = key2.view(batch_size, self.num_heads, -1, self.input_size)
        value2 = value2.view(batch_size, self.num_heads, -1, self.input_size)

        scores1 = torch.matmul(query1, key1.transpose(-2, -1)) * self.scale_factor
        attention_weights1 = F.softmax(scores1, dim=-1)

        scores2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.scale_factor
        attention_weights2 = F.softmax(scores2, dim=-1)

        # Compute the weighted sum of value vectors for output 1
        # 用a乘以v_i
        output1 = torch.matmul(attention_weights1.transpose(-2, -1), value1)
        output1 = output1.view(batch_size, -1, self.input_size * self.num_heads)
        output1 = self.fc_output(output1)

        output2 = torch.matmul(attention_weights2.transpose(-2, -1), value2)
        output2 = output2.view(batch_size, -1, self.input_size * self.num_heads)
        output2 = self.fc_output(output2)

        return output1, output2
