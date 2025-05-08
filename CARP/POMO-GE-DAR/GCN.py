import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(device))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(device))
        nn.init.xavier_uniform_(self.weight)  # 初始化权重
        nn.init.zeros_(self.bias)  # 初始化偏置

    def forward(self, A, node_features):
        # GCN的前向传播
        batch_size = A.size(0)
        weight = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
        support = torch.matmul(node_features, weight)
        output = torch.matmul(A, support)
        output += self.bias.unsqueeze(0).unsqueeze(0)

        # support = torch.matmul(node_features, self.weight)
        # output = torch.matmul(A, support)
        # output += self.bias[:,None,:]
        return output

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        # self.gc1 = GraphConvolutionLayer(input_dim, 64)
        # self.gc2 = GraphConvolutionLayer(64, 16)
        # self.gc3 = GraphConvolutionLayer(16, output_dim)

        self.gc1 = GraphConvolutionLayer(input_dim, 32)
        self.gc2 = GraphConvolutionLayer(32, 64)
        self.gc3 = GraphConvolutionLayer(64, output_dim)

        # self.gc1 = GraphConvolutionLayer(input_dim, 64)
        # self.gc2 = GraphConvolutionLayer(64, output_dim)


    def forward(self, A, node_features):
        h = F.relu(self.gc1(A, node_features))
        h = F.relu(self.gc2(A, h))
        h = torch.tanh(self.gc3(A, h))
        return h


# class GCN(torch.nn.Module):
#     def __init__(self, input_features, output_features):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_features, 64)
#         self.conv2 = GCNConv(64, 16)
#         self.conv3 = GCNConv(16, output_features)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x, edge_index, edge_weight=None):
#         '''
#         GCN
#         '''
#         x = self.relu(self.conv1(x, edge_index, edge_weight=None))
#         x = F.dropout(x, training=self.training)
#         x = self.relu(self.conv2(x, edge_index, edge_weight=None))
#         x = F.dropout(x, training=self.training)
#         x = torch.tanh(self.conv3(x, edge_index, edge_weight=None))
#
#         return x