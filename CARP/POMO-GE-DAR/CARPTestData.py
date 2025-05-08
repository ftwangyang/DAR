import math
import time
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import random
from torch.utils.data import Dataset, DataLoader
import pickle
from GCN import GCN




class CARPTestDataset(Dataset):
    """Simulated Dataset Generator
    This class can generate random points in euclidean space for
    training and testing the reinforcement learning agent.

    ...

    Parameters
    ----------
    num_samples : int
        number of training/testing examples to be generated
    vertex_size  : int
        number of nodes to be generated in each training example
    edge_size  : int
        number of edges to be generated in each training example
    max_load    : int
        maximum load that a vehicle can carry
    max_demand  : int
        maximum demand that a edge can have
    seed        : int
        random seed for reproducing results

    Methods
    -------
    __len__()
        To be used with class instances. class_instance.len returns the num_samples value

    __getitem__(idx)
        returns the specific example at the given index (idx)

    update_mask(mask, dynamic, chosen_idx=None):
        updates the generated mask to hide any invalid states

    update_dynamic(dynamic, chosen_idx):
        updates the loads and demands for the input index (chosen_idx)
    """
    def __init__(self, num_samples, vertex_size, edge_size, device, max_demand=10, max_dhcost=10):

        super(CARPTestDataset, self).__init__()

        self.depot_features = None
        # shape: (batch, 6)
        self.customer_features = None
        # shape: (batch, edge_size, 6)
        self.graph_dynamic = None
        # shape: (batch, edge_size)
        self.graph_info = None
        # shape: (batch, 4, edge_size+1)
        self.D = None
        # shape: (batch, vertex_size, vertex_size)
        self.A = None
        # shape: (batch, edge_size+1, edge_size+1)

        self.vertex_size = vertex_size
        self.edge_size = edge_size
        self.num_samples = num_samples
        self.device = device

        if vertex_size == 10 and edge_size == 20:
            max_load = 30
        elif vertex_size == 30 and edge_size == 50:
            max_load = 50
        elif vertex_size == 50 and edge_size == 100:
            max_load = 100
        else:
            max_load = 100
            # raise NotImplementedError
        # max_load = 100

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        self.max_load = max_load
        self.max_demand = max_demand
        self.max_dhcost = max_dhcost
        self.min_degree = 1
        self.max_degree = 5

        print(
            "样本数量: {0}, 任务: v{1}e{2}, 车载容量: {3}, 最大需求: {4}, 节点度范围:({5},{6}) ".format(
                self.num_samples, self.vertex_size, self.edge_size, self.max_load, self.max_demand, self.min_degree,
                self.max_degree
            )
        )

    def DataGenerate(self):
        self.dynamic = torch.zeros((self.num_samples, self.edge_size + 1))
        self.graph_info = torch.zeros((self.num_samples, self.edge_size + 1, 5))
        self.D = torch.zeros((self.num_samples, self.vertex_size, self.vertex_size))
        vertex_graph_adj = torch.zeros((self.num_samples, self.edge_size + 1, self.edge_size + 1))
        for sample in tqdm(range(self.num_samples), desc="Processing graphs"):
            # 按度生成图
            G, depot, total_cost, total_demand = generate_graph_degree(self.vertex_size, self.edge_size,
                                                                    self.max_dhcost, self.max_demand,min_degree=self.min_degree, max_degree=self.max_degree)
            D_, _ =floyd(G)
            self.D[sample, :, :] = torch.tensor(D_)

            # 将边权图转换为点权图
            vertex_graph = edge2vertex(G, depot)
            vertex_graph_adj_mat = torch.tensor(nx.to_numpy_array(vertex_graph))
            vertex_graph_adj[sample] = vertex_graph_adj_mat
            i = 0
            for node, attributes in vertex_graph.nodes(data=True):
                edge_dhcost = attributes['dhcost']
                edge_demand = attributes['demand']
                node_ori_1 = attributes['node_ori'][0]
                node_ori_2 = attributes['node_ori'][1]

                dynamic_np = [edge_demand/self.max_load]
                graph_info_ = [node, node_ori_1, node_ori_2, edge_dhcost, edge_demand]

                self.dynamic[sample, i] = torch.tensor(dynamic_np)
                self.graph_info[sample, i, :] = torch.tensor(graph_info_)
                i += 1
        # 获取 depot 信息
        depots = self.graph_info[:, 0, 1].long()
        # 获取节点信息
        node_ori_1 = self.graph_info[:, :, 1].long()
        node_ori_2 = self.graph_info[:, :, 2].long()
        # 计算是否为 depot
        f_node_ori_1 = (node_ori_1 == depots[:, None]).long()
        f_node_ori_2 = (node_ori_2 == depots[:, None]).long()
        # 计算距离
        dist_1 = self.D[torch.arange(self.graph_info.size(0)).unsqueeze(1), depots.unsqueeze(1), node_ori_1]
        dist_2 = self.D[torch.arange(self.graph_info.size(0)).unsqueeze(1), depots.unsqueeze(1), node_ori_2]
        # 提取 graph_info 中的其他特征
        graph_info_dhcost = self.graph_info[:, :, 3].clone()
        graph_info_demand = (self.graph_info[:, :, 4] / self.max_load).clone()
        # 计算 total_cost
        total_cost = graph_info_dhcost.sum(dim=1)
        # 归一化graph_info_dhcost
        graph_info_dhcost /= total_cost[:,None]
        # 组合所有特征
        node_features = torch.stack([f_node_ori_1, f_node_ori_2, dist_1, dist_2, graph_info_dhcost, graph_info_demand],dim=2).float()
        self.graph_info = self.graph_info.permute(0, 2, 1)  # [num_samples ,4, edge_size + 1]
        self.depot_features = node_features[:, 0, :] # [num_samples, _, num_features]
        self.customer_features = node_features[:, 1:, :] # [num_samples , edge_size, num_features]

        self.graph_dynamic = self.graph_dynamic[:, 1:] # [num_samples, edge_size, 1]

    def Dataload(self,file_path):

        path = os.path.join(file_path, 'graph_dynamic.txt')
        self.graph_dynamic = torch.load(path, map_location=self.device)
        path = os.path.join(file_path, 'graph_info.txt')
        self.graph_info = torch.load(path, map_location=self.device)
        path = os.path.join(file_path, 'D.txt')
        self.D = torch.load(path, map_location=self.device)
        path = os.path.join(file_path, 'vertex_graph_adj.txt')
        vertex_graph_adj = torch.load(path, map_location=self.device)

        E = torch.eye(vertex_graph_adj.size(1)).unsqueeze(0).expand(vertex_graph_adj.size(0), -1, -1)
        adjacency_matrix = vertex_graph_adj + E
        degree = torch.sum(adjacency_matrix, dim=2)
        degree = torch.diag_embed(degree).to(self.device)
        # 计算度矩阵的倒数
        degree_inv_sqrt = torch.pow(degree, -0.5).to(self.device)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # 处理度为0的情况，避免出现inf
        # 对称归一化
        self.A = torch.matmul(torch.matmul(degree_inv_sqrt, adjacency_matrix), degree_inv_sqrt).float().to(self.device)

        # 获取 depot 信息
        depots = self.graph_info[:, 0, 1].long()
        # 获取节点信息
        node_ori_1 = self.graph_info[:, :, 1].long()
        node_ori_2 = self.graph_info[:, :, 2].long()
        # 计算是否为 depot
        f_node_ori_1 = (node_ori_1 == depots[:, None]).long()
        f_node_ori_2 = (node_ori_2 == depots[:, None]).long()
        # 计算距离
        dist_1 = self.D[torch.arange(self.graph_info.size(0)).unsqueeze(1), depots.unsqueeze(1), node_ori_1]
        dist_2 = self.D[torch.arange(self.graph_info.size(0)).unsqueeze(1), depots.unsqueeze(1), node_ori_2]
        # 提取 graph_info 中的其他特征
        graph_info_dhcost = self.graph_info[:, :, 3].clone()
        graph_info_demand = (self.graph_info[:, :, 4] / self.max_load).clone()
        # 计算 total_cost
        total_cost = graph_info_dhcost.sum(dim=1)
        # 归一化graph_info_dhcost
        graph_info_dhcost /= total_cost[:,None]
        # 组合所有特征
        node_features = torch.stack([f_node_ori_1, f_node_ori_2, dist_1, dist_2, graph_info_dhcost, graph_info_demand],dim=2).float()


        node_ori_1_exp = node_ori_1.unsqueeze(2).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        node_ori_2_exp = node_ori_2.unsqueeze(2).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        targetnode_ori_1_exp = node_ori_1.unsqueeze(1).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        targetnode_ori_2_exp = node_ori_2.unsqueeze(1).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)

        D_expanded = torch.zeros((self.num_samples, self.edge_size + 1, self.edge_size + 1), device="cuda")
        node_size = self.D.size(1)
        # 将 self.D 的值复制到新的张量中
        D_expanded[:, :node_size, :node_size] = self.D
        # 计算所有节点对之间的距离，并取最小值
        distances_1 = D_expanded.gather(2, targetnode_ori_1_exp).gather(1, node_ori_1_exp)
        distances_2 = D_expanded.gather(2, targetnode_ori_2_exp).gather(1, node_ori_1_exp)
        distances_3 = D_expanded.gather(2, targetnode_ori_1_exp).gather(1, node_ori_2_exp)
        distances_4 = D_expanded.gather(2, targetnode_ori_2_exp).gather(1, node_ori_2_exp)

        edge_distance = (distances_1 + distances_2 + distances_3 + distances_4) / 4

        max_values = torch.amax(edge_distance, dim=(1, 2))  # 形状 (128000,)
        normalized_edge_distance = edge_distance / max_values.reshape(-1, 1, 1)
        batch_size, num_nodes, _ = normalized_edge_distance.shape
        diag_indices = torch.arange(num_nodes)
        normalized_edge_distance[:, diag_indices, diag_indices] = 0


        self.graph_info = self.graph_info.permute(0, 2, 1)  # [num_samples ,5, edge_size + 1]
        self.depot_features = node_features[:, 0, :] # [num_samples, _, num_features]
        self.customer_features = node_features[:, 1:, :] # [num_samples , edge_size, num_features]


        self.graph_dynamic = self.graph_dynamic[:, 1:] # [num_samples, edge_size, 1]

        self.edge_distance = normalized_edge_distance
        print("=================================================================")

    def __len__(self):
        """Returns the number of examples being trained/tested on"""
        return self.num_samples

    def __getitem__(self, idx):
        """Returns the specific example at the given index (idx)
        Parameters
        ----------
        idx : int
            index for which the example has to be returned.
        """
        return ((self.depot_features[idx], self.customer_features[idx], self.graph_dynamic[idx]), self.graph_info[idx], self.D[idx], self.A[idx], self.edge_distance[idx])




def edge2vertex(edge_graph, depot):
    G = nx.Graph()
    edge_info_list = []
    edge_info_list.append((depot, depot, 0, 0))

    G.add_node(0, demand=0, dhcost=0,node_ori=(depot,depot))
    i = 1
    for node1, node2, edge_data in edge_graph.edges(data=True):
        demand = edge_data['demand']
        dhcost = edge_data['dhcost']
        G.add_node(i, demand=demand, dhcost=dhcost,node_ori=(node1,node2) )
        i += 1
        edge_info_list.append((node1, node2, demand, dhcost))


    for index1, (index1_node1, index1_node2, index1_demand, index1_dhcost) in enumerate(edge_info_list):
        for index2 in range(index1 + 1, len(edge_info_list)):
            data_index2 = edge_info_list[index2]
            index2_node1 = data_index2[0]
            index2_node2 = data_index2[1]
            if (index1_node1 == index2_node1 or index1_node1 == index2_node2 or index1_node2 == index2_node1 or index1_node2 == index2_node2):
                G.add_edge(index1, index2)

    return G


def floyd(G):
    num_nodes = G.number_of_nodes()
    # 初始化邻接矩阵为无穷大
    adj_matrix = np.full((num_nodes, num_nodes), np.inf)
    # 将有边特征的边存入邻接矩阵
    for node in G.nodes():
        adj_matrix[node, node] = 0
    for node1, node2, edge_data in G.edges(data=True):
        dhcost = edge_data['dhcost']
        adj_matrix[node1, node2] = dhcost
        adj_matrix[node2, node1] = dhcost

    num_nodes = len(adj_matrix)

    # 初始化距离矩阵
    distance_matrix = np.copy(adj_matrix)

    # 初始化路径矩阵，用于记录最短路径的中间节点
    path_matrix = np.ones((num_nodes, num_nodes), dtype=int) * -1

    # Floyd-Warshall 算法
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                    path_matrix[i, j] = k

    return distance_matrix, path_matrix

def generate_graph_degree(vertex_size, edge_size, max_dhcost, max_demand, min_degree, max_degree):
    errorNum = 0
    while True:
        if errorNum == 100:
            raise ValueError("节点与度的数量设置不合理")

        total_cost, total_demand= 0,0

        # 生成地图 根据点的度生成
        G = nx.Graph()
        G.add_nodes_from(range(vertex_size))

        # 为每个节点添加边，确保图是连通的
        for node in G.nodes():

            # 获取当前节点的度
            degree = G.degree(node)
            degree = random.randint(max(degree,min_degree), max_degree)

            select_nodes = [n for n in G.nodes() if G.degree(n) < max_degree and n != node]

            if len(select_nodes) < max(degree - G.degree(node), 0):
                errorNum += 1
                break

            # 随机选择与当前节点相邻的其他节点
            neighbors = random.sample(select_nodes, max(degree - G.degree(node), 0))

            # 添加边
            for neighbor in neighbors:
                dhcost = random.randint(1, max_dhcost)
                total_cost += dhcost
                #demand = random.randint(1, max_demand) if random.random() < 0.7 else 0
                demand = random.randint(1, max_demand)
                total_demand += demand
                G.add_edge(node, neighbor, dhcost=dhcost,demand=demand)

        is_done = False
        iteration_count = 0
        while not is_done and iteration_count < 100:
            iteration_count += 1
            if G.number_of_edges() < edge_size:
                add_nodes = [n for n in G.nodes() if G.degree(n) < max_degree]
                if not add_nodes:
                    break
                add_node = random.sample(add_nodes,k=1)[0]
                select_nodes = [n for n in G.nodes() if
                                G.degree(n) < max_degree and n != add_node and not G.has_edge(n, add_node)]
                if not select_nodes:
                    break
                neighbor = random.sample(select_nodes,k=1)[0]
                dhcost = random.randint(1, max_dhcost)
                total_cost += dhcost
                demand = random.randint(1, max_demand)
                total_demand += demand
                G.add_edge(add_node, neighbor, dhcost=dhcost,demand=demand)

            elif G.number_of_edges() > edge_size:
                delete_nodes = [n for n in G.nodes() if G.degree(n) > min_degree]
                if not delete_nodes:
                    break
                delete_node = random.sample(delete_nodes,k=1)[0]
                select_nodes = [n for n in G.neighbors(delete_node) if G.degree(n) > min_degree]
                if not select_nodes:
                    break
                neighbor = random.sample(select_nodes,k=1)[0]
                edge_data = G.get_edge_data(delete_node, neighbor)
                if edge_data and 'weight' in edge_data:
                    total_cost -= edge_data['dhcost']
                    total_demand -= edge_data['demand']
                G.remove_edge(delete_node, neighbor)

            if G.number_of_edges() == edge_size:
                is_done = True

        # 随机将一个节点设为depot
        depot = random.choice(list(G.nodes))

        if nx.is_connected(G) and G.number_of_edges() == edge_size:
        #if nx.is_connected(G):
            return G, depot, total_cost, total_demand



