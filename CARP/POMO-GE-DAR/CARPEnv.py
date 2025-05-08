import time
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class Reset_State:
    depot: torch.Tensor = None
    # shape: (batch, 1, 2)
    customer: torch.Tensor = None
    # shape: (batch, problem, 2)
    customer_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_edge: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, edge_size+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CARPEnv:
    def __init__(self,device, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.vertex_size = env_params['vertex_size']
        self.edge_size = env_params['edge_size']
        self.pomo_size = env_params['pomo_size']

        self.norm_IN = nn.InstanceNorm1d(8, affine=True, track_running_stats=False)
        self.norm_BN = nn.BatchNorm1d(8, affine=True)

        # Const @Load_Problem
        ####################################
        self.device = device
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_customer = None
        # shape: (batch, edge_size+1, 2)
        self.depot_customer_demand = None
        # shape: (batch, edge_size+1)
        self.graph_info = None
        # shape: (batch, 4, edge_size+1)
        self.D = None
        # shape: (batch, vertex_size, vertex_size)
        self.A = None
        # shape: (batch, edge_size+1, edge_size+1)
        self.edge_distance = None
        # shape: (batch, edge_size+1, edge_size+1)


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_edge = None
        # shape: (batch, pomo)
        self.selected_edge_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, edge_size+1)
        self.ninf_mask = None
        # shape: (batch, pomo, edge_size+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()


    def load_problems(self, batch_size, batch):
        self.batch_size = batch_size

        (depot, customer, customer_demand), self.graph_info, self.D, self.A, self.edge_distance = batch
        depot = depot.to(self.device)
        customer = customer.to(self.device)
        customer_demand = customer_demand.to(self.device)
        self.graph_info = self.graph_info.to(self.device)
        self.D = self.D.to(self.device)
        self.A = self.A.to(self.device)
        self.edge_distance = self.edge_distance.to(self.device)

        self.depot_customer = torch.cat((depot[:,None,:], customer), dim=1)

        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_customer_demand = torch.cat((depot_demand, customer_demand), dim=1)
        # shape: (batch, edge_size+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot = depot
        self.reset_state.customer = customer
        self.reset_state.customer_demand = customer_demand
        self.reset_state.graph_info = self.graph_info
        self.reset_state.A = self.A
        self.reset_state.edge_distance = self.edge_distance

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_edge = None
        # shape: (batch, pomo)
        self.selected_edge_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.edge_size+1))
        # shape: (batch, pomo, edge_size+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.edge_size+1))
        # shape: (batch, pomo, edge_size+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_edge = self.current_edge

        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.current_near_edges = torch.zeros((self.batch_size, self.pomo_size, 5), dtype=torch.long)
        # shape: (batch, pomo, 5)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_edge = selected
        # shape: (batch, pomo)
        self.selected_edge_list = torch.cat((self.selected_edge_list, self.current_edge[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # edge_indices = self.current_edge.unsqueeze(2).expand(-1, -1, self.edge_distance.size(2))
        # distances = torch.gather(self.edge_distance, 1, edge_indices)
        # _, nearest_edge_indices = torch.topk(distances, 21, dim=2, largest=False)
        # nearest_edge_indices = nearest_edge_indices[:,:,1:]
        # self.current_near_edges = nearest_edge_indices
        # self.current_near_edges = None

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_customer_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, edge_size+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, edge_size+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, edge_size+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load

        self.step_state.current_edge = self.current_edge
        self.step_state.current_near_edges = self.current_near_edges
        self.step_state.edge_distance = self.edge_distance

        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):

        total_dhcost = torch.zeros(self.batch_size*self.pomo_size)

        pi = self.selected_edge_list
        pi_num_samples, pomo_size, tour_length = pi.size()
        pi = pi.view(pi_num_samples * pomo_size, tour_length)

        idx = pi.unsqueeze(1).expand(-1, self.graph_info.size(1), -1)
        graph_info = self.graph_info.unsqueeze(1).repeat(1, pomo_size, 1, 1)
        graph_info_num_samples, feature_size, edge_size = self.graph_info.size()
        graph_info = graph_info.view(graph_info_num_samples * pomo_size, feature_size, edge_size)
        tour = torch.gather(graph_info, 2, idx).to(int)

        D = self.D.unsqueeze(1).repeat(1, pomo_size, 1, 1)
        D_num_samples, _, node_size = self.D.size()
        D = D.view(D_num_samples * pomo_size, node_size, node_size)

        num_samples, _, tour_length = tour.size()
        f_1 = torch.zeros(num_samples)
        f_2 = torch.zeros(num_samples)
        depot = torch.zeros(num_samples)
        depot = graph_info[:, 1, 0].long()
        indices = torch.arange(num_samples)

        for i in range(1, tour_length + 1):
            if i == 1:
                node_1_front = tour[:, 1, -i - 1]
                node_2_front = tour[:, 2, -i - 1]
                node_1_behind = tour[:, 1, -i]
                node_2_behind = tour[:, 2, -i]
                f_1 = tour[indices, -2, -i] + torch.min(
                    D[indices, node_2_front, node_1_behind] + D[indices, node_2_behind, depot],
                    D[indices, node_2_front, node_2_behind] + D[indices, node_1_behind, depot])
                f_2 = tour[indices, -2, -i] + torch.min(
                    D[indices, node_1_front, node_1_behind] + D[indices, node_2_behind, depot],
                    D[indices, node_1_front, node_2_behind] + D[indices, node_1_behind, depot])

            elif i == tour_length:
                node_1 = tour[:, 1, -i]
                node_2 = tour[:, 2, -i]
                total_dhcost = tour[indices, -2, -i] + torch.min(D[indices, depot, node_1] + f_1,
                                                                      D[indices, depot, node_2] + f_2)
                total_dhcost = total_dhcost.view(self.batch_size, self.pomo_size)

            else:
                node_1_front = tour[indices, 1, -i - 1]
                node_2_front = tour[indices, 2, -i - 1]
                node_1_behind = tour[indices, 1, -i]
                node_2_behind = tour[indices, 2, -i]
                f_1_ = tour[indices, -2, -i] + torch.min(D[indices, node_2_front, node_1_behind] + f_1,
                                                         D[indices, node_2_front, node_2_behind] + f_2)
                f_2_ = tour[indices, -2, -i] + torch.min(D[indices, node_1_front, node_1_behind] + f_1,
                                                         D[indices, node_1_front, node_2_behind] + f_2)
                f_1 = f_1_
                f_2 = f_2_
        return total_dhcost


        # total_dhcost = torch.zeros(self.batch_size*self.pomo_size)
        #
        # pi = self.selected_edge_list
        # pi_num_samples, pomo_size, tour_length = pi.size()
        # pi = pi.view(pi_num_samples * pomo_size, tour_length)
        # idx = pi.unsqueeze(1).expand(-1, self.graph_info.size(1), -1)
        # graph_info = self.graph_info.unsqueeze(1).repeat(1, pomo_size, 1, 1)
        # graph_info_num_samples, feature_size, edge_size = self.graph_info.size()
        # graph_info = graph_info.view(graph_info_num_samples * pomo_size, feature_size, edge_size)
        # tour = torch.gather(graph_info, 2, idx).to(int)
        #
        # D = self.D.unsqueeze(1).repeat(1, pomo_size, 1, 1)
        # D_num_samples, _, node_size = self.D.size()
        # D = D.view(D_num_samples * pomo_size, node_size, node_size)
        #
        # num_samples, _, tour_length = tour.size()
        # f_1 = torch.zeros(num_samples)
        # f_2 = torch.zeros(num_samples)
        # depot = torch.zeros(num_samples)
        # depot = graph_info[:, 1, 0].long()
        # indices = torch.arange(num_samples)
        #
        # for i in range(1, tour_length + 1):
        #     if i == 1:
        #         node_1_front = tour[:, 1, -i - 1]
        #         node_2_front = tour[:, 2, -i - 1]
        #         node_1_behind = tour[:, 1, -i]
        #         node_2_behind = tour[:, 2, -i]
        #         f_1 = tour[indices, -2, -i] + torch.min(
        #             D[indices, node_2_front, node_1_behind] + D[indices, node_2_behind, depot],
        #             D[indices, node_2_front, node_2_behind] + D[indices, node_1_behind, depot])
        #         f_2 = tour[indices, -2, -i] + torch.min(
        #             D[indices, node_1_front, node_1_behind] + D[indices, node_2_behind, depot],
        #             D[indices, node_1_front, node_2_behind] + D[indices, node_1_behind, depot])
        #
        #     elif i == tour_length:
        #         node_1 = tour[:, 1, -i]
        #         node_2 = tour[:, 2, -i]
        #         total_dhcost = tour[indices, -2, -i] + torch.min(D[indices, depot, node_1] + f_1,
        #                                                               D[indices, depot, node_2] + f_2)
        #         total_dhcost = total_dhcost.view(self.batch_size, self.pomo_size)
        #
        #     else:
        #         node_1_front = tour[indices, 1, -i - 1]
        #         node_2_front = tour[indices, 2, -i - 1]
        #         node_1_behind = tour[indices, 1, -i]
        #         node_2_behind = tour[indices, 2, -i]
        #         f_1_ = tour[indices, -2, -i] + torch.min(D[indices, node_2_front, node_1_behind] + f_1,
        #                                                  D[indices, node_2_front, node_2_behind] + f_2)
        #         f_2_ = tour[indices, -2, -i] + torch.min(D[indices, node_1_front, node_1_behind] + f_1,
        #                                                  D[indices, node_1_front, node_2_behind] + f_2)
        #         f_1 = f_1_
        #         f_2 = f_2_
        # return total_dhcost