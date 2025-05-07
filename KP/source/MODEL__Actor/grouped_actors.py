import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# For debugging
# from IPython.core.debugger import set_trace # Keep if needed

# Hyper Parameters
from HYPER_PARAMS import *
from TORCH_OBJECTS import *

########################################
# ACTOR
########################################

class ACTOR(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_group()

        self.batch_s = None
        self.encoded_nodes_and_dummy = None
        self.encoded_nodes = None
        self.encoded_graph = None
        # Store ratios for easy access if needed, though passed directly is fine too
        self.normalized_ratios = None

    def reset(self, group_state):
        self.batch_s = group_state.item_data.size(0)
        self.encoded_nodes_and_dummy = Tensor(np.zeros((self.batch_s, PROBLEM_SIZE+1, EMBEDDING_DIM)))
        # Encode item data (weights and values)
        self.encoded_nodes_and_dummy[:, :PROBLEM_SIZE, :] = self.encoder(group_state.item_data)
        self.encoded_nodes = self.encoded_nodes_and_dummy[:, :PROBLEM_SIZE, :]
        # shape = (batch, problem, EMBEDDING_DIM)
        self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
        # shape = (batch, 1, EMBEDDING_DIM)

        # Store normalized ratios from the state
        self.normalized_ratios = group_state.normalized_value_ratios
        # shape = (batch, problem)

        # Reset the probability calculator with encoded nodes
        self.node_prob_calculator.reset(self.encoded_nodes)

    def get_action_probabilities(self, group_state):

        # Pass graph, capacity, mask, AND normalized ratios to the calculator
        probs = self.node_prob_calculator(graph=self.encoded_graph,
                                          capacity=group_state.capacity,
                                          ninf_mask=group_state.fit_ninf_mask,
                                          normalized_value_ratios=self.normalized_ratios) # Pass ratios here
        # shape = (batch, group, problem)

        return probs


########################################
# ACTOR_SUB_NN : ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dimension is 2 (weight, value)
        self.embedding = nn.Linear(2, EMBEDDING_DIM)
        self.layers = nn.ModuleList([Encoder_Layer() for _ in range(ENCODER_LAYER_NUM)])

    def forward(self, item_data):
        # item_data.shape = (batch, problem, 2)

        embedded_input = self.embedding(item_data)
        # shape = (batch, problem, EMBEDDING_DIM)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class Encoder_Layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.Wq = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.addAndNormalization1 = Add_And_Normalization_Module()
        self.feedForward = Feed_Forward_Module()
        self.addAndNormalization2 = Add_And_Normalization_Module()

    def forward(self, input1):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=HEAD_NUM)
        k = reshape_by_heads(self.Wk(input1), head_num=HEAD_NUM)
        v = reshape_by_heads(self.Wv(input1), head_num=HEAD_NUM)
        # q shape = (batch, HEAD_NUM, problem, KEY_DIM)

        # Mask is not typically used in self-attention within encoder
        out_concat = multi_head_attention(q, k, v)
        # shape = (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


########################################
# ACTOR_SUB_NN : Next_Node_Probability_Calculator
########################################

class Next_Node_Probability_Calculator_for_group(nn.Module):
    def __init__(self):
        super().__init__()

        # Query includes graph embedding (EMBEDDING_DIM) + capacity (1)
        self.Wq = nn.Linear(EMBEDDING_DIM + 1, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)

        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem, EMBEDDING_DIM)
        batch_s = encoded_nodes.size(0)

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=HEAD_NUM)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=HEAD_NUM)
        # shape = (batch, HEAD_NUM, problem, KEY_DIM)

        # Use batch_norm1d for single-head attention key? Optional.
        # Or just transpose:
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch, EMBEDDING_DIM, problem)

    # Modified forward to accept and use normalized_value_ratios
    def forward(self, graph, capacity, ninf_mask, normalized_value_ratios):
        # graph.shape = (batch, 1, EMBEDDING_DIM)
        # capacity.shape = (batch, group)
        # ninf_mask.shape = (batch, group, problem)
        # normalized_value_ratios.shape = (batch, problem)

        batch_s = capacity.size(0)
        group_s = capacity.size(1)
        problem_s = normalized_value_ratios.size(1) # Should be PROBLEM_SIZE

        # Prepare Query: Combine graph embedding and capacity
        #######################################################
        # Expand graph embedding to match group size
        graph_expanded = graph.expand(batch_s, group_s, EMBEDDING_DIM)
        # Add dimension for concatenation
        capacity_expanded = capacity.unsqueeze(-1) # shape = (batch, group, 1)

        # Concatenate along the feature dimension
        query_input = torch.cat((graph_expanded, capacity_expanded), dim=2)
        # shape = (batch, group, EMBEDDING_DIM + 1)

        # Project query
        q = reshape_by_heads(self.Wq(query_input), head_num=HEAD_NUM)
        # shape = (batch, HEAD_NUM, group, KEY_DIM)

        # Multi-Head Attention (Query attends to Keys/Values derived from all items)
        #######################################################
        # Pass mask to multi_head_attention
        # Mask shape needs to align: (batch, group, problem) -> (batch, 1, group, problem) or handled inside
        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=ninf_mask)
        # shape = (batch, group, HEAD_NUM*KEY_DIM)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch, group, EMBEDDING_DIM)


        # Single-Head Attention, for probability calculation
        #######################################################
        # Calculate compatibility score
        # mh_atten_out shape: (batch, group, EMBEDDING_DIM)
        # self.single_head_key shape: (batch, EMBEDDING_DIM, problem)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch, group, problem)

        score_scaled = score / np.sqrt(EMBEDDING_DIM)
        # shape = (batch, group, problem)

        # --- Value Ratio Bonus Calculation ---
        if RATIO_BONUS_FACTOR > 0:
            # Expand ratios to match score shape: (batch, problem) -> (batch, 1, problem) -> (batch, group, problem)
            ratios_expanded = normalized_value_ratios[:, None, :].expand_as(score_scaled)
            topk_indices = torch.topk(ratios_expanded, 100, dim=-1, largest=False).indices
            topk_indices = topk_indices[:, :, 1:]
            # 根据索引选取对应的距离值，计算对数函数，得到新的分数
            topk_scores = -torch.log(ratios_expanded.gather(-1, topk_indices))
            # 将其余距离数值取负数，保证索引位置不变
            ratio_bonus = -ratios_expanded
            # Calculate log-ratio bonus (add small epsilon to log input for stability)
            # Ensure ratios are slightly > 0 for log
            ratio_bonus.scatter_(-1, topk_indices, topk_scores)
            # Add bonus to the scaled score
            score_final = score_scaled + ratio_bonus
        else:
            # If factor is 0, use original score
            score_final = score_scaled
        # --- End Value Ratio Bonus ---

        # Clip the score for stability
        score_clipped = LOGIT_CLIPPING * torch.tanh(score_final) # Use score_final here

        # Apply the mask (items already selected or overweight)
        # Make sure mask has -np.inf for masked items
        score_masked = score_clipped + ninf_mask # ninf_mask should be 0 for valid, -inf for invalid

        # Calculate final probabilities
        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, group, problem)

        # Handle potential NaNs if any step introduced them (e.g., log(0))
        # probs = torch.nan_to_num(probs, nan=0.0) # Optional: Replace NaNs with 0 probability

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

# pick_nodes_for_each_group seems unused in the provided ACTOR, keep if used elsewhere
def pick_nodes_for_each_group(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape = (batch, problem, EMBEDDING_DIM)
    # node_index_to_pick.shape = (batch, group_s)
    batch_s = node_index_to_pick.size(0)
    group_s = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2) # Use dynamic dim

    gathering_index = node_index_to_pick[:, :, None].expand(batch_s, group_s, embedding_dim)
    # shape = (batch, group, EMBEDDING_DIM)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape = (batch, group, EMBEDDING_DIM)

    return picked_nodes


def reshape_by_heads(qkv, head_num):
    # qkv shape = (batch, C, head_num*key_dim)
    batch_s = qkv.size(0)
    C = qkv.size(1)
    # Calculate key_dim dynamically
    key_dim = qkv.size(2) // head_num

    qkv_reshaped = qkv.reshape(batch_s, C, head_num, key_dim)
    # shape = (batch, C, head_num, key_dim)

    # Transpose to get heads dimension first
    qkv_transposed = qkv_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return qkv_transposed


def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be group_s
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2) # group_s
    key_dim = q.size(3)
    problem_s = k.size(2) # problem_size

    # Calculate attention scores
    # Matmul: (batch, head_num, n, key_dim) x (batch, head_num, key_dim, problem) -> (batch, head_num, n, problem)
    score = torch.matmul(q, k.transpose(2, 3))

    score_scaled = score / np.sqrt(key_dim)

    # Apply mask if provided
    # Mask shape: (batch, group, problem) -> needs (batch, 1, group, problem)
    if ninf_mask is not None:
        # Add unsqueezed mask to scores
        score_scaled = score_scaled + ninf_mask.unsqueeze(1) # Add head_num dim

    # Calculate attention weights
    weights = F.softmax(score_scaled, dim=3) # Softmax over 'problem' dimension
    # shape = (batch, head_num, n, problem)

    # Apply weights to values
    # Matmul: (batch, head_num, n, problem) x (batch, head_num, problem, key_dim) -> (batch, head_num, n, key_dim)
    out = torch.matmul(weights, v)

    # Transpose and reshape for output
    # (batch, head_num, n, key_dim) -> (batch, n, head_num, key_dim)
    out_transposed = out.transpose(1, 2).contiguous() # Use contiguous for efficiency

    # (batch, n, head_num, key_dim) -> (batch, n, head_num * key_dim)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self):
        super().__init__()
        # Consider LayerNorm instead of BatchNorm if sequences vary or batch size is small
        # self.norm = nn.LayerNorm(EMBEDDING_DIM)
        self.norm_by_EMB = nn.BatchNorm1d(EMBEDDING_DIM, affine=True) # Keep BatchNorm if it works well

    def forward(self, input1, input2):
        # input.shape = (batch, problem_or_group, EMBEDDING_DIM)
        batch_s = input1.size(0)
        seq_len = input1.size(1) # problem or group

        added = input1 + input2
        # Reshape for BatchNorm1d: needs (N, C) where C is EMBEDDING_DIM
        reshaped_added = added.reshape(batch_s * seq_len, EMBEDDING_DIM)
        normalized = self.norm_by_EMB(reshaped_added)

        # Reshape back to original shape
        return normalized.reshape(batch_s, seq_len, EMBEDDING_DIM)
        # LayerNorm alternative:
        # added = input1 + input2
        # return self.norm(added)


class Feed_Forward_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(EMBEDDING_DIM, FF_HIDDEN_DIM)
        self.W2 = nn.Linear(FF_HIDDEN_DIM, EMBEDDING_DIM)
        # Optional: Dropout
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, input1):
        # input.shape = (batch, problem_or_group, EMBEDDING_DIM)
        # return self.W2(self.dropout(F.relu(self.W1(input1)))) # With dropout
        return self.W2(F.relu(self.W1(input1))) # Original