import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from GCN import GCN


class CARPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CARP_Encoder(**model_params)
        self.decoder = CARP_Decoder(**model_params)
        self.encoded_edges = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot = reset_state.depot
        # shape: (batch, 1, 8)
        customer = reset_state.customer
        # shape: (batch, problem, 8)
        customer_demand = reset_state.customer_demand
        # shape: (batch, problem)

        A = reset_state.A
        self.dist = reset_state.edge_distance

        self.encoded_edges = self.encoder(depot, customer, A)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_edges)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        # encoded_near_edges = torch.zeros((batch_size, pomo_size, 5, self.encoded_edges.size(2)), dtype=torch.float32)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_edge = _get_encoding(self.encoded_edges, state.current_edge)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_edge, state.load, state.current_edge, self.dist ,ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_edges, edge_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)
    batch_size = edge_index_to_pick.size(0)
    pomo_size = edge_index_to_pick.size(1)
    embedding_dim = encoded_edges.size(2)
    gathering_index = edge_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)
    picked_edges = encoded_edges.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)
    return picked_edges


########################################
# ENCODER
########################################

class CARP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.gcn_model = GCN(6, embedding_dim)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])


    def forward(self,depot, customer, A):
        node_feature = torch.cat((depot[:,None,:], customer), dim=1)

        out = self.gcn_model(A, node_feature)

        for layer in self.layers:
            out = layer(out)
        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CARP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.mask_num = self.model_params['mask_num']
        self.omega = self.model_params['omega']

        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def set_kv(self, encoded_edges):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_edges), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_edges), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_edges.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_edge, load, current_edge, dist, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_edge, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        edge_indices = current_edge.unsqueeze(2).expand(-1, -1, dist.size(2))
        cur_dist = torch.gather(dist, 1, edge_indices)

        # 在problem维度上选取距离最短的前K个值，返回索引
        topk_indices = torch.topk(cur_dist, self.mask_num, dim=-1, largest=False).indices

        topk_indices = topk_indices[:, :, 1:]

        # topk_dist = cur_dist.gather(-1, topk_indices)

        topk_scores = -torch.log(cur_dist.gather(-1, topk_indices))

        cur_dist = -cur_dist

        cur_dist.scatter_(-1, topk_indices, topk_scores)

        score_reshaping = self.omega * score_scaled + (1-self.omega) * cur_dist

        score_clipped = logit_clipping * torch.tanh(score_reshaping)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################
def reshape_by_heads_near(qkv, head_num):
    # q.shape: (batch, n, 5,head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)
    near_s = qkv.size(2)

    q_reshaped = qkv.reshape(batch_s, n, near_s, head_num, -1)
    # shape: (batch, n, 5,head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 3)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
