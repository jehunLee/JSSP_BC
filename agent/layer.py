import torch
from params import configs
import torch.nn as nn
import torch.nn.functional as F


# layer ###################################################################
def get_GNN_layer():
    if 'GAT2_simple' in configs.model_type:
        return GAT2_simple_layer
    elif 'GAT2' in configs.model_type:
        return GAT2_layer
    elif 'GAT' in configs.model_type:
        return GAT_layer
    elif 'node_attn' in configs.model_type:
        return Node_attn
    elif 'GCN' in configs.model_type:
        return GCN_layer
    else:
        return GCN_layer


class GCN_layer(nn.Module):
    def __init__(self, in_dim, out_dim, in_dim_to=1):
        super().__init__()
        em_dim = configs.em_dim

        self.f_init = get_mlp(in_dim, em_dim * configs.attn_head_n)
        self.f_final = get_mlp(em_dim * configs.attn_head_n, out_dim)  # concat 적용 효과 고려

    def forward(self, x_from, edge_index, x_to=None):
        h = self.f_init(x_from)
        h_ = aggr(h, edge_index)

        return self.f_final(h_)  # concat 적용 효과 고려


class GAT_layer(nn.Module):
    # e(h_i, h_j) = LeakyReLU( a [W h_i||W h_j] )
    # alpha = softmax(e)
    # h_i' = alpha * W h_j
    def __init__(self, in_dim, out_dim, in_dim_to=1):
        super().__init__()
        em_dim = configs.em_dim

        self.f_inits = nn.ModuleList([get_mlp(in_dim, em_dim)
                                      for _ in range(configs.attn_head_n)])  # W^k <- h_i, h_j 동일한 mlp 활용

        self.As = nn.ModuleList([get_mlp(em_dim * 2, 1, last_act_TF=True)
                                 for _ in range(configs.attn_head_n)])  # GAT1에서는 alpha 구할 때 activation 거침
        self.f_final = get_mlp(em_dim * configs.attn_head_n, out_dim)  # multi-head attention concat 후 적용

    def forward(self, x_from, edge_index, x_to=None):
        src, dst = edge_index
        hs = torch.stack([f_init(x_from) for f_init in self.f_inits], dim=0)  # W^k  # (attn_head_n, node_n, em_dim)
        hs_to = torch.stack([f_init(x_from) for f_init in self.f_inits], dim=0)  # W^k  # (attn_head_n, node_n, em_dim)

        h_cat = torch.stack([torch.cat([hs[k][src], hs_to[k][dst]], dim=1)
                             for k in range(configs.attn_head_n)], dim=0)  # (attn_head_n, edge_n, em_dim * 2)
        a = torch.stack([A(h_cat[k]) for k, A in enumerate(self.As)], dim=0)  # (attn_head_n, edge_n, 1)

        # softmax -> alpha
        e = torch.exp(a)  # (attn_head_n, edge_n, 1)
        node_e_sum = torch.zeros(configs.attn_head_n, x_from.shape[0], 1, dtype=e.dtype).to(configs.device)  # (attn_head_n, node_n, 1)
        src_node = src.view(-1, 1)
        node_e_sum = torch.stack([node_e_sum[k].scatter_add_(0, src_node, e[k])
                                  for k in range(configs.attn_head_n)], dim=0)  # (attn_head_n, node_n, 1)
        alpha = torch.stack([e[k] / node_e_sum[k][src] for k in range(configs.attn_head_n)], dim=0)  # (attn_head_n, edge_n, 1)

        # final
        h_ = torch.cat([aggr(hs_to[k], edge_index, alpha=alpha[k]) for k in range(configs.attn_head_n)], dim=1)  # (node_n, em_dim * attn_head_n)
        return self.f_final(h_)  # multi-head concat 후 적용


class GAT2_layer(nn.Module):  # GAT2 - ICLR 2022  # HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS?
    # e(h_i, h_j) = a LeakyReLU( W [h_i||h_j] )
    # alpha = softmax(e)
    # h_i' = alpha * W h_j
    def __init__(self, in_dim, out_dim, in_dim_to=1):
        super().__init__()
        em_dim = configs.em_dim

        self.f_concats = nn.ModuleList([get_mlp(in_dim * 2, em_dim, last_act_TF=True)
                                      for _ in range(configs.attn_head_n)])  # W^k
        self.f_inits = nn.ModuleList([get_mlp(in_dim, em_dim)
                                       for _ in range(configs.attn_head_n)])  # W1^k

        self.As = nn.ModuleList([get_mlp(em_dim, 1)
                                 for _ in range(configs.attn_head_n)])  # GAT2에서는 alpha 구할 때 activation x
        self.f_final = get_mlp(em_dim * configs.attn_head_n, out_dim)  # multi-head attention concat 후 적용

    def forward(self, x_from, edge_index, x_to=None):
        src, dst = edge_index

        h_cat = torch.stack([torch.cat([x_from[src], x_from[dst]], dim=1)
                             for _ in range(configs.attn_head_n)], dim=0)  # (attn_head_n, edge_n, in_dim * 2)
        h_cat_ = torch.stack([self.f_concats[k](h_cat[k]) for k in range(configs.attn_head_n)], dim=0)  # W^k  # (attn_head_n, edge_n, em_dim)
        a = torch.stack([A(h_cat_[k]) for k, A in enumerate(self.As)], dim=0)  # (attn_head_n, edge_n, 1)

        # softmax -> alpha
        if len(a[0]) and a.max() > 80:
            a = a - a.max() + 70
        e = torch.exp(a) + 1e-32  # (attn_head_n, edge_n, 1)  # over 80 -> inf, under -100 -> 0
        node_e_sum = torch.zeros(configs.attn_head_n, x_from.shape[0], 1, dtype=e.dtype).to(configs.device)  # (attn_head_n, node_n, 1)
        dst_node = dst.view(-1, 1)
        node_e_sum = torch.stack([node_e_sum[k].scatter_add_(0, dst_node, e[k])
                                  for k in range(configs.attn_head_n)], dim=0)  # (attn_head_n, node_n, 1)
        alpha = torch.stack([e[k] / node_e_sum[k][dst] for k in range(configs.attn_head_n)], dim=0)  # (attn_head_n, edge_n, 1)

        # final
        hs_to = torch.stack([f_init(x_from) for f_init in self.f_inits], dim=0)  # W^k  # (attn_head_n, node_n, em_dim)
        h_ = torch.cat([aggr(hs_to[k], edge_index, alpha=alpha[k]) for k in range(configs.attn_head_n)], dim=1)  # (node_n, em_dim * attn_head_n)
        return self.f_final(h_)  # multi-head concat 후 적용


class GAT2_simple_layer(nn.Module):  # GAT2 - ICLR 2022  # HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS?
    # e(h_i, h_j) = a LeakyReLU( W [h_i||h_j] ) -> simple version: a LeakyReLU (W h_i + W h_j)
    # alpha = softmax(e)
    # h_i' = alpha * W h_j
    def __init__(self, in_dim, out_dim, in_dim_to=1):
        super().__init__()
        em_dim = configs.em_dim

        self.f_inits = nn.ModuleList([get_mlp(in_dim * 2, em_dim, last_act_TF=True)
                                      for _ in range(configs.attn_head_n)])  # W^k

        self.As = nn.ModuleList([get_mlp(em_dim, 1)
                                 for _ in range(configs.attn_head_n)])  # GAT2에서는 alpha 구할 때 activation x
        self.f_final = get_mlp(em_dim * configs.attn_head_n, out_dim)  # multi-head attention concat 후 적용

    def forward(self, x_from, edge_index, x_to=None):
        src, dst = edge_index
        hs_to = torch.stack([f_init(x_from) for f_init in self.f_inits], dim=0)  # W^k

        h_cat_ = torch.stack([hs_to[src] + hs_to[dst]
                             for _ in range(configs.attn_head_n)], dim=0)
        a = torch.stack([A(h_cat_[k]) for k, A in enumerate(self.As)], dim=0)

        # softmax -> alpha
        e = torch.exp(a)
        node_e_sum = torch.zeros(configs.attn_head_n, x_from.shape[0], 1, dtype=e.dtype).to(configs.device)
        src_node = src.view(-1, 1)
        node_e_sum = torch.stack([node_e_sum[k].scatter_add_(0, src_node, e[k])
                                  for k in range(configs.attn_head_n)], dim=0)
        alpha = torch.stack([e[k] / node_e_sum[k][src] for k in range(configs.attn_head_n)], dim=0)

        # final
        h_ = torch.cat([aggr(hs_to[k], edge_index, alpha=alpha[k]) for k in range(configs.attn_head_n)], dim=1)
        return self.f_final(h_)  # multi-head concat 후 적용


class Node_attn(nn.Module):
    def __init__(self, in_dim, out_dim, in_dim_to=1):
        super().__init__()
        em_dim = configs.em_dim

        self.f_init = get_mlp(in_dim, em_dim)
        self.As = nn.ModuleList([get_mlp(em_dim, 1, last_act_TF=True)
                                 for _ in range(configs.attn_head_n)])  # attn에서는 alpha 구할 때 activation 거침

        self.f_final = get_mlp(em_dim * configs.attn_head_n, out_dim)  # concat 후 적용

    def forward(self, x_from, edge_index, x_to=None):
        h = self.f_init(x_from)

        w = torch.stack([A(h) for A in self.As], dim=0)
        w = F.softmax(w, dim=1)
        h_ws = w * h
        h_ = torch.cat([aggr(h_w, edge_index, x_to) for h_w in h_ws], dim=1)

        return self.f_final(h_)  # concat 후 적용


#############################################################################################
def aggr(x_from, edge_index, x_to=None, alpha=1):
    src, dst = edge_index
    if x_to is None:
        x_to = x_from

    x_src = x_from[src] * alpha
    x_ = torch.zeros(x_to.shape[0], x_from.shape[1], dtype=x_src.dtype).to(configs.device)
    dst = dst.view(-1, 1)
    dsts = torch.cat([dst for _ in range(x_from.shape[1])], dim=1)
    x_ = x_.scatter_add_(0, dsts, x_src)

    if 'GAT' not in configs.model_type and 'mean_aggr' in configs.model_type:
        # normalization ################################
        one_v = torch.ones(dst.shape[0], 1, dtype=x_src.dtype).to(configs.device)
        node_degree = torch.zeros(x_to.shape[0], 1, dtype=one_v.dtype).to(configs.device)
        node_degree = node_degree.scatter_add_(0, dst, one_v)
        node_degree[node_degree == 0] = 1
        x_ = x_ / node_degree

    return x_


def get_mlp(in_dim, out_dim, last_act_TF=False):
    layers = list()
    act_f = get_act_f()

    if configs.layer_n == 1:  # layer 1개뿐
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, configs.hi_dim))
    if configs.dropout_p > 0:
        layers.append(nn.Dropout(configs.dropout_p))
    layers.append(act_f)

    # multiple layers ##################
    for i in range(configs.layer_n-1):
        if i == configs.layer_n-2:  # last layer
            layers.append(nn.Linear(configs.hi_dim, out_dim))
        else:
            layers.append(nn.Linear(configs.hi_dim, configs.hi_dim))
            if configs.dropout_p > 0:
                layers.append(nn.Dropout(configs.dropout_p))
        layers.append(act_f)

    if not last_act_TF:  # last activation function 제거
        layers.pop(-1)
    if configs.batch_norm_TF:  # batch normalization 여부
        layers.append(nn.LayerNorm(out_dim))

    return nn.Sequential(*layers)


def get_act_f():
    if configs.act_type == "leaky_relu":
        act_f = nn.LeakyReLU()
    elif configs.act_type == "relu":
        act_f = nn.ReLU()
    elif configs.act_type == "tanh":  # xavier init
        act_f = nn.Tanh()
    elif configs.act_type == 'sigmoid':
        act_f = nn.Sigmoid()
    elif configs.act_type == 'prelu':
        act_f = nn.PReLU()
    elif configs.act_type == 'relu6':
        act_f = nn.ReLU6()
    elif configs.act_type == 'rrelu':
        act_f = nn.RReLU()
    elif configs.act_type == 'selu':
        act_f = nn.SELU()
    elif configs.act_type == 'celu':
        act_f = nn.CELU()
    elif configs.act_type == 'identity':
        act_f = nn.Identity()
    else:
        raise ValueError("Unknown activation function.")

    return act_f


def get_loss_f():
    if configs.loss_type == "Huber":
        loss_f = nn.SmoothL1Loss()
    elif configs.loss_type == "MSE":
        loss_f = nn.MSELoss()
    elif configs.loss_type == "MAE":
        loss_f = nn.L1Loss()
    elif configs.loss_type == "CE":
        loss_f = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss function.")

    return loss_f
