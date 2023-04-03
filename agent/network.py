import torch
import torch.nn as nn
import torch.nn.functional as F
from params import configs
from agent.layer import *
import numpy as np



class GNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super().__init__()
        self.action_type = action_type
        self.layer = get_GNN_layer()
        self.loss_f = get_loss_f()

        # layers ##########################################
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n

        self.f_init = get_mlp(in_dim, em_dim)

        in_em_dim = em_dim
        if configs.self_loop:
            in_em_dim += in_dim
        self.fs_all = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])

        if configs.self_loop:
            self.fs = nn.ModuleList([get_mlp(em_dim + in_em_dim, em_dim) for _ in range(aggr_n)])
        else:
            self.fs = nn.ModuleList([get_mlp(em_dim, em_dim) for _ in range(aggr_n)])

        # final ##########################################
        if configs.global_embed_type == 'mean':
            final_input_dim = configs.em_dim * 2
        elif configs.global_embed_type == 'mmm':
            final_input_dim = configs.em_dim * 4
        else:
            final_input_dim = configs.em_dim

        self.final_v = get_mlp(final_input_dim, out_dim)
        if configs.duel_TF:
            if configs.global_embed_type == 'mmm':
                final_input_dim2 = configs.em_dim * 3
            else:
                final_input_dim2 = configs.em_dim
            self.final_v_ = get_mlp(final_input_dim2, out_dim)

    def final_value(self, h, remain_idxs=""):
        """
        get final embedding vector
        global embedding vector
        """
        if configs.global_embed_type == 'mean':
            # global feature
            if configs.dyn_env_TF:
                g = h[remain_idxs].mean(dim=0)
            else:
                g = h.mean(dim=0)
            h_g = [g for _ in range(h.shape[0])]
            h_g = torch.stack(h_g, dim=0)  # node 개수 만큼 stack
            h = torch.cat([h, h_g], dim=1)
        elif configs.global_embed_type == 'mmm':
            # global feature
            if configs.dyn_env_TF:
                g = h[remain_idxs].mean(dim=0)
            else:
                g = h.mean(dim=0)
            h_g = [g for _ in range(h.shape[0])]
            h_g = torch.stack(h_g, dim=0)  # node 개수 만큼 stack
            if configs.dyn_env_TF:
                g2 = h[remain_idxs].max(dim=0)[0]
            else:
                g2 = h.max(dim=0)[0]
            h_g2 = [g2 for _ in range(h.shape[0])]
            h_g2 = torch.stack(h_g2, dim=0)  # node 개수 만큼 stack
            if configs.dyn_env_TF:
                g3 = h[remain_idxs].min(dim=0)[0]
            else:
                g3 = h.min(dim=0)[0]
            h_g3 = [g3 for _ in range(h.shape[0])]
            h_g3 = torch.stack(h_g3, dim=0)  # node 개수 만큼 stack
            h = torch.cat([h, h_g, h_g2, h_g3], dim=1)

        if configs.duel_TF:
            v = self.final_v(h)
            a = self.final_a(h)
            if configs.duel_bias == "max":
                return v + a - a.max()  # mean(1)[1].view(-1, 1)
            elif configs.duel_bias == "mean":
                return v + a - a.mean()  # mean(1)[1].view(-1, 1)
            else:
                return v + a
        else:
            return self.final_v(h)

    def final_policy(self, z):
        """
        return a policy or a value
        """
        if self.action_type == "policy":
            if configs.softmax_tau != 1:
                z /= configs.softmax_tau
            return F.softmax(z, dim=-2)
        else:
            return z

    def forward_partial(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        op_mask = obs['op'].mask
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)
            h_ = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])

            if configs.self_loop:
                h_ = torch.cat([h_, h], dim=1)
            h = self.fs[i](torch.cat([h_], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z

    def forward(self, obs):
        z = self.forward_partial(obs)
        return self.final_policy(z)

    def loss(self, data):
        """
        sum loss for mini-batch of torch_geometric
        """
        z = self.forward_partial(data)

        target = data.target_policy
        graph_index = data['op'].batch
        losses = list()
        for batch_i in range(data.num_graphs):
            indices = torch.where(graph_index == batch_i)
            z_ = z[indices]
            policy = self.final_policy(z_)

            losses.append(self.loss_f(policy.squeeze(), target[
                indices]))  # for cross entropy, loss_f(policy, target[indices].view(-1, 1))

        loss = torch.stack(losses).to(configs.device).sum()
        return loss

    def get_action_logprob(self):

        return 0


class Hetero_GNN_type_aware(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super(Hetero_GNN_type_aware, self).__init__(in_dim, out_dim, in_dim_rsc, action_type)
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n
        in_em_dim = em_dim
        if configs.self_loop:
            in_em_dim += in_dim  # h = [h||x]

        self.fs_prec = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_succ = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_disj = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])

        if configs.self_loop:
            self.fs = nn.ModuleList([get_mlp(4 * em_dim + in_em_dim, em_dim) for _ in range(aggr_n)])
        else:
            self.fs = nn.ModuleList([get_mlp(4 * em_dim, em_dim) for _ in range(aggr_n)])

    def forward_partial(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        op_mask = obs['op'].mask
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)

            h_p = self.fs_prec[i](h, edge_index_dict['op', 'prec', 'op'])
            h_s = self.fs_succ[i](h, edge_index_dict['op', 'succ', 'op'])
            h_d = self.fs_disj[i](h, edge_index_dict['op', 'disj', 'op'])
            h_a = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])

            if configs.self_loop:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h], dim=1))
            else:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z


class Hetero_GNN_type_all_prec(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super(Hetero_GNN_type_all_prec, self).__init__(in_dim, out_dim, in_dim_rsc, action_type)
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n
        in_em_dim = em_dim
        if configs.self_loop:
            in_em_dim += in_dim  # h = [h||x]

        self.fs_prec = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_succ = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_disj = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all_prec = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all_succ = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])

        if configs.self_loop:
            self.fs = nn.ModuleList([get_mlp(6 * em_dim + in_em_dim, em_dim) for _ in range(aggr_n)])
        else:
            self.fs = nn.ModuleList([get_mlp(6 * em_dim, em_dim) for _ in range(aggr_n)])

    def forward_partial(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        op_mask = obs['op'].mask
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)

            h_p = self.fs_prec[i](h, edge_index_dict['op', 'prec', 'op'])
            h_s = self.fs_succ[i](h, edge_index_dict['op', 'succ', 'op'])
            h_d = self.fs_disj[i](h, edge_index_dict['op', 'disj', 'op'])
            h_a = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])
            h_p2 = self.fs_all_prec[i](h, edge_index_dict['op', 'all_prec', 'op'])
            h_s2 = self.fs_all_succ[i](h, edge_index_dict['op', 'all_succ', 'op'])

            if configs.self_loop:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h_p2, h_s2, h], dim=1))
            else:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h_p2, h_s2], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        # if np.isnan(z[0].item()):
        #     print()
        return z


class Hetero_GNN_no_succ(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super(Hetero_GNN_no_succ, self).__init__(in_dim, out_dim, in_dim_rsc, action_type)
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n

        if configs.self_loop:
            self.fs = nn.ModuleList([get_mlp(4 * em_dim + in_dim, em_dim) for _ in range(aggr_n)])
        else:
            self.fs = nn.ModuleList([get_mlp(3 * em_dim, em_dim) for _ in range(aggr_n)])

    def forward_partial(self, obs):
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict

        edge_index_dict_all = torch.cat([edge_index_dict['op', 'prec', 'op'],
                                         edge_index_dict['op', 'disj', 'op']], dim=1)

        op_mask = obs['op'].mask
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)

            h_p = self.fs_prec[i](h, edge_index_dict['op', 'prec', 'op'])
            h_d = self.fs_disj[i](h, edge_index_dict['op', 'disj', 'op'])
            h_a = self.fs_all[i](h, edge_index_dict_all)

            if configs.self_loop:
                h = self.fs[i](torch.cat([h_p, h_d, h_a, h], dim=1))
            else:
                h = self.fs[i](torch.cat([h_p, h_d, h_a], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z


class Hetero_GNN_IJPR(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super(Hetero_GNN_IJPR, self).__init__(in_dim, out_dim, in_dim_rsc, action_type)
        self.f_act = get_act_f()()
        self.layer = Basic_layer

        em_dim = configs.em_dim
        aggr_n = configs.aggr_n

        self.fs_prec = nn.ModuleList([self.layer(em_dim, em_dim) if i > 0
                                      else self.layer(in_dim, em_dim) for i in range(aggr_n)])
        self.fs_succ = nn.ModuleList([self.layer(em_dim, em_dim) if i > 0
                                      else self.layer(in_dim, em_dim) for i in range(aggr_n)])
        self.fs_disj = nn.ModuleList([self.layer(em_dim, em_dim) if i > 0
                                      else self.layer(in_dim, em_dim) for i in range(aggr_n)])

        self.fs = nn.ModuleList([get_mlp(5 * em_dim + in_dim, em_dim) if i > 0
                                 else get_mlp(4 * em_dim + 2 * in_dim, em_dim)
                                 for i in range(aggr_n)])

    def forward_partial(self, obs):
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        op_mask = obs['op'].mask
        x = x_dict['op']

        h = x
        for i in range(configs.aggr_n):
            h_p = self.fs_prec[i](h, edge_index_dict['op', 'prec', 'op'])
            h_s = self.fs_succ[i](h, edge_index_dict['op', 'succ', 'op'])
            h_d = self.fs_disj[i](h, edge_index_dict['op', 'disj', 'op'])
            h_a = self.f_act(aggr(h, edge_index_dict['op', 'all', 'op']))

            h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h, x], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z


class Hetero_multiplex_GNN(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super(Hetero_multiplex_GNN, self).__init__(in_dim, out_dim, in_dim_rsc, action_type)
        em_dim = configs.em_dim

        # self.f_init = get_mlp(in_dim, em_dim)
        self.f_init_prec = get_mlp(in_dim, em_dim)
        self.f_init_succ = get_mlp(in_dim, em_dim)
        self.f_init_disj = get_mlp(in_dim, em_dim)

        self.fs = self.layer(4 * em_dim + in_dim, em_dim)

    def forward_partial(self, obs):
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        op_mask = obs['op'].mask
        x = x_dict['op']

        h = self.f_init(x)
        h_p = self.f_init_prec(x)
        h_s = self.f_init_succ(x)
        h_d = self.f_init_disj(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)
                h_p = torch.cat([h_p, x], dim=1)
                h_s = torch.cat([h_s, x], dim=1)
                h_d = torch.cat([h_d, x], dim=1)

            h = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])
            h_p = self.fs_prec[i](h_p, edge_index_dict['op', 'prec', 'op'])
            h_s = self.fs_succ[i](h_s, edge_index_dict['op', 'succ', 'op'])
            h_d = self.fs_disj[i](h_d, edge_index_dict['op', 'disj', 'op'])

        h = self.fs(torch.cat([h_p, h_s, h_d, h, x], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z


class Hetero_GNN_mc(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, action_type="policy"):
        super(Hetero_GNN_mc, self).__init__(in_dim, out_dim, in_dim_rsc, action_type)
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n

        if configs.self_loop:
            in_em_dim = em_dim + in_dim
            in_em_dim_rsc = em_dim + in_dim
        else:
            in_em_dim = em_dim
            in_em_dim_rsc = em_dim + in_dim

        self.f_init_rsc = get_mlp(in_dim_rsc, em_dim)

        self.fs_op_to_r = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_r_to_op = nn.ModuleList([self.layer(in_em_dim_rsc, em_dim) for _ in range(aggr_n)])

        self.fs = nn.ModuleList([get_mlp(3 * em_dim + in_dim, em_dim) for _ in range(aggr_n)])
        self.fs_rsc = nn.ModuleList([get_mlp(2 * em_dim + in_dim_rsc, em_dim) for _ in range(aggr_n)])

    def forward_partial(self, obs):
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        op_mask = obs['op'].mask

        x = x_dict['op']
        x_rsc = x_dict['rsc']

        h = self.f_init(x)
        h_rsc = self.f_init_rsc(x_rsc)

        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)
                h_rsc = torch.cat([h_rsc, x_rsc], dim=1)

            h_p = self.fs_prec[i](h, edge_index_dict['op', 'prec', 'op'])
            h_s = self.fs_succ[i](h, edge_index_dict['op', 'succ', 'op'])
            h_r = self.fs_r_to_op[i](h_rsc, edge_index_dict['rsc', 'rsc_to_op', 'op'])
            h_rsc_from_op = self.fs_op_to_r[i](h, edge_index_dict['op', 'op_to_rsc', 'rsc'])

            if configs.self_loop:
                h = self.fs[i](torch.cat([h_p, h_s, h_r, h], dim=1))
                h_rsc = self.fs_rsc[i](torch.cat([h_rsc_from_op, h_rsc], dim=1))
            else:
                h = self.fs[i](torch.cat([h_p, h_s, h_r, h, x], dim=1))
                h_rsc = self.fs_rsc[i](torch.cat([h_rsc_from_op, h_rsc, x_rsc], dim=1))

        z = self.final_value(h, obs['op'].remain)
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z






