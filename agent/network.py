from layer import *


class GNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, agent_type='policy'):
        super().__init__()
        self.agent_type = agent_type
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
        if 'mean_global' in configs.model_global_type:
            final_input_dim = configs.em_dim * 2
        elif 'mmm_global' in configs.model_global_type:
            final_input_dim = configs.em_dim * 4
        else:
            final_input_dim = configs.em_dim
        self.final_v = get_mlp(final_input_dim, out_dim)

        if configs.duel_TF:
            if 'mmm_global' in configs.model_global_type:
                final_input_dim2 = configs.em_dim * 3
            else:
                final_input_dim2 = configs.em_dim
            self.final_v_ = get_mlp(final_input_dim2, out_dim)

    def forward(self, obs):
        z = self.forward_partial(obs)

        probs = list()
        graph_index = obs['op'].batch
        for batch_i in range(obs.num_graphs):
            indices = torch.where(graph_index == batch_i)
            probs.append(self.final_policy(z[indices]))

        return probs

    def forward_partial(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        h = self.forward_h(obs)
        z = self.final_value(h)

        op_mask = obs['op_mask'].x
        z += torch.log(op_mask.to(torch.float32)).view(-1, 1)

        return z

    def forward_h(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)
            h_ = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])

            if configs.self_loop:
                h_ = torch.cat([h_, h], dim=1)
            h = self.fs[i](torch.cat([h_], dim=1))

        graph_index = obs['op'].batch
        if graph_index.max().item() > 0:
            return self.final_h(h, graph_index)
        else:
            return self.final_h(h)

    def final_h(self, h, graph_index=None):
        if graph_index is None:
            if 'mean_global' in configs.model_global_type:
                h_g = h.mean(dim=0).unsqueeze(0).expand(h.shape[0], -1)
                h = torch.cat([h, h_g], dim=1)

            elif 'mmm_global' in configs.model_global_type:
                h_g = torch.cat([h.mean(dim=0), h.max(dim=0)[0], h.min(dim=0)[0]],
                                dim=0).unsqueeze(0).expand(h.shape[0], -1)
                h = torch.cat([h, h_g], dim=1)

        else:
            if 'mean_global' in configs.model_global_type:
                max_i = graph_index.max().item()
                g_list = list()
                for batch_i in range(max_i+1):
                    indices = torch.where(graph_index == batch_i)
                    if indices[0].shape[0]:
                        g = h[indices].mean(dim=0).unsqueeze(0).expand(indices[0].shape[0], -1)
                        g_list.append(g)

                h_g = torch.cat(g_list, dim=0)
                h = torch.cat([h, h_g], dim=1)

            elif 'mmm_global' in configs.model_global_type:
                max_i = graph_index.max().item()
                g_list = list()
                for batch_i in range(max_i+1):
                    indices = torch.where(graph_index == batch_i)
                    if indices[0].shape[0]:
                        g = torch.cat([h[indices].mean(dim=0), h[indices].max(dim=0)[0], h[indices].min(dim=0)[0]],
                                      dim=0).unsqueeze(0).expand(indices[0].shape[0], -1)
                        g_list.append(g)

                h_g = torch.cat(g_list, dim=0)
                h = torch.cat([h, h_g], dim=1)

        return h

    def final_value(self, h):
        """
        get final embedding vector
        global embedding vector
        """
        if configs.duel_TF:
            v = self.final_v(h)
            a = self.final_a(h)
            if configs.duel_bias == 'max':
                return v + a - a.max()  # mean(1)[1].view(-1, 1)
            elif configs.duel_bias == 'mean':
                return v + a - a.mean()  # mean(1)[1].view(-1, 1)
            else:
                return v + a
        else:
            return self.final_v(h)

    def final_policy(self, z):
        """
        return a policy or a value
        """
        if 'policy' in self.agent_type:
            if configs.softmax_tau != 1:
                z /= configs.softmax_tau
            return F.softmax(z, dim=-2)
        else:
            return z

    def loss(self, data):
        """
        sum loss for mini-batch of torch_geometric
        """
        z = self.forward_partial(data)

        op_mask = data['op_mask'].x
        target = data.target_policy
        graph_index = data['op'].batch

        losses = list()
        for batch_i in range(data.num_graphs):
            indices = torch.where(graph_index == batch_i)
            z_ = z[indices]

            op_mask_ = op_mask[indices].nonzero()
            losses.append(self.loss_f(self.final_policy(z_).squeeze()[op_mask_], target[indices][op_mask_]))

        loss = torch.stack(losses).to(configs.device).mean()
        return loss

    def get_model_weight(self):
        for name, child in self.named_children():
            for param in child.parameters():
                print(name, param)

    def get_prob(self, data, a_tensors):
        z = self.forward_partial(data)

        graph_index = data['op'].batch
        policies = list()
        for batch_i in range(data.num_graphs):
            indices = torch.where(graph_index == batch_i)
            z_ = z[indices]
            policy = self.final_policy(z_)

            policies.append(policy[a_tensors[batch_i]])
        policies = torch.stack(policies).to(configs.device)

        return policies.squeeze(dim=-1)


#####################################################################################################
class Hetero_GNN_type_aware_all_pred(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, agent_type='policy'):
        super().__init__(in_dim, out_dim, in_dim_rsc, agent_type)
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n
        in_em_dim = em_dim
        if configs.self_loop:
            in_em_dim += in_dim  # h = [h||x]

        self.fs_pred = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_succ = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_disj = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all_pred = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all_succ = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])

        if configs.self_loop:
            self.fs = nn.ModuleList([get_mlp(6 * em_dim + in_em_dim, em_dim) for _ in range(aggr_n)])
        else:
            self.fs = nn.ModuleList([get_mlp(6 * em_dim, em_dim) for _ in range(aggr_n)])

    def forward_h(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)

            h_p = self.fs_pred[i](h, edge_index_dict['op', 'pred', 'op'])
            h_s = self.fs_succ[i](h, edge_index_dict['op', 'succ', 'op'])
            h_d = self.fs_disj[i](h, edge_index_dict['op', 'disj', 'op'])
            h_a = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])
            h_p2 = self.fs_all_pred[i](h, edge_index_dict['op', 'all_pred', 'op'])
            h_s2 = self.fs_all_succ[i](h, edge_index_dict['op', 'all_succ', 'op'])

            if configs.self_loop:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h_p2, h_s2, h], dim=1))
            else:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h_p2, h_s2], dim=1))

        graph_index = obs['op'].batch
        if graph_index.max().item() > 0:
            return self.final_h(h, graph_index)
        else:
            return self.final_h(h)
        # torch.isnan(h.sum())


#####################################################################################################
class Hetero_GNN_type_aware(GNN):
    def __init__(self, in_dim, out_dim, in_dim_rsc=1, agent_type='policy'):
        super().__init__(in_dim, out_dim, in_dim_rsc, agent_type)
        em_dim = configs.em_dim
        aggr_n = configs.aggr_n
        in_em_dim = em_dim
        if configs.self_loop:
            in_em_dim += in_dim  # h = [h||x]

        self.fs_pred = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_succ = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_disj = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])
        self.fs_all = nn.ModuleList([self.layer(in_em_dim, em_dim) for _ in range(aggr_n)])

        if configs.self_loop:
            self.fs = nn.ModuleList([get_mlp(4 * em_dim + in_em_dim, em_dim) for _ in range(aggr_n)])
        else:
            self.fs = nn.ModuleList([get_mlp(4 * em_dim, em_dim) for _ in range(aggr_n)])

    def forward_h(self, obs):
        """
        this is used to 'forward' and 'loss' function
        """
        x_dict = obs.x_dict
        edge_index_dict = obs.edge_index_dict
        x = x_dict['op']

        h = self.f_init(x)
        for i in range(configs.aggr_n):
            if configs.self_loop:
                h = torch.cat([h, x], dim=1)

            h_p = self.fs_pred[i](h, edge_index_dict['op', 'pred', 'op'])
            h_s = self.fs_succ[i](h, edge_index_dict['op', 'succ', 'op'])
            h_d = self.fs_disj[i](h, edge_index_dict['op', 'disj', 'op'])
            h_a = self.fs_all[i](h, edge_index_dict['op', 'all', 'op'])

            if configs.self_loop:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a, h], dim=1))
            else:
                h = self.fs[i](torch.cat([h_p, h_s, h_d, h_a], dim=1))

        graph_index = obs['op'].batch
        if graph_index.max().item() > 0:
            return self.final_h(h, graph_index)
        else:
            return self.final_h(h)

