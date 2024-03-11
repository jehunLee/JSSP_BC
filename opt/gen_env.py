from environment.env import JobShopEnv
import copy, torch
from params import configs
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


class JobShopGenEnv(JobShopEnv):
    def __init__(self, problems: list=[], pomo_n: int=1, load_envs=None):
        super().__init__(problems, pomo_n, load_envs)

    def reset(self):
        # static ##########################################
        self.job_durations = self.init_job_durations.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_mcs = self.init_job_mcs.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.op_mcs = self.job_mcs[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1)

        self.job_tails = self.init_job_tails.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_tail_ns = self.init_job_tail_ns.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_flow_due_date = self.init_job_flow_due_date.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)

        self.job_step_n = self.init_job_step_n.unsqueeze(dim=1).expand(-1, self.pomo_n, -1)

        # dynamic #########################################
        self.job_last_step = torch.zeros(self.env_n, self.pomo_n, self.max_job_n+1, dtype=torch.long)

        self.job_ready_t = self.init_job_ready.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_ready_t_precedence = self.init_job_ready.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1).clone()
        self.job_ready_t_mc_gap = torch.zeros(self.env_n, self.pomo_n, self.max_job_n+1, self.max_mc_n+1,
                                              dtype=torch.long)
        self.job_done_t = self.init_job_done.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_arrival_t = self.init_job_arrival.unsqueeze(dim=1).expand(-1, self.pomo_n, -1)

        self.mc_last_job = torch.full(size=(self.env_n, self.pomo_n, self.max_mc_n), fill_value=self.max_job_n)
        self.mc_last_job_step = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)

        self.decision_n = torch.zeros(self.env_n, self.pomo_n, dtype=torch.long)

        if 'rule' not in configs.agent_type:
            self.init_disj_info()

        ##################################################
        assign_pairs = self.next_state()

        done = self.done()
        if done:
            reward = -self.get_LB()  # (env_n, pomo_n)
        else:
            reward = None

        # add #################################################
        if self.target_mc.sum() > 0:
            target_mc = [self.target_mc.argmax(dim=2)[0, 0].item()]
        else:
            target_mc = []

        return self.get_obs(), reward, done, assign_pairs, target_mc

    ##########################################################################################################
    def step(self, a):
        a = self.get_assign_job(torch.tensor([a], dtype=torch.int64).view(self.env_n, self.pomo_n))
        self.assign(a)

        # add ######################################
        mc = self.target_mc.argmax(dim=2)[0, 0].item()
        assign_pairs = [(mc, a[0, 0, mc].item())]

        #######################################
        a_sum = a.sum(dim=2)
        a_sum = torch.where(a_sum < self.max_mc_n * self.max_job_n, 1, 0)
        self.decision_n += a_sum

        #######################################
        assign_pairs += self.next_state()

        #######################################
        done = self.done()
        if done:
            reward = -self.get_LB()  # (env_n, pomo_n)
        else:
            reward = None

        # add ######################################
        if self.target_mc.sum() > 0:
            target_mc = [self.target_mc.argmax(dim=2)[0, 0].item()]
        else:
            target_mc = []

        return self.get_obs(), reward, done, assign_pairs, target_mc

    def next_state(self):
        next_state = False
        assign_pairs = list()

        while not next_state:
            # candidate job ################################################################################
            job_mask = torch.where(self.job_last_step < self.job_step_n, 0, self.M)
            start_t = self.job_ready_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step] + job_mask
            self.curr_t = torch.min(start_t, dim=2)[0]  # (env_n, pomo_n)

            # job_mcs = self.init_job_mcs.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)  # (env_n, pomo_n, max_job_n)
            job_mcs = self.job_mcs[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step][:, :, :-1]

            if 'conflict' in configs.action_type:
                # conflict ################################################
                done_t = self.job_done_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step] + job_mask
                c_min = torch.min(done_t, dim=2)[0].unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)  # (env_n, pomo_n)

                min_jobs = torch.where(done_t == c_min, 1, 0)
                min_mcs = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
                min_mcs.scatter_add_(2, index=job_mcs, src=min_jobs)
                min_mcs = torch.where(min_mcs > 0, 1, 0)

                cand_jobs = torch.where(start_t < c_min, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            elif 'buffer' in configs.action_type:
                # buffer ################################################
                curr_t = self.curr_t.unsqueeze(dim=2).expand(-1, -1, self.max_job_n + 1)
                cand_jobs = torch.where(start_t == curr_t, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            else:
                cand_jobs = 0

            # multiple actions #############################################################################
            mc_count = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
            mc_count.scatter_add_(2, index=job_mcs, src=cand_jobs)  # (env_n, pomo_n, max_mc_n)
            if 'conflict' in configs.action_type:
                mc_count = mc_count.mul(min_mcs)

            # automatic assign job #####################
            if len(sum(torch.where(mc_count == 1))) > 0:
                if 'conflict' in configs.action_type:
                    cand_jobs_ = torch.where(start_t < c_min, self.JOB_IDX, 0)  # (env_n, pomo_n, max_job_n+1)
                elif 'buffer' in configs.action_type:
                    cand_jobs_ = torch.where(start_t == curr_t, self.JOB_IDX, 0)  # (env_n, pomo_n, max_job_n+1)
                else:
                    cand_jobs_ = 0

                mc_job_sum = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
                mc_job_sum.scatter_add_(2, index=job_mcs, src=cand_jobs_)  # (env_n, pomo_n, max_mc_n)

                # auto assign ###############################
                self.target_mc = torch.where(mc_count == 1, 1, 0)  # (env_n, pomo_n, max_mc_n)

                while torch.any(self.target_mc.sum(dim=2) > 1):
                    index = self.MC_PRIOR.mul(self.target_mc)
                    index_max = index.max(dim=2)[0].view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)
                    target_mc = copy.deepcopy(self.target_mc)
                    self.target_mc = torch.where((index == index_max) & (index > 0), 1, 0)  # (env_n, pomo_n, max_mc_n)

                    mc_auto_job = torch.where(self.target_mc == 1, mc_job_sum, self.max_job_n)
                    self.assign(mc_auto_job)

                    mc = self.target_mc.argmax(dim=2)[0, 0].item()
                    assign_pairs.append((mc, mc_auto_job[0, 0, mc].item()))

                    self.target_mc = target_mc - self.target_mc

                mc_auto_job = torch.where(self.target_mc == 1, mc_job_sum, self.max_job_n)  # null job: dummy job
                self.assign(mc_auto_job)

                ##################################################
                if self.target_mc.sum() > 0:
                    mc = self.target_mc.argmax(dim=2)[0, 0].item()
                    assign_pairs.append((mc, mc_auto_job[0, 0, mc].item()))

            # next state #####################
            else:
                self.target_mc = torch.where(mc_count > 0, 1, 0)  # (env_n, pomo_n, max_mc_n)
                if torch.any(self.target_mc.sum(dim=2) > 1):
                    index = self.MC_PRIOR.mul(self.target_mc)
                    index_max = index.max(dim=2)[0].view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)
                    self.target_mc = torch.where((index == index_max) & (index > 0), 1, 0)  # (env_n, pomo_n, max_mc_n)

                job_mc = F.one_hot(self.job_last_step, num_classes=self.max_mc_n + 1).mul(self.job_mcs).sum(dim=3)
                cand_jobs = self.target_mc[self.ENV_IDX_J, self.POMO_IDX_J, job_mc].mul(cand_jobs)
                cand_jobs_ = cand_jobs[:, :, :-1].unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n)

                job_last_step_one_hot = F.one_hot(self.job_last_step, num_classes=self.max_mc_n + 1)[:, :, :-1, :-1]
                self.op_mask = job_last_step_one_hot.mul(cand_jobs_).view(self.env_n, self.pomo_n, -1)

                next_state = True

        return assign_pairs

    def get_torch_geom(self):
        if 'rule' in configs.agent_type:
            # torch_geom ############################
            data = HeteroData()

            data['op'].x = self.get_op_x()
            data['op_mask'].x = self.op_mask

            return data

        else:
            x = self.get_op_x()

            if 'dyn' in configs.env_type:
                curr_t_ = self.curr_t.view(self.env_n, self.pomo_n, 1, 1).expand(
                    -1, -1, self.max_job_n+1, self.max_mc_n)
                op_remain = torch.where(self.job_done_t[:, :, :, :-1] > curr_t_, 1, 0)[:, :, :-1, :].view(
                    self.env_n, self.pomo_n, -1)

            def get_data(i, j):
                # torch_geom ############################
                data = HeteroData()

                data['op'].x = x[i, j]
                data['op_mask'].x = self.op_mask[i, j]

                # pred edges
                data['op', 'pred', 'op'].edge_index = self.e_pred[i]
                data['op', 'succ', 'op'].edge_index = self.e_succ[i]

                # disj edges
                e_disj = self.get_disj_edge(i, j)
                data['op', 'disj', 'op'].edge_index = e_disj

                # all: for simple GNN
                data['op', 'all', 'op'].edge_index = torch.cat([self.e_pred[i], self.e_succ[i], e_disj], dim=1)

                # meta edges
                if 'all_pred' in configs.model_type:
                    data['op', 'all_pred', 'op'].edge_index = self.e_all_pred[i]
                    data['op', 'all_succ', 'op'].edge_index = self.e_all_succ[i]

                #######################################################################################################
                if 'dyn' in configs.env_type:
                    op_remain__ = op_remain[i, j].mul(self.JOB_OP[i, j, :-1, :-1].reshape(-1))
                    op_remain_ = torch.nonzero(op_remain__).view(-1)
                    data = data.subgraph({'op': op_remain_, 'op_mask': op_remain_})
                    data['op_remain'] = op_remain_
                else:
                    op_remain_ = torch.nonzero(self.JOB_OP[i, j, :-1, :-1].reshape(-1)).view(-1)
                    data = data.subgraph({'op': op_remain_, 'op_mask': op_remain_})
                    data['op_remain'] = op_remain_

                return data

            # torch_geom ############################
            data = get_data(0, 0)

            # if data['op'].x.sum().isnan().item():
            #     print('error: loss nan')

            return data