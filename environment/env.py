import copy, torch

from utils import load_data
from collections import defaultdict
from params import configs
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader  # https://github.com/pyg-team/pytorch_geometric/issues/2961 ######


class JobShopEnv:
    def __init__(self, problems: list=[], pomo_n: int=1, load_envs=None):
        self.M = 1e4
        self.pomo_n = pomo_n

        if load_envs:
            self.load(load_envs)
            self.init_index()

        else:
            # environment ################################################################
            self.env_n = len(problems)

            self.max_job_n = max([x[1] for x in problems])
            self.max_mc_n = max([x[2] for x in problems])

            self.op_n = self.max_job_n * self.max_mc_n
            self.op_map = torch.arange(self.op_n, dtype=torch.long)[None, :].expand(
                self.env_n, self.op_n).view(self.env_n, self.max_job_n, -1)

            # static ######################################################################
            self.load_problems(problems)
            self.init_index()

        UB = self.init_job_durations.max() * self.max_mc_n * self.max_job_n * self.max_mc_n
        while UB > self.M:
            self.M *= 10

    def load(self, envs) -> (dict, list):
        # environment ################################################################
        self.env_n = len(envs)

        self.max_job_n = envs[0].max_job_n
        self.max_mc_n = envs[0].max_mc_n

        self.op_n = envs[0].op_n
        self.op_map = envs[0].op_map

        # static #####################################################################
        self.job_durations = envs[0].job_durations.repeat(self.env_n, self.pomo_n, 1, 1)
        self.job_mcs = envs[0].job_mcs.repeat(self.env_n, self.pomo_n, 1, 1)
        self.job_step_n = envs[0].job_step_n.repeat(self.env_n, self.pomo_n, 1)
        self.init_job_step_n = envs[0].init_job_step_n.repeat(self.env_n, 1, 1)

        self.job_tails = envs[0].job_tails.repeat(self.env_n, self.pomo_n, 1, 1)
        self.job_tail_ns = envs[0].job_tail_ns.repeat(self.env_n, self.pomo_n, 1, 1)
        self.job_flow_due_date = envs[0].job_flow_due_date.repeat(self.env_n, self.pomo_n, 1, 1)

        self.op_mcs = envs[0].op_mcs.repeat(self.env_n, self.pomo_n, 1)

        # dynamic #####################################################################
        self.job_last_step = torch.cat([env.job_last_step for env in envs], dim=0).repeat(
            1, self.pomo_n, 1)

        self.job_ready_t = torch.cat([env.job_ready_t for env in envs], dim=0).repeat(
            1, self.pomo_n, 1, 1)
        self.job_ready_t_precedence = torch.cat([env.job_ready_t_precedence for env in envs], dim=0).repeat(
            1, self.pomo_n, 1, 1)
        self.job_ready_t_mc_gap = torch.cat([env.job_ready_t_mc_gap for env in envs], dim=0).repeat(
            1, self.pomo_n, 1, 1)
        self.job_done_t = torch.cat([env.job_done_t for env in envs], dim=0).repeat(
            1, self.pomo_n, 1, 1)

        self.mc_last_job = torch.cat([env.mc_last_job for env in envs], dim=0).repeat(
            1, self.pomo_n, 1)
        self.mc_last_job_step = torch.cat([env.mc_last_job_step for env in envs], dim=0).repeat(
            1, self.pomo_n, 1)

        self.decision_n = torch.cat([env.decision_n for env in envs], dim=0).repeat(
            1, self.pomo_n)
        self.target_mc = torch.cat([env.target_mc for env in envs], dim=0).repeat(
            1, self.pomo_n, 1)
        self.op_mask = torch.cat([env.op_mask for env in envs], dim=0).repeat(
            1, self.pomo_n, 1)

        if 'rule' not in configs.agent_type:
            self.curr_t = torch.cat([env.curr_t for env in envs], dim=0).expand(-1, self.pomo_n)

            self.mc_ops = defaultdict()
            self.mc_prev_op = defaultdict()

            self.e_pred = defaultdict()
            self.e_succ = defaultdict()
            self.e_all_pred = defaultdict()
            self.e_all_succ = defaultdict()
            self.e_disj = defaultdict()

            # self.reserved_e_disj = defaultdict()
            self.reserved = defaultdict()

            for i in range(self.env_n):
                self.e_pred[i] = envs[i].e_pred[0]
                self.e_succ[i] = envs[i].e_succ[0]
                self.e_all_pred[i] = envs[i].e_all_pred[0]
                self.e_all_succ[i] = envs[i].e_all_succ[0]

                for j in range(self.pomo_n):
                    self.mc_ops[i, j] = copy.deepcopy(envs[i].mc_ops[0, 0])
                    self.mc_prev_op[i, j] = copy.deepcopy(envs[i].mc_prev_op[0, 0])

                    self.e_disj[i, j] = copy.deepcopy(envs[i].e_disj[0, 0])

                    # self.reserved_e_disj[i, j] = copy.deepcopy(envs[i].reserved_e_disj[0, 0])
                    self.reserved[i, j] = copy.deepcopy(envs[i].reserved[0, 0])

    def gen_init_idxs(self):
        self.init_job_durations = torch.zeros(self.env_n, self.max_job_n+1, self.max_mc_n+1, dtype=torch.long)
        self.init_job_mcs = torch.zeros(self.env_n, self.max_job_n+1, self.max_mc_n+1, dtype=torch.long)
        self.init_job_step_n = torch.zeros(self.env_n, self.max_job_n+1, dtype=torch.long)

        self.init_job_tails = torch.zeros(self.env_n, self.max_job_n, self.max_mc_n, dtype=torch.long)
        self.init_job_tail_ns = torch.zeros(self.env_n, self.max_job_n, self.max_mc_n, dtype=torch.long)
        self.init_job_flow_due_date = torch.zeros(self.env_n, self.max_job_n, self.max_mc_n, dtype=torch.long)

        self.init_job_ready = torch.zeros(self.env_n, self.max_job_n+1, self.max_mc_n+1, dtype=torch.long)
        self.init_job_done = torch.zeros(self.env_n, self.max_job_n+1, self.max_mc_n+1, dtype=torch.long)
        self.init_job_arrival = torch.zeros(self.env_n, self.max_job_n + 1, dtype=torch.long)

        if 'rule' not in configs.agent_type:
            self.init_mc_ops = defaultdict()
            self.init_mc_prev_op = defaultdict()

            self.init_e_pred = defaultdict()
            self.init_e_all_pred = defaultdict()
            self.init_e_disj = defaultdict()

            for i in range(self.env_n):
                self.init_mc_ops[i] = defaultdict(list)
                self.init_mc_prev_op[i] = dict()

                self.init_e_pred[i] = list()
                self.init_e_all_pred[i] = list()
                self.init_e_disj[i] = list()

    def load_problems(self, problems) -> (dict, list):
        self.gen_init_idxs()

        for i, (benchmark, job_n, mc_n, instance_i) in enumerate(problems):
            job_mcs, job_prts = load_data(benchmark, job_n, mc_n, instance_i)

            for j, prts in enumerate(job_prts):
                self.init_job_durations[i, j, :len(prts)] = torch.tensor(prts, dtype=torch.long)
                self.init_job_mcs[i, j, :len(prts)] = torch.tensor(job_mcs[j], dtype=torch.long)
                self.init_job_step_n[i, j] = len(prts)

                for k in range(len(prts)):
                    self.init_job_tails[i, j, :k] += prts[k]
                    self.init_job_tail_ns[i, j, :k] += 1
                    self.init_job_flow_due_date[i, j, k:] += prts[k]

                    self.init_job_ready[i, j, (k+1):] += prts[k]
                    self.init_job_done[i, j, k:] += prts[k]

                    if 'rule' not in configs.agent_type:
                        op = self.op_map[i, j, k].item()
                        m = job_mcs[j][k]

                        # pred arc #####
                        if k < len(prts) - 1:
                            self.init_e_pred[i].append((op, op+1))

                            # all_pred arc #####
                            for l in range(len(prts) - 1 - k):
                                self.init_e_all_pred[i].append((op, op+1+l))

                        # disj arc #####
                        for op2 in self.init_mc_ops[i][m]:
                            self.init_e_disj[i].append((op, op2))
                            self.init_e_disj[i].append((op2, op))
                        self.init_mc_ops[i][m].append(op)

    def init_index(self):
        self.ENV_IDX_J = torch.arange(self.env_n, dtype=torch.long)[:, None, None].expand(
            self.env_n, self.pomo_n, self.max_job_n+1)
        self.POMO_IDX_J = torch.arange(self.pomo_n, dtype=torch.long)[None, :, None].expand(
            self.env_n, self.pomo_n, self.max_job_n+1)
        self.JOB_IDX = torch.arange(self.max_job_n+1, dtype=torch.long)[None, None, :].expand(
            self.env_n, self.pomo_n, self.max_job_n+1)

        self.ENV_IDX_M = torch.arange(self.env_n, dtype=torch.long)[:, None, None].expand(
            self.env_n, self.pomo_n, self.max_mc_n)
        self.POMO_IDX_M = torch.arange(self.pomo_n, dtype=torch.long)[None, :, None].expand(
            self.env_n, self.pomo_n, self.max_mc_n)

        self.ENV_IDX_O_ = torch.arange(self.env_n, dtype=torch.long)[:, None, None, None].expand(
            self.env_n, self.pomo_n, self.max_job_n+1, self.max_mc_n+1)
        self.POMO_IDX_O_ = torch.arange(self.pomo_n, dtype=torch.long)[None, :, None, None].expand(
            self.env_n, self.pomo_n, self.max_job_n+1, self.max_mc_n+1)

        self.JOB_STEP_IDX = torch.arange(self.max_mc_n+1, dtype=torch.long)[None, None, None, :].expand(
            self.env_n, self.pomo_n, self.max_job_n+1, self.max_mc_n+1)

        self.ENV_IDX_O = torch.arange(self.env_n, dtype=torch.long)[:, None, None].expand(
            self.env_n, self.pomo_n, self.op_n)
        self.POMO_IDX_O = torch.arange(self.pomo_n, dtype=torch.long)[None, :, None].expand(
            self.env_n, self.pomo_n, self.op_n)
        self.OP_IDX = torch.arange(self.op_n, dtype=torch.long)[None, None, :].expand(
            self.env_n, self.pomo_n, self.op_n)

        self.MC_PRIOR = torch.arange(self.max_mc_n, dtype=torch.long)[None, None, :].flip(dims=[2]) + 1
        self.MC_PRIOR = self.MC_PRIOR.expand(self.env_n, self.pomo_n, self.max_mc_n)

        self.JOB_ONE = torch.full(size=(self.env_n, self.pomo_n, self.max_job_n+1), fill_value=1)
        self.JOB_OP = torch.where(self.JOB_STEP_IDX < self.init_job_step_n.view(
            self.env_n, 1, self.max_job_n+1, 1).expand(-1, self.pomo_n, -1, self.max_mc_n+1), 1, 0)

    #######################################################################################################
    def init_disj_info(self):
        self.mc_ops = defaultdict()
        self.mc_prev_op = defaultdict()

        self.e_pred = defaultdict()
        self.e_succ = defaultdict()

        self.e_all_pred = defaultdict()
        self.e_all_succ = defaultdict()

        self.e_disj = defaultdict()

        for i in range(self.env_n):
            e_pred = torch.tensor(self.init_e_pred[i], dtype=torch.long).view(-1, 2).to(torch.long).t().detach()
            e_succ = torch.zeros_like(e_pred)
            e_succ[0, :] = e_pred[1, :]
            e_succ[1, :] = e_pred[0, :]

            e_all_pred = torch.tensor(self.init_e_all_pred[i], dtype=torch.long).view(-1, 2).to(
                torch.long).t().detach()
            e_all_succ = torch.zeros_like(e_all_pred)
            e_all_succ[0, :] = e_all_pred[1, :]
            e_all_succ[1, :] = e_all_pred[0, :]

            self.e_pred[i] = e_pred
            self.e_succ[i] = e_succ

            self.e_all_pred[i] = e_all_pred
            self.e_all_succ[i] = e_all_succ

            for j in range(self.pomo_n):
                self.mc_ops[i, j] = copy.deepcopy(self.init_mc_ops[i])
                self.mc_prev_op[i, j] = copy.deepcopy(self.init_mc_prev_op[i])

                self.e_disj[i, j] = copy.deepcopy(self.init_e_disj[i])

    def reset_idxs(self):
        # static ##########################################
        self.job_durations = self.init_job_durations.unsqueeze(dim=1).repeat(1, self.pomo_n, 1, 1)
        self.job_mcs = self.init_job_mcs.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.op_mcs = self.job_mcs[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1)

        self.job_tails = self.init_job_tails.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_tail_ns = self.init_job_tail_ns.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)
        self.job_flow_due_date = self.init_job_flow_due_date.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)

        self.job_step_n = self.init_job_step_n.unsqueeze(dim=1).expand(-1, self.pomo_n, -1)

        # dynamic #########################################
        self.job_last_step = torch.zeros(self.env_n, self.pomo_n, self.max_job_n+1, dtype=torch.long)

        self.job_ready_t = self.init_job_ready.unsqueeze(dim=1).repeat(1, self.pomo_n, 1, 1)
        self.job_ready_t_precedence = self.init_job_ready.unsqueeze(dim=1).repeat(1, self.pomo_n, 1, 1).clone()
        self.job_ready_t_mc_gap = torch.zeros(self.env_n, self.pomo_n, self.max_job_n+1, self.max_mc_n+1,
                                              dtype=torch.long)
        self.job_done_t = self.init_job_done.unsqueeze(dim=1).repeat(1, self.pomo_n, 1, 1)
        self.job_arrival_t_ = self.init_job_arrival.unsqueeze(dim=1).repeat(1, self.pomo_n, 1)

        self.mc_last_job = torch.full(size=(self.env_n, self.pomo_n, self.max_mc_n), fill_value=self.max_job_n)
        self.mc_last_job_step = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)

        self.decision_n = torch.zeros(self.env_n, self.pomo_n, dtype=torch.long)

    def reset(self):
        self.reset_idxs()

        if 'rule' not in configs.agent_type:
            self.init_disj_info()

        ##################################################
        self.next_state()

        done = self.done()
        if done:
            reward = -self.get_LB()  # (env_n, pomo_n)
        else:
            reward = None

        return self.get_obs(), reward, done

    def step(self, a):
        if 'rule' in configs.agent_type:
            self.assign(a)
        elif self.pomo_n > 1:
            a = self.get_assign_job(torch.concat(a).to('cpu').view(
                self.pomo_n, self.env_n).transpose(0, 1).contiguous())
            self.assign(a)
        else:
            a = self.get_assign_job(torch.concat(a).to('cpu').view(self.env_n, 1))
            self.assign(a)

        ##############################################################################
        a_sum = a.sum(dim=2)
        a_sum = torch.where(a_sum < self.max_mc_n * self.max_job_n, 1, 0)
        self.decision_n += a_sum

        ##############################################################################
        self.next_state()

        ##############################################################################
        done = self.done()
        if done:
            reward = -self.get_LB()  # (env_n, pomo_n)
        else:
            reward = None

        return self.get_obs(), reward, done

    ########################################################################################################
    def assign(self, mc_job):  # assign for a machine
        # assign job remain ops ###############################################################
        assign_job = torch.zeros(self.env_n, self.pomo_n, self.max_job_n+1, dtype=torch.long)
        assign_job.scatter_(2, index=mc_job, src=self.JOB_ONE)[:, :, -1] = 0
        assign_job_ = assign_job.view(self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(
            -1, -1, -1, self.max_mc_n+1)

        last_step = self.job_last_step.view(self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(
            -1, -1, -1, self.max_mc_n+1)
        assign_job_follow = torch.where(last_step <= self.JOB_STEP_IDX, 1, 0).mul(assign_job_)

        # ready_t update - assigned job ########################################################
        assign_mc_gap = self.job_ready_t_mc_gap[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX,
                                                self.job_last_step].mul(assign_job)
        assign_mc_gap = assign_mc_gap.view(self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(
            -1, -1, -1, self.max_mc_n+1).mul(assign_job_follow)
        self.job_ready_t_mc_gap -= assign_mc_gap
        self.job_ready_t_precedence += assign_mc_gap

        # disjunctive arc update ###############################################################
        if 'rule' not in configs.agent_type:
            mcs_ = mc_job.argmin(dim=2)
            assign_job_ = assign_job.argmax(dim=2)

            for i in range(self.env_n):
                for j in range(self.pomo_n):
                    mc = mcs_[i, j].item()
                    if mc_job[i, j, mc].item() == self.max_job_n:
                        continue

                    job = assign_job_[i, j].item()
                    if job == self.max_job_n:
                        continue

                    op = self.op_map[i, job, self.job_last_step[i, j, job]].item()
                    self.mc_ops[i, j][mc].remove(op)

                    del_list = [(op2, op) for op2 in self.mc_ops[i, j][mc]]
                    prev_op = None
                    if mc in self.mc_prev_op[i, j]:
                        prev_op = self.mc_prev_op[i, j][mc]
                        del_list += [(prev_op, op2) for op2 in self.mc_ops[i, j][mc]]
                    self.mc_prev_op[i, j][mc] = op

                    # remove: unassigned ops -> selected op / previous op -> unassigned ops
                    self.e_disj[i, j] = list(set(self.e_disj[i, j]) - set(del_list))

        # mc_last_job, job_last_step update  ##################################################################
        self.mc_last_job = torch.where(mc_job < self.max_job_n, mc_job, self.mc_last_job)
        mc_assigned_job_last_step = self.job_last_step[self.ENV_IDX_M, self.POMO_IDX_M, mc_job]
        self.mc_last_job_step = torch.where(mc_job < self.max_job_n, mc_assigned_job_last_step, self.mc_last_job_step)

        self.job_last_step[self.ENV_IDX_M, self.POMO_IDX_M, mc_job] += 1
        self.job_last_step[:, :, -1] = 0

        # same mc jobs mc_gap update #########################################################################
        # job mc_t ##############################################################################
        mc_t = self.job_done_t[self.ENV_IDX_M, self.POMO_IDX_M, self.mc_last_job, self.mc_last_job_step]
        job_mc_t = mc_t[self.ENV_IDX_O_, self.POMO_IDX_O_, self.job_mcs]

        # remain operations #####################################################################
        last_step = self.job_last_step.view(self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(
            -1, -1, -1, self.max_mc_n+1)
        job_remain = torch.where(last_step <= self.JOB_STEP_IDX, 1, 0)  # (env_n, pomo_n, max_job_n+1, max_mc_n+1)
        job_remain[:, :, -1, :] = 0
        job_remain[:, :, :, -1] = 0

        # same step #############################################################################
        same_mc_step = self.target_mc[self.ENV_IDX_O_, self.POMO_IDX_O_, self.job_mcs].mul(job_remain).mul(self.JOB_OP)

        step_pos = same_mc_step.argmax(dim=3).view(self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(
            -1, -1, -1, self.max_mc_n+1)
        step_sum = same_mc_step.sum(dim=3)
        step_sum = torch.where(step_sum > 0, 1, 0).view(self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(
            -1, -1, -1, self.max_mc_n+1)
        same_mc_step_follow = torch.where(step_pos <= self.JOB_STEP_IDX, 1, 0).mul(step_sum)

        mc_update = (job_mc_t - self.job_ready_t_precedence).mul(same_mc_step).sum(dim=3).view(
            self.env_n, self.pomo_n, self.max_job_n+1, -1).expand(-1, -1, -1, self.max_mc_n+1).mul(same_mc_step_follow)

        self.job_ready_t_mc_gap = torch.where(self.job_ready_t_mc_gap > mc_update, self.job_ready_t_mc_gap, mc_update)

        # ready_t update ######################################################################
        self.job_ready_t = self.job_ready_t_precedence + self.job_ready_t_mc_gap
        self.job_done_t = self.job_ready_t + self.job_durations

    def next_state(self):
        next_state = False

        while not next_state:
            job_mask = torch.where(self.job_last_step < self.job_step_n, 0, self.M)
            start_t = self.job_ready_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step] + job_mask
            self.curr_t = torch.min(start_t, dim=2)[0]  # (env_n, pomo_n)

            # candidate job ################################################################################
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
                curr_t = self.curr_t.unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)
                cand_jobs = torch.where(start_t == curr_t, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            else:
                cand_jobs = 0

            # count mc actions #############################################################################
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

                    self.target_mc = target_mc - self.target_mc

                mc_auto_job = torch.where(self.target_mc == 1, mc_job_sum, self.max_job_n)  # null job: dummy job
                self.assign(mc_auto_job)

            # next state #####################
            else:
                self.target_mc = torch.where(mc_count > 0, 1, 0)  # (env_n, pomo_n, max_mc_n)
                if torch.any(self.target_mc.sum(dim=2) > 1):
                    index = self.MC_PRIOR.mul(self.target_mc)
                    index_max = index.max(dim=2)[0].view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)
                    self.target_mc = torch.where((index == index_max) & (index > 0), 1, 0)  # (env_n, pomo_n, max_mc_n)

                job_mc = F.one_hot(self.job_last_step, num_classes=self.max_mc_n+1).mul(self.job_mcs).sum(dim=3)
                cand_jobs = self.target_mc[self.ENV_IDX_J, self.POMO_IDX_J, job_mc].mul(cand_jobs)
                cand_jobs_ = cand_jobs[:, :, :-1].unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n)

                job_last_step_one_hot = F.one_hot(self.job_last_step, num_classes=self.max_mc_n+1)[:, :, :-1, :-1]
                self.op_mask = job_last_step_one_hot.mul(cand_jobs_).view(self.env_n, self.pomo_n, -1)

                next_state = True

    #####################################################################################################
    def get_obs(self):
        """
        get observation(=state features)
        """
        if self.done():
            return 0
        return self.get_torch_geom()

    def get_op_x(self):
        if 'rule' in configs.agent_type:  # rule
            prt = self.job_durations[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1)

            tail_t = self.job_tails.view(self.env_n, self.pomo_n, -1)
            tail_n = self.job_tail_ns.view(self.env_n, self.pomo_n, -1)

            fdd_mwkr = self.job_flow_due_date.view(self.env_n, self.pomo_n, -1) / (prt + tail_t)
            fdd_mwkr = torch.nan_to_num(fdd_mwkr, nan=1)

            # tie: FIFO -> SPT
            enter_t = torch.zeros(self.env_n, self.pomo_n, self.max_job_n, self.max_mc_n)
            enter_t[:, :, :, 1:] = self.job_done_t[:, :, :-1, :-2]
            enter_t = enter_t.reshape(self.env_n, self.pomo_n, -1)  # tie

            return torch.stack([prt, tail_t, tail_n, prt+tail_t, fdd_mwkr, enter_t], dim=3)

        else:  # normalized
            curr_t_ = self.curr_t.view(self.env_n, self.pomo_n, 1, -1).expand(-1, -1, self.max_job_n, self.max_mc_n)

            # op_status ##############################
            # [non_available / waiting / being_processed / reserved / reservable / (done)]
            last_step = self.job_last_step.view(self.env_n, self.pomo_n, self.max_job_n + 1, -1).expand(
                -1, -1, -1, self.max_mc_n+1)
            job_assigned = torch.where(last_step > self.JOB_STEP_IDX, 2, 0)[:, :, :-1, :-1]
            job_candidate = torch.where(last_step == self.JOB_STEP_IDX, 1, 0)[:, :, :-1, :-1]
            start_t = self.job_ready_t[:, :, :-1, :-1]

            # 3: reserved / 2: being_processed (2, 3: assigned)
            op_status = torch.where((job_assigned == 2) & (start_t > curr_t_), 3, job_assigned)

            # 4: reservable
            op_status = torch.where((job_candidate == 1), 4, op_status) #########

            # 1: waiting
            op_status = torch.where((job_candidate == 1) & (start_t == curr_t_), 1, op_status)

            # 5: done (no dynamic environment -> add done status)
            if 'dyn' not in configs.env_type:
                op_status = torch.where(self.job_done_t[:, :, :-1, :-1] <= curr_t_, 5, op_status)
                op_status = op_status.view(self.env_n, self.pomo_n, -1)
                op_status = F.one_hot(op_status, 6)
            else:
                op_status = op_status.view(self.env_n, self.pomo_n, -1)
                op_status = F.one_hot(op_status, 5)

            # prt ##############################
            if 'dyn' in configs.env_type:
                remain_done = self.job_done_t[:, :, :-1, :-1] - curr_t_
                prt = self.job_durations[:, :, :-1, :-1]
                prt = torch.where(remain_done < prt, remain_done, prt).reshape(self.env_n, self.pomo_n, -1)
            else:
                prt = self.job_durations[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1)

            if 'prt_norm' in configs.state_type:
                max_prt = prt.max(dim=2)[0]
                max_prt = max_prt.view(self.env_n, self.pomo_n, 1).expand(-1, -1, prt.size()[2])
                prt = prt / max_prt
            else:
                prt = prt.to(torch.float32)

            # ready_t ##############################
            r_t_ = self.job_ready_t[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1)
            if 'dyn' in configs.env_type:
                if 'prt_norm' in configs.state_type:
                    r_t = (r_t_ - curr_t_.reshape(self.env_n, self.pomo_n, -1)) / max_prt
                else:
                    r_t = (r_t_ - curr_t_.reshape(self.env_n, self.pomo_n, -1))
            else:
                if 'prt_norm' in configs.state_type:
                    r_t = r_t_ / max_prt
                else:
                    r_t = r_t_

            # tail_t ##############################
            if 'prt_norm' in configs.state_type:
                tail_t = self.job_tails.view(self.env_n, self.pomo_n, -1) / max_prt
            else:
                tail_t = self.job_tails.view(self.env_n, self.pomo_n, -1)

            # tail_n ##############################
            tail_n = self.job_tail_ns.view(self.env_n, self.pomo_n, -1)
            # max_tail_n = tail_n.max(dim=2)[0].view(self.env_n, self.pomo_n, 1).expand(-1, -1, tail_n.size()[2])
            # tail_n = tail_n / max_tail_n

            # relative prt ##############################
            prt_r = prt / (prt + tail_t)

            # delay_op, delay_mc ##############################
            mc_t = self.job_done_t[self.ENV_IDX_M, self.POMO_IDX_M, self.mc_last_job, self.mc_last_job_step]
            if 'mc_breakdown' in configs.dyn_type:
                if 'known' in configs.dyn_type:
                    mc_t = torch.where(self.mc_repair > mc_t, self.mc_repair, mc_t)

            curr_t__ = self.curr_t.view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)
            mc_t = torch.where(mc_t < curr_t__, curr_t__, mc_t)
            op_mc_t = mc_t[self.ENV_IDX_O, self.POMO_IDX_O, self.op_mcs]
            gap = op_mc_t - r_t_

            if 'prt_norm' in configs.state_type:
                delay_op = torch.where(gap > 0, gap, 0) / max_prt
                delay_op = torch.where(job_assigned.reshape(self.env_n, self.pomo_n, -1) > 0, 0, delay_op)
                delay_mc = torch.where(-gap > 0, -gap, 0) / max_prt
            else:
                delay_op = torch.where(gap > 0, gap, 0)
                delay_op = torch.where(job_assigned.reshape(self.env_n, self.pomo_n, -1) > 0, 0, delay_op)
                delay_mc = torch.where(-gap > 0, -gap, 0)

            # mc load ##############################
            prt = torch.where(prt > 0, prt, 0)
            mc_load = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.float32)
            mc_load.scatter_add_(2, index=self.job_mcs[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1), src=prt)
            mc_load = mc_load / mc_load.sum(dim=2).view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)

            op_mc_load = mc_load[self.ENV_IDX_O, self.POMO_IDX_O, self.op_mcs]

            # return ###############################
            x_list = [prt_r, r_t]
            if 'mc_gap' in configs.state_type:
                x_list += [delay_mc, delay_op]
            if 'mc_load' in configs.state_type:
                x_list.append(op_mc_load)

            return torch.cat([torch.stack([prt, tail_t, tail_n], dim=3), op_status, torch.stack(x_list, dim=3)], dim=3)

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
                    -1, -1, self.max_job_n + 1, self.max_mc_n)
                op_remain = torch.where(self.job_done_t[:, :, :, :-1] > curr_t_, 1, 0)
                if 'job_arrival' in configs.dyn_type:
                    job_arrival_t_ = self.job_arrival_t_.unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n)
                    job_avail = torch.where(curr_t_ >= job_arrival_t_, 1, 0)  # reflect non-arrival
                    op_remain = op_remain.mul(job_avail)
                op_remain = op_remain[:, :, :-1, :].view(self.env_n, self.pomo_n, -1)

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

            if self.pomo_n > 1:
                total_data_list = list()
                for j in range(self.pomo_n):
                    data_list = list()
                    for i in range(self.env_n):
                        # torch_geom ############################
                        data = get_data(i, j)
                        data_list.append(data)

                    loader = DataLoader(data_list, len(data_list), shuffle=False)
                    total_data_list.append(next(iter(loader)).to(configs.device))

                return total_data_list

            else:
                data_list = list()
                for i in range(self.env_n):
                    for j in range(self.pomo_n):
                        # torch_geom ############################
                        data = get_data(i, j)
                        data_list.append(data)

                loader = DataLoader(data_list, len(data_list), shuffle=False)

                return next(iter(loader)).to(configs.device)

    #####################################################################################################
    def get_disj_edge(self, i, j):
        """
        get disjunctive edges
        """
        return torch.tensor(self.e_disj[i, j], dtype=torch.long).view(-1, 2).to(torch.long).t().detach()

    #####################################################################################################
    def get_LB(self):
        # makespan ##############
        max_done_t = self.job_done_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_step_n-1].max(dim=2)[0]

        # total_complete_t ##############
        done_t = (self.job_done_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_step_n-1] -
                  self.job_arrival_t_)[:, :, :-1]
        total_c_t = done_t.sum(dim=2)

        # mc LB ##############
        # mc_t = self.job_done_t[self.ENV_IDX_M, self.POMO_IDX_M, self.mc_last_job, self.mc_last_job_step]
        # last_step = self.job_last_step.view(self.env_n, self.pomo_n, self.max_job_n + 1, -1).expand(
        #     -1, -1, -1, self.max_mc_n + 1)
        # prt = torch.where(last_step <= self.JOB_STEP_IDX, self.job_durations, 0)[:, :, :-1, :-1].reshape(
        #     self.env_n, self.pomo_n, -1)  # assigned job 만 prt 유지
        # mc_load = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
        # mc_load.scatter_add_(2, index=self.job_mcs[:, :, :-1, :-1].reshape(self.env_n, self.pomo_n, -1), src=prt)

        # return torch.stack((max_done_t, (mc_t+mc_load).max(dim=2)[0]), dim=2).max(dim=2)[0]
        return max_done_t

    def done(self):
        finished = (self.job_last_step == self.job_step_n).all(dim=2)  # (env_n, pomo_n)
        return finished.all().item()

    #####################################################################################################
    def get_action_rule(self, obs, rules):
        def get_index_pos(rule):
            if rule == 'LPT':
                return 1, 0
            elif rule == 'SPT':
                return -1, 0
            elif rule == 'LTT':
                return 1, 1
            elif rule == 'STT':
                return -1, 1
            elif rule == 'MOR':
                return 1, 2
            elif rule == 'LOR':
                return -1, 2
            elif rule == 'LRPT':
                return 1, 3
            elif rule == 'SRPT':
                return -1, 3
            elif rule == 'FDD/MWKR':
                return -1, 4
            else:
                return 0

        features = obs['op'].x
        op_mask = obs['op_mask'].x

        index_pos = torch.zeros(self.pomo_n, dtype=torch.long)
        index_prod = torch.zeros(self.pomo_n, dtype=torch.long)

        for i in range(self.pomo_n):
            rule = rules[i]
            prod, pos = get_index_pos(rule)

            index_pos[i] = pos
            index_prod[i] = prod

        index_pos = index_pos.view(1, self.pomo_n, 1).expand(self.env_n, -1, op_mask.size()[2])
        index_prod = index_prod.view(1, self.pomo_n, 1).expand(self.env_n, -1, op_mask.size()[2])

        index = features[self.ENV_IDX_O, self.POMO_IDX_O, self.OP_IDX, index_pos].mul(index_prod)
        index -= (1 - op_mask) * 1e4

        # tie: FIFO -> SPT
        index_FIFO = -features[self.ENV_IDX_O, self.POMO_IDX_O, self.OP_IDX, 5]
        index_SPT = -features[self.ENV_IDX_O, self.POMO_IDX_O, self.OP_IDX, 0]
        index += index_FIFO * 1e-4 + index_SPT * 1e-8

        return self.get_assign_job(index.argmax(dim=2))

    def get_assign_job(self, selected_index):
        assign_job = F.one_hot(selected_index, num_classes=self.op_n).view(
            self.env_n, self.pomo_n, self.max_job_n, -1).max(dim=3)[0].argmax(dim=2)
        assign_job = assign_job.view(self.env_n, self.pomo_n, 1).expand(-1, -1, self.max_mc_n)

        return torch.where(self.target_mc == 1, assign_job, self.max_job_n)

    #####################################################################################################
    def run_episode_rule(self, rules):
        obs, reward, done = self.reset()
        while not done:
            a = self.get_action_rule(obs, rules)
            obs, reward, done = self.step(a)
        return reward, self.decision_n

    def show_gantt_plotly(self, env_i, pomo_i, title: str=''):
        import pandas as pd
        import plotly.express as px
        import os

        first_list = [(0, 0, 0, 0, 0, 0, "")]
        col_name = ['op_i', 'job_id', 'ith', 'resource', 'start_t', 'end_t', 'text']
        df = pd.DataFrame(first_list, columns=col_name)

        ready_t = self.job_ready_t[env_i, pomo_i, :, :]
        done_t = self.job_done_t[env_i, pomo_i, :, :]
        prts = self.job_durations[env_i, pomo_i, :, :]
        mcs = self.job_mcs[env_i, pomo_i, :, :]

        for i in range(self.max_job_n):
            step_n = self.job_step_n[env_i, pomo_i, i]
            if step_n:
                for j in range(step_n):
                    op_idx = self.max_job_n * i + j
                    mc_i = mcs[i, j].item()

                    s_t = ready_t[i, j].item()
                    e_t = done_t[i, j].item()
                    prt = prts[i, j].item()

                    df2 = pd.DataFrame([(op_idx, i, j, mc_i, s_t, e_t,
                                         "{}, {}, {}".format(i, j, prt))], columns=col_name)
                    df = pd.concat([df, df2])

        # breakdown
        if 'mc_breakdown' in configs.dyn_type:
            if 'known' in configs.dyn_type:
                for k in range(self.max_mc_n):
                    s_t = configs.parameter * k
                    e_t = configs.parameter * (k+1)
                    df2 = pd.DataFrame([(-1, -1, -1, k, s_t, e_t,
                                         "{}, {}, {}".format('break', 'break', configs.parameter))], columns=col_name)
                    df = pd.concat([df, df2])

        df = df[1:]
        df['delta'] = df['end_t'] - df['start_t']

        fig = px.timeline(df, x_start="start_t", x_end="end_t", y="resource",
                          color="job_id", text="text", opacity=0.6,
                          color_continuous_scale='rainbow')  # https://plotly.com/python/colorscales/

        fig.update_yaxes(autorange="reversed")
        fig.layout.xaxis.type = 'linear'
        fig.data[0].x = df.delta.tolist()

        fig.update_layout(plot_bgcolor="white",
                          title={
                              'text': title,
                              'y': 1,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'}
                          )
        fig.update_xaxes(linecolor='black', gridcolor='gray', mirror=True)
        fig.update_yaxes(linecolor='black', mirror=True)

        if title:
            folder_path = './../result/gantt_chart/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            save_path = f'{folder_path}/{title}.png'
            fig.write_image(save_path)
        else:
            fig.show()


if __name__ == "__main__":
    from utils import all_rules, all_benchmarks, action_types, REAL_D, FLOW
    import csv, time
    import cProfile

    # configs.agent_type = 'rule'
    # # configs.action_type = 'conflict'
    # configs.action_type = 'buffer'
    #
    # ##############################################
    # # rules = ['LTT', 'MOR', 'LRPT', 'FDD/MWKR', 'SPT']
    # # # rules = ['LTT']
    # # env = JobShopEnv([('TA', 20, 15, 0)], pomo_n=len(rules))
    # # obs, reward, done = env.reset()
    # #
    # # while not done:
    # #     a = env.get_action_rule(obs, rules)
    # #     obs, reward, done = env.step(a)
    # # # env.show_gantt_plotly(0, 0)
    # # print(reward)
    # # ##############################################
    #
    # # rules = ['LTT', 'MOR', 'FDD/MWKR', 'LRPT', 'SPT']
    # rules = ['LTT']
    #
    # # HUN 4x3_0: 84, 87, 86, 87, 99         / 84, 84, 84, 84, 84             - 84
    # # HUN 6x6_0: 195, 196, 195, 195, 219    / 168, 171, 173, 185, 168        - 168
    # # HUN 6x6_1: 196, 194, 206, 196, 207    / 195, 184, 195, 190, 167        - 167
    # # HUN 8x6_0: 193, 181, 192, 187, 197    / 164, 179, 177, 177, 164        - 164
    # # TA 15x15_0: 1484, 1438, 1491, 1462    / 1498, 1434, 1397, 1385, 1386   - 1386
    #
    # def main():
    #     # env = JobShopEnv([('TA', 15, 15, 0)], pomo_n=len(rules))
    #     # env = JobShopEnv([('HUN', 4, 3, 0)], pomo_n=len(rules))
    #     env = JobShopEnv([('TEST', 4, 4, 0)], pomo_n=len(rules))  # paper
    #
    #     obs, reward, done = env.reset()
    #     while not done:
    #         a = env.get_action_rule(obs, rules)
    #         obs, reward, done = env.step(a)
    #
    #     print(reward)
    #     # print(env.decision_n)
    #     # env.show_gantt_plotly(0, 0)
    #
    # # cProfile.run('main()')
    # main()

    # for configs.action_type in ['single_mc_buffer']:
    #     save_path = './../result/bench_rule.csv'
    #
    #     for (benchmark, job_n, mc_n, instances) in all_benchmarks:
    #         print(benchmark, job_n, mc_n, instances)
    #         for i in instances:
    #             for rule in all_rules:
    #                 envs_info = [(benchmark, job_n, mc_n, i)]
    #                 env = JobShopEnv(envs_info, pomo_n=1)
    #
    #                 ##############################################
    #                 s_t = time.time()
    #                 obs, reward, done = env.reset()
    #
    #                 while not done:
    #                     a = env.get_action_rule(obs, [rule])
    #                     obs, reward, done = env.step(a)
    #                 # env.show_gantt_plotly(0, 0)
    #                 run_t = round(time.time() - s_t, 4)
    #                 print(run_t)
                    #############################################
    #                 print(i, rule, run_t)
    #                 with open(save_path, 'a', newline='') as f:
    #                     wr = csv.writer(f)
    #                     wr.writerow([benchmark, job_n, mc_n, i,
    #                                  configs.agent_type, configs.action_type, configs.state_type, rule,
    #                                  -reward[0, 0].item(), run_t])

    # TA 15x15 rule: 0.152s
    # TA 100x20 rule: 17.6s

    # TA 15x15 rule: 0.075s
    # TA 100x20 rule: 1.13s


    ###########################################################################################################
    import os
    from tqdm import tqdm

    configs.action_type = 'buffer'
    configs.agent_type = 'rule'
    rules = all_rules
    # rules = ['LTT']

    save_folder = f'./../result/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + f'result_rule_flow.csv'

    for (benchmark, job_n, mc_n, instances) in all_benchmarks:
        print(benchmark, job_n, mc_n, instances)
        for i in tqdm(instances):
            envs_info = [(benchmark, job_n, mc_n, i)]
            env = JobShopEnv(envs_info, pomo_n=len(rules))

            ##############################################
            s_t = time.time()
            obs, reward, done = env.reset()

            while not done:
                a = env.get_action_rule(obs, rules)
                obs, reward, done = env.step(a)
            # env.show_gantt_plotly(0, 0)
            run_t = round(time.time() - s_t, 4)
            # print(i, run_t)

            ############################################
            with open(save_path, 'a', newline='') as f:
                wr = csv.writer(f)
                for j in range(len(rules)):
                    wr.writerow([benchmark, job_n, mc_n, i,
                                 configs.agent_type, configs.action_type, rules[j],
                                 -reward[0, j].item(), run_t])