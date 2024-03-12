import copy, torch

from utils import load_data
from collections import defaultdict
from params import configs
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader  # https://github.com/pyg-team/pytorch_geometric/issues/2961 ######
from environment.env import JobShopEnv


class JobShopDynEnv(JobShopEnv):
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
                        if 'job_arrival' in configs.dyn_type:
                            arrival_t = configs.parameter * max(0, j // mc_n - configs.init_batch + 1)  # init: 2M jobs -> batch size M (period configs.parameter)
                            if arrival_t > 0:
                                continue

                        for op2 in self.init_mc_ops[i][m]:
                            self.init_e_disj[i].append((op, op2))
                            self.init_e_disj[i].append((op2, op))
                        self.init_mc_ops[i][m].append(op)

        # dyn #########################################
        self.init_events = defaultdict(list)
        if 'job_arrival' in configs.dyn_type:
            for i, (_, job_n, mc_n, _) in enumerate(problems):
                for j in range(job_n):
                    arrival_t = configs.parameter * max(0, j // mc_n - configs.init_batch + 1)  # init: 2M jobs -> batch size M (period configs.parameter)
                    self.init_job_arrival[i, j] = arrival_t
                    self.init_job_ready[i, j] += arrival_t
                    self.init_job_done[i, j] += arrival_t
                    if arrival_t > 0:
                        self.init_events[i].append((arrival_t, 'job_arrival', j, None))

        if 'mc_breakdown' in configs.dyn_type:
            for i, (_, _, mc_n, _) in enumerate(problems):
                for k in range(mc_n):
                    self.init_events[i].append((configs.parameter * k, 'mc_breakdown', k, configs.parameter * (k + 1)))

            if 'known' in configs.dyn_type:
                self.init_mc_break = torch.zeros(self.env_n, self.max_mc_n, dtype=torch.long)
                self.init_mc_repair = torch.zeros(self.env_n, self.max_mc_n, dtype=torch.long)

            else:
                self.init_mc_break = torch.zeros(self.env_n, self.max_mc_n+1, dtype=torch.long)
                self.init_mc_repair = torch.zeros(self.env_n, self.max_mc_n+1, dtype=torch.long)
                # for i, (_, _, mc_n, _) in enumerate(problems):
                #     for k in range(mc_n):
                #         # self.init_mc_break[i, k] = configs.parameter * k
                #         # self.init_mc_repair[i, k] = configs.parameter * (k+1)

        if 'prt_stochastic' in configs.dyn_type:
            self.init_job_real_durations = torch.zeros(self.env_n, self.max_job_n+1, self.max_mc_n+1, dtype=torch.long)
            if 'known' in configs.dyn_type:
                uncertainty = torch.zeros(self.env_n, self.max_job_n + 1, self.max_mc_n + 1, dtype=torch.float16)
                from math import sin, pi

                for i, (_, job_n, mc_n, _) in enumerate(problems):
                    # # LCG: linear congruential generator
                    # m = job_n * mc_n * mc_n
                    # # c = job_n * mc_n - 1
                    # c = 1
                    # a = 2 * job_n * mc_n + 1
                    # x = instance_i
                    # for job_i in range(job_n):
                    #     for mc_i in range(mc_n):
                    #         x = (x * a + c) % m
                    #         uncertainty[i, job_i, mc_i] = x
                    # uncertainty[i] = (2 * uncertainty[i] / m - 1)

                    # sign function
                    for mc_i in range(mc_n):
                        for job_i in range(job_n):
                            x = sin(2 * pi * (mc_i * job_n + job_i) / (job_n * mc_n))
                            # simple .long() -> floor, not round
                            self.init_job_real_durations[i, job_i, mc_i] = round(
                                (1 + configs.parameter * x) * self.init_job_durations[i, job_i, mc_i].item())
            else:
                if 'related' in configs.dyn_type:
                    for i, (benchmark, job_n, mc_n, instance_i) in enumerate(problems):
                        job_mcs, job_prts = load_data(benchmark, job_n, mc_n, instance_i)

                        # idx = (2*i - job_n - 1) / (job_n - 1)
                        # ratio = 1 + configs.parameter * idx
                        # for j, prts in enumerate(job_prts):
                        #     self.init_job_real_durations[i, j, :len(prts)] = torch.tensor(ratio * prts, dtype=torch.long)

    def reset(self):
        self.reset_idxs()

        if 'rule' not in configs.agent_type:
            self.init_disj_info()

        if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
            self.reserved = defaultdict(list)

        ##################################################
        self.events = defaultdict()
        for i in range(self.env_n):
            for j in range(self.pomo_n):
                self.events[i, j] = copy.deepcopy(self.init_events[i])

        if 'mc_breakdown' in configs.dyn_type and 'known' in configs.dyn_type:
            self.mc_break = self.init_mc_break.unsqueeze(dim=1).expand(-1, self.pomo_n, -1)
            self.mc_repair = self.init_mc_repair.unsqueeze(dim=1).expand(-1, self.pomo_n, -1)

        if 'prt_stochastic' in configs.dyn_type:
            self.job_real_durations = self.init_job_real_durations.unsqueeze(dim=1).expand(-1, self.pomo_n, -1, -1)

        ##################################################
        self.next_state()

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

        # reservation update  ##################################################################
        if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
            reserved_idxs = list()
            for i in range(self.env_n):
                for j in range(self.pomo_n):
                    target_mc = self.target_mc[i, j].argmax().item()
                    job = mc_job[i, j, target_mc]
                    mc_start_t = self.job_ready_t[i, j, job, self.job_last_step[i, j, job]]
                    if (self.curr_t[i, j] < mc_start_t):
                        reserved_idxs.append((i, j))
                        # start_t, (assign_job, job_step, mc), (mc_last_job, mc_last_step), ready_t_pred, (e_info)
                        self.reserved[i, j].append([
                            mc_start_t, (job, copy.deepcopy(self.job_last_step[i, j, job]), target_mc),
                            (self.mc_last_job[i, j, target_mc], self.mc_last_job_step[i, j, target_mc]),
                            copy.deepcopy(self.job_ready_t_precedence[i, j, job]), None])

        # dynamic stochastic duration ##########################################################
        if 'prt_stochastic' in configs.dyn_type:
            last_step_ = self.job_last_step[self.ENV_IDX_M, self.POMO_IDX_M, mc_job]
            dur_gap = self.job_real_durations[self.ENV_IDX_M, self.POMO_IDX_M, mc_job, last_step_] - \
                      self.job_durations[self.ENV_IDX_M, self.POMO_IDX_M, mc_job, last_step_]
            self.job_durations[self.ENV_IDX_M, self.POMO_IDX_M, mc_job, last_step_] += dur_gap

            assign_job_dur_gap = torch.zeros(self.env_n, self.pomo_n, self.max_job_n+1, dtype=torch.long)
            assign_job_dur_gap[self.ENV_IDX_M, self.POMO_IDX_M, mc_job] += dur_gap
            assign_job_follow_ = torch.where(last_step < self.JOB_STEP_IDX, 1, 0).mul(assign_job_)
            add_dur_gap = assign_job_follow_.mul(assign_job_dur_gap.unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n+1))
            self.job_ready_t_precedence += add_dur_gap

            # update t_mc_gap #######################
            mc_t = self.job_done_t[self.ENV_IDX_M, self.POMO_IDX_M, self.mc_last_job, self.mc_last_job_step]
            job_mc_t = mc_t[self.ENV_IDX_O_, self.POMO_IDX_O_, self.job_mcs]

            last_step = self.job_last_step.unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n + 1)
            job_remain = torch.where(last_step <= self.JOB_STEP_IDX, 1, 0)
            job_remain[:, :, -1, :] = 0

            gap = (job_mc_t - self.job_ready_t_precedence).mul(job_remain)
            gap = torch.where(gap > 0, gap, 0)
            gap_max = gap.max(dim=3)[0].unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n + 1)
            gap_arg_max = gap.argmax(dim=3).unsqueeze(dim=3).expand(-1, -1, -1, self.max_mc_n + 1)

            tails = torch.where(self.JOB_STEP_IDX > gap_arg_max, 1, 0).mul(job_remain)
            gap_ = torch.where(self.JOB_STEP_IDX <= gap_arg_max, gap, 0)
            self.job_ready_t_mc_gap = gap_ + tails * gap_max

            self.job_ready_t = self.job_ready_t_precedence + self.job_ready_t_mc_gap
            self.job_done_t = self.job_ready_t + self.job_durations

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

                    # save for reservation cancel #######################################
                    if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
                        if (i, j) in reserved_idxs:
                            self.reserved[i, j][-1][4] = (prev_op, del_list)

        # mc_last_job, job_last_step update  ##################################################################
        self.mc_last_job = torch.where(mc_job < self.max_job_n, mc_job, self.mc_last_job)
        mc_assigned_job_last_step = self.job_last_step[self.ENV_IDX_M, self.POMO_IDX_M, mc_job]
        self.mc_last_job_step = torch.where(mc_job < self.max_job_n, mc_assigned_job_last_step, self.mc_last_job_step)

        self.job_last_step[self.ENV_IDX_M, self.POMO_IDX_M, mc_job] += 1
        self.job_last_step[:, :, -1] = 0

        # same mc jobs mc_gap update #########################################################################
        # job mc_t ##############################################################################
        mc_t = self.job_done_t[self.ENV_IDX_M, self.POMO_IDX_M, self.mc_last_job, self.mc_last_job_step]
        if 'mc_breakdown' in configs.dyn_type:
            if 'known' in configs.dyn_type:
                mc_t = torch.where(self.mc_repair > mc_t, self.mc_repair, mc_t)
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

            if 'job_arrival' in configs.dyn_type:
                curr_t_ = self.curr_t.unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)
                job_mask = torch.where(curr_t_ >= self.job_arrival_t_, job_mask, self.M)  # reflect non-arrival
                start_t += job_mask

            # update started jobs #################################################################
            if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
                for i in range(self.env_n):
                    for j in range(self.pomo_n):
                        if self.reserved[i, j]:
                            curr_t_ = self.curr_t[i, j]
                            del_list = list()
                            for pos, info in enumerate(self.reserved[i, j]):
                                if curr_t_ > info[0]:
                                    del_list.append(i)

                            for pos in reversed(del_list):
                                del self.reserved[i, j][pos]

            # dynamic event #########################################################################
            event = False
            for i in range(self.env_n):
                for j in range(self.pomo_n):
                    while True:
                        if not self.events[i, j]:
                            break
                        if self.events[i, j][0][0] > self.curr_t[i, j]:
                            break

                        event = True
                        s_t, event_type, target, etc = self.events[i, j].pop(0)  # (t, 'job_arrival', j, None)

                        # dynamic reservation reset ###############################
                        if 'conflict' in configs.action_type and configs.dyn_reserve_reset:
                            # cancel reservations ##########################
                            # mc_update_t = defaultdict(lambda: float('inf'))
                            if self.reserved[i, j]:
                                for _, (job, _, mc), (mc_last_job, mc_last_step), ready_t_pred, e_info \
                                        in reversed(self.reserved[i, j]):
                                    self.job_last_step[i, j, job] -= 1
                                    self.mc_last_job[i, j, mc] = mc_last_job
                                    self.mc_last_job_step[i, j, mc] = mc_last_step

                                    # update precedence_t ###################
                                    self.job_ready_t_precedence[i, j, job] = ready_t_pred

                                    # update t_mc_gap #######################
                                    mc_t = self.job_done_t[i, j, self.mc_last_job[i, j], self.mc_last_job_step[i, j]]
                                    job_mc_t = mc_t[self.job_mcs[i, j]]

                                    last_step = self.job_last_step[i, j].view(self.max_job_n + 1, -1).expand(
                                        -1, self.max_mc_n + 1)
                                    job_remain = torch.where(last_step <= torch.arange(self.max_mc_n + 1), 1, 0)
                                    job_remain[-1, :] = 0

                                    gap = (job_mc_t - self.job_ready_t_precedence[i, j]).mul(job_remain)
                                    gap = torch.where(gap > 0, gap, 0)
                                    gap_max = gap.max(dim=1)[0].view(-1, 1).expand(-1, self.max_mc_n+1)
                                    gap_arg_max = gap.argmax(dim=1).view(-1, 1).expand(-1, self.max_mc_n+1)

                                    tails = torch.where(self.JOB_STEP_IDX[i, j] > gap_arg_max, 1, 0).mul(job_remain)
                                    gap_ = torch.where(self.JOB_STEP_IDX[i, j] <= gap_arg_max, gap, 0)
                                    self.job_ready_t_mc_gap[i, j] = gap_ + tails * gap_max

                                    # cancel e_disj #########################
                                    if 'rule' not in configs.agent_type:
                                        prev_op, e_disj_list = e_info
                                        self.e_disj[i, j] += e_disj_list
                                        self.mc_ops[i, j][mc].append(self.mc_prev_op[i, j][mc])
                                        if prev_op is None:
                                            del self.mc_prev_op[i, j][mc]
                                        else:
                                            self.mc_prev_op[i, j][mc] = prev_op

                                # last update #######################################
                                self.job_ready_t = self.job_ready_t_precedence + self.job_ready_t_mc_gap
                                self.job_done_t = self.job_ready_t + self.job_durations
                                del self.reserved[i, j]

                        # new job arrival -> dynamic add e_disj ###################
                        if 'job_arrival' in event_type:
                            # target: arrival job idx
                            if 'rule' not in configs.agent_type:
                                for k in range(self.job_step_n[i, j, target]):
                                    op = self.op_map[i, target, k].item()
                                    m = self.job_mcs[i, j][target][k].item()

                                    for op2 in self.mc_ops[i, j][m]:
                                        self.e_disj[i, j].append((op, op2))
                                        self.e_disj[i, j].append((op2, op))
                                        if m in self.mc_prev_op[i, j].keys():
                                            self.e_disj[i, j].append((self.mc_prev_op[i, j][m], op))
                                    self.mc_ops[i, j][m].append(op)

                        # machine breakdown #######################################
                        if 'mc_breakdown' in event_type:
                            # target: target mc idx, etc: repairing timing
                            if 'known' in configs.dyn_type:  # add only repair time
                                target_mc = F.one_hot(torch.tensor(target), num_classes=self.max_mc_n)
                                # self.mc_break[i, j, target] = s_t
                                self.mc_repair[i, j, target] = etc

                                # being process job update ##########################
                                mc_last_job = self.mc_last_job[i, j, target]
                                if mc_last_job < self.max_job_n:
                                    mc_last_step = self.mc_last_job_step[i, j, target]
                                    if self.job_done_t[i, j, mc_last_job, mc_last_step] > s_t:
                                        self.job_durations[i, j, mc_last_job, mc_last_step] += (etc - s_t)
                                        self.job_ready_t_precedence[i, j, mc_last_job, mc_last_step+1:] += (etc - s_t)
                                self.job_done_t[i, j] = self.job_ready_t[i, j] + self.job_durations[i, j]

                                # mc t #################################
                                mc_t = self.job_done_t[i, j][self.mc_last_job[i, j], self.mc_last_job_step[i, j]]
                                # if mc_t[target] > s_t:
                                #     mc_t[target] += (etc - s_t)
                                # else:
                                #     mc_t[target] = etc
                                mc_t = torch.where(self.mc_repair[i, j] > mc_t, self.mc_repair[i, j], mc_t)
                                job_mc_t = mc_t[self.job_mcs[i, j]]

                                # remain operations #################################
                                last_step = self.job_last_step[i, j].view(self.max_job_n + 1,
                                                                          -1).expand(-1, self.max_mc_n + 1)
                                job_remain = torch.where(last_step <= self.JOB_STEP_IDX[i, j], 1, 0)
                                job_remain[-1, :] = 0
                                job_remain[:, -1] = 0

                                # same step #########################################
                                same_mc_step = target_mc[self.job_mcs[i, j]].mul(job_remain).mul(self.JOB_OP[i, j])

                                step_pos = same_mc_step.argmax(dim=1).view(self.max_job_n + 1,
                                                                           -1).expand(-1, self.max_mc_n + 1)
                                step_sum = same_mc_step.sum(dim=1)
                                step_sum = torch.where(step_sum > 0, 1, 0).view(self.max_job_n + 1,
                                                                                -1).expand(-1, self.max_mc_n + 1)
                                same_mc_step_follow = torch.where(step_pos <= self.JOB_STEP_IDX[i, j],
                                                                  1, 0).mul(step_sum)

                                mc_update = (job_mc_t - self.job_ready_t_precedence[i, j]).mul(
                                    same_mc_step).sum(dim=1).view(self.max_job_n + 1, -1).expand(
                                    -1, self.max_mc_n + 1).mul(same_mc_step_follow)

                                self.job_ready_t_mc_gap[i, j] = torch.where(self.job_ready_t_mc_gap[i, j] > mc_update,
                                                                            self.job_ready_t_mc_gap[i, j], mc_update)

                                # ready_t update ####################################
                                self.job_ready_t[i, j] = self.job_ready_t_precedence[i, j] + self.job_ready_t_mc_gap[i, j]
                                self.job_done_t[i, j] = self.job_ready_t[i, j] + self.job_durations[i, j]

            if event:
                continue

            # candidate job ########################################################################
            job_mcs = self.job_mcs[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step][:, :, :-1]

            if 'conflict' in configs.action_type:
                done_t = self.job_done_t[self.ENV_IDX_J, self.POMO_IDX_J, self.JOB_IDX, self.job_last_step] + job_mask
                c_min = torch.min(done_t, dim=2)[0].unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)  # (env_n, pomo_n)
                min_jobs = torch.where(done_t == c_min, 1, 0)

                min_mcs = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
                min_mcs.scatter_add_(2, index=job_mcs, src=min_jobs)
                min_mcs = torch.where(min_mcs > 0, 1, 0)
                cand_jobs = torch.where(start_t < c_min, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            elif 'buffer' in configs.action_type:
                curr_t = self.curr_t.unsqueeze(dim=2).expand(-1, -1, self.max_job_n+1)
                cand_jobs = torch.where(start_t == curr_t, 1, 0)  # (env_n, pomo_n, max_job_n+1)
            else:
                cand_jobs = 0

            # count mc actions #####################################################################
            mc_count = torch.zeros(self.env_n, self.pomo_n, self.max_mc_n, dtype=torch.long)
            mc_count.scatter_add_(2, index=job_mcs, src=cand_jobs)  # (env_n, pomo_n, max_mc_n)
            if 'conflict' in configs.action_type:
                mc_count = mc_count.mul(min_mcs)

            # automatic assign job ################################################################
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

            # next state ##########################################################################
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


if __name__ == "__main__":
    from utils import all_rules, all_benchmarks, action_types
    import csv, time
    import cProfile

    configs.agent_type = 'rule'
    # configs.action_type = 'conflict'
    configs.action_type = 'buffer'

    ##########################################################
    configs.dyn_reserve_reset = True

    # configs.dyn_type = 'mc_breakdown'
    # parameters = [50, 100, 150, 200]
    # configs.parameter = 0.4

    # configs.dyn_type = 'job_arrival'
    # parameters = [100, 200, 300, 400]
    # configs.parameter = 50  # 50
    # configs.init_batch = 1

    # configs.dyn_type = 'prt_stochastic'
    # parameters = [0.1, 0.2, 0.3, 0.4]
    # configs.parameter = 0.4

    ##############################################
    # rules = ['LTT', 'MOR', 'LRPT', 'FDD/MWKR', 'SPT']
    # # rules = ['LTT']
    # env = JobShopEnv([('TA', 20, 15, 0)], pomo_n=len(rules))
    # obs, reward, done = env.reset()
    #
    # while not done:
    #     a = env.get_action_rule(obs, rules)
    #     obs, reward, done = env.step(a)
    # # env.show_gantt_plotly(0, 0)
    # print(reward)
    # ##############################################

    # rules = ['LTT', 'MOR', 'FDD/MWKR', 'LRPT', 'SPT']
    # rules = ['LTT']  # , 'SPT', 'MOR', 'LPT', 'STT', 'LRPT', 'FDD/MWKR']

    # HUN 4x3_0: 84, 87, 86, 87, 99         / 84, 84, 84, 84, 84             - 84
    # HUN 6x6_0: 195, 196, 195, 195, 219    / 168, 171, 173, 185, 168        - 168
    # HUN 6x6_1: 196, 194, 206, 196, 207    / 195, 184, 195, 190, 167        - 167
    # HUN 8x6_0: 193, 181, 192, 187, 197    / 164, 179, 177, 177, 164        - 164
    # TA 15x15_0: 1484, 1438, 1491, 1462    / 1498, 1434, 1397, 1385, 1386   - 1386

    # def main(i):
    #     # env = JobShopDynEnv([('TA', 15, 15, 0)], pomo_n=len(rules))
    #     # env = JobShopDynEnv([('HUN', 4, 3, 0)], pomo_n=len(rules))
    #     # env = JobShopDynEnv([('TEST', 4, 4, 0)], pomo_n=len(rules))  # paper
    #     # env = JobShopDynEnv([('HUN', 8, 3, 0)], pomo_n=len(rules))
    #     # env = JobShopDynEnv([('HUN', 5, 2, 0)], pomo_n=len(rules))
    #     # env = JobShopDynEnv([('TEST', 4, 3, 1)], pomo_n=len(rules))
    #     # env = JobShopDynEnv([('TEST', 4, 3, i)], pomo_n=len(rules))  # 3 - 1, 11, 42, 46 / 10 - 0, 14
    #     env = JobShopDynEnv([('HUN', 6, 4, i)], pomo_n=len(rules))  # 50 - 11
    #
    #     obs, reward, done = env.reset()
    #     while not done:
    #         a = env.get_action_rule(obs, rules)
    #         obs, reward, done = env.step(a)

        # print(env.decision_n)
        # for j in range(len(rules)):
        #     env.show_gantt_plotly(0, j)
        # print(reward)

    # cProfile.run('main()')
    # for i in range(11, 50):
    #     print(i)
    #     main(i)

    ###########################################################################################################
    # for configs.action_type in ['single_mc_buffer']:
    #     save_path = './../result/bench_rule.csv'
    #
    #     for (benchmark, job_n, mc_n, instances) in all_benchmarks:
    #         print(benchmark, job_n, mc_n, instances)
    #         for i in instances:
    #             for rule in all_rules:
    #                 envs_info = [(benchmark, job_n, mc_n, i)]
    #                 env = JobShopDynEnv(envs_info, pomo_n=1)
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
    #                 ############################################
    #                 print(i, rule, run_t)
    #                 with open(save_path, 'a', newline='') as f:
    #                     wr = csv.writer(f)
    #                     wr.writerow([benchmark, job_n, mc_n, i,
    #                                  configs.agent_type, configs.action_type, configs.state_type, rule,
    #                                  -reward[0, 0].item(), run_t])

    # TA 15x15 rule: 0.152s
    # TA 100x20 rule: 17.6s
    #
    # TA 15x15 rule: 0.075s
    # TA 100x20 rule: 1.13s

    ###########################################################################################################
    from utils import all_dyn_benchmarks, all_benchmarks
    import os

    # configs.dyn_type = 'job_arrival'
    # parameters = [100, 200, 300, 400]
    # configs.init_batch = 2

    # configs.dyn_type = 'mc_breakdown_known'
    # parameters = [100, 200, 300, 400]

    configs.dyn_type = 'prt_stochastic_known'
    parameters = [0.1, 0.2, 0.3, 0.4]

    #########################################################
    configs.action_type = 'buffer'
    # configs.action_type = 'conflict'
    configs.agent_type = 'rule'
    rules = all_rules
    # rules = ['LTT']
    # rules = ['LTT', 'SPT']

    save_folder = f'./../result/{configs.dyn_type}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = save_folder + f'result_rule.csv'

    for configs.dyn_reserve_reset in [True]:
        for configs.parameter in parameters:
            for (benchmark, job_n, mc_n, instances) in all_benchmarks:  # [['HUN', 6, 4, [0]]]  all_benchmarks
                print(benchmark, job_n, mc_n, instances)
                for i in instances:
                    envs_info = [(benchmark, job_n, mc_n, i)]
                    env = JobShopDynEnv(envs_info, pomo_n=len(rules))

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
                                         -reward[0, j].item(), run_t,
                                         configs.dyn_type, configs.parameter, configs.dyn_reserve_reset])