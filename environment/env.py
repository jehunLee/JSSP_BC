from environment.simulator import JobShopSim
from environment.job_shop_graph import JobShopGraph
import copy, torch
from params import configs
import networkx as nx
import numpy as np
from torch_geometric.data import HeteroData
import torch.nn.functional as F


class JobShopEnv(JobShopSim):
    def __init__(self, benchmark: str, job_n: int, mc_n: int, instance_i: int):
        super().__init__(benchmark, job_n, mc_n, instance_i)

        self.move_next_state()
        self.target_mc_i = -1

    def load(self, env) -> None:
        """
        synchronize with another environment
        """
        self.js = copy.deepcopy(env.js)
        self.mc_conflict_ops = copy.deepcopy(env.mc_conflict_ops)

        self.job_arrival = copy.deepcopy(env.init_job_arrival)
        self.arrive_t = env.arrive_t

        self.move_next_state()

    def reset(self):
        """
        reset environment
        """
        self.js = JobShopGraph()
        self.mc_conflict_ops.clear()  # for Giffler and Thompson (1960), ops candidates

        self.job_arrival = copy.deepcopy(self.init_job_arrival)
        self.arrive_t = 0
        self.arrive_jobs()

        self.move_next_state()

        return self.get_obs(), self.get_reward(), self.is_done()

    ##########################################################################################################
    def step(self, op_idx):
        """
        transit to the next decision point(next state) by an action
        """
        node = self.js.op_node_map[op_idx]
        self.js.assign_node(node)  # step 5

        target_mc = node[0]
        del self.mc_conflict_ops[target_mc]

        if not self.mc_conflict_ops:  # no action
            self.move_next_state()
        else:
            if 'being' in configs.action_type:  # independent conflict
                self.action_add_to_mc_conflict_ops(node)
            if 'single_mc' in configs.action_type:  # move to next target machine
                self.target_mc_i = min(self.mc_conflict_ops.keys())
            if 'dyn' in configs.env_type:
                self.remove_edges()  # curr_t can be changed

        return self.get_obs(), self.get_reward(), self.is_done()

    def move_next_state(self) -> None:
        """
        transit to the next decision point(next state)
        """
        while True:
            self.update_mc_conflict_ops()

            if self.is_done():
                break
            self.assign_single_candidate()  # automatic assign

            if self.mc_conflict_ops:  # need action selection
                if 'single' in configs.action_type:
                    self.target_mc_i = min(self.mc_conflict_ops.keys())
                break

        if 'dyn' in configs.env_type:
            self.remove_edges()

    def action_add_to_mc_conflict_ops(self, node) -> None:
        """
        for 'being' action
        assigned node add to current action set to reserve
        """
        mc = node[0]
        if mc in self.mc_conflict_ops.keys():
            prec_node = self.js.prec_node[node]
            if self.js.nodes[prec_node]['S'] <= self.get_current_t():  # being
                min_end_t = min([self.js.nodes[node]['C'] for node in self.mc_conflict_ops[mc]])
                if self.js.nodes[node]['S'] < min_end_t:
                    self.mc_conflict_ops[mc].append(node)

    ##########################################################################################################
    def is_done(self) -> bool:
        """
        investigate end of environment
        """
        if not self.js.candidate_nodes:
            return True
        return False

    def get_reward(self) -> float:
        """
        get reward
        r = - transition_t (cumulative of self.sim_t -> makespan)
        r = - makespan (only end of environment: episodic reward)
        """
        if self.is_done():
            return -self.js.makespan()
        return 0

    def get_obs(self):
        """
        get observation(=state features)
        """
        if self.is_done():
            return 0
        return self.get_torch_geom()

    # obs #####################################################################################################
    def get_torch_geom(self) -> HeteroData:
        """
        get observation
        form: torch_geometric
        """
        # torch_geom ############################
        data = HeteroData()

        # op #############
        data['op'].mask = self.get_op_mask()
        data['op'].x, data['op'].remain = self.get_op_x()

        if 'mc' in configs.state_type:
            data['op'].mc = self.js.get_op_mc_type()

        # visual pos
        # data['op'].pos = self.js.get_op_node()  # mc_i, job_id, ith

        if 'GNN' in configs.agent_type:
            # prec edges
            e_prec, e_succ = self.js.get_prec_succ_edge()
            data['op', 'prec', 'op'].edge_index = e_prec
            data['op', 'succ', 'op'].edge_index = e_succ

            # disj edges
            e_disj = self.js.get_disj_edge()
            data['op', 'disj', 'op'].edge_index = e_disj

            # all: for simple GNN
            data['op', 'all', 'op'].edge_index = torch.cat([e_prec, e_succ, e_disj], dim=1)

            if 'all_prec' in configs.state_type:
                # meta edges
                e_all_prec, e_all_succ = self.js.get_all_prec_succ_edge()
                data['op', 'all_prec', 'op'].edge_index = e_all_prec
                data['op', 'all_succ', 'op'].edge_index = e_all_succ

        return data

        # e_disj_prec, e_disj_succ = self.get_disj_prec_succ_edge()
        # e_disj_all_prec, e_disj_all_succ = self.get_disj_all_prec_succ_edge()
        # e_prec_disj, e_succ_disj = self.get_prec_succ_disj_edge()
        # e_all_prec_disj, e_all_succ_disj = self.get_all_prec_disj_edge()
        #
        # # rsc #############
        # data['rsc'].x = self.get_rsc_x()
        # data['rsc'].node_type = [-1] * self.mc_n
        #
        # # mc disj edges
        # e_op_to_rsc, e_rsc_to_op = self.get_op_to_rsc_edge()
        # data['op', 'op_to_rsc', 'rsc'].edge_index = e_op_to_rsc
        # data['rsc', 'rsc_to_op', 'op'].edge_index = e_rsc_to_op
        #
        # # visual pos
        # rsc_pos = list()
        # for i in range(self.mc_n):
        #     rsc_pos.append((i, -1))
        # data['rsc'].pos = rsc_pos

        # edge_index_dict = data.edge_index_dict
        # src, dst = edge_index_dict['op', 'all', 'op']
        # x = data['op'].x
        # x_src = x[src]
        #
        # b = torch.min(x_src)
        # if b.item() < 0:
        #     print(b)

    def get_op_mask(self) -> torch.IntTensor:
        """
        get mask
        1: selectable nodes
        0: otherwise
        """
        op_mask = np.zeros(len(self.js.op_node_map))

        if 'single' in configs.action_type:
            nodes = self.mc_conflict_ops[self.target_mc_i]
            for node in nodes:
                op_idx = self.js.node_op_map[node]
                op_mask[op_idx] = 1

        else:
            for nodes in self.mc_conflict_ops.values():
                for node in nodes:
                    op_idx = self.js.node_op_map[node]
                    op_mask[op_idx] = 1

        op_mask = torch.IntTensor(op_mask).to(configs.device)
        return op_mask

    def get_op_x(self):
        """
        get node feature
        remain_idxs: for global_info <- torch.mean(dim=0)
        remove_idxs: for dynamic graph representation

        op_status: [non-available, /reservable, waiting, /reserved, being-processed, /done]
        non-available: 0 / waiting: 1 / being_processed: 2
        reservable: 3 / reserved: 4 / done: (3 or 5)
        """
        curr_t = self.get_current_t()

        # op feature #############################################################################
        op_prt = list(nx.get_node_attributes(self.js, 'prt').values())[2:]
        op_prt = torch.tensor(op_prt).view(-1, 1)

        op_ready = list(nx.get_node_attributes(self.js, 'S').values())[2:]
        op_ready = torch.tensor(op_ready).view(-1, 1)

        op_done = list(nx.get_node_attributes(self.js, 'C').values())[2:]
        op_done = torch.tensor(op_done).view(-1, 1)

        op_tail_prt = list(nx.get_node_attributes(self.js, 'tail').values())
        op_tail_prt = torch.tensor(op_tail_prt).view(-1, 1)

        op_succ_n = list(nx.get_node_attributes(self.js, 'tail_n').values())
        op_succ_n = torch.tensor(op_succ_n).view(-1, 1)

        if 'rule' in configs.agent_type:
            op_x = torch.cat([op_prt, op_tail_prt, op_succ_n], dim=1).to(configs.device)
            return op_x.to(torch.float32), []

        # add op feature #############################################################################
        if 'add' in configs.state_type:
            # next mc ready ###########################################
            mc_ready = dict()
            for node in self.js:
                if node in ['U', 'V']:
                    continue
                (mc, job_i, i) = node
                if mc not in mc_ready.keys():
                    mc_ready[mc] = list()
                if self.js.nodes[node]['assign']:  # 할당된 것만 고려
                    mc_ready[mc].append(self.js.nodes[node]['C'])

            # next mc remaining prts ###########################################
            mc_remains = dict()
            for node in self.js:
                if node in ['U', 'V']:
                    continue
                (mc, job_i, i) = node
                if mc not in mc_remains.keys():
                    mc_remains[mc] = list()
                if self.js.nodes[node]['S'] >= curr_t:  # 현재 시점부터 남은것만 고려
                    mc_remains[mc].append(self.js.nodes[node]['prt'])
                elif self.js.nodes[node]['S'] < curr_t and self.js.nodes[node]['C'] > curr_t:  # 진행중인 것도 고려
                    mc_remains[mc].append(self.js.nodes[node]['C'] - curr_t)

            total_remain_t = 0
            for mc in mc_remains.keys():
                if mc_remains[mc]:
                    mc_remains[mc] = sum(mc_remains[mc])
                    total_remain_t += mc_remains[mc]
                else:
                    mc_remains[mc] = 0

            if total_remain_t > 0:
                for mc in mc_remains.keys():
                    mc_remains[mc] = mc_remains[mc] / total_remain_t

            # save features ###########################################
            next_mc_ready = torch.zeros_like(op_prt)
            next_mc_remains = torch.zeros_like(op_prt)
            node_mc_ready = torch.zeros_like(op_prt)
            node_mc_remains = torch.zeros_like(op_prt)
            for op_i in range(op_prt.shape[0]):
                node = self.js.op_node_map[op_i]

                # node mc ready
                if not mc_ready[node[0]]:
                    mc_ready_t = 0
                else:
                    mc_ready_t = max(mc_ready[node[0]])
                node_mc_ready[op_i] = mc_ready_t

                # node mc remaining prts
                node_mc_remains[op_i] = mc_remains[node[0]]

                if node in self.js.succ_node.keys():
                    next_node = self.js.succ_node[node]

                    # next mc ready
                    if not mc_ready[next_node[0]]:
                        mc_ready_t = 0
                    else:
                        mc_ready_t = max(mc_ready[next_node[0]])
                    next_mc_ready[op_i] = mc_ready_t

                    # next mc remaining prts
                    next_mc_remains[op_i] = mc_remains[next_node[0]]

        # op status #############################################################################
        op_s_assigned = list(nx.get_node_attributes(self.js, 'assign').values())
        op_s_assigned = torch.tensor(op_s_assigned).view(-1, 1)
        op_status = torch.zeros_like(op_s_assigned)  # non=available: 0
        op_status = torch.where(op_ready == curr_t, 1, op_status)  # waiting: 1
        op_status = torch.where(((op_ready == curr_t) & (op_s_assigned == 1)) |
                                (op_ready < curr_t), 2, op_status)  # being-processed: 2

        if 'buffer' in configs.action_type and 'being' not in configs.action_type:
            # op_status: [non-available, /waiting, /being-processed, /done]
            if 'dyn' in configs.env_type:  # dyn state 이면 done node 제거해버림
                op_status = F.one_hot(op_status, 3).squeeze(dim=1)
            else:
                op_status = torch.where((op_ready < curr_t) & (op_done >= curr_t), 3, op_status)  # done: 3
                op_status = F.one_hot(op_status, 4).squeeze(dim=1)
        else:
            status_v = 3

            if 'simple' not in configs.state_type:
                op_status = torch.where((op_ready > curr_t) | (op_s_assigned == 1), status_v, op_status)  # reserved
                status_v += 1

                op_candidate = torch.zeros_like(op_status).view(-1, 1)
                for node in self.js.candidate_nodes:
                    op_idx = self.js.node_op_map[node]
                    op_candidate[op_idx] = 1
                op_status = torch.where((op_ready > curr_t) | (op_candidate == 1), status_v, op_status)  # reservable
                status_v += 1

            if 'dyn' in configs.env_type:  # dyn state 이면 done node 제거해버림
                op_status = F.one_hot(op_status, status_v + 1).squeeze(dim=1)
            else:
                op_status = torch.where((op_ready < curr_t) & (op_done >= curr_t), status_v, op_status)  # done: 5
                op_status = F.one_hot(op_status, status_v + 1).squeeze(dim=1)

        # data update #############################################################################
        if 'dyn' in configs.env_type:
            op_ready -= curr_t
            op_remain_t = op_done - curr_t
            op_prt = torch.where(op_ready > 0, op_prt, op_remain_t)  # 이미 시작한 것은 remain_t로 대체
            op_ready = torch.where(op_ready < 0, 0, op_ready)
            remain_idxs = torch.where(op_remain_t > 0)[0]

            if 'add' in configs.state_type:  # add
                op_done -= curr_t
                op_done = torch.where(op_done < 0, 0, op_done)
                next_mc_ready -= curr_t
                next_mc_ready = torch.where(next_mc_ready < 0, 0, next_mc_ready)
        else:
            remain_idxs = torch.where(op_ready >= 0)[0]

        op_relative_prt = op_prt / (op_prt + op_tail_prt)  # ratio by total remain prt, 분모 0이면 inf로 자동 처리

        # remove info for dynamic env ############################################################
        if 'dyn' in configs.env_type:
            remove_idxs = torch.where(op_prt.squeeze() <= 0)[0]
            # op_prt[remove_idxs] = 0
            # op_tail_prt[remove_idxs] = 0
            # op_ready[remove_idxs] = 0
            # op_succ_n[remove_idxs] = 0
            op_relative_prt[remove_idxs] = 0

        # normalize by max prt #############################################################################
        if 'norm' in configs.state_type:
            max_prt = copy.deepcopy(max(op_prt))
            op_prt = op_prt / max_prt
            op_tail_prt = op_tail_prt / max_prt
            op_ready = op_ready / max_prt

            if 'add' in configs.state_type:
                op_done = op_done / max_prt
                node_mc_ready = node_mc_ready / max_prt
                next_mc_ready = next_mc_ready / max_prt

        # node feature x #############################################################################
        if configs.state_type == "simple":
            op_x = torch.cat([op_prt, op_tail_prt, op_succ_n, op_status], dim=1).to(configs.device)

        elif configs.state_type == "add1":  # add
            op_x = torch.cat([op_prt, op_tail_prt, op_succ_n, op_status,
                              op_relative_prt, op_ready,
                              node_mc_ready, node_mc_remains],
                             dim=1).to(configs.device)

        elif configs.state_type == "add2":  # add
            op_x = torch.cat([op_prt, op_tail_prt, op_succ_n, op_status,
                              op_relative_prt, op_ready,
                              op_done, next_mc_ready, next_mc_remains],
                             dim=1).to(configs.device)

        elif configs.state_type == "add3":  # add
            op_x = torch.cat([op_prt, op_tail_prt, op_succ_n, op_status,
                              op_relative_prt, op_ready,
                              node_mc_ready, node_mc_remains,
                              op_done, next_mc_ready, next_mc_remains],
                             dim=1).to(configs.device)

        else:
            op_x = torch.cat([op_prt, op_tail_prt, op_succ_n, op_status,
                              op_relative_prt, op_ready],
                             dim=1).to(configs.device)
        # IJPR paper: task ratio: remain op n / total op n (first: 1)

        return op_x.to(torch.float32), remain_idxs

    # def get_rsc_x(self):
    #     """ node feature 반환 함수
    #     resource(=machine) node
    #     """
    #     # rsc feature ###########################################
    #     rsc_remain_prt = [0] * self.mc_n
    #     rsc_remain_prt = torch.tensor(rsc_remain_prt).view(-1, 1)
    #
    #     # max prt로 normalization ###########################################
    #     # max_prt = copy.deepcopy(max(op_prt))
    #     # rsc_remain_prt = rsc_remain_prt / max_prt
    #
    #     # x ###########################################
    #     rsc_x = torch.cat([rsc_remain_prt], dim=1).to(configs.device)
    #     return rsc_x.to(torch.float32)


if __name__ == "__main__":
    env = JobShopEnv('FT', 6, 6, 0)
    print()