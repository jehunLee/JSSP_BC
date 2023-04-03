# reference: https://github.com/Kruszylo/shifting-bottleneck
from environment.cpm_graph import CPM
from params import configs
from collections import defaultdict
import torch


class JobShopGraph(CPM):
    def __init__(self):  # , jobs):
        super().__init__()
        self.mc_nodes = defaultdict(list)

        self.add_node("U", prt=0)
        self.add_node("V", prt=0)

        self.op_idx = 0
        self.op_node_map = dict()  # key: op_idx
        self.node_op_map = dict()  # key: (mc, job, i_th)
        self.succ_node = dict()  # key: (mc, job, i_th)
        self.prec_node = dict()  # key: (mc, job, i_th)

        self.e_prec = dict()  # key: op_idx
        self.e_succ = dict()  # key: op_idx
        self.e_disj = defaultdict(list)  # key: op_idx
        self.e_all_prec = defaultdict(list)  # key: op_idx
        self.e_all_succ = defaultdict(list)  # key: op_idx
        self.e_op_to_rsc = dict()  # key: op_idx

        self.mc_seq = defaultdict(list)  # key: mc_idx

        self.candidate_nodes = list()

    ######################################################################################
    def add_job(self, job) -> None:
        """
        add a job
        """
        self.handle_job_prts(job)
        self.handle_job_routings(job)
        self.gen_disj_arcs(job)  # create disj arcs with assigned mc_seq

    def handle_job_prts(self, job) -> None:
        """
        add nodes for a job
        update self.op_node_map, self.node_op_map
        """
        tail_prt = sum(job.prts)
        tail_n = len(job.prts)
        for i, (m, p) in enumerate(zip(job.mcs, job.prts)):
            tail_prt -= p
            tail_n -= 1
            self.add_node((m, job.id, i), prt=p, ith=i, tail=tail_prt, tail_n=tail_n, assign=0)
            self.op_node_map[self.op_idx] = (m, job.id, i)
            self.node_op_map[(m, job.id, i)] = self.op_idx
            self.op_idx += 1

    def handle_job_routings(self, job) -> None:
        """
        add edges for a job
        update self.e_prec, self.e_succ, self.e_all_prec, self.e_all_succ
        update self.succ_node, self.mc_nodes
        """
        # edge for the first node
        self.add_edge("U", (job.mcs[0], job.id, 0))
        self.candidate_nodes.append((job.mcs[0], job.id, 0))  # first node: possible

        for i, (m, m2) in enumerate(zip(job.mcs[:-1], job.mcs[1:])):
            # pred edge
            self.add_edge((m, job.id, i), (m2, job.id, i+1))  # key: (machine, job)
            op = self.node_op_map[(m, job.id, i)]
            op2 = self.node_op_map[(m2, job.id, i+1)]
            self.e_prec[op] = op2
            self.e_succ[op2] = op

            # all pred edge
            if 'all_pred' in configs.GNN_type:
                if i < len(job.mcs) - 1:
                    for j, m3 in enumerate(job.mcs[(i+1):]):
                        op3 = self.node_op_map[(m3, job.id, i+1+j)]
                        self.e_all_prec[op].append(op3)
                        self.e_all_succ[op3].append(op)

            # info
            self.succ_node[(m, job.id, i)] = (m2, job.id, i+1)
            self.prec_node[(m2, job.id, i+1)] = (m, job.id, i)
            self.mc_nodes[m].append((m, job.id, i))

        # edge for the last node
        self.add_edge((job.mcs[-1], job.id, len(job.mcs)-1), "V")
        # info
        m = job.mcs[-1]
        self.mc_nodes[m].append((m, job.id, len(job.mcs)-1))

    def gen_disj_arcs(self, job) -> None:
        """
        add disjunctive edges for a job
        update self.e_op_to_rsc, self.e_disj by using self.mc_nodes
        """
        # op_to_mc
        for i, m in enumerate(job.mcs):
            op_idx = self.node_op_map[(m, job.id, i)]
            self.e_op_to_rsc[op_idx] = m

        # disj arcs
        for i, m in enumerate(job.mcs):
            op_idx = self.node_op_map[(m, job.id, i)]
            self.e_op_to_rsc[op_idx] = m

            for node in self.mc_nodes[m]:
                if node == (m, job.id, i):  # same job -> no disj edge
                    continue

                op_idx2 = self.node_op_map[node]
                self.e_disj[op_idx2].append(op_idx)
                self.e_disj[op_idx].append(op_idx2)

    ######################################################################################
    def assign_node(self, node) -> None:
        """
        step 5: update candidate nodes
        update cpm_graph, self.e_op_to_rsc, self.e_disj, self.candidate_nodes
        """
        mc = node[0]
        self.mc_seq[mc].append(node)  # (mc, job, i_th)
        self.nodes[node]['assign'] = 1

        # update cpm_graph #############################################
        # add arcs: selected node -> unassigned nodes
        new_disjs = [(node, mc_node) for mc_node in self.mc_nodes[mc] if mc_node not in self.mc_seq[mc]]
        self.add_edges_from(new_disjs)

        # remove arcs: prev node -> unselected nodes
        if len(self.mc_seq[mc]) > 1:
            from_node = self.mc_seq[mc][-2]
            del_disjs = [(from_node, mc_node) for mc_node in self.mc_nodes[mc] if mc_node not in self.mc_seq[mc]]
            self.remove_edges_from(del_disjs)

        # update self.e_disj #############################################
        # remove disj edges: unassigned nodes -> selected node
        op_idx = self.node_op_map[node]

        remove_op_idxs = list()
        for from_op_idx, to_op_idxes in self.e_disj.items():
            for to_op_idx in reversed(to_op_idxes):
                if to_op_idx == op_idx:
                    from_node = self.op_node_map[from_op_idx]
                    if from_node not in self.mc_seq[mc]:
                        to_op_idxes.remove(to_op_idx)
                        if not to_op_idxes:
                            remove_op_idxs.append(from_op_idx)
                    break
        for op_idx2 in remove_op_idxs:  # remove empty
            del self.e_disj[op_idx2]

        # remove disj edges: prev node -> unassigned nodes
        remove_op_idxs = list()
        if len(self.mc_seq[mc]) > 1:
            pre_node = self.mc_seq[mc][-2]
            pre_op_idx = self.node_op_map[pre_node]
            if pre_node in self.e_disj.keys():
                for to_op_idx in reversed(self.e_disj[pre_op_idx]):
                    if to_op_idx == op_idx:  # only remain this arc
                        continue
                    self.e_disj[pre_op_idx].remove(to_op_idx)
                    if not self.e_disj[pre_op_idx]:
                        remove_op_idxs.append(pre_op_idx)
        for op_idx2 in remove_op_idxs:  # remove empty
            del self.e_disj[op_idx2]

        # (step 5) update candidate_nodes #############################
        if node not in self.candidate_nodes:
            print()
        self.candidate_nodes.remove(node)
        if node in self.succ_node.keys():  # none: last op
            self.candidate_nodes.append(self.succ_node[node])

    # edge #################################################################################################
    def get_prec_succ_edge(self) -> torch.tensor:
        """
        get precedent edges, succedent edges
        """
        # prec edges
        e_prec = [(from_op, to_op) for from_op, to_op in self.e_prec.items()]

        # tensor
        e_prec = torch.tensor(e_prec, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_prec = e_prec.t().detach()

        # succ edges
        e_succ = torch.zeros_like(e_prec)
        e_succ[0, :] = e_prec[1, :]
        e_succ[1, :] = e_prec[0, :]

        return e_prec, e_succ

    def get_disj_edge(self) -> torch.tensor:
        """
        get disjunctive edges
        """
        e_disj = [(from_op, to_op) for from_op, to_op_idxs in self.e_disj.items() for to_op in to_op_idxs]

        # tensor
        e_disj = torch.tensor(e_disj, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)

        return e_disj.t().contiguous()

    # for meta-path ############
    def get_disj_prec_succ_edge(self) -> torch.tensor:
        """
        get disjunctive -> precedent edges, succedent edges
        """
        e_disj_prec = list()
        e_disj_succ = list()
        for from_op, to_op_idxs in self.e_disj.items():
            for to_op in to_op_idxs:
                # prec edges
                if to_op in self.e_prec.keys():
                    meta_to_op = self.e_prec[to_op]
                    e_disj_prec.append((from_op, meta_to_op))

                # succ edges
                if to_op in self.e_succ.keys():
                    meta_to_op = self.e_succ[to_op]
                    e_disj_succ.append((from_op, meta_to_op))

        # tensor
        e_disj_prec = torch.tensor(e_disj_prec, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_disj_succ = torch.tensor(e_disj_succ, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)

        return e_disj_prec.t().contiguous(), e_disj_succ.t().contiguous()

    def get_prec_succ_disj_edge(self) -> torch.tensor:
        """
        get precedent, succedent -> disjunctive edges
        """
        # prec edges
        e_prec_disj = list()
        for from_op, to_op in self.e_prec.items():
            if to_op in self.e_disj.keys():
                meta_to_op_idxs = self.e_disj[to_op]
                for meta_to_op in meta_to_op_idxs:
                    e_prec_disj.append((from_op, meta_to_op))

        # succ edges
        e_succ_disj = list()
        for from_op, to_op in self.e_succ.items():
            if to_op in self.e_disj.keys():
                meta_to_op_idxs = self.e_disj[to_op]
                for meta_to_op in meta_to_op_idxs:
                    e_succ_disj.append((from_op, meta_to_op))

        # tensor
        e_prec_disj = torch.tensor(e_prec_disj, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_succ_disj = torch.tensor(e_succ_disj, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)

        return e_prec_disj.t().contiguous(), e_succ_disj.t().contiguous()

    # for length-agnostic meta-path ############
    def get_all_prec_succ_edge(self) -> torch.tensor:
        """
        get all precedent edges, all succedent edges
        """
        # all prec edges
        e_all_prec = [(from_op, to_op) for from_op, to_op_idxs in self.e_all_prec.items() for to_op in to_op_idxs]

        # tensor
        e_all_prec = torch.tensor(e_all_prec, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_all_prec = e_all_prec.t()

        # all succ edges
        e_all_succ = torch.zeros_like(e_all_prec)
        e_all_succ[0, :] = e_all_prec[1, :]
        e_all_succ[1, :] = e_all_prec[0, :]

        return e_all_prec.contiguous(), e_all_succ.contiguous()

    def get_disj_all_prec_succ_edge(self) -> torch.tensor:
        """
        get disjunctive -> all precedent edges, all succedent edges
        """
        e_disj_all_prec = list()
        e_disj_all_succ = list()
        for from_op, to_op_idxs in self.e_disj.items():
            for to_op in to_op_idxs:
                # all prec edges
                if to_op in self.e_all_prec.keys():
                    meta_to_op_idxs = self.e_all_prec[to_op]
                    for meta_to_op in meta_to_op_idxs:
                        e_disj_all_prec.append((from_op, meta_to_op))

                # all succ edges
                if to_op in self.e_all_succ.keys():
                    meta_to_op_idxs = self.e_all_succ[to_op]
                    for meta_to_op in meta_to_op_idxs:
                        e_disj_all_succ.append((from_op, meta_to_op))

       # tensor
        e_disj_all_prec = torch.tensor(e_disj_all_prec, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_disj_all_succ = torch.tensor(e_disj_all_succ, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)

        return e_disj_all_prec.t().contiguous(), e_disj_all_succ.t().contiguous()

    def get_all_prec_disj_edge(self) -> torch.tensor:
        """
        get all precedent, all succedent -> disjunctive edges
        """
        # all prec
        e_all_prec_disj = list()
        for from_op, to_op_idxs in self.e_all_prec.items():
            for to_op in to_op_idxs:
                if to_op in self.e_disj.keys():
                    meta_to_op_idxs = self.e_disj[to_op]
                    for meta_to_op in meta_to_op_idxs:
                        e_all_prec_disj.append((from_op, meta_to_op))

        # all succ
        e_all_succ_disj = list()
        for from_op, to_op_idxs in self.e_all_succ.items():
            for to_op in to_op_idxs:
                if to_op in self.e_disj.keys():
                    meta_to_op_idxs = self.e_disj[to_op]
                    for meta_to_op in meta_to_op_idxs:
                        e_all_succ_disj.append((from_op, meta_to_op))

        # tensor
        e_all_prec_disj = torch.tensor(e_all_prec_disj, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_all_succ_disj = torch.tensor(e_all_succ_disj, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)

        return e_all_prec_disj.t().contiguous(), e_all_succ_disj.t().contiguous()

    # mc node type
    def get_op_to_rsc_edge(self) -> torch.tensor:
        e_op_to_rsc = []
        for op_idx, mc in self.e_op_to_rsc.items():
            e_op_to_rsc.append((op_idx, mc))

        e_op_to_rsc = torch.tensor(e_op_to_rsc, dtype=torch.long).view(-1, 2).to(torch.long).to(configs.device)
        e_op_to_rsc = e_op_to_rsc.t()

        e_rsc_to_op = torch.zeros_like(e_op_to_rsc)
        e_rsc_to_op[0, :] = e_op_to_rsc[1, :]
        e_rsc_to_op[1, :] = e_op_to_rsc[0, :]

        return e_op_to_rsc.contiguous(), e_rsc_to_op.contiguous()

    # for obs ################################################################################
    def get_op_mc_type(self) -> list:
        return [node[1] for _, node in self.op_node_map.items()]

    def get_op_node(self) -> list:
        return [node for _, node in self.op_node_map.items()]

    ######################################################################################
    def mc_his_dict(self) -> dict:
        mc_his_dict = dict()
        mc_n = len(self.mc_seq.keys())

        for m in range(mc_n):
            for (m, id, i) in self.mc_seq[m]:
                node_info = self.nodes[(m, id, i)]
                mc_his_dict[m].append((id, i, node_info['S'], node_info['prt'], node_info['C']))
        return mc_his_dict