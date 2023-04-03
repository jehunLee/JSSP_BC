import copy

from utils import load_data
from collections import defaultdict
from environment.job_shop_graph import JobShopGraph
from params import configs


class Job:
    def __init__(self, id: int, mcs: list, prts: list, arrival_t: int=0, due: int=0):
        self.id = id
        self.mcs = mcs  # route
        self.prts = prts  # processing times
        self.arrival_t = arrival_t
        self.due = due


def get_job_dict(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> (dict, list):
    job_dict = dict()
    job_mcs, job_prts = load_data(benchmark, job_n, mc_n, instance_i)

    job_arrival = list()
    for i, prts in enumerate(job_prts):
        job_dict[i] = Job(i, job_mcs[i], prts)
        job_arrival.append([i, 0, 0])  # job, arrival_t, due

    return job_dict, job_arrival


class JobShopSim:
    def __init__(self, benchmark: str, job_n: int, mc_n: int, instance_i: int):
        # initial #####################################################################
        self.js = JobShopGraph()
        self.mc_conflict_ops = defaultdict(list)  # for Giffler and Thompson (1960), ops candidates
        self.done_op_idxs = list()

        # load data #####################################################################
        self.job_dict, self.job_arrival = get_job_dict(benchmark, job_n, mc_n, instance_i)

        self.init_job_arrival = copy.deepcopy(self.job_arrival)
        self.arrive_t = 0
        self.arrive_jobs()

    # job arrival ##########################################################################################
    def arrive_jobs(self):
        """
        arrival job until self.arrive_t
        add new jobs
        update next self.arrive_t
        """
        while self.job_arrival:  # list((job, arrival_t, due))
            if self.job_arrival[0][1] > self.arrive_t:
                self.arrive_t = self.job_arrival[0][1]  # move to next arrival time
                break

            job_info = self.job_arrival.pop(0)  # (job, arrival_t, due)
            self.js.add_job(self.job_dict[job_info[0]])

    #######################################################################################################
    def get_current_t(self) -> float:
        """
        get the timing of decision point(=current time of the state)
        by considering all job in buffer
        """
        curr_t = min([float('inf')] + [self.js.nodes[node]['S'] for node in self.js.candidate_nodes])
        return curr_t

    def update_mc_conflict_ops(self):
        """
        step 2: search min completion time
        step 3: update conflict set for candidate machines
        """
        # arrive_jobs()

        self.mc_conflict_ops.clear()
        _ = self.js.makespan()  # update critical path

        if 'conflict' in configs.action_type:
            # step 2: search min completion time
            min_t = min([float('inf')] + [self.js.nodes[node]['C'] for node in self.js.candidate_nodes])

            # step 3: update conflict set for candidate machines
            for node in self.js.candidate_nodes:
                mc = node[0]
                if self.js.nodes[node]['C'] == min_t:
                    self.mc_conflict_ops[mc].append(node)

            for node in self.js.candidate_nodes:
                mc = node[0]
                if mc in self.mc_conflict_ops.keys():  # for candidate machines
                    if self.js.nodes[node]['S'] < min_t:  # conflict set
                        if node not in self.mc_conflict_ops[mc]:
                            self.mc_conflict_ops[mc].append(node)

        elif 'buffer' in configs.action_type:
            curr_t = self.get_current_t()

            # current buffer
            for node in self.js.candidate_nodes:
                mc = node[0]
                if self.js.nodes[node]['S'] == curr_t:  # buffer
                    self.mc_conflict_ops[mc].append(node)

            if 'being' in configs.action_type:
                mc_min_end_t = dict()
                for mc, nodes in self.mc_conflict_ops.items():
                    mc_min_end_t[mc] = min([self.js.nodes[node]['C'] for node in nodes])

                for node in self.js.candidate_nodes:
                    mc = node[0]
                    if mc in self.mc_conflict_ops.keys():  # all possible node for candidate machines
                        if node not in self.mc_conflict_ops[mc]:
                            prec_node = self.js.prec_node[node]
                            if self.js.nodes[prec_node]['S'] <= curr_t:  # being
                                if self.js.nodes[node]['S'] < mc_min_end_t[mc]:  # conflict with buffer jobs
                                    self.mc_conflict_ops[mc].append(node)

        else:
            raise ValueError("Unknown configs.action_type.")

    def assign_single_candidate(self):
        """
        step 4: assign job to machine
        only one candidate job -> automatic assign
        """
        del_mcs = list()
        for mc, candidate_nodes in self.mc_conflict_ops.items():
            if len(candidate_nodes) == 1:
                self.js.assign_node(candidate_nodes[0])
                del_mcs.append(mc)

        for mc in del_mcs:
            del self.mc_conflict_ops[mc]

    def remove_edges(self):
        """
        remove edges for dynamic env
        """
        curr_t = self.get_current_t()
        done_op_idxs = [self.js.node_op_map[node] for node in self.js
                        if node not in ['U', 'V'] and self.js.nodes[node]['C'] <= curr_t
                        and self.js.node_op_map[node] not in self.done_op_idxs]

        for op_idx in done_op_idxs:
            if op_idx in self.js.e_prec.keys():
                to_op_idx = self.js.e_prec[op_idx]
                del self.js.e_succ[to_op_idx]
                del self.js.e_prec[op_idx]

                to_op_idxs = self.js.e_all_prec[op_idx]
                for to_op_idx in to_op_idxs:
                    self.js.e_all_succ[to_op_idx].remove(op_idx)
                    if not self.js.e_all_succ[to_op_idx]:
                        del self.js.e_all_succ[to_op_idx]
                del self.js.e_all_prec[op_idx]

            if op_idx in self.js.e_disj.keys():  # edges to op_idx are already removed
                del self.js.e_disj[op_idx]

        self.done_op_idxs += done_op_idxs


if __name__ == "__main__":
    env = JobShopSim('FT', 6, 6, 0)
    print()