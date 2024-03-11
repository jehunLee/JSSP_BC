import copy, csv, os
from tqdm import tqdm
from opt.cpm_graph import CPM
from utils import load_opt_sol
from opt.simulator import get_job_dict


class JobShopGraph(CPM):
    def __init__(self):  # , jobs):
        super().__init__()
        self.add_node("U", prt=0)
        self.add_node("V", prt=0)

        self.prev_node = dict()  # key: (mc, job)
        self.succ_node = dict()  # key: (mc, job)

        self.op_idx = 0
        self.op_node_map = dict()  # key: op_idx
        self.node_op_map = dict()  # key: (mc, job)
        self.succ_node = dict()  # key: (mc, job)

    ######################################################################################
    def add_job(self, job):
        """
        add a job
        """
        self.handle_job_prts(job)
        self.handle_job_routings(job)

    def reversed_add_job(self, job):
        """
        add a job
        """
        self.handle_job_prts(job)
        self.handle_job_reversed_routings(job)

    def handle_job_prts(self, job):
        """
        add nodes for a job
        update self.op_node_map, self.node_op_map
        """
        for i, (m, p) in enumerate(zip(job.mcs, job.prts)):
            self.add_node((m, job.id, i), prt=p, ith=i)
            self.op_node_map[self.op_idx] = (m, job.id, i)
            self.node_op_map[(m, job.id, i)] = self.op_idx
            self.op_idx += 1

    def handle_job_routings(self, job):
        """
        add edges for a job
        update self.e_pred, self.e_succ, self.e_all_pred, self.e_all_succ
        update self.succ_node, self.mc_nodes
        """
        # edge for the first node
        self.add_edge("U", (job.mcs[0], job.id, 0))

        for i, (m, m2) in enumerate(zip(job.mcs[:-1], job.mcs[1:])):
            # pred edge
            self.add_edge((m, job.id, i), (m2, job.id, i+1))  # key: (machine, job)

            # info
            self.succ_node[(m, job.id, i)] = (m2, job.id, i+1)
            self.prev_node[(m2, job.id, i+1)] = (m, job.id, i)

        # edge for the last node
        self.add_edge((job.mcs[-1], job.id, len(job.mcs)-1), "V")

    def handle_job_reversed_routings(self, job):
        """
        add edges for a job
        update self.e_pred, self.e_succ, self.e_all_pred, self.e_all_succ
        update self.succ_node, self.mc_nodes
        """
        # edge for the first node
        self.add_edge((job.mcs[0], job.id, 0), "U")

        for i, (m, m2) in enumerate(zip(job.mcs[:-1], job.mcs[1:])):
            # pred edge
            self.add_edge((m2, job.id, i+1), (m, job.id, i))  # key: (machine, job)

            # info
            self.prev_node[(m, job.id, i)] = (m2, job.id, i+1)
            self.succ_node[(m2, job.id, i+1)] = (m, job.id, i)

        # edge for the last node
        self.add_edge("V", (job.mcs[-1], job.id, len(job.mcs)-1))


class Graph_Env():
    def __init__(self, benchmark: str, job_n: int, mc_n: int, instance_i: int):
        super().__init__()
        self.job_dict, self.job_arrival = get_job_dict(benchmark, job_n, mc_n, instance_i)

        self.js = JobShopGraph()
        self.reversed_js = JobShopGraph()

        for job in self.job_dict.values():
            self.js.add_job(job)
            self.reversed_js.reversed_add_job(job)

    # disjunctive edges #############################################################################
    def add_disj_edges(self, sol):
        for mc_i, job_seq in enumerate(sol):
            prev_node = None
            for j, job_i in enumerate(job_seq):
                if j == 0:
                    i = self.job_dict[job_i].mcs.index(mc_i)
                    prev_node = (mc_i, job_i, i)
                    continue
                else:
                    i = self.job_dict[job_i].mcs.index(mc_i)
                    node = (mc_i, job_i, i)
                    self.js.add_edge(prev_node, node)
                    # next step
                    prev_node = node

    def add_reversed_disj_edges(self, sol):
        for mc_i, job_seq in enumerate(sol):
            prev_node = None
            for j, job_i in enumerate(job_seq):
                if j == 0:
                    i = self.job_dict[job_i].mcs.index(mc_i)
                    prev_node = (mc_i, job_i, i)
                    continue
                else:
                    i = self.job_dict[job_i].mcs.index(mc_i)
                    node = (mc_i, job_i, i)
                    self.reversed_js.add_edge(node, prev_node)
                    # next step
                    prev_node = node

    # predecessor edges #############################################################################
    def get_pred_end_t(self, job_i, i):
        if i > 0:
            pred_i = i - 1
            pred_mc_i = self.job_dict[job_i].mcs[pred_i]
            pred_node = (pred_mc_i, job_i, pred_i)
            pred_end_t = self.js.nodes[pred_node]['C']
        else:
            pred_end_t = 0

        return pred_end_t

    def get_reversed_pred_end_t(self, job_i, i):
        if i < len(self.job_dict[job_i].mcs) - 1:
            pred_i = i + 1
            pred_mc_i = self.job_dict[job_i].mcs[pred_i]
            pred_node = (pred_mc_i, job_i, pred_i)
            pred_end_t = self.reversed_js.nodes[pred_node]['C']
        else:
            pred_end_t = 0

        return pred_end_t

    # left shift ####################################################################################
    def left_shift_once(self, sol):
        for mc_i, job_seq in enumerate(sol):
            time_seq = list()
            node_seq = list()

            for j, job_i in enumerate(job_seq):
                i = self.job_dict[job_i].mcs.index(mc_i)
                node = (mc_i, job_i, i)

                if j > 0:  # first job in mc_seq -> left shift x
                    prt = self.reversed_js.nodes[node]['prt']

                    for k, (s_t1, _) in enumerate(time_seq):
                        node1 = node_seq[k]
                        if k == 0:
                            e_t0 = 0
                            node0 = None
                        else:
                            e_t0 = time_seq[k-1][1]
                            node0 = node_seq[k-1]

                        if s_t1 - e_t0 >= prt:  # 1. whether insert in the machine
                            pred_end_t = self.get_pred_end_t(job_i, i)

                            if s_t1 - pred_end_t >= prt:  # 2. whether insert
                                # disjunctive edges #################
                                node2 = node_seq[-1]
                                self.js.remove_edge(node2, node)
                                self.js.add_edge(node, node1)

                                if node0 is not None:
                                    self.js.remove_edge(node0, node1)
                                    self.js.add_edge(node0, node)

                                if j < len(job_seq) - 1:  # last job in mc_seq -> no remove
                                    next_job_i = job_seq[j+1]
                                    next_i = self.job_dict[next_job_i].mcs.index(mc_i)
                                    next_node = (mc_i, next_job_i, next_i)
                                    self.js.remove_edge(node, next_node)
                                    self.js.add_edge(node2, next_node)

                                sol[mc_i].pop(j)
                                insert_i = k
                                sol[mc_i].insert(insert_i, job_i)

                                return sol, True

                node_seq.append(node)
                time_seq.append((self.js.nodes[node]['S'], self.js.nodes[node]['C']))

        return sol, False

    def left_shift(self, sol):
        diff_TF = False

        while True:
            sol, left_TF = self.left_shift_once(sol)
            if left_TF:
                _ = self.js.makespan()
                diff_TF = True
            else:
                break

        return sol, diff_TF

    # reversed left shift #############################################################################
    def reversed_left_shift_once(self, sol):
        for mc_i, job_seq in enumerate(sol):
            time_seq = list()
            node_seq = list()

            for j, job_i in enumerate(reversed(job_seq)):
                i = self.job_dict[job_i].mcs.index(mc_i)
                node = (mc_i, job_i, i)

                if j > 0:  # 맨 처음 op는 left shift 불가능
                    prt = self.reversed_js.nodes[node]['prt']

                    for k, (s_t1, _) in enumerate(time_seq):
                        node1 = node_seq[k]
                        if k == 0:
                            e_t0 = 0
                            node0 = None
                        else:
                            e_t0 = time_seq[k-1][1]
                            node0 = node_seq[k-1]

                        if s_t1 - e_t0 >= prt:  # 1. whether insert in the machine
                            pred_end_t = self.get_reversed_pred_end_t(job_i, i)

                            if s_t1 - pred_end_t >= prt:  # 2. whether insert
                                # disjunctive edges
                                node2 = node_seq[-1]  # mc_seq 상에서 마지막 node
                                self.reversed_js.remove_edge(node2, node)
                                self.reversed_js.add_edge(node, node1)

                                if node0 is not None:  # 없으면 제거 안함
                                    self.reversed_js.remove_edge(node0, node1)
                                    self.reversed_js.add_edge(node0, node)

                                j2 = len(job_seq) - 1 - j
                                if j < len(job_seq) - 1:  # 마지막이면 제거 안함
                                    next_job_i = job_seq[j2-1]
                                    next_i = self.job_dict[next_job_i].mcs.index(mc_i)
                                    next_node = (mc_i, next_job_i, next_i)
                                    self.reversed_js.remove_edge(node, next_node)
                                    self.reversed_js.add_edge(node2, next_node)

                                sol[mc_i].pop(j2)
                                insert_i = len(job_seq) - 1 - k + 1
                                sol[mc_i].insert(insert_i, job_i)

                                return sol, True

                node_seq.append(node)
                time_seq.append((self.reversed_js.nodes[node]['S'], self.reversed_js.nodes[node]['C']))

        return sol, False

    def reversed_left_shift(self, sol):
        diff_TF = False

        # doing
        while True:
            sol, left_TF = self.reversed_left_shift_once(sol)
            if left_TF:
                _ = self.reversed_js.makespan()
                # show_gantt_plotly(env.reversed_js)
                diff_TF = True
            else:
                break

        return sol, diff_TF

    # save ###########################################################################################
    def save_start_t(self, sol):
        start_t_list = list()
        for mc_i, job_seq in enumerate(sol):
            sub_start_t_list = list()
            for j, job_i in enumerate(job_seq):
                i = self.job_dict[job_i].mcs.index(mc_i)
                node = (mc_i, job_i, i)
                sub_start_t_list.append(int(self.js.nodes[node]['S']))
            start_t_list.append(sub_start_t_list)

        return start_t_list

    def sol_save(self, mc_seq1, mc_seq2, benchmark: str, job_n: int, mc_n: int, instance_i: int, sol_type: str=''):
        problem = f'{benchmark}{job_n}x{mc_n}'
        folder_path = f'./../benchmark/{benchmark}/{problem}'
        if not os.path.isdir(folder_path):
            folder_path = f'./benchmark/{benchmark}/{problem}'

        # check diff sol ################################################
        diff_TF = False
        for mc_i, seq in enumerate(mc_seq2):
            opt_seq = mc_seq1[mc_i]
            if seq != opt_seq:
                diff_TF = True
                break

        # save ########################################################
        if diff_TF:
            start_t_list = self.save_start_t(mc_seq2)
            obj = self.js.makespan()

            with open(f'{folder_path}/opt_{sol_type}_{instance_i}.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow([obj, benchmark, job_n, mc_n, instance_i])

                for seq in mc_seq2:
                    wr.writerow(seq)

                for start_ts in start_t_list:
                    wr.writerow(start_ts)


if __name__ == "__main__":
    from utils import get_opt_makespan_once, HUN, all_benchmarks, HUN_40_3
    # from environment.visualize import show_gantt_plotly

    ######################################################################################################
    for (benchmark, job_n, mc_n, instance_n) in HUN_40_3:
        problem = str(benchmark) + str(job_n) + "x" + str(mc_n)
        data_path = './../bench_data/' + benchmark + '/' + problem

        print("\n=========", benchmark, job_n, mc_n, "=============================================")
        for instance_i in tqdm(range(instance_n)):
            for sol_type in ['active', 'full_active']:
                opt_mc_seq, _ = load_opt_sol(benchmark, job_n, mc_n, instance_i, sol_type=sol_type)
                if not opt_mc_seq:
                    continue

                sol = copy.deepcopy(opt_mc_seq)

                opt_makespn = get_opt_makespan_once(benchmark, job_n, mc_n, instance_i)
                env = Graph_Env(benchmark, job_n, mc_n, instance_i)

                if 'full_active' in sol_type:
                    # left_shifting for reverse #################################
                    env.add_reversed_disj_edges(sol)
                    mk2 = env.reversed_js.makespan()
                    # show_gantt_plotly(env.reversed_js)
                    if mk2 != opt_makespn:
                        print("error: none optimal but a solution is saved", mk2, opt_makespn)
                    sol, diff_TF = env.reversed_left_shift(sol)
                    # show_gantt_plotly(env.reversed_js)

                # left_shifting ###############################################
                env.add_disj_edges(sol)
                mk = env.js.makespan()
                # show_gantt_plotly(env.js)
                if mk != opt_makespn:
                    print("error: none optimal but a solution is saved", mk, opt_makespn)
                sol, diff_TF = env.left_shift(sol)
                # show_gantt_plotly(env.js)

                env.sol_save(opt_mc_seq, sol, benchmark, job_n, mc_n, instance_i, sol_type=sol_type)
