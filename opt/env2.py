from environment.env import JobShopEnv
from environment.job_shop_graph import JobShopGraph
import copy
from params import configs


class JobShopGenEnv(JobShopEnv):
    def __init__(self, benchmark: str, job_n: int, mc_n: int, instance_i: int):
        super().__init__(benchmark, job_n, mc_n, instance_i)

    def reset(self):
        """
        reset environment
        """
        self.js = JobShopGraph()

        self.job_arrival = copy.deepcopy(self.init_job_arrival)
        self.arrive_t = 0
        self.arrive_jobs()

        # additional function
        assign_pairs = self.move_next_state()

        return self.get_obs(), self.get_reward(), self.is_done(), assign_pairs, self.mc_conflict_ops.keys()

    ##########################################################################################################
    def step(self, op_idx):
        """
        transit to the next decision point(next state) by an action
        """
        node = self.js.op_node_map[op_idx]
        self.js.assign_node(node)  # step 5

        target_mc = node[0]
        del self.mc_conflict_ops[target_mc]

        # additional function
        assign_pairs = [(target_mc, node[1])]

        if not self.mc_conflict_ops:  # need action selection
            assign_pairs += self.move_next_state()  # additional function
        else:
            if not configs.policy_total_mc:  # move to next target machine
                self.target_mc = min(self.mc_conflict_ops.keys())
            if configs.dyn_env_TF:
                self.remove_edges()  # curr_t may be changed

        return self.get_obs(), self.get_reward(), self.is_done(), assign_pairs, self.mc_conflict_ops.keys()

    def move_next_state(self):
        """
        transit to the next decision point(next state)
        """
        # additional function
        assign_pairs = list()

        while True:
            self.update_mc_conflict_ops()

            if self.is_done():
                break
            single_assign_pairs = self.assign_single_candidate()
            assign_pairs += single_assign_pairs  # additional function

            if self.mc_conflict_ops:  # need action selection
                if not configs.policy_total_mc:
                    self.target_mc = min(self.mc_conflict_ops.keys())
                break

        if configs.dyn_env_TF:
            self.remove_edges()

        return assign_pairs

    def assign_single_candidate(self):
        """
        step 4: assign job to machine
        only one candidate job -> automatic assign
        """
        # additional function
        single_assign_pairs = list()

        del_mcs = list()
        for mc, nodes in self.mc_conflict_ops.items():
            if len(nodes) == 1:
                node = nodes[0]
                self.js.assign_node(node)  # step 5
                del_mcs.append(mc)
                single_assign_pairs.append((mc, node[1]))  # additional function

        for mc in del_mcs:
            del self.mc_conflict_ops[mc]

        return single_assign_pairs