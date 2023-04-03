import time
import numpy as np
import torch


def rule_values(obs, rule) -> torch.Tensor:
    """
    return values of operations
    for argmax selection
    """
    op_x = obs['op'].x  # op_prt, op_tail_prt, op_succ_n, ...
    if rule == 'SPT':  # shortest processing time
        values = - op_x[:, 0].view(-1, 1)
    elif rule == 'LPT':  # longest processing time
        values = op_x[:, 0].view(-1, 1)
    elif rule == 'SRPT':  # shortest remaining processing time
        values = - (op_x[:, 0].view(-1, 1) + op_x[:, 1].view(-1, 1))
    elif rule == 'LRPT':  # longest remaining processing time
        values = op_x[:, 0].view(-1, 1) + op_x[:, 1].view(-1, 1)
    elif rule == 'STT':  # shortest tail time
        values = - op_x[:, 1].view(-1, 1)
    elif rule == 'LTT':  # longest tail time
        values = op_x[:, 1].view(-1, 1)
    elif rule == 'LOR':  # least operation remaining
        values = - op_x[:, 2].view(-1, 1)
    elif rule == 'MOR':  # most operation remaining
        values = op_x[:, 2].view(-1, 1)
    else:
        raise ValueError("Unknown dispatching rule.")

    op_mask = obs['op'].mask
    values -= 1e4 * (1 - op_mask.view(-1, 1))

    return values


# agent #########################################################################################
class Agent:
    def __init__(self, agent_type='rule'):
        self.agent_type = agent_type

    def get_action(self, obs, rule='') -> torch.Tensor:
        """ rule 이 max, min 도출하는지 구분해서 action 선택
        """
        op_mask = obs['op'].mask
        if rule == 'random':
            prob = op_mask / op_mask.sum()
            prob = np.array(prob.to('cpu'))
            return np.random.choice(range(op_mask.shape[0]), 1, p=prob)[0]
        else:
            values = rule_values(obs, rule)

        return torch.argmax(values)

    # run episode ###################################################################################
    def run_episode(self, env, rule='', test_TF=False, model=None) -> (float, float, float, float, list):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        obs, reward, done = env.reset()
        cum_r = reward
        decision_n = 0

        s_t = time.time()
        while not done:
            a_tensor = self.get_action(obs, rule)
            obs, reward, done = env.step(a_tensor.item())
            cum_r += reward
            decision_n += 1
        run_t = round(time.time() - s_t, 4)

        if decision_n == 0:
            mean_decision_t = 0
        else:
            mean_decision_t = round(run_t / decision_n, 4)

        return cum_r, run_t, mean_decision_t, decision_n, []

    def perform_envs(self, envs, rule='', model=None) -> (float, float, float, float):
        """
        perform for envs
        return: avg_cum_r, avg_run_t, avg_decision_t
        """
        total_cum_r = 0
        total_t = 0
        total_transit_t = 0
        total_decision_n = 0
        for i, env in enumerate(envs):
            cum_r, t, transit_t, decision_n, _ = self.run_episode(env, rule=rule, test_TF=True, model=model)
            total_cum_r += cum_r
            total_t += t
            total_transit_t += transit_t
            total_decision_n += decision_n

        return round(total_cum_r / len(envs), 2), round(total_t / len(envs), 5), \
               round(total_transit_t / len(envs), 5), round(total_decision_n / len(envs), 2)

    def perform_benchmarks(self, benchmarks, save_path='', rules=None, model=None) -> None:
        """
        perform for envs
        return: avg_cum_r, avg_run_t, avg_decision_t
        """
        import csv
        from tqdm import tqdm

        for (benchmark, job_n, mc_n, instance_i_n) in benchmarks:
            print(benchmark, job_n, mc_n, instance_i_n)
            envs = list()
            for instance_i in range(instance_i_n):
                envs.append(JobShopEnv(benchmark, job_n, mc_n, instance_i))

            for instance_i in tqdm(range(instance_i_n)):
                for rule in rules:
                    cum_r, run_t, transit_t, decision_n, _ = self.run_episode(envs[instance_i], rule=rule,
                                                                              test_TF=True, model=model)
                    with open(save_path, 'a') as f:
                        wr = csv.writer(f)
                        wr.writerow([benchmark, job_n, mc_n, instance_i,
                                     configs.agent_type, configs.action_type, configs.state_type, rule,
                                     -cum_r, run_t, transit_t, decision_n])

if __name__ == "__main__":
    from environment.env import JobShopEnv
    from params import configs
    from utils import all_benchmarks, all_rules

    configs.env_type = ''  # 'dyn', ''
    configs.state_type = 'simple'  # 'norm', '', 'add_norm', 'simple', 'simple_norm', 'meta', 'mc'

    configs.agent_type = 'rule'  # 'rule', 'GNN_BC', 'GNN_RL'
    if configs.agent_type == 'rule':
        configs.state_type = 'simple'
        configs.env_type = ''

    agent = Agent()

    # for configs.action_type in ['single_mc_buffer', 'single_mc_buffer_being', 'conflict', 'single_mc_conflict']:
    #     env = JobShopEnv('FT', 6, 6, 0)
    #     cum_r, run_t, mean_decision_t, decision_n, _ = agent.run_episode(env, 'LTT')
    #     print(-cum_r, run_t, mean_decision_t, decision_n)
    #
    # for configs.action_type in ['single_mc_buffer', 'single_mc_buffer_being', 'conflict', 'single_mc_conflict']:
    #     envs = list()
    #     for instance_i in range(10):
    #         envs.append(JobShopEnv('ORB', 10, 10, instance_i))
    #
    #     mean_cum_r, mean_run_t, mean_decision_t, mean_decision_n = agent.perform_envs(envs, 'LTT')
    #     print(-mean_cum_r, mean_run_t, mean_decision_t, mean_decision_n)

    for configs.action_type in ['single_mc_buffer', 'single_mc_buffer_being', 'conflict', 'single_mc_conflict']:
        save_path = './../result/bench_rule.csv'
        agent.perform_benchmarks(all_benchmarks, save_path=save_path, rules=all_rules)

    print()