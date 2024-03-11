import csv, os
from tqdm import tqdm
import numpy as np
from environment.env import JobShopEnv
# from agent.agent_GNN import AgentGNN
from opt.CP import JSSP_solver
from params import configs


# params for running agent using dispatching rule #######################################
def gen_instances(benchmark: str, job_n: int, mc_n: int, instance_range: list,
                  prt_range: list, mc_skip_ratio: float, separate_TF: bool, improve_criteria=0.1):
    agent_type = configs.agent_type
    state_type = configs.state_type
    env_type = configs.env_type
    action_type = configs.action_type
    # configs.agent_type = 'rule'  # 'rule', 'GNN_BC', 'GNN_RL'
    # configs.state_type = 'simple'
    # configs.env_type = ''

    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark'

    folder_path = f'{folder_path}/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    rules = ['LTT', 'MOR', 'SPT']

    print(f'{benchmark} {job_n}x{mc_n} {prt_range} {mc_skip_ratio} {separate_TF} =================')
    for instance_i in tqdm(range(instance_range[0], instance_range[1])):
        # print(f'{benchmark} {job_n}x{mc_n}_{instance_i}====================================================')

        file_path = f'{folder_path}/instance_{instance_i}'

        # while True:
        gen_instance(benchmark, job_n, mc_n, instance_i, prt_range, mc_skip_ratio, separate_TF, file_path)

            # env = JobShopEnv([(benchmark, job_n, mc_n, instance_i)], pomo_n=len(rules))
            # status, obj_v = JSSP_solver(benchmark, job_n, mc_n, instance_i, 1)

            # if improve_criteria > 0:
            #     # validation ###############
            #     configs.action_type = 'buffer'
            #     cum_r, _ = env.run_episode_rule(rules)
            #     best_rule_perform = -cum_r.max().item()
            #
            #     improve_ratio = round((best_rule_perform - obj_v) / obj_v, 4)
            #     if improve_ratio <= improve_criteria:
            #         continue
            #
            # configs.action_type = 'conflict'
            # cum_r, decision_n = env.run_episode_rule(rules)

            # if decision_n.min().item() > 0:  # check the number of decision points
            #     if status != 4:  # non convergence -> again CP
            #         _, _ = JSSP_solver(benchmark, job_n, mc_n, instance_i, 7200)
            #     break

    # configs.agent_type = agent_type
    # configs.state_type = state_type
    # configs.env_type = env_type
    # configs.action_type = action_type

def gen_instance(benchmark: str, job_n: int, mc_n: int, instance_i: int,
                 prt_range: list, mc_skip_ratio: float, separate_TF: bool, file_path: str):
    job_mcs = list()
    job_prts = list()

    for j in range(job_n):
        mcs = get_mcs(mc_n, mc_skip_ratio, separate_TF)
        prts = np.random.randint(prt_range[0], prt_range[1]+1, size=len(mcs))
        job_mcs.append(mcs)
        job_prts.append(prts)

    # save ################
    with open(file_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([benchmark, job_n, mc_n, instance_i])

        for i in range(job_n):
            add_row = list()
            for j in range(len(job_mcs[i])):
                add_row += [int(job_mcs[i][j]), int(job_prts[i][j])]
            wr.writerow(add_row)


def get_random_mcs(mc_skip_ratio: float, mcs: np.array):
    if mc_skip_ratio:
        choice_mcs = np.random.binomial(n=1, p=1 - mc_skip_ratio, size=len(mcs)).nonzero()
        mcs_ = mcs[choice_mcs]
    else:
        mcs_ = mcs
    if not len(mcs_):
        mcs_ = np.random.choice(mcs, 1)

    np.random.shuffle(mcs_)

    return mcs_


def get_mcs(mc_n: int, mc_skip_ratio: float, separate_TF: bool):
    if not separate_TF:
        mcs = np.arange(mc_n)

        mcs_ = get_random_mcs(mc_skip_ratio, mcs)

    else:
        half_mc_i = int(mc_n/2)
        mcs_1 = np.arange(half_mc_i)
        mcs_2 = np.arange(half_mc_i, mc_n)

        mcs_1_ = get_random_mcs(mc_skip_ratio, mcs_1)
        mcs_2_ = get_random_mcs(mc_skip_ratio, mcs_2)
        mcs_ = np.concatenate([mcs_1_, mcs_2_], axis=0)

    return mcs_


if __name__ == "__main__":
    # data_set = [([0, 40], [1, 30], 0, False), ([40, 80], [1, 100], 0, False),
    #             ([80, 120], [21, 30], 0, False), ([120, 160], [1, 30], 0.3, False),
    #             ([160, 200], [1, 30], 0, True)]
    #
    # data_set = [([0, 1000], [1, 10], 0, False)]

    # data_set = [[[200, 240], [1, 30], 0, False], [[240, 280], [1, 100], 0, False],
    #             [[280, 320], [21, 30], 0, False], [[320, 360], [1, 30], 0.3, False],
    #             [[360, 400], [1, 30], 0, True]]
    start_i = 800
    data_set = [[[start_i, start_i+40], [1, 30], 0, False], [[start_i+40, start_i+80], [1, 100], 0, False],
                [[start_i+80, start_i+120], [21, 30], 0, False], [[start_i+120, start_i+160], [1, 30], 0.3, False],
                [[start_i+160, start_i+200], [1, 30], 0, True]]

    data_set = [[[0, 10], [10, 300], 0, False],
                [[10, 20], [210, 300], 0, False], [[20, 30], [10, 300], 0.3, False],
                [[30, 40], [10, 300], 0, True]]

    data_set = [[[0, 10], [1, 100], 0, True]]

    configs.model_type = ''

    def load_opt_obj(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> (list, list):
        problem = f'{benchmark}{job_n}x{mc_n}'
        folder_path = f'./../benchmark/{benchmark}/{problem}'
        if not os.path.isdir(folder_path):
            folder_path = f'./benchmark/{benchmark}/{problem}'

        file_path = f'{folder_path}/opt_{instance_i}.csv'

        with open(file_path, 'r') as f:
            rdr = csv.reader(f)
            for i, line in enumerate(rdr):
                if i == 0:
                    return line[0]

    benchmark = "FLOW"
    # benchmark = "TEST"
    for job_n in [10, 20, 50]:
        for mc_n in [5, 10]:
            if job_n < mc_n:
                continue
            for (instance_range, prt_range, mc_skip_ratio, separate_TF) in data_set:
                gen_instances(benchmark, job_n, mc_n, instance_range, prt_range, mc_skip_ratio, separate_TF)
                # for instance_i in range(instance_range[0], instance_range[1]):
                    # obj_v = load_opt_obj(benchmark, job_n, mc_n, instance_i)

                    # # save ################
                    # with open('../result/bench_opt.csv', 'a', newline='') as f:
                    #     wr = csv.writer(f)
                    #     wr.writerow([benchmark, job_n, mc_n, instance_i,
                    #                  obj_v, 4, obj_v, '-', separate_TF])
