import csv, os
from tqdm import tqdm
import numpy as np
from environment.env import JobShopEnv
from agent.agent import Agent
from opt.CP import JSSP_solver
from params import configs


# params for running agent using dispatching rule #######################################
configs.env_type = ''  # 'dyn', ''
configs.state_type = 'simple'  # 'norm', '', 'add_norm', 'simple', 'simple_norm', 'meta'

configs.agent_type = 'rule'  # 'rule', 'GNN_BC', 'GNN_RL'
if configs.agent_type == 'rule':
    configs.state_type = 'simple'
    configs.env_type = ''

agent = Agent()


def gen_instances(benchmark: str, job_n: int, mc_n: int, instance_range: list,
                  prt_range: list, mc_skip_ratio: float, separate_TF: bool):
    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark'

    folder_path = f'{folder_path}/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    print(f'{benchmark} {job_n}x{mc_n} ====================================================')
    for instance_i in tqdm(range(instance_range[0], instance_range[1])):
        # print(f'{benchmark} {job_n}x{mc_n}_{instance_i}====================================================')

        file_path = f'{folder_path}/instance_{instance_i}'

        cnt = 0
        over_cnt = 0
        non_opt_cnt = 0
        while True:
            gen_instance(benchmark, job_n, mc_n, instance_i, prt_range, mc_skip_ratio, separate_TF, file_path)

            # validation ###############
            configs.action_type = 'single_mc_buffer'
            env = JobShopEnv(benchmark, job_n, mc_n, instance_i)

            cum_r1, _, _, _, _ = agent.run_episode(env, 'LTT')
            cum_r2, _, _, _, _ = agent.run_episode(env, 'MOR')
            cum_r3, _, _, _, _ = agent.run_episode(env, 'SPT')
            best_rule_perform = min(-cum_r1, -cum_r2, -cum_r3)

            status, obj_v = JSSP_solver(benchmark, job_n, mc_n, instance_i, 60)

            improve_ratio = round((best_rule_perform - obj_v) / obj_v, 4)
            # print(f'improve ratio: {improve_ratio}')

            if status != 4:
                non_opt_cnt += 1
            cnt += 1
            if improve_ratio > 0.1:
                over_cnt += 1
            #     configs.action_type = 'conflict'
            #     env = JobShopEnv(benchmark, job_n, mc_n, instance_i)
            #     _, _, _, decision_n, _ = agent.run_episode(env, 'LTT')
            #
            #     if decision_n > 0:  # check the number of decision points
            #         if status != 4:  # non convergence -> again CP
            #             _, _ = JSSP_solver(benchmark, job_n, mc_n, instance_i, 7200)
            #         break
            if cnt >= 5000:
                print(cnt, over_cnt, non_opt_cnt)
                break


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
        choice_binaray = np.random.binomial(n=1, p=1 - mc_skip_ratio, size=mc_n).nonzero()
        mcs_ = mcs[choice_binaray]
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
    data_set = [([0, 40], [1, 30], 0, False), ([40, 80], [1, 100], 0, False),
                ([80, 120], [21, 30], 0, False), ([120, 160], [1, 30], 0.3, False),
                ([160, 200], [1, 30], 0, True)]

    data_set = [([0, 1], [1, 30], 0, False)]

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

    # benchmark = "HUN"
    benchmark = "TEST"
    for job_n in [6, 8]:
        for mc_n in [4, 6]:
            for (instance_range, prt_range, mc_skip_ratio, separate_TF) in data_set:
                gen_instances(benchmark, job_n, mc_n, instance_range, prt_range, mc_skip_ratio, separate_TF)
                for instance_i in range(instance_range[0], instance_range[1]):
                    obj_v = load_opt_obj(benchmark, job_n, mc_n, instance_i)

                    # save ################
                    with open('../result/bench_opt.csv', 'a', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow([benchmark, job_n, mc_n, instance_i,
                                     obj_v, 4, obj_v, '-', separate_TF])
