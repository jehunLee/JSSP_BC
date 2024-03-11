import csv, os
from tqdm import tqdm
import numpy as np
from environment.env import JobShopEnv
# from agent.agent_GNN import AgentGNN
from opt.CP import JSSP_solver
from params import configs


# params for running agent using dispatching rule #######################################
def gen_instances(benchmark: str, job_n: int, mc_n: int, instance_range: list,
                  job_type_n: int, prt_range: list, mc_skip_ratio: float, separate_TF: bool, improve_criteria=0.1):
    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark'

    folder_path = f'{folder_path}/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    print(f'{benchmark} {job_n}x{mc_n} {prt_range} {mc_skip_ratio} {separate_TF} =================')
    for instance_i in tqdm(range(instance_range[0], instance_range[1])):
        file_path = f'{folder_path}/instance_{instance_i}'

        # while True:
        gen_instance(benchmark, job_type_n, job_n, mc_n, instance_i, prt_range, mc_skip_ratio, separate_TF, file_path)


def gen_instance(benchmark: str, job_type_n: int, job_n: int, mc_n: int, instance_i: int,
                 prt_range: list, mc_skip_ratio: float, separate_TF: bool, file_path: str):
    job_type_info = dict()
    for k in range(job_type_n):
        mcs = get_mcs(mc_n, mc_skip_ratio, separate_TF)
        prts = np.random.randint(prt_range[0], prt_range[1]+1, size=len(mcs))
        job_type_info[k] = (mcs, prts)

    job_mcs = list()
    job_prts = list()
    for _ in range(job_n):
        type = np.random.choice(job_type_n, 1)[0]
        mcs, prts = job_type_info[type]

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
    data_set = [[[30, 40], [700, 1000], 0.2, False]]

    benchmark = "REAL_D"
    job_type_n = 40
    # benchmark = "TEST"
    for job_n in [200, 300]:
        for mc_n in [20]:
            if job_n < mc_n:
                continue
            for (instance_range, prt_range, mc_skip_ratio, separate_TF) in data_set:
                gen_instances(benchmark, job_n, mc_n, instance_range, job_type_n, prt_range, mc_skip_ratio, separate_TF)

