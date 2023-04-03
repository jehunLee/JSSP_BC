import os, csv
import pandas as pd


# total 242 instances
TA = [('TA', 15, 15, 10),
      ('TA', 20, 15, 10), ('TA', 20, 20, 10),
      ('TA', 30, 15, 10), ('TA', 30, 20, 10),
      ('TA', 50, 15, 10), ('TA', 50, 20, 10),
      ('TA', 100, 20, 10), ]  # 80
LA = [('LA', 10, 5, 5), ('LA', 15, 5, 5), ('LA', 20, 5, 5),
      ('LA', 10, 10, 5), ('LA', 15, 10, 5), ('LA', 20, 10, 5),
      ('LA', 30, 10, 5), ('LA', 15, 15, 5), ]  # 40
DMU = [('DMU', 20, 15, 10), ('DMU', 20, 20, 10),
       ('DMU', 30, 15, 10), ('DMU', 30, 20, 10),
       ('DMU', 40, 15, 10), ('DMU', 40, 20, 10),
       ('DMU', 50, 15, 10), ('DMU', 50, 20, 10), ]  # 80
SWV = [('SWV', 20, 10, 5), ('SWV', 20, 15, 5), ('SWV', 50, 10, 10), ]  # 20
ABZ = [('ABZ', 10, 10, 2), ('ABZ', 20, 15, 3), ]  # 5
ORB = [('ORB', 10, 10, 10), ]  # 10
YN = [('YN', 20, 20, 4), ]  # 4
FT = [('FT', 6, 6, 1), ('FT', 10, 10, 1), ('FT', 20, 5, 1), ]  # 3

all_benchmarks = TA + LA + FT + ORB + ABZ + YN + SWV + DMU


all_rules = ['SPT', 'LPT', 'SRPT', 'LRPT', 'STT', 'LTT', 'LOR', 'MOR']


# total 800 instances
HUN = [('HUN', 6, 4, 200), ('HUN', 6, 6, 200),
       ('HUN', 8, 4, 200), ('HUN', 8, 6, 200), ]  # 800


def load_data(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> (list, list):
    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark/{benchmark}/{problem}'

    file_path = f'{folder_path}/instance_{instance_i}'

    job_prts = list()
    job_mcs = list()

    with open(file_path, 'r') as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                continue
            line = [int(i) for i in line]
            job_mcs.append(line[0::2])
            job_prts.append(line[1::2])

    return job_mcs, job_prts


def load_opt_sol(benchmark: str, job_n: int, mc_n: int, instance_i: int, sol_type: str='') -> (list, list):
    """
    load optimal solution
    """
    problem = f'{benchmark}{job_n}x{mc_n}'
    folder_path = f'./../benchmark/{benchmark}/{problem}'
    if not os.path.isdir(folder_path):
        folder_path = f'./benchmark/{benchmark}/{problem}'

    # file_path = f'{folder_path}/instance_{instance_i}'

    opt_full_active_path = folder_path + f'/opt_full_active_{instance_i}.csv'
    opt_active_path = folder_path + f'/opt_active_{instance_i}.csv'
    opt_cp_path = folder_path + f'/opt_{instance_i}.csv'
    opt_path = folder_path + f'/opt_{instance_i}.txt'

    mc_seq = list()
    mc_seq_st = list()
    if sol_type and 'full_active' in sol_type and os.path.isfile(opt_full_active_path):  # full_active 결과 있음
            with open(opt_full_active_path, 'r') as f:
                rdr = csv.reader(f)
                for i, line in enumerate(rdr):
                    if i == 0:
                        continue
                    mc_seq.append([int(i) for i in line])

            mc_seq_st = mc_seq[mc_n:]
            mc_seq = mc_seq[:mc_n]

    elif sol_type and 'active' in sol_type and os.path.isfile(opt_active_path):  # full_active 결과 있음
            with open(opt_active_path, 'r') as f:
                rdr = csv.reader(f)
                for i, line in enumerate(rdr):
                    if i == 0:
                        continue
                    mc_seq.append([int(i) for i in line])

            mc_seq_st = mc_seq[mc_n:]
            mc_seq = mc_seq[:mc_n]

    elif os.path.isfile(opt_cp_path):  # cp 결과 있음
        with open(opt_cp_path, 'r') as f:
            rdr = csv.reader(f)
            for i, line in enumerate(rdr):
                if i == 0:
                    continue
                mc_seq.append([int(i) for i in line])

        mc_seq_st = mc_seq[mc_n:]
        mc_seq = mc_seq[:mc_n]

    elif os.path.isfile(opt_path):  # cp 결과 없음
        with open(opt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                if '\n' in line[-1]:
                    line[-1] = line[-1][:-1]
                mc_seq.append([int(n) for n in line])

    else:  # none optimal solution
        print(f'none optimal sol: problem {problem}')

    return mc_seq, mc_seq_st


def is_in_list(target_idxs, idx) -> bool:
    if idx in target_idxs:
        return True
    return False


# def get_opt_makespans() -> pd.DataFrame:
#     file_path = "../result/bench_opt.csv"
#     if not os.path.exists(file_path):
#         file_path = "./result/bench_opt.csv"
#
#     opt_data = pd.read_csv(file_path)
#     opt_data.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'makespan', 'convergence', 'UB', 'name', 'set']
#     opt_data['problem'] = opt_data['benchmark'] + ' ' + opt_data['job_n'].astype('string') + 'x' \
#                           + opt_data['mc_n'].astype('string')
#
#     opt_data['UB'] = opt_data['UB'].astype('int')
#     opt_data = opt_data.drop(['makespan', 'convergence'], axis=1)
#
#     return opt_data


def get_opt_makespan_once(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> int:
    file_path = "../result/bench_opt.csv"
    if not os.path.exists(file_path):
        file_path = "./result/bench_opt.csv"

    opt_data = pd.read_csv(file_path)
    opt_data.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'makespan', 'convergence', 'UB', 'name', 'set']
    # opt_data['benchmark'] = opt_data['benchmark'].astype('str')
    opt_data['UB'] = opt_data['UB'].astype('int')

    return opt_data.loc[(opt_data['benchmark'] == benchmark) & (opt_data['job_n'] == job_n)
                        & (opt_data['mc_n'] == mc_n) & (opt_data['instance_i'] == instance_i)].iloc[0]['UB']


def get_load_result_data():
    # result data ####################################################################
    result_ref = pd.read_csv("./bench_ref.csv")
    result_ref.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'name',
                          'makespan', 'method']
    result_ref['problem'] = result_ref['benchmark'] + ' ' + result_ref['job_n'].astype('string') + 'x' \
                            + result_ref['mc_n'].astype('string')
    result_ref = result_ref.drop(['benchmark', 'job_n', 'mc_n', 'name'], axis=1)

    # opt_module result data ###################################################################
    result_opt = pd.read_csv("./bench_opt.csv")
    result_opt.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i',
                          'opt_makespan', 'convergence', 'UB', 'name', 'set']
    result_opt['problem'] = result_opt['benchmark'] + ' ' + result_opt['job_n'].astype('string') + 'x' \
                            + result_opt['mc_n'].astype('string')
    result_opt = result_opt.drop(['benchmark', 'opt_makespan', 'convergence', 'set'], axis=1)

    # cp data ###################################################################
    result_cp = pd.read_csv("./bench_cp.csv")
    result_cp.columns = ['time_limit', 'benchmark', 'job_n', 'mc_n', 'instance_i',
                         'makespan', 'time', 'convergence']
    result_cp['problem'] = result_cp['benchmark'] + ' ' + result_cp['job_n'].astype('string') + 'x' \
                           + result_cp['mc_n'].astype('string')
    result_cp['method'] = 'CP (' + result_cp['time_limit'].astype('string') + ' sec)'
    result_cp = result_cp.drop(['time_limit', 'benchmark', 'job_n', 'mc_n', 'time', 'convergence'], axis=1)

    # our data ###################################################################
    result_ours = pd.read_csv("./bench_ours.csv")
    result_ours.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time', 'mean_decision_t', 'decision_n']
    result_ours['problem'] = result_ours['benchmark'] + ' ' + result_ours['job_n'].astype('string') + 'x' \
                             + result_ours['mc_n'].astype('string')
    result_ours = result_ours.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time', 'mean_decision_t', 'decision_n'], axis=1)

    # rule data ###################################################################
    result_rule = pd.read_csv("./bench_rule.csv")
    result_rule.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time', 'mean_decision_t', 'decision_n']
    result_rule['problem'] = result_rule['benchmark'] + ' ' + result_rule['job_n'].astype('string') + 'x' \
                             + result_rule['mc_n'].astype('string')
    result_rule = result_rule[result_rule['action_type'] == 'single_mc_buffer']
    result_rule = result_rule.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time', 'mean_decision_t', 'decision_n'], axis=1)


    # result + opt ##################################################################
    result_opt2 = result_opt.drop(['job_n', 'mc_n', 'UB', 'name'], axis=1)
    result_opt2['makespan'] = result_opt['UB']
    result_opt2['method'] = 'OPT'

    result_data = pd.concat([result_ours, result_ref, result_rule, result_cp, result_opt2])
    merge_data = pd.merge(result_data, result_opt, on=['problem', 'instance_i'], how="left")

    merge_data = merge_data.dropna(subset=['UB'])
    merge_data['opt_gap'] = (merge_data['makespan'] - merge_data['UB']) / merge_data['UB']
    if min(merge_data['opt_gap']) < 0:
        print("error: opt_module gap error ----------------------------------------------")

    merge_data = merge_data.drop(['makespan', 'UB'], axis=1)
    merge_data['job_n'] = merge_data['job_n'].astype('int')
    merge_data['mc_n'] = merge_data['mc_n'].astype('int')

    return merge_data

# def get_random_env():
#     data_set = [([1, 20], 0, False), ([1, 20], 0.2, False), ([1, 20], 0.4, False)]
#     job_n = random.choice(range(6, 11))
#     mc_n = random.choice(range(3, 8))
#
#     (prt_range, op_skip_ratio, flow_TF) = random.choice(data_set)
#     data_path = './agent'
#     if not os.path.isdir(data_path):
#         data_path = '../../GNN_RL5/agent'
#     instance_i = 0
#     problem_gen(data_path, instance_i, prt_range, job_n, mc_n, op_skip_ratio, flow_TF)
#
#     return env.JobShopEnv(data_path, instance_i=instance_i)


if __name__ == "__main__":
    data = load_data('FT', 6, 6, 0)
    opt_mc_seq = load_opt_sol('FT', 6, 6, 0)
    # opt_mc_seq2 = load_opt_sol('SWV', 20, 10, 0)

    print()