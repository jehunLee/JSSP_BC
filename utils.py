import os, csv
from params import configs
import pickle


# load data ##############################################################################################
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

    # else:  # none optimal solution
    #     print(f'none optimal sol: problem {problem} {instance_i}')

    return mc_seq, mc_seq_st


def get_opt_data_path(benchmark: str, job_n: int, mc_n: int, instances: list) -> str:
    save_folder = f'./../opt_policy/{configs.action_type}__{configs.sol_type}'
    # if 'diff_disj_all_pred' in configs.model_type:
    #     save_folder = f'{save_folder}__diff_disj_all_pred'

    if 'all_pred' in configs.model_type:
        save_folder = f'{save_folder}__all_pred'
    else:
        save_folder = f'{save_folder}__all_pred'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    opt_data_type = f'{configs.env_type}__{configs.state_type}__{configs.policy_symmetric_TF}'

    save_path = f'{save_folder}/{opt_data_type}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    problem = f'{benchmark}{job_n}x{mc_n}_{len(instances)}'

    return f'{save_path}/{problem}.p'


def get_opt_data(problem_set):
    opt_data = list()

    for (benchmark, job_n, mc_n, instances) in problem_set:
        save_file = get_opt_data_path(benchmark, job_n, mc_n, instances)

        with open(save_file, 'rb') as file:  # rb: binary read mode
            opt_data += pickle.load(file)

    if 'cuda' in configs.device and not opt_data[0].is_cuda:
        for data in opt_data:
            data.to(configs.device)

    print(f'opt_data load done: {len(opt_data)} optimal policies')
    return opt_data


# load result data #########################################################################################
import pandas as pd


def get_opt_makespan_once(benchmark: str, job_n: int, mc_n: int, instance_i: int) -> int:
    file_path = "../result/basic/bench_opt.csv"
    if not os.path.exists(file_path):
        file_path = "./result/basic/bench_opt.csv"

    opt_data = pd.read_csv(file_path)
    opt_data.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'makespan', 'convergence', 'UB', 'name', 'set']
    opt_data['UB'] = opt_data['UB'].astype('int')

    return opt_data.loc[(opt_data['benchmark'] == benchmark) & (opt_data['job_n'] == job_n)
                        & (opt_data['mc_n'] == mc_n) & (opt_data['instance_i'] == instance_i)].iloc[0]['UB']


def get_load_result_data(test_csv_file_name='./basic/bench_ours.csv') -> pd.DataFrame:
    # opt_module result data #########################################################
    result_opt = pd.read_csv("./basic/bench_opt.csv")
    result_opt.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i',
                          'opt_makespan', 'convergence', 'UB', 'name', 'set']
    result_opt['problem'] = result_opt['benchmark'] + ' ' + result_opt['job_n'].astype('string') + 'x' \
                            + result_opt['mc_n'].astype('string')
    result_opt = result_opt.drop(['opt_makespan', 'convergence', 'set'], axis=1)

    # baseline data ####################################################################
    result_ref = pd.read_csv("./basic/bench_ref.csv")
    result_ref.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'name',
                          'makespan', 'method']
    result_ref['problem'] = result_ref['benchmark'] + ' ' + result_ref['job_n'].astype('string') + 'x' \
                            + result_ref['mc_n'].astype('string')
    result_ref = result_ref.drop(['benchmark', 'job_n', 'mc_n', 'name'], axis=1)

    # cp data #######################################################################
    result_cp = pd.read_csv("./basic/bench_cp.csv")
    result_cp.columns = ['time_limit', 'benchmark', 'job_n', 'mc_n', 'instance_i',
                         'makespan', 'time', 'convergence']
    result_cp['problem'] = result_cp['benchmark'] + ' ' + result_cp['job_n'].astype('string') + 'x' \
                           + result_cp['mc_n'].astype('string')
    result_cp['method'] = 'CP (' + result_cp['time_limit'].astype('string') + ' sec)'
    result_cp = result_cp.drop(['time_limit', 'benchmark', 'job_n', 'mc_n', 'time', 'convergence'], axis=1)

    # our data #####################################################################
    result_ours = pd.read_csv("./basic/bench_ours.csv")
    result_ours.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time', 'mean_decision_t', 'decision_n']
    result_ours['problem'] = result_ours['benchmark'] + ' ' + result_ours['job_n'].astype('string') + 'x' \
                             + result_ours['mc_n'].astype('string')
    result_ours = result_ours.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time', 'mean_decision_t', 'decision_n'], axis=1)

    # rule data #####################################################################
    result_rule = pd.read_csv("./basic/bench_rule.csv")
    result_rule.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time']
    result_rule['problem'] = result_rule['benchmark'] + ' ' + result_rule['job_n'].astype('string') + 'x' \
                             + result_rule['mc_n'].astype('string')
    result_rule = result_rule[result_rule['action_type'] == 'single_mc_buffer']
    result_rule = result_rule.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time'], axis=1)

    # result + opt ##################################################################
    result_opt2 = result_opt.drop(['benchmark', 'job_n', 'mc_n', 'UB', 'name'], axis=1)
    result_opt2['makespan'] = result_opt['UB']
    result_opt2['method'] = 'OPT'

    result_data = pd.concat([result_ours, result_ref, result_rule, result_cp, result_opt2])
    # result_data = pd.concat([result_ref, result_rule,  result_opt2, result_ours])
    merge_data = pd.merge(result_data, result_opt, on=['problem', 'instance_i'], how="left")

    merge_data = merge_data.dropna(subset=['UB'])
    merge_data['opt_gap'] = (merge_data['makespan'] - merge_data['UB']) / merge_data['UB']
    if min(merge_data['opt_gap']) < 0:
        print("error: opt_module gap error ----------------------------------------------")

    merge_data = merge_data.drop(['makespan', 'UB'], axis=1)
    merge_data['job_n'] = merge_data['job_n'].astype('int')
    merge_data['mc_n'] = merge_data['mc_n'].astype('int')

    return merge_data


def get_load_result_data2(test_csv_file_name='') -> pd.DataFrame:
    # opt_module result data #########################################################
    result_opt = pd.read_csv("./basic/bench_opt.csv")
    result_opt.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i',
                          'opt_makespan', 'convergence', 'UB', 'name', 'set']
    result_opt['problem'] = result_opt['benchmark'] + ' ' + result_opt['job_n'].astype('string') + 'x' \
                            + result_opt['mc_n'].astype('string')
    result_opt = result_opt.drop(['opt_makespan', 'convergence', 'set'], axis=1)

    # baseline data ####################################################################
    result_ref = pd.read_csv("./basic/bench_ref.csv")
    result_ref.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'name',
                          'makespan', 'method']
    result_ref['problem'] = result_ref['benchmark'] + ' ' + result_ref['job_n'].astype('string') + 'x' \
                            + result_ref['mc_n'].astype('string')
    result_ref = result_ref.drop(['benchmark', 'job_n', 'mc_n', 'name'], axis=1)

    # cp data #######################################################################
    # result_cp = pd.read_csv("./basic/bench_cp.csv")
    # result_cp.columns = ['time_limit', 'benchmark', 'job_n', 'mc_n', 'instance_i',
    #                      'makespan', 'time', 'convergence']
    # result_cp['problem'] = result_cp['benchmark'] + ' ' + result_cp['job_n'].astype('string') + 'x' \
    #                        + result_cp['mc_n'].astype('string')
    # result_cp['method'] = 'CP (' + result_cp['time_limit'].astype('string') + ' sec)'
    # result_cp = result_cp.drop(['time_limit', 'benchmark', 'job_n', 'mc_n', 'time', 'convergence'], axis=1)

    # our data #####################################################################
    # result_ours = pd.read_csv("./basic/bench_ours.csv")
    # result_ours.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
    #                        'method', 'makespan', 'time', 'mean_decision_t', 'decision_n']
    # result_ours['problem'] = result_ours['benchmark'] + ' ' + result_ours['job_n'].astype('string') + 'x' \
    #                          + result_ours['mc_n'].astype('string')
    # result_ours = result_ours.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
    #                                 'time', 'mean_decision_t', 'decision_n'], axis=1)

    # rule data #####################################################################
    result_rule = pd.read_csv("./basic/bench_rule.csv")
    result_rule.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time']
    result_rule['problem'] = result_rule['benchmark'] + ' ' + result_rule['job_n'].astype('string') + 'x' \
                             + result_rule['mc_n'].astype('string')
    result_rule = result_rule[result_rule['action_type'] == 'single_mc_buffer']
    result_rule = result_rule.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'time'], axis=1)

    # rule data #####################################################################
    result_test = pd.read_csv(test_csv_file_name, header=None)
    try:
        result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                               'method', 'rl', 'decay', 'loss',
                               'makespan', 'time', 'decision_n', 'model_i']
        result_test['dyn_type'] = ''
        result_test['parameter'] = ''
    except:
        result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                               'method', 'rl', 'decay', 'loss',
                               'makespan', 'time', 'decision_n', 'model_i', 'dyn_type', 'parameter']

    result_test['problem'] = result_test['benchmark'] + ' ' + result_test['job_n'].astype('string') + 'x' \
                             + result_test['mc_n'].astype('string')
    result_test['method'] = result_test['method'] + result_test['model_i'].astype('string')
    result_test = result_test.drop(['benchmark', 'job_n', 'mc_n', 'agent_type', 'action_type', 'state_type',
                                    'rl', 'decay', 'loss',
                                    'time', 'decision_n', 'model_i'], axis=1)  # , 'dyn_type', 'parameter'

    # result + opt ##################################################################
    result_opt2 = result_opt.drop(['benchmark', 'job_n', 'mc_n', 'UB', 'name'], axis=1)
    result_opt2['makespan'] = result_opt['UB']
    result_opt2['method'] = 'OPT'

    # result_data = pd.concat([result_ours, result_ref, result_rule, result_cp, result_opt2])
    result_data = pd.concat([result_ref, result_rule,  result_opt2, result_test])
    merge_data = pd.merge(result_data, result_opt, on=['problem', 'instance_i'], how="left")

    merge_data = merge_data.dropna(subset=['UB'])
    merge_data['opt_gap'] = (merge_data['makespan'] - merge_data['UB']) / merge_data['UB']
    if min(merge_data['opt_gap']) < 0:
        print("error: opt_module gap error ----------------------------------------------")

    merge_data = merge_data.drop(['makespan', 'UB'], axis=1)
    merge_data['job_n'] = merge_data['job_n'].astype('int')
    merge_data['mc_n'] = merge_data['mc_n'].astype('int')

    return merge_data


def get_load_result_data_ablation(test_csv_files=[]) -> pd.DataFrame:
    # opt_module result data #########################################################
    result_opt = pd.read_csv("./basic/bench_opt.csv")
    result_opt.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i',
                          'opt_makespan', 'convergence', 'UB', 'name', 'set']
    result_opt['problem'] = result_opt['benchmark'] + ' ' + result_opt['job_n'].astype('string') + 'x' \
                            + result_opt['mc_n'].astype('string')
    result_opt = result_opt.drop(['opt_makespan', 'convergence', 'set'], axis=1)

    # ours result data #########################################################
    result_ours = pd.read_csv("./basic/bench_ours.csv")
    result_ours.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type', 'state_type',
                           'method', 'makespan', 'time', 'unit_t', 'decision_n']
    result_ours['problem'] = result_ours['benchmark'] + ' ' + result_ours['job_n'].astype('string') + 'x' \
                             + result_ours['mc_n'].astype('string')
    result_ours = result_ours.drop(['benchmark', 'job_n', 'mc_n', 'agent_type',
                                    'time', 'unit_t', 'decision_n', 'method'], axis=1)
    result_ours['env_type'] = 'dyn'
    result_ours['model_type'] = 'type_all_pred_GAT2_mean_global'
    result_ours['model_i'] = -1

    # rule data #####################################################################
    def get_data(test_csv_file_name):
        result_test = pd.read_csv(test_csv_file_name, header=None)
        try:
            result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type',
                                   'state_type',
                                   'method', 'rl', 'decay', 'loss',
                                   'makespan', 'time', 'decision_n', 'model_i',
                                   'env_type', 'model_type', 'dyn_type', 'parameter']
        except:
            try:
                result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type',
                                       'state_type',
                                       'method', 'rl', 'decay', 'loss',
                                       'makespan', 'time', 'decision_n', 'model_i', 'dyn_type', 'parameter']
                result_test['env_type'] = ''
                result_test['model_type'] = ''
            except:
                result_test.columns = ['benchmark', 'job_n', 'mc_n', 'instance_i', 'agent_type', 'action_type',
                                       'state_type',
                                       'method', 'rl', 'decay', 'loss',
                                       'makespan', 'time', 'decision_n', 'model_i']
                result_test['dyn_type'] = ''
                result_test['parameter'] = ''
                result_test['env_type'] = ''
                result_test['model_type'] = ''

        result_test['problem'] = result_test['benchmark'] + ' ' + result_test['job_n'].astype('string') + 'x' \
                                 + result_test['mc_n'].astype('string')
        result_test = result_test.drop(['benchmark', 'job_n', 'mc_n', 'agent_type',
                                        'rl', 'decay', 'loss', 'dyn_type', 'parameter',
                                        'time', 'decision_n', 'method'], axis=1)  # , 'dyn_type', 'parameter'
        return result_test

    data = list()
    for test_csv_file_name in test_csv_files:
        data.append(get_data(test_csv_file_name))
    result_data = pd.concat(data + [result_ours])

    # result_data['method'] = result_data['method'] + result_data['model_i'].astype('string')
    # result_data['method'] = ''

    # result + opt ##################################################################
    merge_data = pd.merge(result_data, result_opt, on=['problem', 'instance_i'], how="left")

    merge_data = merge_data.dropna(subset=['UB'])
    merge_data['opt_gap'] = (merge_data['makespan'] - merge_data['UB']) / merge_data['UB']
    if min(merge_data['opt_gap']) < 0:
        print("error: opt_module gap error ----------------------------------------------")

    merge_data = merge_data.drop(['makespan', 'UB'], axis=1)
    merge_data['job_n'] = merge_data['job_n'].astype('int')
    merge_data['mc_n'] = merge_data['mc_n'].astype('int')

    return merge_data


# load data ##############################################################################################
def get_x_dim():
    """
    get feature dimension
    """
    from environment.env import JobShopEnv

    rollout_type = configs.rollout_type
    configs.rollout_type = 'model'

    sample_env = JobShopEnv([('HUN', 4, 3, 0)], pomo_n=1)
    obs, _, _ = sample_env.reset()

    in_dim_op = obs['op'].x.shape[1]
    if 'mc_node' in configs.state_type:
        in_dim_rsc = obs['rsc'].x.shape[1]
    else:
        in_dim_rsc = 0

    configs.rollout_type = rollout_type
    return in_dim_op, in_dim_rsc


# env ##############################################################################################
def get_env(problem_set: list, pomo_n: int=1):
    from environment.env import JobShopEnv

    env_list = list()
    for benchmark, job_n, mc_n, instances in problem_set:
        env_list += [(benchmark, job_n, mc_n, instance_i) for instance_i in instances]

    return JobShopEnv(env_list, pomo_n=pomo_n)


# total 242 instances ####################################################################################
TA = [['TA', 15, 15, list(range(10))],
      ['TA', 20, 15, list(range(10))], ['TA', 20, 20, list(range(10))],
      ['TA', 30, 15, list(range(10))], ['TA', 30, 20, list(range(10))],
      ['TA', 50, 15, list(range(10))], ['TA', 50, 20, list(range(10))],
      ['TA', 100, 20, list(range(10))], ]  # 80
TA_small = [['TA', 15, 15, list(range(10))],
      ['TA', 20, 15, list(range(10))], ['TA', 20, 20, list(range(10))],
      ['TA', 30, 15, list(range(10))], ['TA', 30, 20, list(range(10))], ]
LA = [['LA', 10, 5, list(range(5))], ['LA', 15, 5, list(range(5))], ['LA', 20, 5, list(range(5))],
      ['LA', 10, 10, list(range(5))], ['LA', 15, 10, list(range(5))], ['LA', 20, 10, list(range(5))],
      ['LA', 30, 10, list(range(5))], ['LA', 15, 15, list(range(5))], ]  # 40
DMU = [['DMU', 20, 15, list(range(10))], ['DMU', 20, 20, list(range(10))],
       ['DMU', 30, 15, list(range(10))], ['DMU', 30, 20, list(range(10))],
       ['DMU', 40, 15, list(range(10))], ['DMU', 40, 20, list(range(10))],
       ['DMU', 50, 15, list(range(10))], ['DMU', 50, 20, list(range(10))], ]  # 80
SWV = [['SWV', 20, 10, list(range(5))], ['SWV', 20, 15, list(range(5))], ['SWV', 50, 10, list(range(10))], ]  # 20
ABZ = [['ABZ', 10, 10, list(range(2))], ['ABZ', 20, 15, list(range(3))], ]  # 5
ORB = [['ORB', 10, 10, list(range(10))], ]  # 10
YN = [['YN', 20, 20, list(range(4))]]  # 4
FT = [['FT', 6, 6, list(range(1))], ['FT', 10, 10, list(range(1))], ['FT', 20, 5, list(range(1))], ]  # 3


all_benchmarks = TA + LA + ABZ + FT + ORB + SWV + YN + DMU


# total 86 instances ####################################################################################
TA_dyn = [['TA', 50, 15, list(range(10))], ['TA', 50, 20, list(range(10))], ['TA', 100, 20, list(range(10))]]  # 30
LA_dyn = [['LA', 15, 5, list(range(5))], ['LA', 20, 5, list(range(5))], ['LA', 30, 10, list(range(5))]]  # 15
DMU_dyn = [['DMU', 40, 15, list(range(5))], ['DMU', 50, 15, list(range(5))], ['DMU', 50, 20, list(range(5))],
           ['DMU', 40, 15, list(range(5, 10))], ['DMU', 50, 15, list(range(5, 10))], ['DMU', 50, 20, list(range(5, 10))],
           ]  # 30
SWV_dyn = [['SWV', 50, 10, list(range(10))]]  # 10
FT_dyn = [['FT', 20, 5, list(range(1))]]  # 1

all_dyn_benchmarks = DMU_dyn + TA_dyn + LA_dyn + SWV_dyn + FT_dyn

# real distribution ####################################################################################
REAL = [['REAL', 100, 20, list(range(40))], ['REAL', 100, 50, list(range(40))],
        ['REAL', 300, 20, list(range(40))], ['REAL', 300, 50, list(range(40))]]
REAL_D = [['REAL_D', 200, 20, list(range(10))], ['REAL_D', 300, 20, list(range(10))]]
REAL_D2 = [['REAL_D', 200, 20, list(range(10, 40))], ['REAL_D', 300, 20, list(range(10, 40))]]

FLOW = [['FLOW', 10, 5, list(range(10))], ['FLOW', 10, 10, list(range(10))],
        ['FLOW', 20, 5, list(range(10))], ['FLOW', 20, 10, list(range(10))],
        ['FLOW', 50, 5, list(range(10))], ['FLOW', 50, 10, list(range(10))]]

# total 800 instances ####################################################################################
HUN_40 = [['HUN', 6, 4, list(range(200))], ['HUN', 6, 6, list(range(200))],
       ['HUN', 8, 4, list(range(200))], ['HUN', 8, 6, list(range(200))], ]  # 800

list_30 = [i for i in range(200) if i % 40 < 30]
HUN_30 = [['HUN', 6, 4, list_30], ['HUN', 6, 6, list_30],
       ['HUN', 8, 4, list_30], ['HUN', 8, 6, list_30], ]  # 600

list_20 = [i for i in range(200) if i % 40 < 20]
HUN_20 = [['HUN', 6, 4, list_20], ['HUN', 6, 6, list_20],
       ['HUN', 8, 4, list_20], ['HUN', 8, 6, list_20], ]  # 400

list_10 = [i for i in range(200) if i % 40 < 10]
HUN_10 = [['HUN', 6, 4, list_10], ['HUN', 6, 6, list_10],
       ['HUN', 8, 4, list_10], ['HUN', 8, 6, list_10], ]  # 200

list_5 = [i for i in range(200) if i % 40 < 5]
HUN_5 = [['HUN', 6, 4, list_5], ['HUN', 6, 6, list_5],
       ['HUN', 8, 4, list_5], ['HUN', 8, 6, list_5], ]  # 100

list_2 = [i for i in range(200) if i % 40 < 2]
HUN_2 = [['HUN', 6, 4, list_2], ['HUN', 6, 6, list_2],
       ['HUN', 8, 4, list_2], ['HUN', 8, 6, list_2], ]  # 40

list_1 = [i for i in range(200) if i % 40 < 1]
HUN_1 = [['HUN', 6, 4, list_1], ['HUN', 6, 6, list_1],
         ['HUN', 8, 4, list_1], ['HUN', 8, 6, list_1], ]  # 40

list_60 = list(range(200)) + [i for i in range(200, 400) if i % 40 < 20]
HUN_60 = [['HUN', 6, 4, list_60], ['HUN', 6, 6, list_60],
       ['HUN', 8, 4, list_60], ['HUN', 8, 6, list_60], ]  # 300

list_80 = list(range(400))
HUN_80 = [['HUN', 6, 4, list_80], ['HUN', 6, 6, list_80],
          ['HUN', 8, 4, list_80], ['HUN', 8, 6, list_80], ]  # 400

list_100 = list(range(400)) + [i for i in range(400, 600) if i % 40 < 20]
HUN_100 = [['HUN', 6, 4, list_100], ['HUN', 6, 6, list_100],
          ['HUN', 8, 4, list_100], ['HUN', 8, 6, list_100], ]  # 500

list_120 = list(range(600))
HUN_120 = [['HUN', 6, 4, list_120], ['HUN', 6, 6, list_120],
          ['HUN', 8, 4, list_120], ['HUN', 8, 6, list_120], ]  # 600

list_140 = list(range(600)) + [i for i in range(600, 800) if i % 40 < 20]
HUN_140 = [['HUN', 6, 4, list_140], ['HUN', 6, 6, list_140],
          ['HUN', 8, 4, list_140], ['HUN', 8, 6, list_140], ]  # 700

list_160 = list(range(800))
HUN_160 = [['HUN', 6, 4, list_160], ['HUN', 6, 6, list_160],
          ['HUN', 8, 4, list_160], ['HUN', 8, 6, list_160], ]  # 800

list_180 = list(range(800)) + [i for i in range(800, 1000) if i % 40 < 20]
HUN_180 = [['HUN', 6, 4, list_180], ['HUN', 6, 6, list_180],
          ['HUN', 8, 4, list_180], ['HUN', 8, 6, list_180], ]  # 900

list_200 = list(range(1000))
HUN_200 = [['HUN', 6, 4, list_200], ['HUN', 6, 6, list_200],
          ['HUN', 8, 4, list_200], ['HUN', 8, 6, list_200], ]  # 1000

# dispatching rules ######################################################################################
# all_rules = ['LTT', 'MOR', 'FDD/MWKR', 'LRPT', 'SPT', 'STT', 'SRPT', 'LOR', 'LPT']
all_rules = ['LTT', 'MOR', 'FDD/MWKR', 'LRPT', 'SPT']

# action_types = ['conflict', 'buffer', 'buffer_being']
action_types = ['conflict', 'buffer']
