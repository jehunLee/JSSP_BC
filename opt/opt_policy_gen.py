from opt.gen_env import JobShopGenEnv
from params import configs

import pickle, torch
# from itertools import permutations
from utils import load_opt_sol, get_opt_data_path
from collections import defaultdict


def remove_assign_pairs(opt_mc_seq: tuple, assign_pairs: list) -> (tuple, list):
    """
    repairing optimal mc sequence
    """
    no_matching_TF = False

    assign_dict = defaultdict(list)
    for (mc, job_idx) in assign_pairs:
        assign_dict[mc].append(job_idx)
    for mc, job_list in assign_dict.items():
        for _ in range(len(job_list)):
            if opt_mc_seq[mc].pop(0) not in job_list:
                no_matching_TF = True
                break

    done = True
    for mc_sol in opt_mc_seq:
        if mc_sol:
            done = False
            break

    return opt_mc_seq, no_matching_TF or done


def get_opt_action_set(env: JobShopGenEnv, mcs: list, opt_mc_seq: tuple, op_mask) -> (list, list):
    """
    actions at the current time
    """
    action_set = list()
    assigns = list()

    for mc in mcs:
        job_i = opt_mc_seq[mc][0]
        node_i = env.max_mc_n * job_i + env.job_last_step[0, 0, job_i].item()
        # a = search_node_index(env, mc, job_i)

        # if 'dyn' in configs.env_type:
        #     a_ = torch.where(env.remain_op_idxs == a)[0].item()
        # else:
        #     a_ = a
        # if op_mask[a_]:
        #     action_set.append(a)
        #     assigns.append((mc, job_i))

        action_set.append(node_i)
        assigns.append((mc, job_i))

    return action_set, assigns


# def search_node_index(env: JobShopGenEnv, mc: int, job_i: int) -> int:
#     """
#     search node index
#     """
#     for node in env.js.node_op_map.keys():
#         if node[0] == mc and node[1] == job_i:
#             return env.js.node_op_map[node]
#     print("none matching node")
#     return -1


def get_policy(actions: list, mask: torch.tensor) -> torch.tensor:
    """
    target policy generation
    """
    policy = torch.zeros(mask.size()[0], dtype=torch.float32)
    for action, (_, _) in actions:
        policy[action] = 1
    return policy / policy.sum()


def gen_opt_data(env: JobShopGenEnv, action_set: list, assigns: list) -> list:
    """
    generate optimal decision data
    """
    data = list()

    action_pairs = [(action, (mc, job_id)) for action, (mc, job_id) in zip(action_set, assigns)]
    action_pairs = sorted(action_pairs, key=lambda x: x[1][0])  # sorting by mc_idx

    # gen opt data ################################################################
    if not configs.policy_symmetric_TF:
        env_copy = JobShopGenEnv(load_envs=[env])
        obs = env_copy.get_obs()

        for a, (mc, job_i) in action_pairs:
            env_copy.target_mc = torch.zeros(1, 1, env.max_mc_n, dtype=torch.long)
            env_copy.target_mc[0, 0, mc] = 1

            # if 'dyn' in configs.env_type:
            # skip때도 발생
            a_ = torch.where(obs['op_remain'] == a)[0].item()
            # zero_n = a - obs['op_remain'][:a][-1].item() + 1 - a
            # a_ = a - zero_n
            # else:
            #     a_ = a

            policy = get_policy([(a_, (mc, job_i))], obs.x_dict['op_mask'])
            obs.target_policy = policy
            data.append(obs)

            obs, _, _, _, _ = env_copy.step(a)

    # else:
    #     def is_all_component_in(test_path, passed_paths):
    #         for passed_path in passed_paths:
    #             if len(test_path) != len(passed_path):
    #                 continue
    #             if 'single_mc' in configs.action_type:
    #                 if test_path[-1] != passed_path[-1]:
    #                     continue
    #
    #             component_in_TF = True
    #             for comp in test_path:
    #                 if comp not in passed_path:
    #                     component_in_TF = False  # not in passed_path
    #                     break
    #
    #             if component_in_TF:  # all in passed_path
    #                 return True
    #
    #     all_action_pairs = list(permutations(action_pairs))
    #     already_search_paris = list()
    #
    #     # 2 action_pairs -> 4 (single_mc) or 3 (multiple_mc) states
    #     # 3 action_pairs -> 12 (single_mc) or 7 (multiple_mc) states
    #     # 4 action_pairs -> 32 (single_mc) or 15 (multiple_mc) states
    #     for action_pairs in all_action_pairs:
    #         env_copy = copy.deepcopy(env)
    #         obs = env_copy.get_obs()
    #         for i, (a, (mc, job_i)) in enumerate(action_pairs):
    #             env_copy.target_mc_i = mc
    #
    #             # same passed path -> a target policy is already generated for the state
    #             if 'single_mc' in configs.action_type:
    #                 pass_TF = is_all_component_in(action_pairs[:i+1], already_search_paris)
    #             else:
    #                 pass_TF = is_all_component_in(action_pairs[:i], already_search_paris)
    #
    #             if 'dyn' in configs.env_type:
    #                 a = torch.where(env_copy.remain_op_idxs == a)[0].item()
    #
    #             if not pass_TF:
    #                 if 'single_mc' in configs.action_type:
    #                     policy = get_policy([(a, (mc, job_i))], obs.x_dict['op_mask'])
    #                     already_search_paris.append(action_pairs[:i+1])  # same passed + current mc -> same policy
    #                 else:
    #                     policy = get_policy([(torch.where(env_copy.remain_op_idxs == a_)[0].item(), (mc, job_i))
    #                                           for (a_, (mc, job_i)) in action_pairs[i:]], obs.x_dict['op_mask'])
    #                     already_search_paris.append(action_pairs[:i])  # same passed -> same policy
    #
    #                 obs.target_policy = policy
    #                 data.append(obs)
    #
    #             obs, _, _, _, _ = env_copy.step(a)

    return data


if __name__ == '__main__':
    from utils import get_opt_makespan_once, HUN_100, HUN_140, HUN_120, HUN_180, HUN_200, HUN_160, HUN_2, HUN_5, HUN_10, HUN_20, HUN_30, HUN_40, HUN_60, HUN_80, HUN_100
    from tqdm import tqdm

    configs.agent_type = 'GNN'
    configs.state_type = 'mc_gap_mc_load_prt_norm'  #, 'mc_gap_mc_load_prt_norm', 'mc_load_norm', 'mc_gap_norm', 'basic_norm', 'simple_norm']  # 'simple_norm', 'basic_norm'
    configs.sol_type = 'full_active'
    configs.rollout_type = 'model'

    configs.model_type = 'all_pred'
    configs.env_type = 'dyn'  # dyn
    configs.policy_symmetric_TF = False  # True, False

    ######################################################################################################
    for configs.action_type in ['conflict']:  # action_types   all_buffer_being
        print(f'\naction: {configs.action_type}\tsol_type: {configs.sol_type} ============================')
        print(f'state: {configs.state_type}\tsymmetric: {configs.policy_symmetric_TF} ----------')

        for problem_set in [HUN_100]:  # , HUN_2]  HUN_60, HUN_40
            for (benchmark, job_n, mc_n, instances) in problem_set:
                # benchmark, job_n, mc_n, instances = 'HUN', 6, 4, [121]
                print(f'{benchmark} {job_n}x{mc_n} -------------')

                # data gen #########################################################################
                opt_data = list()
                pass_idxs = list()
                for instance_i in tqdm(instances):

                    opt_mc_seq, _ = load_opt_sol(benchmark, job_n, mc_n, instance_i, sol_type=configs.sol_type)
                    if not opt_mc_seq:
                        continue

                    opt_makespn = get_opt_makespan_once(benchmark, job_n, mc_n, instance_i)
                    env = JobShopGenEnv([(benchmark, job_n, mc_n, instance_i)])

                    # repeat ######################################################################
                    obs, reward, done, assign_pairs, mcs = env.reset()  # mcs: mc_conflict_ops.keys()
                    opt_mc_seq, done2 = remove_assign_pairs(opt_mc_seq, assign_pairs)  # remove assigned jobs
                    done = done or done2
                    cum_r = reward

                    while not done:
                        a_set, assigns = get_opt_action_set(env, mcs, opt_mc_seq, obs['op_mask'].x)  # assigns: [(mc, job_i)]
                        node_i = a_set[0]
                        if env.op_mask[0, 0, node_i].item() == 0:
                            cum_r = 0
                            break

                        # policy gen #################################################
                        partial_opt_data = gen_opt_data(env, a_set, assigns)

                        # step env #####
                        total_assign_pairs = list()
                        for a, (mc, job_i) in zip(a_set, assigns):

                            # if 'dyn' in configs.env_type:
                            #     a = torch.where(env.remain_op_idxs == a)[0].item()

                            obs, reward, done, assign_pairs, mcs = env.step(a)
                            total_assign_pairs += assign_pairs

                            # remove assigned operations
                            opt_mc_seq, done2 = remove_assign_pairs(opt_mc_seq, assign_pairs)
                            done = done or done2

                        # save #######################################################
                        save_TF = True
                        if len(set(assigns) & set(total_assign_pairs)) != len(assigns):
                            print("error: none executed")
                            save_TF = False
                            done = True  # end while

                        if save_TF:  # save opt data before error
                            opt_data += partial_opt_data
                            cum_r = reward
                        else:
                            break

                    # check feasibility: different obj -> MDP wrong! ##################
                    if cum_r == 0 or cum_r == None:
                        pass_idxs.append(instance_i)
                    else:
                        if opt_makespn != -cum_r[0, 0].item():
                            pass_idxs.append(instance_i)

                    # complete schedule ####################################
                    # env.show_gantt_plotly(0, 0)

                # pass index #########################################################################
                if len(pass_idxs) > 0:
                    print(f'pass_num: {len(pass_idxs)}\tidx: {pass_idxs}')
                #
                # print(f'opt_data num: {len(opt_data)}')

                # data save #########################################################################
                save_file = get_opt_data_path(benchmark, job_n, mc_n, instances)

                with open(save_file, 'wb') as file:  # wb: binary write mode
                    pickle.dump(opt_data, file)