import copy, os
from opt.env2 import JobShopGenEnv
from params import configs
import pickle, torch
from environment import visualize
from itertools import permutations
from utils import load_opt_sol


def get_opt_action_set(env, mcs, opt_sol):
    action_set = list()
    assigns = list()
    for mc in mcs:
        job_i = opt_sol[mc][0]
        a = search_node_index(env, mc, job_i)
        action_set.append(a)
        assigns.append((mc, job_i))

    return action_set, assigns


def search_node_index(env, mc, job_i):
    for node in env.js.node_op_map.keys():
        if node[0] == mc and node[1] == job_i:
            return env.js.node_op_map[node]
    print("none matching node")
    return -1


def remove_assign_pairs(opt_sol, assign_pairs):
    no_matching_TF = False
    for (mc, job_idx) in assign_pairs:
        job_i = opt_sol[mc].pop(0)
        if job_i != job_idx:
            # print("no matching: mc {}, job_idx {} job_i {}".format(mc, job_idx, job_i))
            no_matching_TF = True
            break

    no_opt_sol = True
    for stage_sol in opt_sol:
        if stage_sol:
            no_opt_sol = False
            break

    return opt_sol, no_matching_TF or no_opt_sol


def get_policy(actions, mask):
    """
    target policy generation
    """
    policy = torch.zeros_like(mask).to(torch.float32)
    for action, (_, _) in actions:
        policy[action] = 1
    return policy / policy.sum()


def check_pairs_in_list(new_list, total_list):
    for (_, remain_pairs) in total_list:
        if new_list == remain_pairs:
            return True
    return False


def gen_opt_data(action_set, env, assigns):
    data = list()

    action_pairs = [(action, (mc, job_id)) for action, (mc, job_id) in zip(action_set, assigns)]
    action_pairs = sorted(action_pairs, key=lambda x: x[1][0])  # mc_idx로 sorting

    # env_remain_pairs = [(env, action_pairs)]

    def is_all_component_in(test_path, passed_paths):
        for passed_path in passed_paths:
            if len(test_path) != len(passed_path):
                continue
            component_in_TF = True
            for comp in test_path:
                if comp not in passed_path:
                    component_in_TF = False
                    break
            if component_in_TF:
                return True

        return False

    def is_single_mc_already_search(test_path, passed_paths):
        for passed_path in passed_paths:
            if len(test_path) != len(passed_path) or test_path[-1] != passed_path[-1]:
                continue
            component_in_TF = True
            for comp in test_path:
                if comp not in passed_path:
                    component_in_TF = False
                    break
            if component_in_TF:
                return True

        return False

    if not configs.policy_symmetric_TF:
        env_copy = copy.deepcopy(env)
        obs = env_copy.get_obs()
        for a, (mc, job_i) in action_pairs:
            # 하나의 remain action pair를 고려해서 policy 생성
            env_copy.target_mc = mc
            policy = get_policy([(a, (mc, job_i))], obs['op'].mask)
            obs.target_policy = policy
            data.append(obs)

            obs, _, _, _, _ = env_copy.step(a)

    else:
        if not configs.policy_total_mc:
            all_action_pairs = list(permutations(action_pairs))
            already_search_paris = list()
            already_search_remain_paris = list()
            for action_pairs in all_action_pairs:
                env_copy = copy.deepcopy(env)
                obs = env_copy.get_obs()
                for i, (a, (mc, job_i)) in enumerate(action_pairs):
                    # 남은 path 가 이미 탐색 했었다면 이미 지난 state 이므로 더이상 탐색 x
                    # 개수가 4개 이상일 때부터 의미
                    if action_pairs[i:] in already_search_remain_paris:
                        break

                    # 하나의 remain action pair를 고려해서 policy 생성
                    env_copy.target_mc = mc

                    # 지나온 것이 동일하다면, 이미 해당 state 대해서 target policy 가 생성됨
                    if not is_single_mc_already_search(action_pairs[:i + 1], already_search_paris):
                        policy = get_policy([(a, (mc, job_i))], obs['op'].mask)
                        obs.target_policy = policy
                        data.append(obs)
                        already_search_remain_paris.append(action_pairs[i:])
                        already_search_paris.append(action_pairs[:i + 1])

                    obs, _, _, _, _ = env_copy.step(a)
        else:
            all_action_pairs = list(permutations(action_pairs))
            already_search_paris = list()
            already_search_remain_paris = list()
            for action_pairs in all_action_pairs:
                env_copy = copy.deepcopy(env)
                obs = env_copy.get_obs()
                for i, (a, (mc, job_i)) in enumerate(action_pairs):
                    # 남은 path 가 이미 탐색 했었다면 이미 지난 state 이므로 더이상 탐색 x
                    # 개수가 4개 이상일 때부터 의미
                    if action_pairs[i:] in already_search_remain_paris:
                        break

                    # 하나의 remain action pair를 고려해서 policy 생성
                    env_copy.target_mc = mc

                    # total_mc 에서 지나온 것이 동일하다면, 이미 해당 state 대해서 target policy 가 생성됨
                    if not is_all_component_in(action_pairs[:i], already_search_paris):
                        policy = get_policy([(a, (mc, job_i))], obs['op'].mask)
                        obs.target_policy = policy
                        data.append(obs)
                        already_search_remain_paris.append(action_pairs[i:])
                        already_search_paris.append(action_pairs[:i])

                    obs, _, _, _, _ = env_copy.step(a)

        #
        #     env_copy = copy.deepcopy(env)
        #     total_mc_opt_data_gen(env_copy, action_pairs)
        #
        #     # 모든 remain action pairs를 고려해서 policy 생성
        #     obs = env.get_obs()
        #     policy = get_policy(remain_pairs, obs['op'].mask)
        #     obs.target_policy = policy
        #     data.append(obs)
        #
        #     # next state 생성
        #     for (a, (mc, job_i)) in remain_pairs:
        #         new_remain_pairs = copy.deepcopy(remain_pairs)
        #         new_remain_pairs.remove((a, (mc, job_i)))
        #
        #         if not check_pairs_in_list(new_remain_pairs, new_env_remain_pairs):  # 새로운 state이면 추가
        #             env_copy = copy.deepcopy(env)
        #             _, _, _, _, _ = env_copy.step(a)
        #             new_env_remain_pairs.append((env_copy, new_remain_pairs))
        #
        #
        # # action i+1 개 선택 #######################################
        # new_env_remain_pairs = list()
        # for (env, remain_pairs) in env_remain_pairs:
        #
        #     # 모든 actions 동시에 고려 #####################################
        #     if configs.policy_symmetric_TF:
        #
        #         # state 별로 하나의 policy 생성 ######
        #         if configs.policy_total_mc:
        #             # 모든 remain action pairs를 고려해서 policy 생성
        #             obs = env.get_obs()
        #             policy = get_policy(remain_pairs, obs['op'].mask)
        #             obs.target_policy = policy
        #             data.append(obs)
        #
        #             # next state 생성
        #             for (a, (mc, job_i)) in remain_pairs:
        #                 new_remain_pairs = copy.deepcopy(remain_pairs)
        #                 new_remain_pairs.remove((a, (mc, job_i)))
        #
        #                 if not check_pairs_in_list(new_remain_pairs, new_env_remain_pairs):  # 새로운 state이면 추가
        #                     env_copy = copy.deepcopy(env)
        #                     _, _, _, _, _ = env_copy.step(a)
        #                     new_env_remain_pairs.append((env_copy, new_remain_pairs))
        #
        #         # edge 별로 하나의 policy 생성 ######
        #         else:
        #             for (a, (mc, job_i)) in remain_pairs:
        #                 env_copy = copy.deepcopy(env)
        #                 env_copy.target_mc = mc
        #
        #                 # 하나의 remain action pair를 고려해서 policy 생성
        #                 obs = env_copy.get_obs()
        #                 policy = get_policy([(a, (mc, job_i))], obs['op'].mask)
        #                 obs.target_policy = policy
        #                 data.append(obs)
        #
        #                 # next state 생성
        #                 new_remain_pairs = copy.deepcopy(remain_pairs)
        #                 new_remain_pairs.remove((a, (mc, job_i)))
        #
        #                 if not check_pairs_in_list(new_remain_pairs, new_env_remain_pairs):  # 새로운 state이면 추가
        #                     _, _, _, _, _ = env_copy.step(a)
        #                     new_env_remain_pairs.append((env_copy, new_remain_pairs))

        # env_remain_pairs = new_env_remain_pairs
        # end: action i+1 개 선택 #######################################

    return data



if __name__ == "__main__":
    from utils import get_opt_makespan_once, HUN, all_benchmarks

    configs.agent_type = 'rule'
    configs.env_type = ''  # 'dyn', ''
    ######################################################################################################
    for configs.action_type in ['single_mc_buffer', 'single_mc_buffer_being', 'conflict', 'single_mc_conflict']:

        for configs.target_sol_type in ['full_active', 'active', '']:  # , 'full_active', 'active', 'active'

            print("\n=============", configs.action_set_type, configs.target_sol_type, "============================")

            save_folder = f'./../opt_policy/{configs.action_set_type} {configs.target_sol_type}'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for (benchmark, job_n, mc_n, instance_n) in HUN:
                problem = str(benchmark) + str(job_n) + "x" + str(mc_n)
                data_path = './../bench_data/' + benchmark + '/' + problem

                print("\n--------", benchmark, job_n, mc_n, "--------------")

                for configs.state_type in ['simple']:  # 'norm', '', 'add_norm']:  # 'simple', 'add1', 'add2', 'basic'

                    for configs.policy_symmetric_TF in [False, True]:  # True, False

                        opt_data_type = '{} {} {} {}'.format(
                            configs.env_type, configs.state_type, configs.action_type,
                            configs.policy_symmetric_TF,
                        )

                        save_path = save_folder + '/' + opt_data_type
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_file = save_path + '/{}.p'.format(problem)

                        # data gen #########################################################################
                        opt_data = list()
                        pass_idxs = list()
                        for instance_i in range(instance_n):
                            opt_sol = load_opt_sol(benchmark, job_n, mc_n, instance_i,
                                                   sol_type=configs.target_sol_type)
                            if not opt_sol:
                                continue

                            opt_makespn = get_opt_makespan_once(benchmark, job_n, mc_n, instance_i)
                            env = JobShopGenEnv(benchmark, job_n, mc_n, instance_i)

                            # repeat ##############################################
                            obs, reward, done, assign_pairs, mcs = env.reset()  # mcs: mc_conflict_ops.keys()
                            opt_sol, done2 = remove_assign_pairs(opt_sol, assign_pairs)  # 할당된거 제거
                            done = done or done2

                            cum_r = reward
                            while not done:
                                a_set, assigns = get_opt_action_set(env, mcs, opt_sol)  # assigns: [(mc, job_i)]

                                if configs.policy_total_mc:
                                    if obs['op'].mask[a_set].sum().item() != len(a_set):
                                        cum_r = 0
                                        break
                                else:
                                    if obs['op'].mask[a_set].sum().item() != 1:
                                        cum_r = 0
                                        break

                                    # action 후보에 없는데 opt_sol에서 action_set으로 뽑힘 -> 종료
                                    # 설비의 맨 앞에 남은 job_i가 동일한 경우, opt_sol()에서 같이 나올 수 있음
                                    no_action = False
                                    for mc, job_i in assigns:
                                        no_action = True
                                        for (_, job_i2, _) in env.mc_conflict_ops[mc]:
                                            if job_i2 == job_i:
                                                no_action = False
                                                break
                                        if no_action:  # action 후보에 없는데 opt_sol에서 action_set으로 뽑힘 -> 종료
                                            break
                                    if no_action:  # action 후보에 없는데 opt_sol에서 action_set으로 뽑힘 -> 종료
                                        break

                                # policy gen #####
                                # if len(a_set) >= 2:  # 3개: hun 6x3_11 / 2개: hun 4x2_1
                                #     print('{} actions'.format(len(a_set)))
                                partial_opt_data = gen_opt_data(a_set, env, assigns)
                                # if len(a_set) >= 2:
                                #     print('{} opt data '.format(len(partial_opt_data)))

                                # step env #####
                                total_assign_pairs = list()
                                for a, (mc, job_i) in zip(a_set, assigns):
                                    env.target_mc = mc
                                    obs, reward, done, assign_pairs, mcs = env.step(a)
                                    total_assign_pairs += assign_pairs

                                    opt_sol, done2 = remove_assign_pairs(opt_sol, assign_pairs)  # 할당된거 제거
                                    done = done or done2

                                # 문제 없으면 저장 #####
                                save_TF = True
                                if len(set(assigns) & set(total_assign_pairs)) != len(assigns):
                                    print("error: none executed")
                                    save_TF = False
                                    done = True  # while 종료

                                if save_TF:  # 문제 있기 전까지의 data는 저장tqdm
                                    opt_data += partial_opt_data
                                    cum_r = reward + configs.gamma * cum_r
                                else:
                                    # 지금까지 스케줄
                                    # visualize.get_gantt_plotly(env)
                                    break

                            # check feasibility: different obj -> MDP wrong! ##################
                            if opt_makespn != -cum_r:
                                pass_idxs.append(instance_i)
                                # visualize.get_gantt_plotly(env)

                            # 완성 스케줄
                            visualize.get_gantt_plotly(env)

                        # pass index #########################################################################
                        if len(pass_idxs) > 0:
                            print(opt_data_type)
                            print("pass_num: {}\tidx: {}".format(len(pass_idxs), pass_idxs))

                        print("opt_data num: {}".format(len(opt_data)))

                        # data save #########################################################################
                        with open(save_file, 'wb') as file:  # wb: binary write mode
                            pickle.dump(opt_data, file)