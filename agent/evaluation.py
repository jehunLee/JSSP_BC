from utils import all_benchmarks, HUN_100
from params import configs
from agent_BC import AgentBC

configs.agent_type = 'GNN_BC_policy'

for data_set in [HUN_100]:  # HUN_2, HUN_5, HUN_10, HUN_20, HUN_30, HUN_40,
    for j in range(5):

        # if j == 0:
        #     configs.state_type = 'mc_gap_mc_load'
        # elif j == 1:
        #     configs.state_type = 'prt_norm'
        # else:
        #     configs.state_type = 'mc_gap_mc_load_prt_norm'
        #
        # if j == 2:
        #     configs.env_type = ''
        # else:
        #     configs.env_type = 'dyn'
        #
        # if j == 3:
        #     configs.model_type = 'type_GAT2'
        # else:
        #     configs.model_type = 'type_all_pred_GAT2'
        #
        # if j == 4:
        #     configs.action_type = 'buffer'
        # else:
        #     configs.action_type = 'conflict'

        configs.training_len = len(data_set[0][3])

        for i in [0, 1, 2, 3, 4]:
            agent = AgentBC(model_i=i)
            agent.model_load()
            save_path = f'./../result/result_ablation_{configs.training_len}_{configs.lr}_{configs.L2_norm_w}_{i}.csv'

            agent.perform_model_benchmarks(all_benchmarks, save_path=save_path)

    # from environment.env import JobShopEnv
    # env = JobShopEnv('HUN', 6, 4, 0)
    # cum_r, run_t, _, _, _ = agent.run_episode(env, test_TF=True)
    # print(cum_r, run_t)
    # cum_r, run_t, _, _, _ = agent.run_episode(env, test_TF=True)
    # print(cum_r, run_t)

# tai15x15 - OPT_mean: 1228.9 / schedule_Net: 1417.2 / Park: 1476.2 / MOR: 1802.0 / LTT: 1657.4
# tai15x15 - / Informs: 1404.3, 1401.9 / ICML: 1397.5

# TA 15x15 - OPT_mean: 1228.9 / schedule_Net: 1417.2 / TASE: 1352.3
# TA 20x15 - OPT_mean: 1364.8 / schedule_Net: 1630.0 / TASE: 1508.2
# TA 20x20 - OPT_mean: 1617.5 / schedule_Net: 1896.0 / TASE: 1839.0
# TA 30x15 - OPT_mean: 1790.2 / schedule_Net: 2129.5 / TASE: 1983.0
# TA 30x20 - OPT_mean: 1948.8 / schedule_Net: 2411.3 / TASE: 2236.0
# TA 50x15 - OPT_mean: 2773.8 / schedule_Net: 3157.4 / TASE: 3004.5
# TA 50x20 - OPT_mean: 2843.9 / schedule_Net: 3229.4 / TASE: 3117.0
# TA 100x20 - OPT_mean: 5365.7 / schedule_Net: 5723.6 / TASE: 5562.9