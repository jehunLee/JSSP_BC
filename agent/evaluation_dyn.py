from utils import all_dyn_benchmarks, HUN_100, all_benchmarks
from params import configs
from agent_BC import AgentBC
import os


configs.agent_type = 'GNN_BC_policy'

##########################################################
# configs.dyn_reserve_reset = False

# configs.dyn_type = 'mc_breakdown_known'
# parameters = [100, 200, 300, 400]

configs.dyn_type = 'job_arrival'
parameters = [100, 200, 300, 400]
configs.init_batch = 2

# configs.dyn_type = 'prt_stochastic_known'
# parameters = [0.1, 0.2, 0.3, 0.4]

# test_set = all_dyn_benchmarks
test_set = all_benchmarks

save_folder = f'./../result/{configs.dyn_type}/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for data_set in [HUN_100]:  # HUN_2, HUN_5, HUN_10, HUN_20, HUN_30, HUN_40,
    configs.training_len = len(data_set[0][3])

    for configs.action_type in ['conflict']:  # 'conflict', 'buffer'
        for configs.parameter in parameters:
            for configs.dyn_reserve_reset in [True, False]:
                for i in [3]:  # 0, 1, 2,
                    agent = AgentBC(model_i=i)
                    agent.model_load()
                    save_path = save_folder + f'result_{configs.dyn_type}_{configs.training_len}_{configs.lr}_{configs.L2_norm_w}_{i}.csv'

                    agent.perform_model_benchmarks(test_set, save_path=save_path)
