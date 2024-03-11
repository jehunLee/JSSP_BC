from params import configs

from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_

from agent_GNN import AgentGNN
from utils import get_opt_data, get_env
from tqdm import tqdm
import torch


class AgentBC(AgentGNN):
    def __init__(self, model_i=0):
        super().__init__(model_i)

    # learning #####################################################################################
    def update_BC(self, batch, model, optimizer):
        """
        update weight of model
        """
        model.train()  # (dropout=True)
        optimizer.zero_grad()

        loss = model.loss(batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), configs.max_grad_norm)

        optimizer.step()

        return loss

    def learning_opt(self, valid_problem_set, opt_problem_set) -> None:
        """
        learn optimal policies
        """
        env_valid = get_env(valid_problem_set)
        losses = list()
        valids = list()

        # init valid ###############################################################
        reward, mean_run_t, _ = self.run_episode_model_once(env_valid, test_TF=True)
        reward = round(reward.to(torch.float).mean(dim=0).item(), 3)
        valids.append(reward)
        print(f'initial valid mean_cum_r: {reward}\tmean_run_t: {mean_run_t} sec')

        # repeat ##################################################################
        self.model_save()
        opt_data = get_opt_data(opt_problem_set)
        loader = DataLoader(opt_data, configs.batch_size, shuffle=True)  # https://github.com/pyg-team/pytorch_geometric/issues/2961 ######

        for epoch in tqdm(range(configs.max_ep_n)):
            total_loss = 0
            for batch in loader:
                loss = self.update_BC(batch, self.model, self.optimizer)
                # if loss.isnan().item():
                #     print('error: loss nan')
                total_loss += loss.item()

            losses.append(round(total_loss, 4))

            if self.scheduler:
                self.scheduler.step()

            # valid ######################
            mean_cum_r, mean_run_t, _ = self.run_episode_model_once(env_valid, test_TF=True)
            reward = round(mean_cum_r.to(torch.float).mean(dim=0).item(), 3)
            valids.append(reward)

            if max(valids) <= reward:
                self.model_save()
                print(f'-- epoch: {epoch} \ttotal loss: {round(total_loss, 1)} '
                      f'\tvalid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec '
                      f'\tbest valid mean_cum_r: {max(valids)} \tlearning rate: {self.get_current_lr()}')

            if (epoch + 1) % 100 == 0:
                print(f'-- epoch: {epoch} \ttotal loss: {round(total_loss, 1)} '
                      f'\tvalid mean_cum_r: {reward} \tmean_run_t: {mean_run_t} sec '
                      f'\tbest valid mean_cum_r: {max(valids)} \tlearning rate: {self.get_current_lr()}')

                self.save_training_process(valids, losses)
                # self.training_process_fig(valids, losses, save_TF=False)

        # save result #############################################################
        self.save_training_process(valids, losses)
        self.training_process_fig(valids, losses, save_TF=True)
        # from environment import visualize
        # visualize.get_gantt_plotly(envs_valid[0])


if __name__ == '__main__':
    from utils import HUN_1

    configs.agent_type = 'GNN_BC_policy'
    configs.env_type = 'dyn'  # 'dyn', ''
    configs.state_type = 'mc_gap_mc_load_prt_norm'  # 'basic_norm', 'simple_norm', 'norm', 'mc_load_norm'
    configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'
    # configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'

    valid_problem_set = [('TA', 15, 15, list(range(10)))]

    for data_set in [HUN_1]:  # HUN_2, HUN_5, HUN_10, HUN_20, HUN_30, HUN_40,
        configs.training_len = len(data_set[0][3])

        for configs.action_type in ['conflict']:  # action_types ['single_mc_conflict', 'conflict', 'buffer_being', 'single_mc_buffer', 'single_mc_buffer_being']
            for i in [0, 1, 2, 3, 4]:
                agent = AgentBC(model_i=i)
                agent.learning_opt(valid_problem_set, data_set)

                # traing_process = agent.get_training_process()
                # print(traing_process[0].index(max(traing_process[0])))

        # save_path = './../result/bench_test.csv'
        # agent.perform_benchmarks(all_benchmarks, save_path=save_path)

    # TA15x15 - OPT: 1228.9 / schedule_Net: 1417.2 / ICML: 1397.5 / ours: 1350.7

    # LA10x10 - OPT: 864.2 / schedule_Net: 966.8 / ours: 953.6 / ICML: 932.4
    # LA15x10 - OPT: 983.4 / schedule_Net: 1127.6 / ours: 1066.8 / ICML: 1123
    # LA15x15 - OPT: 1263.2 / schedule_Net: 1466.6 / ours: 1411.2 / ICML: 1400

    ###########################################################
    # import torch
    # from environment.visualize import show_gantt_plotly
    #
    # configs.action_type = 'single_mc_conflict'
    # from utils import get_envs
    #
    # ###########################################################
    # def main():
    #     agent = AgentBC(model_i=0)
    #     agent.model_load()
    #
    #     # env = get_envs([('TA', 50, 15, [0])])[0]
    #     env = get_envs([('HUN', 6, 6, [0])])[0]
    #
    #     agent.model.eval()
    #     with torch.no_grad():
    #         obs, reward, done = env.reset()
    #
    #         while not done:
    #             a_tensor = agent.get_action(obs)
    #             # print(obs['op_mask'].x.nonzero())
    #             # print(a_tensor)
    #             obs, reward, done = env.step(a_tensor.item())
    #     show_gantt_plotly(env.js)
    #
    # ###########################################################
    # import cProfile
    # # cProfile.run('main()')
    # main()

    ###########################################################
    # def print_top_stats(pr, num, sort):
    #     """
    #     sort: -1 or "time"
    #     """
    #     import pstats
    #     pstats.Stats(pr).strip_dirs().sort_stats(sort).print_stats(num)
    #
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # main()
    #
    # pr.disable()
    # print_top_stats(pr, num=30, sort='time')
