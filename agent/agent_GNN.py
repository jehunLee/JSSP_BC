import copy

from params import configs
import pickle, torch, os, time

from torch.distributions import Categorical
from utils import get_x_dim
from network import GNN, Hetero_GNN_type_aware_all_pred, Hetero_GNN_type_aware


class AgentGNN():
    def __init__(self, model_i=0):
        self.model_i = model_i
        self.agent_type = configs.agent_type
        self.update_seed()

        # input dim, output dim ########################
        configs.in_dim_op, configs.in_dim_rsc = get_x_dim()

        # model generation ########################
        if 'multi_policy' in self.agent_type:
            self.model = self.get_model('multi_policy')
        elif 'policy' in self.agent_type:
            self.model = self.get_model('policy')
        elif 'actor_critic' in self.agent_type:
            self.model = self.get_model('actor_critic')
        elif 'value' in self.agent_type:
            self.model = self.get_model('value')
        else:
            print("Unknown agent_type.")

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    # model value ####################################################################################
    def get_action_model(self, obs, model=None):
        """
        get greedy action with the highest value
        """
        if not model:
            model = self.model
        probs = model(obs)

        actions = list()
        graph_index = obs['op'].batch
        for batch_i, prob in enumerate(probs):
            if not prob.size()[0]:
                actions.append(torch.tensor([0], dtype=torch.int64).to(configs.device))
                continue

            a_tensor = torch.argmax(prob).view(-1)
            indices = torch.where(graph_index == batch_i)
            actions.append(obs['op_remain'][indices][a_tensor])

        return actions

    def get_action_models(self, obs_list, models, env_n=1):
        """
        get greedy action with the highest value
        """
        actions = list()
        for j, obs in enumerate(obs_list):
            if len(obs['op_mask'].x) == 0:
                for _ in range(env_n):
                    actions.append(torch.tensor([0], dtype=torch.int64).to(configs.device))
                continue

            probs = models[j](obs)
            graph_index = obs['op'].batch
            for batch_i, prob in enumerate(probs):
                if not prob.size()[0]:
                    actions.append(torch.tensor([0], dtype=torch.int64).to(configs.device))
                    continue

                a_tensor = torch.argmax(prob).view(-1)
                indices = torch.where(graph_index == batch_i)
                actions.append(obs['op_remain'][indices][a_tensor])

        return actions

    def get_action_sample(self, obs, model=None) -> (list, torch.Tensor):
        """
        get an action according value
        """
        if not model:
            model = self.model
        probs = model(obs)

        actions = list()
        actions_ = list()
        log_probs = list()

        graph_index = obs['op'].batch
        for batch_i, prob in enumerate(probs):
            if not prob.size()[0]:
                actions.append(torch.tensor([0], dtype=torch.int64).to(configs.device))
                actions_.append(torch.tensor([None], dtype=torch.int64).to(configs.device))
                continue

            policy = Categorical(prob.view(1, -1))
            a_tensor = policy.sample()
            log_prob = policy.log_prob(a_tensor)
            log_probs.append(log_prob)

            indices = torch.where(graph_index == batch_i)
            actions.append(obs['op_remain'][indices][a_tensor])
            actions_.append(a_tensor)

        return actions, actions_, log_probs

    def log_prob(self, obs, a_tensor, model=None):
        """
        get log π(a|s) based on current policy
        """
        if not model:
            model = self.model

        prob = model(obs)
        policy = Categorical(prob.view(1, -1))
        log_prob = policy.log_prob(a_tensor)

        return log_prob

    # model ###########################################################################################
    def get_model(self, agent_type='policy'):
        """
        get GNN model <- configs.agent_type
        """
        out_dim = 1
        if 'multi_policy2' in agent_type:
            out_dim = configs.policy_n * configs.pomo_n
        elif 'multi_policy' in agent_type:
            out_dim = configs.policy_n

        if 'type_all_pred' in configs.model_type:
            Model = Hetero_GNN_type_aware_all_pred
        elif 'type' in configs.model_type:
            Model = Hetero_GNN_type_aware
        else:
            Model = GNN

        return Model(configs.in_dim_op, out_dim, agent_type=agent_type).to(configs.device)

    def get_optimizer(self, model=None, lr=None):
        if not model:
            model = self.model
        if not lr:
            lr = configs.lr

        if configs.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=configs.L2_norm_w)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=configs.L2_norm_w)
        return optimizer

    def get_scheduler(self, optimizer=None, lr=None, scheduler_type=None):
        if not optimizer:
            optimizer = self.optimizer
        if not lr:
            lr = configs.lr
        if scheduler_type is None:
            scheduler_type = configs.scheduler_type

        if 'cyclic' in scheduler_type:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr=configs.min_lr, max_lr=lr,
                                                          step_size_up=1, step_size_down=49,
                                                          mode='triangular2', cycle_momentum=False)
        elif 'step' in scheduler_type:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[250, 500, 750], gamma=0.5)
        else:
            scheduler = None

        return scheduler

    # torch model save, load ##########################################################################
    def model_path(self) -> str:
        if 'BC' in configs.agent_type:
            save_folder = f'./../opt_model/{configs.agent_type}/{configs.action_type}__{configs.sol_type}'
            opt_data_type = f'{configs.env_type}__{configs.state_type}__{configs.policy_symmetric_TF}'
        else:
            save_folder = f'./../opt_model/{configs.agent_type}/{configs.action_type}'
            opt_data_type = f'{configs.env_type}__{configs.state_type}'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = f'{save_folder}/{opt_data_type}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = f'{save_path}/{self.get_model_name()}'
        return model_path

    def get_model_name(self) -> str:
        # model_name = f'{configs.model_type}_{configs.model_global_type} {configs.hi_dim}' \
        #              f' {configs.loss_type} {configs.lr} {configs.dropout_p} {configs.batch_size}' \
        #              f' {configs.softmax_tau} {configs.L2_norm_w} {configs.optimizer_type} {configs.attn_head_n}' \
        #              f' {configs.scheduler_type} {configs.max_ep_n} {configs.training_len} {self.model_i}'
        model_name = 'best'
        return model_name

    def model_save(self) -> None:
        """
        save torch model parameter
        """
        torch.save(self.model.state_dict(), f'{self.model_path()}.pt')

    def model_load(self) -> None:
        """ torch model parameter load
        """
        self.model.load_state_dict(torch.load(f'{self.model_path()}.pt'))

    def models_load(self, model_i_list) -> list:
        """ torch model parameter load
        """
        models = list()
        for i in model_i_list:
            self.model_i = i
            self.model.load_state_dict(torch.load(f'{self.model_path()}.pt'))
            models.append(copy.deepcopy(self.model))

        return models

    def get_current_lr(self) -> float:
        return round(self.optimizer.param_groups[0]["lr"], 7)

    # run episode ####################################################################################
    def update_seed(self) -> None:
        import random
        import numpy as np

        torch.manual_seed(self.model_i)
        torch.cuda.manual_seed(self.model_i)
        torch.cuda.manual_seed_all(self.model_i)  # multiple gpu

        random.seed(self.model_i)
        np.random.seed(self.model_i)

        torch.backends.cudnn.deterministic = True  # type: ignore #slow computation
        torch.backends.cudnn.benchmark = False  # type: ignore

    def run_episode_model_once(self, env, test_TF=True, model=None):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t, decision_n, trajectory
        """
        if not model:
            model = self.model

        if not test_TF:
            model.train()  # dropout: True
            obs, reward, done = env.reset()

            s_t = time.time()
            while not done:
                a, a_, log_p = self.get_action_sample(obs, model=model)
                obs_, reward, done = env.step(a)

                obs = obs_
        else:
            model.eval()
            with torch.no_grad():
                obs, reward, done = env.reset()

                s_t = time.time()
                while not done:
                    a = self.get_action_model(obs, model=model)
                    obs, reward, done = env.step(a)

        run_t = round(time.time() - s_t, 4)
        # env.show_gantt_plotly(0, 0)

        return reward, run_t, env.decision_n

    def perform_model_benchmarks(self, benchmarks, save_path='') -> None:
        """
        perform for envs
        """
        import csv
        from tqdm import tqdm
        from environment.env import JobShopEnv
        from environment.dyn_env import JobShopDynEnv

        self.model_load()
        self.model.to(configs.device)
        for (benchmark, job_n, mc_n, instances) in benchmarks:
            print(benchmark, job_n, mc_n, len(instances))
            mean_r = 0

            for i in tqdm(instances):
                if configs.dyn_type:
                    env = JobShopDynEnv([(benchmark, job_n, mc_n, i)], pomo_n=1)
                else:
                    env = JobShopEnv([(benchmark, job_n, mc_n, i)], pomo_n=1)

                reward, run_t, decision_n = self.run_episode_model_once(env, test_TF=True)
                mean_r += -reward[0, 0].item()

                # env.show_gantt_plotly(0, 0)
                with open(save_path, 'a', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow([benchmark, job_n, mc_n, i,
                                 configs.agent_type, configs.action_type, configs.state_type, 'model',
                                 configs.lr, configs.L2_norm_w, configs.loss_type,
                                 -reward[0, 0].item(), run_t, decision_n[0, 0].item(), self.model_i,
                                 configs.env_type, configs.model_type, configs.dyn_type, configs.parameter,
                                 configs.dyn_reserve_reset])

            print(round(mean_r/len(instances), 2))
            # mean_r = round(mean_r/len(instances), 2)

    # etc function for learning #######################################################################
    def save_training_process(self, valids: list, losses: list) -> None:
        """
        save loss and training validation performance
        """
        result_path = self.model_path() + '_result.pkl'
        result = [valids, losses]
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

    def get_training_process(self) -> (list, list):
        """
        load loss and training validation performance
        """
        result_path = self.model_path() + '_result.pkl'
        with open(result_path, 'rb') as f:
            valids, losses = pickle.load(f)

        return valids, losses

    def training_process_fig(self, valids: list, losses: list, save_TF: bool=False) -> None:
        from matplotlib import pyplot as plt

        plt.clf()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('repeat_n')
        ax1.set_ylabel('loss')
        line1 = ax1.plot(losses, color='green', label='train loss')
        # ax1.set_ylim([200, 350])

        ax2 = ax1.twinx()
        ax2.set_ylabel('makespan')
        line2 = ax2.plot(valids, color='deeppink', label='valid makespan')
        ax2.set_ylim([-1450, -1320])

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        plt.title(self.get_model_name())

        if save_TF:
            result_path = self.model_path() + '_result.png'
            plt.savefig(result_path)
        else:
            plt.show()


if __name__ == '__main__':
    from utils import all_benchmarks, HUN_100

    configs.agent_type = 'GNN_BC_policy'
    configs.env_type = 'dyn'  # 'dyn', ''
    configs.state_type = 'mc_gap_mc_load_prt_norm'  # 'basic_norm', 'simple_norm', 'norm', 'mc_load_norm'
    configs.model_type = 'type_all_pred_GAT2'  # 'type_all_pred_GAT2'

    configs.training_len = len(HUN_100[0][3])

    for configs.action_type in ['conflict']:
        for i in range(5):
            agent = AgentGNN(model_i=i)

            save_path = f'./../result/bench_model_{i}.csv'
            agent.perform_model_benchmarks(all_benchmarks, save_path=save_path)
