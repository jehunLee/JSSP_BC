from agent import Agent
from params import configs
import pickle, torch, os, time
from network import Hetero_GNN_type_all_prec



# model ##################################################################################
def get_model(action_type="policy"):
    """
    get GNN model <- configs.GNN_type
    """
    if configs.GNN_type == "type_aware":
        GNN_model = Hetero_GNN_type_aware
    elif configs.GNN_type == "type_all_prec":
        GNN_model = Hetero_GNN_type_all_prec
    elif configs.GNN_type == "ijpr":
        GNN_model = Hetero_GNN_IJPR
    elif configs.GNN_type == "simple":
        GNN_model = GNN
    else:
        GNN_model = GNN

    return GNN_model(configs.in_dim_op, 1, configs.in_dim_rsc, action_type).to(configs.device)


def get_optimizer_scheduler(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr,
                                 weight_decay=configs.L2_norm_weight_decay)
    if 'lr_scheduler' in configs.model_type:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=configs.min_lr, max_lr=configs.lr,
                                                      step_size_up=5, step_size_down=15,
                                                      mode='triangular2', cycle_momentum=False)
    else:
        scheduler = None

    return optimizer, scheduler





def get_action_model_sample(obs, model):
    """
    get an action according value
    """
    prob = model(obs)
    policy = Categorical(prob.view(1, -1))  # 나중에 샘플링을 위해서 필요한 함수
    action = policy.sample()
    log_prob = policy.log_prob(action)
    return prob, log_prob.item(), action


def log_prob(obs, model, action):  # log π(a|s) based on current policy
    prob = model(obs)
    policy = Categorical(prob.view(1, -1))  # 나중에 샘플링을 위해서 필요한 함수
    log_prob = policy.log_prob(action)
    return log_prob


def get_x_dim():
    """
    get input_dim, output_dim
    """
    sample_data_path = './bench_data/hun/' + str("hun") + str(3) + "x" + str(2)
    if not os.path.isdir(sample_data_path):
        sample_data_path = './../bench_data/hun/' + str("hun") + str(3) + "x" + str(2)

    sample_env = env.JobShopEnv(sample_data_path, instance_i=0)
    obs, reward, done = sample_env.reset()

    in_dim_op = obs['op'].x.shape[1]
    if configs.mc_node_TF:
        in_dim_rsc = obs['rsc'].x.shape[1]
    else:
        in_dim_rsc = 0
    return in_dim_op, in_dim_rsc


def get_model_name(model_i):
    model_name = '{} {} {}' \
                 ' {} {} {} {} {}' \
                 ' {} {} {} {} {}'.format(
        configs.GNN_type, configs.layer_type, configs.aggr_type,
        configs.global_embed_type, configs.loss_type, configs.lr_type, configs.lr, configs.min_lr,
        configs.dropout_p, configs.batch_size, configs.softmax_tau, configs.opt_max_ep_n, model_i
    )
    return model_name


# agent #########################################################################################
class AgentGNN(Agent):
    def __init__(self, agent_type='policy'):
        super().__init__(agent_type)

        # input dim, output dim ########################
        # configs.in_dim_op, configs.in_dim_rsc = get_x_dim()

        # model generation ########################
        if agent_type == 'policy':
            self.p_net = get_model('policy')
            self.model = self.p_net
            if 'adv' in configs.model_type:
                self.v_net = get_model('value')
        elif agent_type == 'value':
            self.v_net = get_model('value')
            self.model = self.v_net
        else:
            print("Unknown agent_type.")

        # learning setting ########################
        self.optimizer, self.scheduler = get_optimizer_scheduler(self.model)

    def get_action(self, obs, model) -> torch.Tensor:
        """
        get greedy action with highest value
        """
        prob = model(obs)
        return torch.argmax(prob)

    # torch model save, load ##########################################################################
    def model_path(self) -> str:
        folder_path = f'./../opt_model/{configs.agent_type, configs.action_type, configs.state_type}'
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        model_path = f'{folder_path}/{configs.model_type}'
        return model_path

    def model_save(self) -> None:
        """
        save torch model parameter
        """
        if 'p_net' in dir(self):
            torch.save(self.p_net.state_dict(), self.model_path() + "_policy.pt")
        if 'v_net' in dir(self):
            torch.save(self.v_net.state_dict(), self.model_path() + "_value.pt")

    def model_load(self) -> None:
        """ torch model parameter load
        """
        if 'p_net' in dir(self):
            self.p_net.load_state_dict(torch.load(self.model_path() + "_policy.pt"))
        if 'v_net' in dir(self):
            self.v_net.load_state_dict(torch.load(self.model_path() + "_value.pt"))

    def get_current_lr(self):
        return round(self.optimizer.param_groups[0]["lr"], 8)

    # run episode ###################################################################################
    def run_episode(self, env, rule='', test_TF=False, model=None) -> (float, float, float, float, list):
        """
        perform by GNN model
        """
        if not model:
            model = self.model

        obs, reward, done = env.reset()
        cum_r = reward
        decision_n = 0
        trajectory = list()

        s_t = time.time()

        if not test_TF:
            model.train()  # (dropout=True)
            while not done:
                a_tensor, log_p = self.get_action_sample(obs, model)
                obs_, reward, done = env.step(a_tensor.item())
                trajectory.append((obs, a_tensor, log_p, reward, done))
                obs = obs_
                cum_r += reward
                decision_n += 1

        else:
            model.eval()
            with torch.no_grad():
                while not done:
                    a_tensor = self.get_action(obs, model)
                    obs, reward, done = env.step(a_tensor.item())
                    cum_r += reward
                    decision_n += 1

        run_t = round(time.time() - s_t, 4)

        return cum_r, run_t, round(run_t / decision_n, 4), decision_n, trajectory

    # etc function for learning ##########################################################################
    def training_process_save(self, valid_performs, total_losses) -> None:
        """
        save loss and training validation performance
        """
        result_path = self.model_path() + '_result.pkl'
        result = [valid_performs, total_losses]
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    from environment.env import JobShopEnv
    from utils import all_benchmarks, all_rules

    configs.env_type = ''  # 'dyn', ''
    configs.state_type = 'add'  # 'norm', '', 'add_norm', 'simple', 'simple_norm', 'meta'
    configs.agent_type = 'rule'  # 'rule', 'GNN_BC', 'GNN_RL'
    if configs.agent_type == 'rule':
        configs.state_type = 'simple'
        configs.env_type = ''

    configs.model_type = ''

    agent = AgentGNN()
