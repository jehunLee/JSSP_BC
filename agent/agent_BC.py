from agent_GNN import AgentNN




from params import configs



from environment import env
import time, os
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import pickle
import random



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


def get_optimizer_scheduler(model, scheduler_TF=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr,
                                 weight_decay=configs.L2_norm_weight_decay)
    # if scheduler_TF:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  base_lr=configs.min_lr, max_lr=configs.lr,
                                                  step_size_up=5, step_size_down=15,
                                                  mode='triangular2', cycle_momentum=False)
    return optimizer, scheduler


def get_action_model(obs, model):
    """
    get greedy action with highest value
    """
    prob = model(obs)
    return prob, torch.argmax(prob)


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


def update_BC(batch, model, optimizer):
    """
    tree model 로 performance evaluation
    """
    model.train()  # (dropout=True)
    optimizer.zero_grad()

    loss = model.loss(batch)

    loss.backward()
    clip_grad_norm_(model.parameters(), configs.max_grad_norm)
    optimizer.step()

    return loss



def get_random_env():
    data_set = [([1, 20], 0, False), ([1, 20], 0.2, False), ([1, 20], 0.4, False)]
    job_n = random.choice(range(6, 11))
    mc_n = random.choice(range(3, 8))

    (prt_range, op_skip_ratio, flow_TF) = random.choice(data_set)
    data_path = './agent'
    if not os.path.isdir(data_path):
        data_path = '../../GNN_RL5/agent'
    instance_i = 0
    problem_gen(data_path, instance_i, prt_range, job_n, mc_n, op_skip_ratio, flow_TF)

    return env.JobShopEnv(data_path, instance_i=instance_i)


# agent #########################################################################################
class AgentBC(AgentNN):
    def __init__(self, agent_type="policy", adv_TF=False):
        self.agent_type = agent_type
        self.adv_TF = adv_TF

        # input dim, output dim ########################
        configs.in_dim_op, configs.in_dim_rsc = get_x_dim()

        # model generation ########################
        if agent_type == "rule":
            self.run_episode = self.run_episode_rule
        elif agent_type == "policy":
            self.p_net = get_model("policy")
            self.model = self.p_net
            if adv_TF:
                self.v_net = get_model("value")
            self.run_episode = self.run_episode_model
        elif agent_type == "value":
            self.v_net = get_model("value")
            self.model = self.v_net
            self.run_episode = self.run_episode_model
        elif agent_type == "tree":
            self.tree_model = get_model(agent_type)
            self.run_episode = self.run_episode_tree_search
        else:
            print("no defined agent_type")

    def perform_envs(self, envs, rule=""):
        """
        perform for envs
        return: avg_cum_r, avg_run_t, avg_decision_t
        """
        total_cum_r = 0
        total_t = 0
        total_transit_t = 0
        for i, env in enumerate(envs):
            _, cum_r, t, transit_t = self.perform_env(env, rule=rule, test_TF=True)
            total_cum_r += cum_r
            total_t += t
            total_transit_t += transit_t

        return round(total_cum_r / len(envs), 2), round(total_t / len(envs), 5), round(total_transit_t / len(envs), 5)

    def perform_env(self, env, rule="", test_TF=False):
        """
        perform for an env
        return: cum_r, avg_run_t, avg_decision_t
        """
        return self.run_episode(env, rule, test_TF)

    # torch model save, load ##########################################################################
    def model_save(self, save_path):
        """
        save torch model parameter
        """
        if 'p_net' in dir(self):
            torch.save(self.p_net.state_dict(), save_path + "_policy.pt")
        if 'v_net' in dir(self):
            torch.save(self.v_net.state_dict(), save_path + "_value.pt")

    def model_load(self, save_path):
        """ torch model parameter load
        """
        if 'p_net' in dir(self):
            self.p_net.load_state_dict(torch.load(save_path + "_policy.pt"))
        if 'v_net' in dir(self):
            self.v_net.load_state_dict(torch.load(save_path + "_value.pt"))

    def get_current_lr(self):
        return round(self.optimizer.param_groups[0]["lr"], 8)

    # run episode ###################################################################################
    # rule
    def run_episode_rule(self, env, rule="", test_TF=False, old_model=None):
        """
        perform by rule
        """
        s_t = time.time()
        obs, reward, done = env.reset()
        cum_r = reward
        i = 0
        while not done:
            a_idx = get_action_rule(obs, rule)
            obs, reward, done = env.step(a_idx)
            cum_r = reward + configs.gamma * cum_r
            i += 1
        total_t = round(time.time() - s_t, 5)
        return cum_r, total_t, round(total_t / i, 5)

    # nn model
    def run_episode_model(self, env, rule="", test_TF=False, old_model=None):
        """
        perform by NN model
        """
        if old_model != None:
            model = old_model
        else:
            model = self.model

        trajectory = list()
        s_t = time.time()
        obs, reward, done = env.reset()
        cum_r = reward

        if not test_TF:
            model.train()  # (dropout=True)
            i = 0
            while not done:
                prob, log_p, a_tensor = get_action_model_sample(obs, model)
                obs_, reward, done = env.step(a_tensor.item())
                trajectory.append((obs, a_tensor, log_p, reward, done))
                obs = obs_
                cum_r += reward
                i += 1
            total_t = round(time.time() - s_t, 5)

        else:
            model.eval()
            with torch.no_grad():
                i = 0
                while not done:
                    prob, a_tensor = get_action_model(obs, model)
                    obs, reward, done = env.step(a_tensor.item())
                    cum_r += reward
                    i += 1
                total_t = round(time.time() - s_t, 5)

        return trajectory, cum_r, total_t, round(total_t / i, 5)

    # tree model
    def run_episode_tree_search(self, env, rule="", test_TF=False, old_model=None):
        """
        perform by tree
        """
        s_t = time.time()
        obs, reward, done = env.reset()
        cum_r = reward
        i = 0
        while not done:
            a_idx = self.tree_model.search(env)
            obs, reward, done = env.step(a_idx)
            cum_r = reward + configs.gamma * cum_r
            i += 1
        total_t = round(time.time() - s_t, 5)
        return cum_r, total_t, round(total_t / i, 5)

    # etc function for learning ######################################################################################
    def result_save(self, save_path, valids, total_losses):  # loss 결과 저장
        """
        save torch model parameter
        """
        result_path = save_path + "_result.pkl"
        result = [valids, total_losses]
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

    def get_opt_save_path(self, model_i, opt_problem_set):
        save_path3 = './opt_model'
        if not os.path.isdir(save_path3):
            save_path3 = './../opt_model'

        save_path2 = save_path3 + '/{} {}'.format(configs.action_type, configs.target_sol_type)
        if not os.path.exists(save_path2):
            os.makedirs(save_path2)

        opt_data_type = '{} {}' \
                        ' {} {} {} {}'.format(
            configs.state_type, configs.dyn_env_TF,
            configs.mc_node_TF, configs.prt_norm_TF,
            configs.single_target_mc_TF, configs.policy_symmetric_TF,
        )
        save_path = save_path2 + '/' + opt_data_type
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        benchmark, job_n, mc_n, _ = opt_problem_set[-1]
        problem = str(benchmark) + str(job_n) + "x" + str(mc_n)
        save_path4 = save_path + '/{}_{}'.format(problem, len(opt_problem_set))
        if not os.path.exists(save_path4):
            os.makedirs(save_path4)

        return save_path4 + '/' + get_model_name(model_i)

    # learning ######################################################################################
    def learning_opt(self, valid_problem, opt_problem_set, model_i=0):
        """
        learn optimal policies
        """
        save_path = self.get_opt_save_path(model_i, opt_problem_set)

        # envs ####################################
        envs_valid = get_envs(valid_problem)
        # torch.cuda.manual_seed_all(model_i)  # if use multi-GPU

        # opt_module data ##################################
        opt_data = get_opt_data(opt_problem_set)
        loader = DataLoader(opt_data, configs.batch_size, shuffle=True)  # https://github.com/pyg-team/pytorch_geometric/issues/2961 ######

        # optimizer, scheduler ####################################
        self.optimizer, self.scheduler = get_optimizer_scheduler(self.model)

        # learning ##################################################################
        total_losses = list()
        valids = list()
        self.model_save(save_path)

        # init valid ######################
        mean_cum_r, total_t, mean_decision_t = self.perform_envs(envs_valid)
        valids.append(mean_cum_r)
        print("initial mean cum_r: ", mean_cum_r, "\ttotal_t: ", total_t, "\tmean_decision_t: ", mean_decision_t)
        best_valid_r = mean_cum_r

        # repeat ######################
        for epoch in range(configs.opt_max_ep_n):
            total_loss = 0
            for batch in loader:
                loss = update_BC(batch, self.model, self.optimizer)
                if loss.isnan().item():
                    print("error: loss nan")
                total_loss += loss.item()
            total_losses.append(round(total_loss, 4))
            self.scheduler.step()

            # valid ######################
            mean_cum_r, total_t, mean_decision_t = self.perform_envs(envs_valid)
            valids.append(mean_cum_r)
            if best_valid_r < mean_cum_r:
                best_valid_r = mean_cum_r
                self.model_save(save_path)
            print("---- epoch: {} \ttotal loss: {} \tvalidation max: {} \tnow mean cum_r: {} \t decision_t: {} "
                  "\t lr {}".format(epoch, round(total_loss, 4), best_valid_r, mean_cum_r,
                                    mean_decision_t, self.get_current_lr()))

            if (epoch + 1) % 100 == 0:
                self.result_save(save_path, valids, total_losses)

        # save result ##################
        self.result_save(save_path, valids, total_losses)
        # from environment import visualize
        # visualize.get_gantt_plotly(envs_valid[0])

        return best_valid_r, min(total_losses)



# env, opt_data ##################################################################################


def get_opt_data(problem_set):
    opt_data = list()  # obs

    save_path2 = './../opt_policy/{} {}'.format(configs.action_type, configs.target_sol_type)
    if not os.path.isdir(save_path2):
        save_path2 = './opt_policy/{} {}'.format(configs.action_type, configs.target_sol_type)

    opt_data_type = '{} {}' \
                    ' {} {} {} {}'.format(
        configs.state_type, configs.dyn_env_TF,
        configs.mc_node_TF, configs.prt_norm_TF,
        configs.single_target_mc_TF, configs.policy_symmetric_TF,
    )
    save_path = save_path2 + '/' + opt_data_type

    for (benchmark, job_n, mc_n, _) in problem_set:
        problem = str(benchmark) + str(job_n) + "x" + str(mc_n)

        save_file = save_path + '/{}.p'.format(problem)
        with open(save_file, 'rb') as file:
            opt_data += pickle.load(file)

    print("opt_data load done: {} optimal policies".format(len(opt_data)))
    return opt_data


