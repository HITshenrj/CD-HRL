from argparse import Namespace
from gym import Env
from gym.spaces import Box, MultiDiscrete
import networkx as nx
import numpy as np
from collections import defaultdict
from memory import Memory
from utils import count_accuracy, simulate_sem, CondIndepParCorr
from low_level_policy import Low_Level_Policy, loss_function
import copy
import torch
import torch.optim as optim
from scipy.linalg import expm as matrix_exponential


class Environment(Env):
    def __init__(self,
                 args:Namespace
                 ) -> None:
        super(Environment).__init__()
        self.G = args.ground_truth_G
        self.d = nx.to_numpy_array(self.G).shape[0]
        self.n = args.data_sample_size
        self.sem_type = args.graph_sem_type
        self.linear_type = args.graph_linear_type
        # load state
        if self.sem_type == 'load':
            self.state = np.load(args.ground_truth_data_path)
        else:
            self.state = simulate_sem(
                self.G, self.n, self.sem_type, self.linear_type).T
        self.cipc = CondIndepParCorr(self.state, self.n)
        self.entropy_gate = args.entropy_gate
        self.pre_set_gate = args.pre_set_gate
        self.edge_gate = args.edge_gate

        self.memory_decay = args.memory_decay
        self.reward_amplifier = args.reward_amplifier
        self.acyclicity_amplifier = args.acyclicity_amplifier
        self.ground_truth_dag = args.ground_truth_dag

        self.need_to_find = []

        # pre-process for fewer action dimension
        if args.pre_set is True:
            for i in range(self.d):
                for j in range(i+1, self.d):
                    i_j_condition = tuple(
                        [k_II for k_II in range(self.d) if k_II != i and k_II != j])
                    fisher_z_i_j = self.cipc.calc_statistic(i, j, i_j_condition)
                    if fisher_z_i_j < self.pre_set_gate:
                        self.need_to_find.append([i, j])
        else:
            for i in range(self.d):
                for j in range(i+1, self.d):
                    self.need_to_find.append([i, j])
        self.need_to_find_graph = np.zeros((self.d, self.d))
        for i, j in self.need_to_find:
            self.need_to_find_graph[i, j] = 1
            self.need_to_find_graph[j, i] = 1

        # RL environment defination
        self.action_input = [2 for _ in range(len(self.need_to_find)*self.d)]
        self.action_space = MultiDiscrete(self.action_input)
        self.observation_space = Box(-np.inf, np.inf, (self.d, self.n))

        self.logger = None
        self.log_data = defaultdict(int)
        self.num_step_per_episode = 10
        self.done_index = 0
        self.ep_rew = 0.0
        self.log_interval = args.log_interval
        self.cnt_log_interval = 0

        self.memory = Memory(buffer_size=args.buffer_size)

        # two type of HRL
        # HRL-I
        self.model_I = Low_Level_Policy(
            args.low_level_policy_dim).to(args.device)
        self.sdy_I = torch.tensor([args.low_level_policy_sdy], device=args.device,
                                    dtype=torch.float, requires_grad=True)
        self.optimizer_I = optim.Adam([{'params': self.model_I.parameters()}, {'params': self.sdy_I}],
                                        lr=args.low_level_policy_lr)
        self.score_I = []
        self.train_loss_list_I = []
        self.train_loss_I = np.zeros_like(self.ground_truth_dag)
        # HRL-II
        self.max_reward = -float("inf")
        self.max_reward_graph = np.zeros_like(self.need_to_find_graph)
        self.model_II = Low_Level_Policy(
            args.low_level_policy_dim).to(args.device)
        self.sdy_II = torch.tensor([args.low_level_policy_sdy], device=args.device,
                                dtype=torch.float, requires_grad=True)
        self.optimizer_II = optim.Adam([{'params': self.model_II.parameters()}, {'params': self.sdy_II}],
                                    lr=args.low_level_policy_lr)
        self.score_II = []
        self.train_loss_list_II = []
        self.train_loss_II = np.zeros_like(self.ground_truth_dag)

        self.device = args.device
        self.traindata = torch.from_numpy(self.state).float().to(self.device)
        self.low_level_policy_beta = args.low_level_policy_beta

    def reset(self):
        self.done_index = 0
        self.ep_rew = 0.0

        self.score_I = []
        self.train_loss_list_I = []
        self.train_loss_I = np.zeros_like(self.ground_truth_dag)
        self.score_II = []
        self.train_loss_list_II = []
        self.train_loss_II = np.zeros_like(self.ground_truth_dag)
        return self.state

    def step(self, action):
        action = action.reshape(len(self.need_to_find), self.d)

        # curiosity bonus reward
        bonus = float("inf")
        pre_actions = self.memory.buffer
        if self.memory.count() == 0:
            bonus = 0
        else:
            for i in pre_actions:
                distance = np.sum(np.abs(i-action))
                bonus = min(bonus, distance)
        self.memory.add(action)
        if self.cnt_log_interval < self.memory_decay:
            bonus /= self.d
        else:
            bonus /= self.d*self.d*self.d

        # caculate reward
        reward = 0
        graph_I = copy.deepcopy(self.need_to_find_graph)

        for index, value in enumerate(self.need_to_find):
            i = value[0]
            j = value[1]
            action[index, i] = 0
            action[index, j] = 0
            i_j_condition = tuple(np.where(action[index] == 1)[0])
            fisher_z_value = self.cipc.calc_statistic(i, j, i_j_condition)
            reward += (fisher_z_value-self.entropy_gate)*self.reward_amplifier
            if fisher_z_value > self.edge_gate:
                graph_I[i, j] = 0
                graph_I[j, i] = 0

        if reward > self.max_reward:
            self.max_reward = reward
            self.max_reward_graph = graph_I
        reward += bonus
        self.cnt_log_interval += 1

        # HRL-I low level policy train
        g_cuda_I = torch.tensor(graph_I).to(self.device)
        self.optimizer_I.zero_grad()
        y_I, yhat_I, mu_I, logvar_I = self.model_I(
            g_cuda_I, self.traindata)

        loss_I = loss_function(
            y_I, yhat_I, mu_I, logvar_I, self.sdy_I, self.low_level_policy_beta)
        back_loss_I = torch.sum(loss_I)
        back_loss_I.backward()

        self.train_loss_I += loss_I.detach().cpu().numpy()
        self.optimizer_I.step()
        # Ensuring the sdy-I larger than 0.01 to avoid the NAN loss.
        if self.sdy_I < 0.01:
            self.sdy_I = self.sdy_I + 0.01
        self.score_I.append(-self.train_loss_I)
        self.train_loss_list_I.append(np.sum(self.train_loss_I))

        # HRL-II low level policy train
        g_cuda_II = torch.tensor(self.max_reward_graph).to(self.device)
        self.optimizer_II.zero_grad()
        y_II, yhat_II, mu_II, logvar_II = self.model_II(g_cuda_II, self.traindata)

        loss_II = loss_function(y_II, yhat_II, mu_II, logvar_II, self.sdy_II,
                             self.low_level_policy_beta)
        back_loss_II = torch.sum(loss_II)
        back_loss_II.backward()

        self.train_loss_II += loss_II.detach().cpu().numpy()
        self.optimizer_II.step()
        # Ensuring the sdy-II larger than 0.01 to avoid the NAN loss.
        if self.sdy_II < 0.01:
            self.sdy_II = self.sdy_II + 0.01
        self.score_II.append(-self.train_loss_II)
        self.train_loss_list_II.append(np.sum(self.train_loss_II))

        # acyclicity reward for HRL-I and HRL-II
        rt_I = np.max(np.array(self.score_I), axis=0)
        k_I = np.maximum(rt_I-rt_I.T, 0) * graph_I
        k_I = np.where(k_I, 1, 0)

        rt_II = np.max(np.array(self.score_II), axis=0)
        k_II = np.maximum(rt_II-rt_II.T, 0) * self.max_reward_graph
        k_II = np.where(k_II > 0, 1, 0)

        cycness_I = np.trace(matrix_exponential(k_I))- self.d
        cycness_II = np.trace(matrix_exponential(k_II))- self.d

        reward-=(cycness_I+cycness_II)*self.acyclicity_amplifier

        info = dict()
        self.done_index = self.done_index + 1
        if self.done_index == self.num_step_per_episode:
            if self.cnt_log_interval % self.log_interval == 0:
                print("========HRL-I==========")
                rt_I = np.max(np.array(self.score_I), axis=0)
                k_I = np.maximum(rt_I-rt_I.T, 0) * graph_I
                k_I = np.where(k_I > 0, 1, 0)
                print('dag accuracy:fdr, tpr, fpr, shd, pred_size')
                acc_I = count_accuracy(self.ground_truth_dag, k_I.T)
                print(acc_I)

                print("========HRL-II==========")
                rt_II = np.max(np.array(self.score_II), axis=0)
                k_II = np.maximum(rt_II-rt_II.T, 0) * self.max_reward_graph
                k_II = np.where(k_II > 0, 1, 0)
                print('dag accuracy:fdr, tpr, fpr, shd, pred_size')
                acc_II = count_accuracy(self.ground_truth_dag, k_II.T)
                print(acc_II)
            info["episode"] = dict()
            info["episode"]["r"] = self.ep_rew  # episode reward
            info["episode"]["l"] = self.num_step_per_episode  # the number of episode step
            done = True
        else:
            self.ep_rew = self.ep_rew + reward  # accumulated rewards
            done = False

        return self.state, reward, done, info

    def log_callback(self):
        for k_II, v in self.log_data.items():
            self.logger.logkv(k_II, v)
        self.log_data = defaultdict(int)

    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
