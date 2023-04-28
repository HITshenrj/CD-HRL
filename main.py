import argparse
import numpy as np
from Environment import Environment
from stable_baselines3  import PPO
from utils import simulate_random_dag
import networkx as nx
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


arg_parser = argparse.ArgumentParser()
# Synthetic data
arg_parser.add_argument('--data_dim', type=int, default=10,
                        help='the number of variables in synthetic generated data')
arg_parser.add_argument('--data_sample_size', type=int, default=128,
                        help='the number of samples of data')
arg_parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        help='the type of DAG graph by generation method')
arg_parser.add_argument('--graph_degree', type=int, default=3,
                        help='the number of degree in generated DAG graph')
arg_parser.add_argument('--graph_sem_type', type=str, default="linear-gauss",
                        help='the structure equation model (SEM) parameter type')
arg_parser.add_argument('--graph_linear_type', type=str, default='linear_sin',
                        help='the synthetic data type')
arg_parser.add_argument('--w_range', type=tuple, default=(0.5, 2),
                        help='the weight of the graph')
# Training parameters
arg_parser.add_argument('--cuda', action='store_true', default=True,
                        help='Enable CUDA training.')
arg_parser.add_argument('--seed', type=int, default=0, help='Random seed.')
arg_parser.add_argument('--epochs', type=int, default=100000,
                        help='Number of epochs to train.')
arg_parser.add_argument('--high_level_policy_lr', type=float, default=0.00001,
                        help='High level policy learning rate.')
# Load data
arg_parser.add_argument('--ground_truth_data_path', type=str, default=None,
                        help='the path where the dag store')
arg_parser.add_argument('--ground_truth_dag_path', type=str, default=None,
                        help='the path where the dag store')
arg_parser.add_argument('--ground_truth_G', type=nx.DiGraph, default=None,
                        help='the ground truth Graph')
arg_parser.add_argument('--ground_truth_undirected_graph', type=torch.Tensor, default=None,
                        help='the ground truth undirected graph')
# Environment parameters
arg_parser.add_argument('--entropy_gate', type=float, default=0.3,
                        help='The entropy gate for reward')
arg_parser.add_argument('--log_interval', type=int, default=10,
                        help='Action log interval')
arg_parser.add_argument('--buffer_size', type=int, default=100000,
                        help='curiosity buffer size')
arg_parser.add_argument('--memory_decay', type=int, default=10000,
                        help='curiosity reward decay')
arg_parser.add_argument('--reward_amplifier', type=int, default=10,
                        help='Reward amplifier for Fisher-Z')
arg_parser.add_argument('--acyclicity_amplifier', type=int, default=0,
                        help='Reward amplifier for acyclicity')
arg_parser.add_argument('--pre_set', type=bool, default=True,
                        help='Pre fisher-Z process for low dim')
arg_parser.add_argument('--pre_set_gate', type=float, default=0.9,
                        help='Pre fisher-Z gate for low dim')
arg_parser.add_argument('--edge_gate', type=float, default=0.5,
                        help='Edge gate to judge causal edges')
# High level policy parameters
arg_parser.add_argument('--High_level_policy_dim', type=int, nargs='+', default=['512', '512'],
                        help='The dimension of layers in high level policy')
# Low level policy parameters
arg_parser.add_argument('--low_level_policy_dim', type=int, default=20,
                        help='The dimension of hidden layer in low level policy')
arg_parser.add_argument('--low_level_policy_lr', type=float, default=1e-3,
                        help='Low level policy learning rate')
arg_parser.add_argument('--low_level_policy_sdy', type=float, default=0.5,
                        help='low level policy learnable parameter for sample')
arg_parser.add_argument('--low_level_policy_beta', type=float, default=1,
                        help='Low level policy beta')

args = arg_parser.parse_args()
args.High_level_policy_dim = [int(i) for i in args.High_level_policy_dim]
args.cuda = args.cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

print(args)

if args.ground_truth_dag_path is None:
    args.ground_truth_G = simulate_random_dag(
        args.data_dim, args.graph_degree, args.graph_type, args.w_range)

    ground_truth_dag = nx.to_numpy_array(args.ground_truth_G)
    args.ground_truth_dag = ground_truth_dag
else:
    ground_truth_dag = np.load(args.ground_truth_dag_path)
    args.ground_truth_G = nx.DiGraph(ground_truth_dag)
    args.ground_truth_dag = ground_truth_dag

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def train(args):
    env = Environment(
        args=args)
    policy_kwargs = {"net_arch" : args.High_level_policy_dim}
    model = PPO(policy="MlpPolicy",
                env=env,
                n_steps=10,
                n_epochs=10,
                learning_rate=args.high_level_policy_lr,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log='logs',
                seed=94566
                )

    model.learn(
        total_timesteps=args.epochs
    )

if __name__ == "__main__":
    train(args)
