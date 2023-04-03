import argparse
import torch

parser = argparse.ArgumentParser(description='GNN_JSSP')

# args for device
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='computing device type: cpu or gpu')

# method parameter
parser.add_argument('--prt_norm_TF', type=bool, default=True, help='GNN structure')
parser.add_argument('--dyn_env_TF', type=bool, default=True, help='UB of duration')
parser.add_argument('--action_set_type', type=str, default="conflict", help='UB of duration')
parser.add_argument('--GNN_type', type=str, default="multiplex", help='GNN structure')
parser.add_argument('--loss_type', type=str, default="MSE", help='GNN structure')
parser.add_argument('--policy_symmetric_TF', type=bool, default=True, help='GNN structure')
parser.add_argument('--policy_total_mc', type=bool, default=True, help='GNN structure')
parser.add_argument('--aggr_type', type=str, default="sum", help='GNN structure')
parser.add_argument('--global_embed_type', type=str, default='mean', help='GNN structure')
parser.add_argument('--layer_type', type=str, default="GAT2", help='GNN structure')
parser.add_argument('--lr_type', type=str, default="scheduler", help='GNN structure')
parser.add_argument('--self_loop', type=bool, default=True, help='GNN structure')
parser.add_argument('--simple_state_TF', type=bool, default=True, help='GNN structure')
parser.add_argument('--target_sol_type', type=str, default='full_active', help='GNN structure')
parser.add_argument('--add_state_TF', type=bool, default=True, help='GNN structure')

configs = parser.parse_args()

