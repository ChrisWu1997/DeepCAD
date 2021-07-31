import h5py
from tqdm import tqdm
import os
import argparse
import numpy as np
import sys
sys.path.append("..")
from cadlib.macro import *

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
args = parser.parse_args()

TOLERANCE = 3

result_dir = args.src
filenames = sorted(os.listdir(result_dir))

# overall accuracy
avg_cmd_acc = [] # ACC_cmd
avg_param_acc = [] # ACC_param

# accuracy w.r.t. each command type
each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

# accuracy w.r.t each parameter
args_mask = CMD_ARGS_MASK.astype(np.float)
N_ARGS = args_mask.shape[1]
each_param_cnt = np.zeros([*args_mask.shape])
each_param_acc = np.zeros([*args_mask.shape])

for name in tqdm(filenames):
    path = os.path.join(result_dir, name)
    with h5py.File(path, "r") as fp:
        out_vec = fp["out_vec"][:].astype(np.int)
        gt_vec = fp["gt_vec"][:].astype(np.int)

    out_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    out_param = out_vec[:, 1:]
    gt_param = gt_vec[:, 1:]

    cmd_acc = (out_cmd == gt_cmd).astype(np.int)
    param_acc = []
    for j in range(len(gt_cmd)):
        cmd = gt_cmd[j]
        each_cmd_cnt[cmd] += 1
        each_cmd_acc[cmd] += cmd_acc[j]
        if cmd in [SOL_IDX, EOS_IDX]:
            continue

        if out_cmd[j] == gt_cmd[j]: # NOTE: only account param acc for correct cmd
            tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(np.int)
            # filter param that do not need tolerance (i.e. requires strictly equal)
            if cmd == EXT_IDX:
                tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(np.int)[-2:]
            elif cmd == ARC_IDX:
                tole_acc[3] = (out_param[j] == gt_param[j]).astype(np.int)[3]

            valid_param_acc = tole_acc[args_mask[cmd].astype(np.bool)].tolist()
            param_acc.extend(valid_param_acc)

            each_param_cnt[cmd, np.arange(N_ARGS)] += 1
            each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc

    param_acc = np.mean(param_acc)
    avg_param_acc.append(param_acc)
    cmd_acc = np.mean(cmd_acc)
    avg_cmd_acc.append(cmd_acc)

save_path = result_dir + "_acc_stat.txt"
fp = open(save_path, "w")
# overall accuracy (averaged over all data)
avg_cmd_acc = np.mean(avg_cmd_acc)
print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)
avg_param_acc = np.mean(avg_param_acc)
print("avg param acc (ACC_param):", avg_param_acc, file=fp)

# acc of each command type
each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)
print("each command count:", each_cmd_cnt, file=fp)
print("each command acc:", each_cmd_acc, file=fp)

# acc of each parameter type
each_param_acc = each_param_acc * args_mask
each_param_cnt = each_param_cnt * args_mask
each_param_acc = each_param_acc / (each_param_cnt + 1e-6)
for i in range(each_param_acc.shape[0]):
    print(ALL_COMMANDS[i] + " param acc:", each_param_acc[i][args_mask[i].astype(np.bool)], file=fp)
fp.close()

with open(save_path, "r") as fp:
    res = fp.readlines()
    for l in res:
        print(l, end='')
