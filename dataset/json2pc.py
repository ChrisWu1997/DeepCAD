import os
import glob
import json
import numpy as np
import random
import h5py
from joblib import Parallel, delayed
from trimesh.sample import sample_surface
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD
from utils.pc_utils import write_ply, read_ply

DATA_ROOT = "../data"
RAW_DATA = os.path.join(DATA_ROOT, "cad_json")
RECORD_FILE = os.path.join(DATA_ROOT, "train_val_test_split.json")

N_POINTS = 8096 # 4096
WRITE_NORMAL = False
SAVE_DIR = os.path.join(DATA_ROOT, "pc_cad")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

INVALID_IDS = []


def process_one(data_id):
    if data_id in INVALID_IDS:
        print("skip {}: in invalid id list".format(data_id))
        return

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    # if os.path.exists(save_path):
    #     print("skip {}: file already exists".format(data_id))
    #     return

    # print("[processing] {}".format(data_id))
    json_path = os.path.join(RAW_DATA, data_id + ".json")
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        print("create_CAD failed:", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, N_POINTS, data_id.split("/")[-1])
    except Exception as e:
        print("convert point cloud failed:", data_id)
        return None

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    write_ply(out_pc, save_path)


with open(RECORD_FILE, "r") as fp:
    all_data = json.load(fp)

# process_one(all_data["train"][3])
# exit()

parser = argparse.ArgumentParser()
parser.add_argument('--only_test', action="store_true", help="only convert test data")
args = parser.parse_args()

if not args.only_test:
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["train"])
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["validation"])
Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])
