import os
import glob
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file, write_step_file
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from file_utils import ensure_dir


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="file format")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=10, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format(args.form))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]
save_dir = args.src + "_step" if args.outputs is None else args.outputs
ensure_dir(save_dir)

for path in out_paths:
    print(path)
    try:
        if args.form == "h5":
            with h5py.File(path, 'r') as fp:
                out_vec = fp["out_vec"][:].astype(np.float)
                out_shape = vec2CADsolid(out_vec)
        else:
            with open(path, 'r') as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

    except Exception as e:
        print("load and create failed.")
        continue
    
    if args.filter:
        analyzer = BRepCheck_Analyzer(out_shape)
        if not analyzer.IsValid():
            print("detect invalid.")
            continue
    
    name = path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name + ".step")
    write_step_file(out_shape, save_path)

