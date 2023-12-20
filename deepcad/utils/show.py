import os
import glob
import json
import h5py
import numpy as np
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepCheck import BRepCheck_Analyzer
import argparse
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="file format")
parser.add_argument('--idx', type=int, default=0, help="show n files starting from idx.")
parser.add_argument('--num', type=int, default=10, help="number of shapes to show. -1 shows all shapes.")
parser.add_argument('--with_gt', action="store_true", help="also show the ground truth")
parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
args = parser.parse_args()

src_dir = args.src
print(src_dir)
out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format(args.form))))
if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]


def translate_shape(shape, translate):
    trans = gp_Trsf()
    trans.SetTranslation(gp_Vec(translate[0], translate[1], translate[2]))
    loc = TopLoc_Location(trans)
    shape.Move(loc)
    return shape


display, start_display, add_menu, add_function_to_menu = init_display()
cnt = 0
for path in out_paths:
    print(path)
    try:
        if args.form == "h5":
            with h5py.File(path, 'r') as fp:
                out_vec = fp["out_vec"][:].astype(np.float)
                out_shape = vec2CADsolid(out_vec)
                if args.with_gt:
                    gt_vec = fp["gt_vec"][:].astype(np.float)
                    gt_shape = vec2CADsolid(gt_vec)
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

    out_shape = translate_shape(out_shape, [0, 2 * (cnt % 10), 2 * (cnt // 10)])
    if args.form == "h5" and args.with_gt:
        gt_shape = translate_shape(gt_shape, [-2, 2 * (cnt % 10), 2 * (cnt // 10)])
        display.DisplayShape([out_shape, gt_shape], update=True)
    else:
        display.DisplayShape(out_shape, update=True)

    cnt += 1

start_display()


