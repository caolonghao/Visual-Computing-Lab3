import argparse
import numpy as np
import pyvista as pv


def vis_mesh(mesh_path):
    mesh = pv.read(mesh_path)
    mesh.plot()

    pass
if __name__ =="__main__":
    parser=argparse.ArgumentParser(description='Mesh simplify')
    parser.add_argument('-i', type=str, default=None, help='input file path of an existing 3d model.')
    args=parser.parse_args()

    vis_mesh(args.i)