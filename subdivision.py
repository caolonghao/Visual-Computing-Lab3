import numpy as np
import sys
import argparse
from base_model import MeshModel

# Mesh subdivision calss
class mesh_subdivision(MeshModel):
    def __init__(self, input_filepath):
        super().__init__(input_filepath)
        print('Import model: '+str(input_filepath))

    # 根据边中点进行细分
    # 找到所有边中点的位置，将所有中点拼接在self.vertices中，再修改所有三角面片，1个三角面片被细分成4个三角形
    def subdivide_by_mid(self):
        pass

if __name__ =="__main__":

    parser=argparse.ArgumentParser(description='Mesh subdivision')
    parser.add_argument('-i', type=str, default='models/kitten.obj', help='input file path of an existing 3d model.')
    parser.add_argument('-o', type=str, default='results/subdivision_kitten.obj', help='output path of 3d model.')
    args=parser.parse_args()

    model = mesh_subdivision(args.i)
    model.subdivide_by_mid()
    model.save_obj(args.o)

