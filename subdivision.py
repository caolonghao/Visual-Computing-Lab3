import numpy as np
import sys
import argparse

from psutil import swap_memory
from base_model import MeshModel

# Mesh subdivision calss
class mesh_subdivision(MeshModel):
    def __init__(self, input_filepath):
        super().__init__(input_filepath)
        print('Import model: '+str(input_filepath))

    # 根据边中点进行细分
    # 找到所有边中点的位置，将所有中点拼接在self.vertices中，再修改所有三角面片，1个三角面片被细分成4个三角形
    def subdivide_by_mid(self):
        # print("edge_num: ", self.edges.shape[0])
        out_faces = []
        center_dict = {}
        for edge in self.edges:
            p1_id, p2_id = edge
            p1, p2 = self.vertices[p1_id - 1], self.vertices[p2_id - 1]
            p_center = (p1 + p2) / 2
            p_center = p_center.reshape(1, -1)
            self.vertices = np.concatenate((self.vertices, p_center))
            center_dict[(p1_id, p2_id)] = self.vertices.shape[0] #新的点的编号恰等于现在的点个数
        self.numb_v = self.vertices.shape[0]

        for face in self.faces:
            p1_id, p2_id, p3_id = face
            c1_id = center_dict[ (p2_id, p3_id) ]
            c2_id = center_dict[ (p1_id, p3_id) ]
            c3_id = center_dict[ (p1_id, p2_id) ]
            out_faces.append([p1_id, c2_id, c3_id])
            out_faces.append([p2_id, c1_id, c3_id])
            out_faces.append([p3_id, c1_id, c2_id])
            out_faces.append([c1_id, c2_id, c3_id])
        
        self.faces = np.array(out_faces)
        self.numb_f = self.faces.shape[0]

if __name__ =="__main__":

    parser=argparse.ArgumentParser(description='Mesh subdivision')
    parser.add_argument('-i', type=str, default='models/block.obj', help='input file path of an existing 3d model.')
    parser.add_argument('-o', type=str, default='results/subdivision_block.obj', help='output path of 3d model.')
    args=parser.parse_args()

    model = mesh_subdivision(args.i)
    print("subdivision started!")
    model.subdivide_by_mid()
    model.save_obj(args.o)
