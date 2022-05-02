import numpy as np
import sys
import argparse
from base_model import MeshModel

# Mesh smooth calss
class mesh_smooth(MeshModel):
    def __init__(self, input_filepath):
        super().__init__(input_filepath)
        print('Import model: '+str(input_filepath))
    
    # 找到每个顶点的所有邻居
    def find_vertex_neighbors(self):
        vertex_neighbors = []
        pass
        return vertex_neighbors
    
    # 根据邻接关系计算用于平滑的laplacian矩阵
    def laplacian_calculation(self):
        v_neighbors = self.find_vertex_neighbors()
        lap_matrix = None
        pass
        return lap_matrix
    
    # 用计算的laplacian矩阵对顶点位置进行平滑，iterations为迭代次数，lamb是每次迭代的平滑程度
    # new_vertices = lamb * (laplacian * old_vertices) + (1 - lamb) * old_vertices
    def filter_laplacian(self, iterations: int, lamb:float):
        lap_mat = self.laplacian_calculation()
        for _ in range(iterations):
            pass


if __name__ =="__main__":

    parser=argparse.ArgumentParser(description='Mesh smooth')
    parser.add_argument('-i', type=str, default='models/dinosaur.obj', help='input file path of an existing 3d model.')
    parser.add_argument('-o', type=str, default='results/smooth_dinosaur.obj', help='output path of 3d model.')
    parser.add_argument('-n', type=int, default=5, help='iteration number')
    parser.add_argument('-l', type=float, default=0.9, help='lambda used in filter')
    args=parser.parse_args()

    model = mesh_smooth(args.i)
    model.filter_laplacian(args.n, args.l)
    model.save_obj(args.o)

