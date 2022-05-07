import numpy as np
import sys
import argparse
from base_model import MeshModel

# Mesh simplification class
class mesh_simplify(MeshModel):
    def __init__(self, input_filepath, threshold, simplify_ratio):
        super().__init__(input_filepath)
        print('Import model: '+str(input_filepath))
        self.t=threshold
        self.ratio=simplify_ratio
    
    # 选出所有的 valid pairs.
    def select_valid_pairs(self):
        pass
    
    # 对于每一对valid pair (v1, v2)， 计算最优的  target v_opt 
    # 损失方程为 v_opt.T*(Q1+Q2)*v_opt，按照这个数值来排序所有valid pair
    def calculate_all_pairs_cost_and_order(self):
        pass
    
    #迭代的删除 valid pair (v1, v2)  直到剩余的vertices数量= ratio*初始顶点数
    #Hint 
    #每次删除v1 v2时，把v2和v1的位置都改为v_opt, 然后删除v1
    #然后更新 surface参数,Q矩阵, valid pair以及 optimal 【不需要全部重新计算】
    def start_remove_pair_until_iteratively(self):
        pass
    
    # 将简化后的顶点以及平面重新组织成mesh
    def get_new_3d_model(self):
        pass
    
    #为所有平面计算参数
    def calculate_all_plane_equations(self):
        
        pass
    
    # 为每个顶点计算Q矩阵
    def calculate_vertics_Q(self):
        pass
       
    #建议实现一个通过三个顶点计算平面方程的函数 
    # plane equ: ax+by+cz+d=0 a^2+b^2+c^2=1 
    # p1 ,p2, p3 (x, y, z) are three points on a face
    def calculate_plane_equation_for_one_face(self, p1, p2, p3):
        # p[0]->p.x p[1]->p.y p[2]->p.z
        a = ((p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]))
        b = ((p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]))
        c = ((p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]))
        norm = (a*a + b*b + c*c)**0.5
        a, b, c = a/norm, b/norm, c/norm
        d = -(a*p1[0] + b*p1[1] + c*p1[2])
        return a, b, c, d


if __name__ =="__main__":

    parser=argparse.ArgumentParser(description='Mesh simplify')
    parser.add_argument('-i', type=str, default='models/dinosaur.obj', help='input file path of an existing 3d model.')
    parser.add_argument('-o', type=str, default='results/simplify_dinosaur.obj', help='output path of 3d model.')
    parser.add_argument('-ratio', type=np.float, default=0.1, help='Simplification ratio (0<r<=1)')
    parser.add_argument('-t', type=np.float, default=0, help='Threshold for valid pair selection (>=0).')
    args=parser.parse_args()

    obj_path = args.i
    threshold = args.t
    ratio = args.ratio
    out_path = args.o

    model = mesh_simplify(obj_path,threshold,simplify_ratio=ratio)

    model.calculate_all_plane_equations()

    model.calculate_vertics_Q()

    model.select_valid_pairs()

    model.calculate_all_pairs_cost_and_order()

    model.start_remove_pair_until_iteratively()

    model.get_new_3d_model()

    model.save_obj(out_path)
