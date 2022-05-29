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
        self.Q_list=None
        self.plane_Kp_list = None
        self.Pair_heap = []
        self.valid_pairs = set()

        # -1，节点编号从0开始，方便处理
        self.edges -= 1
        self.faces -= 1

    class Pair:
        def __init__(self, pair, v_opt, loss):
            self.pair = pair
            self.v_opt = v_opt
            self.loss = loss

        def __lt__(self, other):
            return self.loss.item() < other.loss.item()
        
        def __eq__(self, __o: object) -> bool:
            return self.pair == __o.pair

    # 选出所有的 valid pairs.
    def select_valid_pairs(self):
        num_nodes = self.vertices.shape[0]
        valid_pairs = set()
        for edge in self.edges:
            if edge[0] > edge[1]:
                edge[0], edge[1] = edge[1], edge[0] # 边的端点交换序号
            valid_pairs.add((edge[0], edge[1]))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (i, j) in valid_pairs:
                    continue
                tmp = self.vertices[i] - self.vertices[j]
                dis = np.sum(tmp**2) ** 0.5
                if dis < self.t:
                    valid_pairs.append((i, j))
        self.valid_pairs = valid_pairs

    
    # 对于每一对valid pair (v1, v2)， 计算最优的  target v_opt 
    # 损失方程为 v_opt.T*(Q1+Q2)*v_opt，按照这个数值来排序所有valid pair
    def calculate_all_pairs_cost_and_order(self):
        valid_pairs = self.valid_pairs
        Q_matrix_list = self.Q_list

        for pair in valid_pairs:
            v1, v2 = pair[0], pair[1]
            Q1, Q2 = Q_matrix_list[v1], Q_matrix_list[v2]
            Q_hat = Q1 + Q2
            tmp = Q_hat[:3]
            tmp = np.concatenate((tmp, [[0, 0, 0, 1]]))
            tmp = np.matrix(tmp)
            zero_vector = np.matrix([0, 0, 0, 1]).reshape(-1, 1)

            if np.linalg.det(tmp) == 0:
                v_opt = (self.vertices[v1] + self.vertices[v2]) / 2
                v_opt = np.matrix(np.concatenate((v_opt, np.zeros(1))).reshape(-1, 1))
            else:
                v_opt = tmp**-1 * zero_vector
            loss = v_opt.T * Q_hat * v_opt

            self.Pair_heap.append(self.Pair(pair, v_opt, loss))
            # pairs_with_loss_list.append(self.Pair(pair, v_opt))

    #迭代的删除 valid pair (v1, v2)  直到剩余的vertices数量= ratio*初始顶点数
    #Hint 
    #每次删除v1 v2时，把v2和v1的位置都改为v_opt, 然后删除v1
    #然后更新 surface参数,Q矩阵, valid pair以及 optimal 【不需要全部重新计算】
    def start_remove_pair_until_iteratively(self):
        num_node_delete = round(self.vertices.shape[0] * (1-self.ratio))
        del_cnt = 0
        # 最后统一删点
        node_need_del = []

        while del_cnt < num_node_delete:
            # data: (pair, v_opt, loss)
            # find the smallest cost pair
            data = self.Pair_heap[0]
            for tmp in self.Pair_heap:
                if tmp < data:
                    data = tmp

            v1, v2 = data.pair

            # 更新 v1, v2 的坐标
            v_opt = np.array(data.v_opt[:3].reshape(1, -1)).squeeze() # 取出4维的 v_opt 前3维
            self.vertices[v1] = v_opt
            self.vertices[v2] = v_opt
            node_need_del.append(v2)

            # 更新 v1, v2 的 Q
            self.Q_list[v1] = self.Q_list[v1] + self.Q_list[v2]
            self.Q_list[v2] = self.Q_list[v1] + self.Q_list[v2]

            # 合并surface, 应该去找 v1, v2 同时出现的面，删除掉
            # 更新 v1 未出现，但是 v2 出现的面，将 v2 改为 v1
            face_need_del = []
            for i, face in enumerate(self.faces):
                if v1 in face and v2 in face:
                    face_need_del.append(i)
                elif v1 not in face and v2 in face:
                    mask = np.nonzero(face == v2)[0]
                    self.faces[i][mask] = v1
            # 编号从 1 开始，恶心炸了

            self.faces = np.delete(self.faces, face_need_del, 0)
            self.numb_f -= len(face_need_del)


            pair_index_need_del = []
            pair_need_add = []
            # 更新其他包含 v1, v2 的 pair
            for index, data in enumerate(self.Pair_heap):
                if v1 in data.pair and v2 in data.pair:
                    pair_index_need_del.append(index)
                elif v1 in data.pair:
                    now_node = v1 # v1 的临时复制，避免交换影响到 v1 的值
                    pair_index_need_del.append(index)
                    pre_v1, pre_v2 = data.pair
                    other_node = pre_v2 if now_node == pre_v1 else pre_v1
                    # print("v1:", v1, "pair:", data.pair)
                    if now_node > other_node:
                        now_node, other_node = other_node, now_node

                    Q1, Q2 = self.Q_list[now_node], self.Q_list[other_node]
                    Q_hat = Q1 + Q2
                    tmp = Q_hat[:3]
                    tmp = np.concatenate((tmp, [[0, 0, 0, 1]]))
                    tmp = np.matrix(tmp)
                    zero_vector = np.matrix([0, 0, 0, 1]).reshape(-1, 1)

                    if np.linalg.det(tmp) == 0:
                        v_opt = (self.vertices[now_node] + self.vertices[other_node]) / 2
                        v_opt = np.matrix(np.concatenate((v_opt, np.zeros(1))).reshape(-1, 1))
                    else:
                        v_opt = tmp**-1 * zero_vector
                    
                    loss = v_opt.T * Q_hat * v_opt
                    pair_need_add.append(self.Pair((now_node, other_node), v_opt, loss))

                elif v2 in data.pair:
                    pair_index_need_del.append(index)
                    now_node = v1
                    pre_v1, pre_v2 = data.pair
                    other_node = pre_v2 if v2 == pre_v1 else pre_v1

                    if now_node > other_node:
                        now_node, other_node = other_node, now_node
                    # print('v1:', v1, 'v2:', v2, 'other:', other_node)
                    Q1, Q2 = self.Q_list[now_node], self.Q_list[other_node]
                    Q_hat = Q1 + Q2
                    tmp = Q_hat[:3]
                    tmp = np.concatenate((tmp, [[0, 0, 0, 1]]))
                    tmp = np.matrix(tmp)
                    zero_vector = np.matrix([0, 0, 0, 1]).reshape(-1, 1)
                    if np.linalg.det(tmp) == 0:
                        v_opt = (self.vertices[now_node] + self.vertices[other_node]) / 2
                        v_opt = np.matrix(np.concatenate((v_opt, np.zeros(1))).reshape(-1, 1))
                    else:
                        v_opt = tmp**-1 * zero_vector
                    loss = v_opt.T * Q_hat * v_opt
                    pair_need_add.append(self.Pair((now_node, other_node), v_opt, loss))

            pair_index_need_del.sort(reverse=True)
            for index in pair_index_need_del:
                del self.Pair_heap[index]
            for data in pair_need_add:
                self.Pair_heap.append(data)
            
            # print(self.check_i_in_pair_list(v2))
            del_cnt += 1
        
        self.vertices = np.delete(self.vertices, node_need_del, 0)
        self.numb_v -= len(node_need_del)
    
    def print_pair_list(self):
        for i, pair in enumerate(self.Pair_heap):
            print(pair.pair)
    
    def check_i_in_pair_list(self, index):
        for pair in self.Pair_heap:
            if index in pair.pair:
                return True
        return False

    def print_i_in_pair_list(self, num):
        for pair in self.Pair_heap:
            if num in pair.pair:
                print(pair.pair)

    # 将简化后的顶点以及平面重新组织成mesh
    def get_new_3d_model(self):
        # 重新整理点的编号，使点的最大编号刚好等于点数
        order_mapping = {}
        all_numbers = set()
        for face in self.faces:
            for num in face:
                all_numbers.add(num)
        all_numbers = list(all_numbers)
        all_numbers.sort()
        for index, number in enumerate(all_numbers):
            order_mapping[number] = index
        for i, face in enumerate(self.faces):
            for j, num in enumerate(face):
                self.faces[i][j] = order_mapping[num]
        
        # 加回 1，节点编号从 1 开始
        self.faces += 1
        self.edges += 1
    
    #为所有平面计算参数
    def calculate_all_plane_equations(self):
        plane_Kp_list = []
        for plane in self.faces:
            p1_id, p2_id, p3_id = plane[0], plane[1], plane[2]
            p1, p2, p3 = self.vertices[p1_id], self.vertices[p2_id], self.vertices[p3_id]
            a, b, c, d = self.calculate_plane_equation_for_one_face(p1, p2, p3)
            Kp_matrix = np.array([ [a*a, a*b, a*c, a*d],
                                    [a*b, b*b, b*c, b*d],
                                    [a*c, b*c, c*c, c*d],
                                    [a*d, b*d, c*d, d*d]])
            plane_Kp_list.append(Kp_matrix)
        
        self.plane_Kp_list = plane_Kp_list
    
    # 为每个顶点计算Q矩阵
    def calculate_vertics_Q(self):
        num_nodes = self.vertices.shape[0]
        Q_list = [np.zeros((4,4)) for _ in range(num_nodes)]
        for index, face in enumerate(self.faces):
            for p in face:
                Q_list[p] += self.plane_Kp_list[index]
        self.Q_list = Q_list
        
       
    #建议实现一个通过三个顶点计算平面方程的函数 
    # plane equ: ax+by+cz+d=0 a^2+b^2+c^2=1 
    # p1 ,p2, p3 (x, y, z) are three points on a face
    def calculate_plane_equation_for_one_face(self, p1, p2, p3):
        norm = np.cross(p2 - p1, p3 - p2)
        norm = norm.astype(np.float32)
        norm /= np.sqrt(norm[0] ** 2 + norm[1] ** 2 + norm[2] ** 2)
        d = -norm[0] * p1[0] - norm[1] * p1[1] - norm[2] * p1[2]
        return np.append(norm, [d])

if __name__ =="__main__":

    parser=argparse.ArgumentParser(description='Mesh simplify')
    parser.add_argument('-i', type=str, default='models/block.obj', help='input file path of an existing 3d model.')
    parser.add_argument('-o', type=str, default='results/simplify_block.obj', help='output path of 3d model.')
    parser.add_argument('-ratio', type=np.float, default=0.8, help='Simplification ratio (0<r<=1)')
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
