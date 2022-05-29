import numpy as np

class MeshModel:
    def __init__(self, obj_path):
        self.obj_path=obj_path
        # mesh顶点位置，size=(number of vertices, 3)
        self.vertices = None
        # mesh面上的顶点序号（从1开始），size=(number of faces, 3)
        self.faces = None
        # mesh边上的顶点序号（有方向，1->2和2->1都会出现），size=(number of edges, 2)
        self.edges = None
        self.load_obj()
        # self.load_obj_file()
        # self.calculate_plane_equations()
        # self.calculate_Q_matrices()
        
    def load_obj(self):
        """
        加载mesh
        """
        mesh_file = open(self.obj_path).readlines()
        self.vertices = []
        self.faces = []
        for line in mesh_file:
            line = line.split(" ")
            if line[0] == 'v':
                self.vertices.append((float(line[1]), float(line[2]), float(line[3])))
            if line[0] == "f":
                self.faces.append((int(line[1]), int(line[2]), int(line[3])))
        self.vertices=np.array(self.vertices)
        self.faces=np.array(self.faces)
        
        self.numb_v=self.vertices.shape[0]
        self.numb_f=self.faces.shape[0]

        edge_1=self.faces[:,0:2]
        edge_2=self.faces[:,1:]
        edge_3=np.concatenate([self.faces[:,:1], self.faces[:,-1:]], axis=1)
        edge_4=np.concatenate([self.faces[:,1:2], self.faces[:,:1]], axis=1)
        edge_5=np.concatenate([self.faces[:,-1:], self.faces[:,1:2]], axis=1)
        edge_6=np.concatenate([self.faces[:,-1:], self.faces[:,:1]], axis=1)

        self.edges=np.concatenate([edge_1, edge_2, edge_3, edge_4, edge_5, edge_6], axis=0)

        _, edge_index=np.unique(self.edges[:,0]*(10**10)+self.edges[:,1], return_index=True)
        self.edges=self.edges[edge_index,:]


    def save_obj(self, output_path):
        """
        保存mesh到指定路径
        """
        with open(output_path, 'w') as file_obj:
            file_obj.write('# '+str(self.numb_v)+' vertices, '+str(self.numb_f)+' faces\n')
            for i in range(self.numb_v):
                file_obj.write('v '+str(self.vertices[i,0])+' '+str(self.vertices[i,1])+' '+str(self.vertices[i,2])+'\n')
            for i in range(self.numb_f):
                file_obj.write('f '+str(self.faces[i,0])+' '+str(self.faces[i,1])+' '+str(self.faces[i,2])+'\n')
        print('Output simplified model: '+str(output_path))
    


   