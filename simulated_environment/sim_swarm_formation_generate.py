import numpy as np
import random
import matplotlib.pyplot as plt

class SwarmFormationGenerate(object):
    """ 仿真生成无人空中集群的队形
        Note：队形识别的时候，将最前面，最右侧的点作为参考点，其余点相对于参考点进行坐标变换
    """
    def __init__(self, num_objs, formtype, min_distance=5, max_distance=20):
        self.num_objs = num_objs
        self.formtype = formtype
        
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        self.fleet_locs = None
        
        if formtype == 'vertical':
            self.fleet_locs = self._vertical_formation(num_objs)
        elif formtype == 'horizontal':
            self.fleet_locs = self._horizontal_formation(num_objs)
        elif formtype == 'echelon':
            self.fleet_locs = self._echelon_formation(num_objs)
        elif formtype == 'wedge':
            self.fleet_locs = self._wedge_formation(num_objs)
        elif formtype == 'circle':
            self.fleet_locs = self._circlar_formation(num_objs)
        
        self.fleet_locs = np.array(self.fleet_locs)
    
    def _vertical_formation(self, num_objs):
        # 生成纵队队形
        # 从原点开始，依次向后面增加个体数量
        _fleet_locs = [[0, 0]]
        _cur_loc = np.array([0, 0])
        
        for _o_iter in range(num_objs - 1):
            _y_add_len = random.uniform(self.min_distance * 0.5, self.max_distance)
            _rnd_distance = random.uniform(max(_y_add_len, self.min_distance), self.max_distance)
            _x_dev_len = random.uniform(0, self.min_distance * 0.8) * random.choice([-1, 1])
            
            _cur_loc[1] = _cur_loc[1] + _y_add_len
            _cur_loc[0] = _x_dev_len
            _fleet_locs.append(_cur_loc.tolist())
        
        return _fleet_locs
    
    def _horizontal_formation(self, num_objs):
        # 生成横队队形
        # 从原点开始，依次向后面增加个体数量
        _fleet_locs = [[0, 0]]
        _cur_loc = np.array([0, 0])
        
        _left_right_sign = random.choice([-1, 1])
                
        for _o_iter in range(num_objs - 1):
            _x_add_len = random.uniform(self.min_distance * 0.5, self.max_distance) * _left_right_sign
            _rnd_distance = random.uniform(max(_x_add_len, self.min_distance), self.max_distance)
            _y_dev_len = random.uniform(0, self.min_distance * 0.8) * random.choice([-1, 1])
            
            _cur_loc[0] = _cur_loc[0] + _x_add_len
            _cur_loc[1] = _y_dev_len
            _fleet_locs.append(_cur_loc.tolist())
        
        return _fleet_locs
    
    def _wedge_formation(self, num_objs, min_angle= 15, max_angle=65):
        # 生成楔形（箭头）队形
        # 首先生成一个典型的横向队列
        _horizontal_locs = np.array(self._horizontal_formation(num_objs))
        
        # 按照x坐标从小打到的顺序排列
        _horizontal_locs = _horizontal_locs[_horizontal_locs[:, 0].argsort()]
        
        # 然后选择中间的一个点作为楔形队形的队首
        _leader_idx = random.randint(1, num_objs - 2)
        _leader_loc = _horizontal_locs[_leader_idx].reshape(1, -1)
        # import pdb; pdb.set_trace()
        _horizontal_locs = _horizontal_locs - _leader_loc
        
        # 对leader左侧的点进行旋转
        _left_locs = _horizontal_locs[:_leader_idx].reshape(_leader_idx, -1)
        _left_tilt_angle = random.uniform(min_angle, max_angle)
        
        _left_tilt_rad = np.deg2rad(_left_tilt_angle)
        _left_rot_mat = np.array([[np.cos(_left_tilt_rad), -np.sin(_left_tilt_rad)],
                                  [np.sin(_left_tilt_rad), np.cos(_left_tilt_rad)]])
        _left_tilt_locs = np.dot(_left_locs, _left_rot_mat)
        
        # 对leader右侧的点进行旋转
        _right_locs = _horizontal_locs[_leader_idx + 1:].reshape(num_objs - _leader_idx - 1, -1)
        _right_tilt_angle = random.uniform(min_angle, max_angle) * -1

        _right_tilt_rad = np.deg2rad(_right_tilt_angle)
        _right_rot_mat = np.array([[np.cos(_right_tilt_rad), -np.sin(_right_tilt_rad)],
                                   [np.sin(_right_tilt_rad), np.cos(_right_tilt_rad)]])
        _right_tilt_locs = np.dot(_right_locs, _right_rot_mat)

        # 将旋转后的点与队首点拼接起来
        _comb_locs = np.concatenate([_left_tilt_locs, np.array([[0, 0]]), _right_tilt_locs], axis=0)
        
        return _comb_locs
    
    def _echelon_formation(self, num_objs, min_angle= 20, max_angle=75):
        # 生成梯形队形(先生成纵队，然后左右旋转)
        # 先生成一个典型的纵向队列
        _vertical_locs = np.array(self._vertical_formation(num_objs))
        
        _tilt_angle = random.uniform(min_angle, max_angle) * random.choice([-1, 1])
        _tilt_rad = np.deg2rad(_tilt_angle)
        _rot_matrix = np.array([[np.cos(_tilt_rad), -np.sin(_tilt_rad)],
                                [np.sin(_tilt_rad), np.cos(_tilt_rad)]])

        # import pdb; pdb.set_trace()
        _echelon_locs = np.dot(_vertical_locs, _rot_matrix)

        return _echelon_locs
        
    
    def _circlar_formation(self, num_objs):
        # 生成圆形队形
        pass
    
    def show_formation(self):
        # 显示队形
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))  # 设置图像大小
        # import pdb; pdb.set_trace()
        
        _min_x, _max_x = np.min(self.fleet_locs[:, 0]), np.max(self.fleet_locs[:, 0])
        _min_y, _max_y = np.min(self.fleet_locs[:, 1]), np.max(self.fleet_locs[:, 1])
        
        plt.scatter(self.fleet_locs[:, 0], self.fleet_locs[:, 1], color='red', s=50, alpha=0.7)  # 绘制散点图
        axes.set_xlim(_min_x - 0.1, _max_x + 0.1)
        
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.title(f"Scatter Plot of {self.num_objs} Objects in Formation {self.formtype}", fontsize=14)
        
        plt.grid(True)  # 添加网格
        axes.set_aspect('equal')  # 设置坐标轴比例相等
        plt.show()