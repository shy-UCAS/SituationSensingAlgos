import os, os.path as osp
import shutil
import yaml
import json

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
        elif formtype == 'circular':
            self.fleet_locs = self._circular_formation(num_objs)
        elif formtype == 'random':
            self.fleet_locs = self._random_formation(num_objs)
        
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

        _echelon_locs = np.dot(_vertical_locs, _rot_matrix)

        return _echelon_locs
    
    def _circular_formation(self, num_objs):
        _spacing_angle_min = 360 / (num_objs + 1)
        _spacing_angle_max = 360 / (num_objs - 1) 
        
        # 首先生成num_obj个圆形角度划分
        _spacing_angles = np.random.uniform(_spacing_angle_min, _spacing_angle_max, num_objs)
        _spacing_angles = _spacing_angles / np.sum(_spacing_angles) * 360
        
        # 然后生成每个分割角度上对应的半径长度
        _sim_radius_list = np.random.uniform(self.max_distance * 0.6 * 0.7, 
                                             self.max_distance * 0.6 * 1.3, num_objs)
        
        _spacing_edges = []
        for _e_iter in range(num_objs):
            _cur_side_a = _sim_radius_list[_e_iter]
            _cur_side_b = _sim_radius_list[(_e_iter + 1) % num_objs]
            
            _cur_side_c = np.sqrt(_cur_side_a ** 2 + _cur_side_b ** 2 - 2 * _cur_side_a * _cur_side_b * np.cos(np.deg2rad(_spacing_angles[_e_iter])))
            _spacing_edges.append(_cur_side_c)
        
        _radius_rescale_factor = (self.min_distance + self.max_distance) / 2 / min(_spacing_edges)
        _sim_radius_list = _sim_radius_list * _radius_rescale_factor
        
        # 然后根据生成的半径长度、旋转角度，生成每个对象的坐标
        _accum_rot_angle = 0
        _circular_locs = []
        for _r_iter in range(num_objs):
            _cur_x = _sim_radius_list[_r_iter] * np.cos(np.deg2rad(_accum_rot_angle))
            _cur_y = _sim_radius_list[_r_iter] * np.sin(np.deg2rad(_accum_rot_angle))
            _accum_rot_angle += _spacing_angles[_r_iter]
            
            _circular_locs.append([_cur_x, _cur_y])

        _circular_locs = np.array(_circular_locs)
        
        _rand_rot_rad = np.deg2rad(np.random.uniform(0, 360))
        _circular_locs = _circular_locs @ np.array([[np.cos(_rand_rot_rad), -np.sin(_rand_rot_rad)],
                                                    [np.sin(_rand_rot_rad), np.cos(_rand_rot_rad)]])
        
        # 最后将生成的坐标进行随机旋转，并根据y轴坐标最小的点，平移到原点的位置上面
        _min_y_idx = np.argmin(_circular_locs[:, 1])
        _circular_locs = _circular_locs - _circular_locs[_min_y_idx]
        
         # import pdb; pdb.set_trace()
        return _circular_locs
    
    def _random_formation(self, num_objs):
        # 首先确定需要随机生成的区域的长度和宽度（行数和列数）
        _num_cols = np.random.randint(2, int(np.ceil(np.sqrt(num_objs))))
        _num_rows = int(np.ceil(num_objs / _num_cols))
        
        # 根据生成的列数和行数，确定随机分布空间区域的长度和宽度
        _reg_width = self.min_distance * _num_cols
        _reg_length = self.min_distance * _num_rows

        # 生成随机分布空间分布坐标
        _rand_xs = np.random.uniform(0, _reg_width, num_objs)
        _rand_ys = np.random.uniform(0, _reg_length, num_objs)
        
        _rand_locs = np.stack([_rand_xs, _rand_ys], axis=1)
        
        # 根据区域中y值最小的坐标，平移所有坐标点，使得y值最小的点在原点
        _min_y_idx = np.argmin(_rand_locs[:, 1])
        _rand_locs = _rand_locs - _rand_locs[_min_y_idx]
        
        return _rand_locs
    
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
    
    def save_to_file(self, filepath):
        with open(filepath, 'wt') as wf:
            # wf.write(f"formtype:{self.formtype}\n")
            for loc in self.fleet_locs:
                wf.write(f"{loc[0]},{loc[1]}\n")

class MaskSwarmFormationDataset(object):
    def __init__(self, num_objs:list[int], form_types:list[str], data_dir:str, num_samples:int=1000):
        # 只针对包含3个以上成员的分组构建特定的队形
        self.num_objs = num_objs
        assert np.all(np.array(num_objs) > 2), "生成队形的群组中包含成员个数不能少于3个"

        self.form_types = form_types
        self.num_samples = num_samples

        self.data_dir = data_dir
        self.data_file = osp.join(self.data_dir, f"swarm_formations_{self.num_samples}.txt")

        self.gene_parms = self._make_generate_parms()
    
    def _make_generate_parms(self):
        # 生成队形参数
        _gene_parms = [] # 给出每一条记录的生成参数

        _num_forms = len(self.form_types)
        _each_form_samples = int(self.num_samples / _num_forms)

        for _form_type in self.form_types:
            for _s_iter in range(_each_form_samples):
                _fleet_size = random.choice(self.num_objs)
                _gene_parms.append({'formtype': _form_type, 'num_objs': _fleet_size})

        return _gene_parms
    
    def generate(self, data_file=None):
        if data_file is None:
            data_file = self.data_file

        if osp.exists(data_file):
            os.remove(data_file)

        _gene_iter = 0 # 数据生成计数器
        _app_quant_th = 100 # 每次追加100条数据，防止内存使用过量

        while _gene_iter < self.num_samples:
            _sect_fleet_locs = []

            for _g_iter in range(_gene_iter, min(self.num_samples, _gene_iter + _app_quant_th)):
                _cur_parm = self.gene_parms[_g_iter]

                _form_type = _cur_parm['formtype']
                _fleet_size = _cur_parm['num_objs']
                _s_former = SwarmFormationGenerate(_fleet_size, _form_type)

                _sect_fleet_locs.append({"formtype": _form_type, "fleet_size": _fleet_size,
                                         "fleet_locs": [{'x': _x, 'y': _y} for _x, _y in _s_former.fleet_locs]})
            
            _gene_iter += _app_quant_th
            # import pdb; pdb.set_trace()

            with open(data_file, 'at') as wf:
                yaml.dump(_sect_fleet_locs, wf, allow_unicode=True)

            print("Generated fleet-locs %d - %d generated." % (_gene_iter - _app_quant_th, _gene_iter))

    def generate_filewise(self, data_dir=None):
        if data_dir is None:
            data_dir = self.data_dir

        if osp.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir)

        _dump_data_collection = []

        for _iter, _parm in enumerate(self.gene_parms):
            _form_type = _parm['formtype']
            _fleet_size = _parm['num_objs']
            _s_former = SwarmFormationGenerate(_fleet_size, _form_type)

            _dump_data = {'formtype': _form_type, 'fleet_size': _fleet_size,
                          'fleet_locs': [{'x': int(_x), 'y': int(_y)} for _x, _y in _s_former.fleet_locs]}
            _dump_data_collection.append(_dump_data)

            # _cur_data_file = osp.join(data_dir, f"form_%s_%05d.json" % (_form_type, _iter))

            # with open(_cur_data_file, 'w') as wf:
            #     # import pdb; pdb.set_trace()
            #     json.dump([_dump_data], wf)
            
            if (_iter + 1) % 100 == 0:
                print("Generated fleet-locs %d - %d generated." % (_iter - 99, _iter + 1))

        with open(self.data_file, 'wt') as wf:
            json.dump(_dump_data_collection, wf)

        # import pdb; pdb.set_trace()