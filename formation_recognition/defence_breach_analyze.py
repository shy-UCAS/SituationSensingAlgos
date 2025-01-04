""" 给出我方防御圈I和防御II里面有哪些敌方的集群编组
"""
import math
import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from formation_recognition import basic_units

class DefRingBreach(object):
    def __init__(self, rings_cfg_file=None):
        if rings_cfg_file is not None:
            self.rings_cfg = basic_units.BasicFacilities(rings_cfg_file)
        else:
            self.rings_cfg = basic_units.BasicFacilities()
        
        self.ring1_coords = self.rings_cfg.RING1_XYS
        self.ring2_coords = self.rings_cfg.RING2_XYS

        self._validate_ring_coords()

        self.ring1_path = mpltPath.Path(self.ring1_coords)
        self.ring2_path = mpltPath.Path(self.ring2_coords)
    
    def _validate_ring_coords(self):
        if len(self.ring1_coords) < 3:
            raise ValueError("DEFENCE_RING1 的坐标点不足，无法形成多边形。")
        
        if len(self.ring2_coords) < 3:
            raise ValueError("DEFENCE_RING2 的坐标点不足，无法形成多边形。")
        
        if np.any(self.ring1_coords[0] != self.ring1_coords[-1]):
            self.ring1_coords = np.vstack([self.ring1_coords, self.ring1_coords[0]])

        if np.any(self.ring2_coords[0] != self.ring2_coords[-1]):
            self.ring2_coords = np.vstack([self.ring2_coords, self.ring2_coords[0]])
        
        # hull1 = mpltPath.Path(self.ring1_coords)
        hull2 = mpltPath.Path(self.ring2_coords)

        # 检查防御圈I的所有点是否在防御圈II内
        for point in self.ring1_coords:
            if not hull2.contains_point(point):
                raise ValueError("防御圈II 不完全包含防御圈I。请确保圈2在圈1的外部。")

    def infer_rings_breach_groups(self, objs_trjs:list[basic_units.ObjTracks], cluster_labels, formated_output=True):
        # 初始化结果列表
        ring1_objs_idxs = []
        ring2_objs_idxs = []
        formated_result_jsonstr = []

        # 获取唯一的集群标签
        unique_clusters = np.unique(cluster_labels)
        trj_ts = objs_trjs[0].ts
        
        for _c_lbl in unique_clusters:
            # 获取属于当前集群的无人机索引
            cluster_member_idxs = np.where(cluster_labels == _c_lbl)[0]

            # 获取集群中所有无人机的最后位置
            members_positions = [objs_trjs[idx].last_location() for idx in cluster_member_idxs]

            # 判断是否突破防御圈I
            breach_I = any(self.ring1_path.contains_point(pos[:2]) for pos in members_positions)

            # 判断是否突破防御圈II（前提是未突破I）
            if not breach_I:
                breach_II = any(self.ring2_path.contains_point(pos[:2]) for pos in members_positions)
                breach_circle = "C2" if breach_II else None
            else:
                breach_II = False
                breach_circle = "C1"

            if breach_I or breach_II:
                if breach_I:
                    ring1_objs_idxs.extend(cluster_member_idxs)
                elif breach_II:
                    ring2_objs_idxs.extend(cluster_member_idxs)
                
                # 计算集群的中心位置
                cluster_centroid = (np.mean([_pos[0] for _pos in members_positions]), 
                                    np.mean([_pos[1] for _pos in members_positions]))
                
                # 获取集群成员的名称（假设无人机按索引顺序命名为eUav1, eUav2, ...）
                members_names = [objs_trjs[idx].id for idx in cluster_member_idxs]
                cluster_timestamp = trj_ts[-1]

                swarm_no = "swarm%02d" % (_c_lbl + 1)
                breach_record = {
                    'timestamp': str(cluster_timestamp),
                    'location': cluster_centroid,
                    'swarm_no': swarm_no,
                    'members': members_names,
                    'breach_circle': breach_circle
                }

                # 添加到格式化结果列表
                formated_result_jsonstr.append(breach_record)

        return ring1_objs_idxs, ring2_objs_idxs, formated_result_jsonstr

class DefenceTimeEstimate(object):
    def __init__(self, airport_location, objs_trjs:list[basic_units.ObjTracks], speed=10):
        # 基于当前无人机机场的位置、无人机的飞行轨迹，估计我方派出拦截力量的到达时间
        self.airport_location = airport_location
        self.objs_trjs = objs_trjs
        self.uav_speed = speed
        
        self.intercept_infos = self._estimate_intecept_times()
    
    def _estimate_one_intercept_time(self, airport_location, obj_track:basic_units.ObjTracks):
        """ 计算其中一个敌方无人机被拦截的时间 """
        _euav_xs_shifted = obj_track.xs[-2:] - airport_location[0]
        _euav_ys_shifted = obj_track.ys[-2:] - airport_location[1]
        _euav_move_interval = obj_track.ts[-1] - obj_track.ts[-2]
        
        _euav_movement = np.array([_euav_xs_shifted[-1] - _euav_xs_shifted[-2], 
                                   _euav_ys_shifted[-1] - _euav_ys_shifted[-2]]) / _euav_move_interval
        
        # 根据追击时间的一元二次方程，获得方程求解所需的主要系数
        _coeff_a = _euav_movement[0]**2 + _euav_movement[1]**2 - self.uav_speed**2
        _coeff_b = 2 * (_euav_xs_shifted[-1] * _euav_movement[0] + _euav_ys_shifted[-1] * _euav_movement[1])
        _coeff_c = _euav_xs_shifted[-1]**2 + _euav_ys_shifted[-1]**2
        
        # 然后判断当前方程是否有解
        _solution1_cond = -_coeff_b + np.sqrt(_coeff_b**2 - 4 * _coeff_a * _coeff_c) / (2 * _coeff_a)
        _solution2_cond = -_coeff_b - np.sqrt(_coeff_b**2 - 4 * _coeff_a * _coeff_c) / (2 * _coeff_a)
        
        _num_solutions = 0
        _chase_times = []
        _chase_locations = []
        _chase_directions = []
        
        if _solution1_cond > 0:
            _chase_times.append(_solution1_cond)
            _chased_location = np.array([obj_track.xs[-1] + _euav_movement[0] * _solution1_cond,
                                         obj_track.ys[-1] + _euav_movement[1] * _solution1_cond])
            _chase_locations.append(_chased_location)
            _chase_directions.append(math.degrees(np.arctan2(_chased_location[1], _chased_location[0])))
            _num_solutions += 1
            
        if _solution2_cond > 0:
            _chase_times.append(_solution2_cond)
            _chased_location = np.array([obj_track.xs[-1] + _euav_movement[0] * _solution2_cond,
                                         obj_track.ys[-1] + _euav_movement[1] * _solution2_cond])
            _chase_locations.append(_chased_location)
            _chase_directions.append(math.degrees(np.arctan2(_chased_location[1], _chased_location[0])))
            _num_solutions += 1
        
        # import pdb; pdb.set_trace()
        
        if len(_chase_times) <= 0:
            return None, None, None
        else:
            if len(_chase_times) == 1:
                _chase_time = _chase_times[0]
                _chase_location = _chase_locations[0]
                _chase_direction = _chase_directions[0]
            else:
                _argmin_idx = np.argmin(_chase_times)
                _chase_time = _chase_times[_argmin_idx]
                _chase_location = _chase_locations[_argmin_idx]
                _chase_direction = _chase_directions[_argmin_idx]
                
            return _chase_time, _chase_direction, _chase_location
    
    def _estimate_intecept_times(self):
        """ 计算输入的一组敌方无人机的拦截时间、位置和我方出击角度 """
        _intercept_infos = {}
        
        for _obj_trj in self.objs_trjs:
            _intrcpt_time, _intrcpt_deg, _intrcpt_loc = self._estimate_one_intercept_time(self.airport_location, _obj_trj)
            _intercept_infos[_obj_trj.id] = {'time': _intrcpt_time, 'degree': _intrcpt_deg, 'location': _intrcpt_loc}
        
        return _intercept_infos
    
    def get_intercept_infos(self, vis=False):
        """ 获取对敌拦截情况信息 """
        
        if vis:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))

            for _obj_trj in self.objs_trjs:
                _trj_xs = _obj_trj.xs[-2:]
                _trj_ys = _obj_trj.ys[-2:]
                
                ax.scatter(_trj_xs[0], _trj_ys[0], c='red', s=50)
                ax.scatter(_trj_xs[1], _trj_ys[1], c='blue', s=50)
                ax.arrow(_trj_xs[0], _trj_ys[0], _trj_xs[1] - _trj_xs[0], _trj_ys[1] - _trj_ys[0], head_width=0.5, head_length=0.5, fc='red', ec='red')
                
                _cur_intrcpt_info = self.intercept_infos[_obj_trj.id]
                if _cur_intrcpt_info['time'] is not None:
                    _cur_intrcpt_loc = _cur_intrcpt_info['location']
                    ax.scatter(_cur_intrcpt_loc[0], _cur_intrcpt_loc[1], c='green', s=50)
                    ax.plot([_trj_xs[-1], _cur_intrcpt_loc[0]], [_trj_ys[-1], _cur_intrcpt_loc[1]], '--', c='green')
                    
                    _cur_intrcpt_rad = math.radians(_cur_intrcpt_info['degree'])
                    _cur_intrcpt_x = _cur_intrcpt_loc[0] + self.uav_speed * math.cos(_cur_intrcpt_rad)
                    _cur_intrcpt_y = _cur_intrcpt_loc[1] + self.uav_speed * math.sin(_cur_intrcpt_rad)
                    ax.arrow(self.airport_location[0], self.airport_location[1], 
                             _cur_intrcpt_x - self.airport_location[0], _cur_intrcpt_y - self.airport_location[1], 
                             head_width=0.8, head_length=1.0, fc='green', ec='green')
                    
            # ax.set_xlim(-40, 40)  # 设置x轴范围
            # ax.set_ylim(-40, 40)  # 设置y轴范围
            ax.grid(True)
            plt.show()
            
        return self.intercept_infos