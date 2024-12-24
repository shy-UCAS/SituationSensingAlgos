""" 基于轨迹、队形等信息，识别敌方可能的行为意图，识别特征包括如下：
    1）敌方分群情况；
    2）队形编码信息；
    3）敌方和我方重要目标之间的距离；
    4）敌方与我方防御圈层之间的空间关系。
"""
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from formation_recognition import basic_units

class SingleUavBehavior(object):
    def __init__(self, obj_track:basic_units.ObjTracks, analyze_win=8):
        self.track = obj_track
        self.win_size = analyze_win
        
        self.config_parms = basic_units.GlobalConfigs()
    
    def speed_up(self, speed_up_threshold=None, return_val=False):
        """ 识别敌方无人机是否加速 """
        if speed_up_threshold is None:
            speed_up_threshold = self.config_parms.SPEED_CHANGE_THRESHOLD
            
        _movements = self.track.move_speeds(self.win_size - 1)
        _half_win_size = int(self.win_size / 2)
        
        _argmin_idx = np.argmin(_movements)
        _argmax_idx = np.argmax(_movements)
        _max_acc_ratio = np.abs(_movements[_argmax_idx] - _movements[_argmin_idx]) / _movements[_argmin_idx]

        if (_max_acc_ratio > speed_up_threshold) \
            and (_argmax_idx > _argmin_idx) \
            and (np.mean(_movements[-_half_win_size:]) > np.mean(_movements[:_half_win_size])):
            if not return_val:
                return True
            else:
                return True, _max_acc_ratio
        else:
            if not return_val:
                return False
            else:
                return False, _max_acc_ratio
    
    def slow_down(self, slow_down_threshold=None, return_val=False):
        """ 识别敌方无人机是否减速 """
        if slow_down_threshold is None:
            slow_down_threshold = self.config_parms.SPEED_CHANGE_THRESHOLD
        
        _movements = self.track.move_speeds(self.win_size - 1)
        _half_win_size = int(self.win_size / 2)
        
        _argmin_idx = np.argmin(_movements)
        _argmax_idx = np.argmax(_movements)
        _max_acc_ratio = np.abs(_movements[_argmax_idx] - _movements[_argmin_idx]) / _movements[_argmin_idx]
        
        if (_max_acc_ratio > slow_down_threshold) \
            and (_argmax_idx < _argmin_idx) \
            and (np.mean(_movements[-_half_win_size:]) < np.mean(_movements[:_half_win_size])):
            if not return_val:
                return True
            else:
                return True, _max_acc_ratio
        else:
            if not return_val:
                return False
            else:
                return False, _max_acc_ratio
    
    def orient2start_diffs_rectify(self, orient_angles):
        """ 如果前后方向角差值超过180度，则进行修正，假设角度在-180 ~ 180度之间 """
        _cur_angle = orient_angles[1]
        _ori2start_diff_angles = [0]

        for _i in range(2, len(orient_angles)):
            _cur_diff_angle = orient_angles[_i] - _cur_angle

            if _cur_diff_angle > 180: # 说明前从-180跨越到了180
                _cur_diff_angle = _cur_diff_angle - 360

            elif _cur_diff_angle < -180: # 说明前从180跨越到了-180
                _cur_diff_angle = _cur_diff_angle + 360
            
            _ori2start_diff_angles.append(_ori2start_diff_angles[-1] + _cur_diff_angle)
            _cur_angle = orient_angles[_i]

        return np.array(_ori2start_diff_angles)

    def orient_change(self, angle_change_threshold=None, return_val=False):
        """ 识别敌方无人机是否转向 """
        if angle_change_threshold is None:
            angle_change_threshold = self.config_parms.ORIENT_CHANGE_THRESHOLD
        
        _orient_angles = self.track.move_direct_angles(lookback=self.win_size - 1)
        _orient_angles = self.orient2start_diffs_rectify(_orient_angles)

        _max2min_angle_diff = np.max(_orient_angles) - np.min(_orient_angles)

        if _max2min_angle_diff > angle_change_threshold:
            if not return_val:
                return True
            else:
                return True, _max2min_angle_diff
        else:
            if not return_val:
                return False
            else:
                return False, _max2min_angle_diff
    
    def turning_frequency(self, change_freq_threshold=None, angle_change_threshold=None, return_val=False):
        """ 识别敌方飞机转向频率是否高 """
        if change_freq_threshold is None:
            change_freq_threshold = self.config_parms.ORIENT_CHANGE_FREQ_THRESHOLD
        
        if angle_change_threshold is None:
            angle_change_threshold = self.config_parms.ORIENT_CHANGE_THRESHOLD
        
        _orient_angles = self.track.move_direct_angles(lookback=self.win_size - 1)
        _orient_angles = self.orient2start_diffs_rectify(_orient_angles)
        _ori2mean_diff_angles = np.array(_orient_angles) - np.mean(_orient_angles)

        _peaks, _preak_props = find_peaks(_ori2mean_diff_angles, 
                                    height=angle_change_threshold / 2 * 0.8,
                                    distance=2)
        _valleys, _valley_props = find_peaks(-_ori2mean_diff_angles, 
                                    height=angle_change_threshold / 2 * 0.8,
                                    distance=2)

        # 按照时间顺序排列后，遍历peak和valley值，查看其中角度差异是否符合条件
        _pv_idxs = np.concatenate([_peaks, _valleys])
        _pv_types = np.array(['p' for _i in range(len(_peaks))] + ['v' for _i in range(len(_valleys))])

        _pv_argsort = np.argsort(_pv_idxs)
        _pv_idxs = _pv_idxs[_pv_argsort]
        _pv_types = _pv_types[_pv_argsort]

        # 无人机转向的次数统计
        _turning_count = 0

        if len(_pv_idxs) <= 0:
            if max(_ori2mean_diff_angles) - min(_ori2mean_diff_angles) > angle_change_threshold:
                _turning_count = 1
            else:
                _turning_count = 0
            
            if _turning_count >= change_freq_threshold:
                if not return_val:
                    return True
                else:
                    return True, _turning_count
            else:
                if not return_val:
                    return False
                else:
                    return False, _turning_count

        # 在序列的开头和结束部分补充必要的peak和valley标记
        if (_pv_types[0] == 'v') and (_ori2mean_diff_angles[0] > _ori2mean_diff_angles[_pv_idxs[0]] + angle_change_threshold):
            _pv_idxs = np.concatenate([np.array([0]), _pv_idxs])
            _pv_types = np.concatenate([np.array(['p']), _pv_types])
        elif (_pv_types[0] == 'p') and (_ori2mean_diff_angles[0] < _ori2mean_diff_angles[_pv_idxs[0]] - angle_change_threshold):
            _pv_idxs = np.concatenate([np.array([0]), _pv_idxs])
            _pv_types = np.concatenate([np.array(['v']), _pv_types])
        
        if (_pv_types[-1] == 'v') and (_ori2mean_diff_angles[-1] > _ori2mean_diff_angles[_pv_idxs[-1]] + angle_change_threshold):
            _pv_idxs = np.concatenate([_pv_idxs, np.array([len(_ori2mean_diff_angles) - 1])])
            _pv_types = np.concatenate([_pv_types, np.array(['p'])])
        elif (_pv_types[-1] == 'p') and (_ori2mean_diff_angles[-1] < _ori2mean_diff_angles[_pv_idxs[-1]] - angle_change_threshold):
            _pv_idxs = np.concatenate([_pv_idxs, np.array([len(_ori2mean_diff_angles) - 1])])
            _pv_types = np.concatenate([_pv_types, np.array(['v'])])

        # 检查开始与结束位置的角度与相邻peak、valley的差异
        _cur_pv_type = None
        _cur_pv_idx = None

        for _pv_i, (_pv_idx, _pv_type) in enumerate(zip(_pv_idxs, _pv_types)):
            if _pv_i == 0:
                _cur_pv_type = _pv_type
                _cur_pv_idx = _pv_idx
                continue

            if _cur_pv_type == 'p':
                if _pv_type == 'p':
                    if _ori2mean_diff_angles[_pv_idx] > _ori2mean_diff_angles[_cur_pv_idx]:
                        _cur_pv_idx = _pv_idx
                elif _pv_type == 'v':
                    _turning_count += 1
                    _cur_pv_type = _pv_type
            elif _cur_pv_type == 'v':
                if _pv_type == 'v':
                    if _ori2mean_diff_angles[_pv_idx] < _ori2mean_diff_angles[_cur_pv_idx]:
                        _cur_pv_idx = _pv_idx
                elif _pv_type == 'p':
                    _turning_count += 1
                    _cur_pv_type = _pv_type

        if _turning_count >= change_freq_threshold:
            if return_val:
                return True, _turning_count
            else:
                return True
        else:
            if return_val:
                return False, _turning_count
            else:
                return False

class MultiUavsBehavior(object):
    def __init__(self, objs_tracks:list[basic_units.ObjTracks], analyze_win=8):
        self.tracks = objs_tracks
        self.win_size = analyze_win

        self.config_parms = basic_units.GlobalConfigs()
        
        self.tracks_inter_dists, self.tracks_mean_dists = self._inter_distances()
    
    def _inter_distances(self):
        # 计算一组x、y坐标位置之间的相对距离
        # 计算每帧中所有无人机之间的相对距离
        _comb_tracks_xys = np.array([np.stack([_trk.xs, _trk.ys], axis=1) for _trk in self.tracks])
        _track_dists = []
        _mean_dists = []
        
        for _iter in range(_comb_tracks_xys.shape[1]):
            _cur_objs_xys = _comb_tracks_xys[:, _iter, :]
            _cur_objs_dists = np.linalg.norm(_cur_objs_xys[:, np.newaxis, :] - _cur_objs_xys[np.newaxis, :, :], axis=-1)
        
            _track_dists.append(_cur_objs_dists[np.triu(np.ones_like(_cur_objs_dists, dtype=bool), k=1)])
            _mean_dists.append(np.mean(_track_dists[-1]))

        return _track_dists, _mean_dists
    
    def shrink_fleet(self, shrink_ratio_threshold=None, return_val=False):
        if shrink_ratio_threshold is None:
            shrink_ratio_threshold = self.config_parms.DIST_CHANGE_RATIO_THRESHOLD

        _track_len = len(self.tracks[0])
        _prev_mean_dist = np.mean(self.tracks_mean_dists[:int(_track_len / 2)])
        _cur_mean_dist = np.min(self.tracks_mean_dists[-2:])
        _shrink_ratio = (_prev_mean_dist - _cur_mean_dist) / _prev_mean_dist
        
        if _shrink_ratio > shrink_ratio_threshold:
            if return_val:
                return True, _shrink_ratio
            else:
                return True
        else:
            if return_val:
                return False, _shrink_ratio
            else:
                return False
