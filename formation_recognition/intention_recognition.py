""" 基于轨迹、队形等信息，识别敌方可能的行为意图，识别特征包括如下：
    1）敌方分群情况；
    2）队形编码信息；
    3）敌方和我方重要目标之间的距离；
    4）敌方与我方防御圈层之间的空间关系。
"""
import time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from formation_recognition import basic_units
from formation_recognition import clusters_recognition as clust_rec
from formation_recognition import formation_recognition as form_rec
class SingleUavBehavior(object):
    def __init__(self, obj_track:basic_units.ObjTracks, analyze_win=8):
        self.track = obj_track
        self.win_size = analyze_win
        
        self.config_parms = basic_units.GlobalConfigs()
    
    def speed_up(self, speed_up_threshold=None, return_val=False):
        """ 识别敌方无人机是否加速 """
        if speed_up_threshold is None:
            speed_up_threshold = self.config_parms.SPEED_CHANGE_THRESHOLD
        
        _movements = self.track.move_speeds(self.win_size)
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
        
        _, _orient_angles = self.track.move_direct_angles(lookback=self.win_size)
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
        
        _, _orient_angles = self.track.move_direct_angles(lookback=self.win_size)
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

    def targeting_judges(self, move_direct_angles, targeting_angles, targeting_angle_eps=None):
        if targeting_angle_eps is None:
            targeting_angle_eps = self.config_parms.DIRECTING_ANGLE_EPS

        _m2t_diff_angles = np.array(move_direct_angles) - np.array(targeting_angles)
        for _i, _ddeg in enumerate(_m2t_diff_angles):
            if _ddeg > 180:
                _m2t_diff_angles[_i] = _ddeg - 360
            elif _ddeg < -180:
                _m2t_diff_angles[_i] = _ddeg + 360

        _targeting_bools = np.abs(_m2t_diff_angles) < targeting_angle_eps
        return _targeting_bools

    def directing_facilities(self, facilities:basic_units.BasicFacilities, lookback=None, return_val=False):
        """ 计算当前无人机的设施朝向情况 """
        if lookback is None:
            lookback = self.win_size
            
        _move_tstamps, _move_direct_degs = self.track.move_direct_angles(lookback=lookback)
        # import pdb; pdb.set_trace()
        
        _to_facilities_bools = {}
        _to_facilities_degs = {}

        _ttl_facilities_locs = facilities.total_facilities

        for _fac in _ttl_facilities_locs:
            _to_facilities_degs[_fac] = self.track.to_position_angles(_ttl_facilities_locs[_fac], lookback=lookback)
            _to_facilities_bools[_fac] = self.targeting_judges(_move_direct_degs, _to_facilities_degs[_fac])
        
        _facilities_names_arr = np.array([_key for _key in _ttl_facilities_locs.keys()])
        _to_facilities_bools_mat = np.stack([_to_facilities_bools[_fac] for _fac in _to_facilities_bools], axis=1).reshape(-1, len(_facilities_names_arr))
        _to_facilities_degs_mat = np.stack([_to_facilities_degs[_fac] for _fac in _to_facilities_degs], axis=1).reshape(-1, len(_facilities_names_arr))

        # 将上述结果转换为朝向的设施名称的列表
        _targeting_facilities_names = []
        _targeting_facilities_degs = []
        
        for _targeted_bools, _targeted_degs in zip(_to_facilities_bools_mat, _to_facilities_degs_mat):
            if np.sum(_targeted_bools) <= 0:
                _targeting_facilities_names.append(None)
                _targeting_facilities_degs.append(None)
            else:
                _targeting_facilities_names.append(_facilities_names_arr[_targeted_bools].tolist())
                _targeting_facilities_degs.append(_targeted_degs[_targeted_bools].tolist())

        if return_val:
            return _move_tstamps, _targeting_facilities_names, _targeting_facilities_degs
        else:
            return _move_tstamps, _targeting_facilities_names
    
    def distance_to_facilities(self, facilities:basic_units.BasicFacilities, return_val=False):
        """ 计算当前无人机是否靠近设施 """
        _ttl_facilities_locs = facilities.total_facilities
        _facs_coords = np.stack([_ttl_facilities_locs[_fac] for _fac in _ttl_facilities_locs], axis=0)

        _uav_trj_ts, _uav_trj_xs, _uav_trj_ys = self.track.last_n_locations(self.win_size)
        _uav_trj_xys = np.stack([_uav_trj_xs, _uav_trj_ys], axis=1)

        _uav_trj_dists = np.linalg.norm(_uav_trj_xys[:, np.newaxis, :] - _facs_coords[np.newaxis, :, :], axis=-1)

        # 上半段的最远距离和下半段的最近距离的差值，如果小于某个阈值则认为接近相应的设施
        _half_len = int(self.win_size / 2)

        _first_half_maxdists = _uav_trj_dists[:_half_len, :].max(axis=0)
        _second_half_mindists = _uav_trj_dists[_half_len:, :].min(axis=0)
        
        # 基于无人机的运动速度和记录轨迹的时间长度，评估无人机是否切实靠近建筑物
        _interval_len = max(_uav_trj_ts) - min(_uav_trj_ts)
        _prob_move_dist = _interval_len * self.config_parms.SWARM_AVERAGE_SPEED * 0.5

        _distances_changes = _first_half_maxdists - _second_half_mindists

        _closing_to_bools = _distances_changes > _prob_move_dist
        _staying_distance_bools = np.abs(_distances_changes) <= _prob_move_dist
        _keeping_away_bools = _distances_changes < -_prob_move_dist

        _distance_to_facilities_labels = ['none' for _iter in range(len(_ttl_facilities_locs))]
        for _iter, _cl_bool, _sd_bool, _ka_bool in zip(np.arange(len(_distances_changes)), _closing_to_bools, _staying_distance_bools, _keeping_away_bools):
            if _cl_bool:
                _distance_to_facilities_labels[_iter] = 'closing'
            elif _sd_bool:
                _distance_to_facilities_labels[_iter] = 'staying'
            elif _ka_bool:
                _distance_to_facilities_labels[_iter] = 'away_from'

        if return_val:
            return _distance_to_facilities_labels, np.divide(_first_half_maxdists - _second_half_mindists, _first_half_maxdists)
        else:
            return _distance_to_facilities_labels
    
    def probed_facilities(self, facilities:basic_units.BasicFacilities, return_val=False):
        """ 给出当前无人机探测到我方设施的时间（最早时间） """
        _ttl_facilities_locs = facilities.total_facilities
        _facs_coords = np.stack([_ttl_facilities_locs[_fac] for _fac in _ttl_facilities_locs], axis=0)

        _uav_trj_ts, _uav_trj_xs, _uav_trj_ys = self.track.last_n_locations(self.win_size)
        _uav_trj_xys = np.stack([_uav_trj_xs, _uav_trj_ys], axis=1)

        # 计算每帧中所有无人机之间的相对距离
        _uav_trj_dists = np.linalg.norm(_uav_trj_xys[:, np.newaxis, :] - _facs_coords[np.newaxis, :, :], axis=-1)
        _uav2fac_obsrv_bools = _uav_trj_dists <= self.config_parms.OPTICAL_OBSERVE_RANGE

        _uav2fac_obsrv_states = []

        # 识别、提取轨迹中最早出现我方设施被覆盖的序号，和对应的时间点
        for _fac_i, _fac_nm in enumerate(_ttl_facilities_locs):
            _cur_fac_obsrv_bools = _uav2fac_obsrv_bools[:, _fac_i]
            _first_observed_origidx = np.sum(np.logical_not(_cur_fac_obsrv_bools))

            if _first_observed_origidx >=1 and _first_observed_origidx < len(_cur_fac_obsrv_bools):
                _obsrv_idx = _first_observed_origidx - 1
                _obsrv_ts = _uav_trj_ts[_obsrv_idx]

                _uav2fac_obsrv_states.append({'tstamp': _obsrv_ts, 'facility': _fac_nm})

        return _uav2fac_obsrv_states
    
    def estimate_arrive_time(self, facilities:basic_units.BasicFacilities, arrive_at_eps=None):
        """ 基于当前的飞行速度，估计到达我方设施的时间 """
        if arrive_at_eps is None:
            arrive_at_eps = self.config_parms.ARRIVE_AT_EPS

        # 首先计算最近一段轨迹中，和我方设施之间距离的平均缩小速度，然后由此推算到达的时间
        _ttl_facilities_locs = facilities.total_facilities
        _facs_coords = np.stack([_ttl_facilities_locs[_fac] for _fac in _ttl_facilities_locs], axis=0)

        _uav_trj_ts, _uav_trj_xs, _uav_trj_ys = self.track.last_n_locations(self.win_size)
        _uav_trj_xys = np.stack([_uav_trj_xs, _uav_trj_ys], axis=1)

        _uav_trj_dists = np.linalg.norm(_uav_trj_xys[:, np.newaxis, :] - _facs_coords[np.newaxis, :, :], axis=-1)

        # 首先分析当前无人机和设施之间的距离是否缩小
        _check_last_n = 3
        _last_n_shrink_dists = (_uav_trj_dists[-1, :] - _uav_trj_dists[-_check_last_n, :]) / (_check_last_n - 1)
        _arrive_bools = _last_n_shrink_dists < 0
        _arrive_times = []

        # 然后计算距离缩小的设施的到达时间
        for _o_i, _arrv_bool in enumerate(_arrive_bools):
            if _arrv_bool:
                _arrv_time = _uav_trj_ts[-1] + np.abs(_uav_trj_dists[-1, _o_i] / _last_n_shrink_dists[_o_i]) * (_uav_trj_ts[-1] - _uav_trj_ts[0]) / (len(_uav_trj_ts) - 1)
                _arrive_times.append(_arrv_time)
            else:
                _arrive_times.append(None)
        
        return _arrive_bools, _arrive_times

class MultiUavsBehavior(object):
    def __init__(self, objs_tracks:list[basic_units.ObjTracks], analyze_win=8):
        self.tracks = objs_tracks
        self.win_size = analyze_win

        self.config_parms = basic_units.GlobalConfigs()
        
        self.objs_behavs = [SingleUavBehavior(_trk, self.win_size) for _trk in self.tracks]
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

        _track_len = self.win_size
        _prev_mean_dist = np.mean(self.tracks_mean_dists[:int(_track_len / 2)]) # 取前半段轨迹的平均距离
        _cur_mean_dist = np.min(self.tracks_mean_dists[-2:]) # 取最后两帧的平均距离
        _shrink_ratio = (_prev_mean_dist - _cur_mean_dist) / _prev_mean_dist # 计算缩小比例
        
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

    def spread_fleet(self, spread_ratio_threshold=None, return_val=False):
        if spread_ratio_threshold is None:
            spread_ratio_threshold = self.config_parms.DIST_CHANGE_RATIO_THRESHOLD

        _track_len = self.win_size
        _prev_mean_dist = np.mean(self.tracks_mean_dists[:int(_track_len / 2)]) # 取前半段轨迹的平均距离
        _cur_mean_dist = np.min(self.tracks_mean_dists[-2:]) # 取最后两帧的平均距离
        _spread_ratio = (_cur_mean_dist - _prev_mean_dist) / _prev_mean_dist # 计算扩大比例
        
        if _spread_ratio > spread_ratio_threshold:
            if return_val:
                return True, _spread_ratio
            else:
                return True
        else:
            if return_val:
                return False, _spread_ratio
            else:
                return False
        
    def targeting_same_facilities(self, facilities:basic_units.BasicFacilities):
        """ 识别并挑出有共同指向目标的敌方无人机 """
        # _tic = time.time()
        
        _uavs_targeting_facilities = []
        _facs_names = [_nm for _nm in facilities.total_facilities.keys()]
        
        for _u_i, _u_behav in enumerate(self.objs_behavs):
            _, _u_tgt_facs = _u_behav.directing_facilities(facilities, lookback=1)
            _arrive_bools, _arrive_times = _u_behav.estimate_arrive_time(facilities)
            
            if _u_tgt_facs[0] is None:
                _uavs_targeting_facilities.append([])
            else:
                _uavs_targeting_facilities.append([(_fac_nm, _arrive_times[_facs_names.index(_fac_nm)]) for _fac_nm in _u_tgt_facs[0]])
            # _uavs_targeting_facilities.append([] if _u_tgt_facs[0] is None else _u_tgt_facs[0])
        
        # 提取出所有无人机指向的设施名称
        _uniq_tgt_facs = np.unique(np.concatenate([[_fac_time[0] for _fac_time in _tgt_info] for _tgt_info in _uavs_targeting_facilities]))
        
        # 然后遍历这些指向的区域，提取共同指向的无人机名称
        _tgt2fac_uavs = {}
        
        for _u_fac in _uniq_tgt_facs:
            _tgt2fac_uavs_idxs = [i for i, _u_facs in enumerate(_uavs_targeting_facilities) if _u_fac in [_fac_time[0] for _fac_time in _u_facs]]
            _tgt2fac_uavs_ids = [self.objs_behavs[_i].track.id for _i in _tgt2fac_uavs_idxs]
            
            _tgt2fac_uavs[_u_fac] = {'uav_idxs': _tgt2fac_uavs_idxs, 'uav_ids': _tgt2fac_uavs_ids, 
                                     'arrive_times': []}
            
            for _i in _tgt2fac_uavs_idxs:
                _cur_uav_arrive_times = _uavs_targeting_facilities[_i]
                _cur_fac_arive_time = [_cur_uav_arrive_times[_idx][1] for _idx in range(len(_cur_uav_arrive_times)) if _cur_uav_arrive_times[_idx][0] == _u_fac][0]
                _tgt2fac_uavs[_u_fac]['arrive_times'].append(_cur_fac_arive_time)
        
        # print("infer time: %.3f" % (time.time() - _tic))
        # import pdb; pdb.set_trace()
        return _tgt2fac_uavs

class EventsPacker(object):
    """ 基于探测行为分析结果，构建典型单体和多体无人机事件 """
    def __init__(self):
        pass