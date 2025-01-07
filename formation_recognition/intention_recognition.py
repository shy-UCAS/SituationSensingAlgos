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

from problog.program import PrologFile, PrologString
from problog import get_evaluatable

from formation_recognition import basic_units
from formation_recognition import clusters_recognition as clus_rec
from formation_recognition import formation_recognition as form_rec
from formation_recognition import defence_breach_analyze as def_brch

class SingleUavBehavior(object):
    def __init__(self, obj_track:basic_units.ObjTracks, analyze_win=8):
        self.track = obj_track
        self.win_size = analyze_win
        
        self.config_parms = basic_units.GlobalConfigs()
    
    def flying_speed(self, speed_thresholds=[5, 10], lookback=3, return_val=False):
        """ 对无人机飞行速度进行等级划分 """
        _avg_end_speed = self.track.move_speed(lookback=lookback)
        
        _speed_level = 'none'
        _level_score = None
        
        """ slow阶段：越低越接近1.0；
            fast阶段：越高越接近1.0；
            medium阶段：越接近中间速度越接近1.0.
        """
        if _avg_end_speed < speed_thresholds[0]:
            _speed_level = 'slow'
            _level_score = 0.7 + 0.3 * (speed_thresholds[0] - _avg_end_speed) / speed_thresholds[0]
        elif _avg_end_speed > speed_thresholds[1]:
            _speed_level = 'fast'
            _level_score = 0.7 + 0.3 * (1 - speed_thresholds[1]/_avg_end_speed)
        else:
            _speed_level = 'medium'
            _speed_thresholds_center = np.mean(speed_thresholds)
            
            if _avg_end_speed < np.mean(speed_thresholds):
                _level_score = 0.7 + 0.3 * (_avg_end_speed - np.mean(speed_thresholds)) / (_speed_thresholds_center - speed_thresholds[0])
            else:
                _level_score = 0.7 + 0.3 * (speed_thresholds[1] - _avg_end_speed) / (speed_thresholds[1] - _speed_thresholds_center)
        
        if not return_val:
            return _speed_level, _level_score
        else:
            return _speed_level, _level_score, _avg_end_speed
    
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
            
            # 对无人机的加速比例进行分阶段分类
            # 特征值：加速speed_up_threshold的时候，给出估计值为0.8
            _acc_score_p = 0.8
            _acc_score_alpha = (1 - _acc_score_p) / _acc_score_p * speed_up_threshold
            _acc_score = 1 - _acc_score_alpha / (_max_acc_ratio + _acc_score_alpha)
        
            if not return_val:
                return True, _acc_score
            else:
                return True, _acc_score, _max_acc_ratio
        else:
            if not return_val:
                return False, 0.0
            else:
                return False, 0.0, _max_acc_ratio
    
    def slow_down(self, slow_down_threshold=None, return_val=False):
        """ 识别敌方无人机是否减速 """
        if slow_down_threshold is None:
            slow_down_threshold = self.config_parms.SPEED_CHANGE_THRESHOLD
        
        _movements = self.track.move_speeds(self.win_size - 1)
        _half_win_size = int(self.win_size / 2)
        
        _argmin_idx = np.argmin(_movements)
        _argmax_idx = np.argmax(_movements)
        _max_dac_ratio = np.abs(_movements[_argmax_idx] - _movements[_argmin_idx]) / _movements[_argmax_idx]
        
        if (_max_dac_ratio > slow_down_threshold) \
            and (_argmax_idx < _argmin_idx) \
            and (np.mean(_movements[-_half_win_size:]) < np.mean(_movements[:_half_win_size])):
            
            # 计算减速程度的分值
            # 当减速比例为slow_down_threshold的时候，分值为0.8
            _dac_score_p = 0.8
            _dac_score_alpha = _dac_score_p * slow_down_threshold / (1 - slow_down_threshold)
            _dac_score = _dac_score_alpha * (1 / _max_dac_ratio - 1)
            
            if not return_val:
                return True, _dac_score
            else:
                return True, _dac_score, _max_dac_ratio
        else:
            if not return_val:
                return False, 0.0
            else:
                return False, 0.0, _max_dac_ratio
    
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
            # 计算转向程度的分值
            # 当转向角度为angle_change_threshold的时候，分值为0.
            _ori_chg_score_p = 0.7
            _ori_chg_alpha = (1 - _ori_chg_score_p) * angle_change_threshold
            _ori_chg_score = 1.0 - _ori_chg_alpha / _max2min_angle_diff
            
            if not return_val:
                return True, _ori_chg_score
            else:
                return True, _ori_chg_score, _max2min_angle_diff
        else:
            if not return_val:
                return False, 0.0
            else:
                return False, 0.0, _max2min_angle_diff
    
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
        
        else:
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
            # 计算转向频率的分值，当刚好为 change_freq_threshold 时，转向次数分值为 0.7
            _turn_freq_score_p = 0.7
            _turn_freq_score_alpha = change_freq_threshold * (1 - _turn_freq_score_p)
            _turn_freq_score = 1.0 - _turn_freq_score_alpha / _turning_count
            
            if return_val:
                return True, _turn_freq_score, _turning_count
            else:
                return True, _turn_freq_score
        else:
            if return_val:
                return False, 0.0, _turning_count
            else:
                return False, 0.0

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
        return _targeting_bools, np.abs(_m2t_diff_angles)

    def directing_facilities(self, facilities:basic_units.BasicFacilities, targeting_angle_eps=None, lookback=None, return_val=False):
        """ 计算当前无人机的设施朝向情况 """
        if targeting_angle_eps is None:
            targeting_angle_eps = self.config_parms.DIRECTING_ANGLE_EPS
            
        if lookback is None:
            lookback = self.win_size
            
        _move_tstamps, _move_direct_degs = self.track.move_direct_angles(lookback=lookback)
        # import pdb; pdb.set_trace()
        
        _to_facilities_bools = {}
        _to_facilities_degs = {}
        _to_facilities_diffdegs = {}

        _ttl_facilities_locs = facilities.total_facilities

        for _fac in _ttl_facilities_locs:
            _to_facilities_degs[_fac] = self.track.to_position_angles(_ttl_facilities_locs[_fac], lookback=lookback)
            _to_facilities_bools[_fac], _to_facilities_diffdegs[_fac] = self.targeting_judges(_move_direct_degs, _to_facilities_degs[_fac], targeting_angle_eps)
        
        _facilities_names_arr = np.array([_key for _key in _ttl_facilities_locs.keys()])
        _to_facilities_bools_mat = np.stack([_to_facilities_bools[_fac] for _fac in _to_facilities_bools], axis=1).reshape(-1, len(_facilities_names_arr))
        _to_facilities_degs_mat = np.stack([_to_facilities_diffdegs[_fac] for _fac in _to_facilities_degs], axis=1).reshape(-1, len(_facilities_names_arr))

        # 将上述结果转换为朝向的设施名称的列表
        _targeting_facilities_names = []
        _targeting_facilities_degs = []
        
        _tgt_deg_score_p = 0.4
        _tgt_deg_score_alpha = targeting_angle_eps * _tgt_deg_score_p / (1 - _tgt_deg_score_p)
        _targeting_facilities_scores = []

        for _targeted_bools, _targeted_degs in zip(_to_facilities_bools_mat, _to_facilities_degs_mat):
            if np.sum(_targeted_bools) <= 0:
                _targeting_facilities_names.append(None)
                _targeting_facilities_degs.append(None)
                _targeting_facilities_scores.append(None)
            else:
                _targeting_facilities_names.append(_facilities_names_arr[_targeted_bools].tolist())
                _targeting_facilities_degs.append(_targeted_degs[_targeted_bools].tolist())
                
                _tgt_deg_scores = [(_tgt_deg_score_alpha / (_tgt_deg + _tgt_deg_score_alpha)) if _tgt_deg is not None else 0.0
                                   for _tgt_deg in _targeted_degs[_targeted_bools]]
                _targeting_facilities_scores.append(_tgt_deg_scores)
        
        if return_val:
            return _move_tstamps, _targeting_facilities_names, _targeting_facilities_scores, _targeting_facilities_degs
        else:
            return _move_tstamps, _targeting_facilities_names, _targeting_facilities_scores
    
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
        
        _dist_change_ratio_th = 0.2 # 在距离已经比较近的情况下，需要使用距离变化的比例进行判别
        _distances_changes = _first_half_maxdists - _second_half_mindists
        _distances_change_ratios = np.divide(_distances_changes, _first_half_maxdists)

        _closing_to_bools = np.logical_or(_distances_changes > _prob_move_dist, np.abs(_distances_change_ratios) > _dist_change_ratio_th)
        _staying_distance_bools = np.logical_and(np.abs(_distances_changes) <= _prob_move_dist, np.abs(_distances_change_ratios) <= _dist_change_ratio_th)
        _keeping_away_bools = np.logical_or(_distances_changes < -_prob_move_dist, np.abs(_distances_change_ratios) > _dist_change_ratio_th)

        _distance_to_facilities_labels = ['none' for _iter in range(len(_ttl_facilities_locs))]
        for _iter, _cl_bool, _sd_bool, _ka_bool in zip(np.arange(len(_distances_changes)), _closing_to_bools, _staying_distance_bools, _keeping_away_bools):
            if _cl_bool:
                _distance_to_facilities_labels[_iter] = 'closing'
                _dist_change = _distances_changes[_iter]
            elif _sd_bool:
                _distance_to_facilities_labels[_iter] = 'staying'
                _dist_change = _distances_changes[_iter]
            elif _ka_bool:
                _distance_to_facilities_labels[_iter] = 'away_from'
                _dist_change = _distances_changes[_iter]

        if return_val:
            return _distance_to_facilities_labels, _uav_trj_dists[-1, :], np.divide(_first_half_maxdists - _second_half_mindists, _first_half_maxdists)
        else:
            return _distance_to_facilities_labels, _uav_trj_dists[-1, :]
    
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
        
        try:
            for _iter in range(_comb_tracks_xys.shape[1]):
                _cur_objs_xys = _comb_tracks_xys[:, _iter, :]
                _cur_objs_dists = np.linalg.norm(_cur_objs_xys[:, np.newaxis, :] - _cur_objs_xys[np.newaxis, :, :], axis=-1)
            
                _track_dists.append(_cur_objs_dists[np.triu(np.ones_like(_cur_objs_dists, dtype=bool), k=1)])
                _mean_dists.append(np.mean(_track_dists[-1]))
        except:
            import pdb; pdb.set_trace()

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
            _, _u_tgt_facs, _u_tgt_scores = _u_behav.directing_facilities(facilities, lookback=1)
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
        
        return _tgt2fac_uavs
    
    def split_up_swarm(self):
        """ 对当前的无人机集群进行编组划分 """
        _tic = time.time()
        
        _multi_uavs_cluster = clus_rec.SplitClusters(self.tracks, spatial_scale=15)
        
        _cluster_labels = _multi_uavs_cluster.last_clustering()
        _uniq_cluster_labels = np.unique(_cluster_labels)
        
        _cluster_elm_idxs = [np.where(_cluster_labels == _lbl)[0] for _lbl in _uniq_cluster_labels]

        print("splitting up swarm in %.3fsecs" % (time.time() - _tic))
        return _multi_uavs_cluster.last_clustering(), _cluster_elm_idxs
    
    def infer_cluster_formations(self):
        """ 分析当前无人机集群中的群组关联性，队形模式等信息 """
        _tic = time.time()

        # _multi_uavs_cluster = clus_rec.SplitClusters(self.tracks, spatial_scale=15)
        _uavs_clustering, _ = self.split_up_swarm()
        _cur_locs = np.array([[self.tracks[_iter].xs[-1], self.tracks[_iter].ys[-1]] for _iter in range(len(self.tracks))])

        _form_recognizer = form_rec.FormationRecognizer(form_types=self.config_parms.FORMATION_RECOG_TYPES, 
                                                        num_layers=3, 
                                                        hidden_size=64, 
                                                        pretrained_weights=self.config_parms.FORMATION_RECOG_MODEL_FILE)
        
        print("formation-recog model loaded in %.3fsecs" % (time.time() - _tic))

        _form_types, _form_typenames = _form_recognizer.infer_swarm_formations(self.tracks, _uavs_clustering)

        return _uavs_clustering, _form_types, _form_typenames

class IntentFactorExtractor(object):
    """ 基于规则的意图识别总函数：
        基于上述单体和多体的行为要素提取结果，使用Problog表达式描述意图识别要素
    """
    def __init__(self, objs_tracks:list[basic_units.ObjTracks], basic_facilities:basic_units.BasicFacilities, analyze_win=18):
        self.config_parms = basic_units.GlobalConfigs()
        
        self.tracks = objs_tracks
        self.facilities = basic_facilities
        self.win_size = analyze_win
        
        self.objs_behavs = [SingleUavBehavior(_trk, analyze_win=self.win_size) for _trk in self.tracks]
        self.mult_behavs = MultiUavsBehavior(self.tracks, analyze_win=self.win_size)
        
        self.intent_knows = []
        self.intent_knows.extend(self._uav_factors())
    
    def get_knows(self):
        return self.intent_knows
    
    def get_uav_knows(self, uav_id):
        pass

    def _uav_factors(self):
        """ 生成面向单个无人机行为分析的认知知识 """
        # 遍历单个无人机，获取主要的行为要素
        _uavs_sgl_facts = []
        
        for _u_i, _sgl_behav in enumerate(self.objs_behavs):
            _speed_level, _speed_level_score, _avg_speed = _sgl_behav.flying_speed(return_val=True)
            _flying_speed_knowstr = self._pack_flying_speed(_sgl_behav.track.id, _speed_level, _speed_level_score)
            _uavs_sgl_facts.append(_flying_speed_knowstr)
            
            _speed_up_bool, _speed_up_score, _speed_up_ratio = _sgl_behav.speed_up(return_val=True)
            _slow_down_bool, _slow_down_score, _slow_down_ratio = _sgl_behav.slow_down(return_val=True)
            if _speed_up_bool:
                _acc_dac_knowstr = self._pack_speed_up(_sgl_behav.track.id, _speed_up_score)
            elif _slow_down_bool:
                _acc_dac_knowstr = self._pack_slow_down(_sgl_behav.track.id, _slow_down_score)
            else:
                _acc_dac_knowstr = self._pack_steady_speed(_sgl_behav.track.id, 0.7)
            _uavs_sgl_facts.append(_acc_dac_knowstr)
            
            _orient_change_bool, _orient_change_score, _orient_change_ratio = _sgl_behav.orient_change(return_val=True)
            _orient_change_knowstr = self._pack_orient_change(_sgl_behav.track.id, _orient_change_bool, _orient_change_score)
            _uavs_sgl_facts.append(_orient_change_knowstr)
            
            _highfreq_turn_bool, _highfreq_turn_score, _turning_count = _sgl_behav.turning_frequency(return_val=True)
            _turn_freq_knowstr = self._pack_turning_frequency(_sgl_behav.track.id, _highfreq_turn_bool, _highfreq_turn_score)
            _uavs_sgl_facts.append(_turn_freq_knowstr)
            
            # 分析无人机朝向我方重要设施的情况
            _facilities_names = [_key for _key in self.facilities.total_facilities.keys()]
            
            _probed_states = _sgl_behav.probed_facilities(self.facilities)
            _probed_facilities_knowstr = self._pack_probed_state(_sgl_behav.track.id, _probed_states)
            _uavs_sgl_facts.append(_probed_facilities_knowstr)
            
            _directed_ts, _directed_facilities, _directed_scores, _directed_degs = _sgl_behav.directing_facilities(self.facilities, lookback=None, return_val=True)
            _directed_facilities_knowstr = self._pack_directing_to_facilities(_sgl_behav.track.id, _directed_ts, _directed_facilities, _directed_scores)
            _uavs_sgl_facts.append(_directed_facilities_knowstr)
            
            _closing_labels, _dist2facilities, _closing_ratios = _sgl_behav.distance_to_facilities(self.facilities, return_val=True)
            _closing_facilities_names = [_facilities_names[_iter] for _iter in range(len(self.facilities)) if _closing_labels[_iter]]
            _closing_to_facilites_knowstr = self._pack_closing_to_facilities(_sgl_behav.track.id, _closing_facilities_names, _closing_labels)
            _uavs_sgl_facts.append(_closing_to_facilites_knowstr)
            
        # 获取多机朝向主要建筑设施的情况
        _same_targeting_states = self.mult_behavs.targeting_same_facilities(self.facilities)
        _same_targeting_knowstr = self._pack_arrival_at_same_facility(_same_targeting_states)
        _uavs_sgl_facts.append(_same_targeting_knowstr)
        
        _clustering, _, _form_typenames = self.mult_behavs.infer_cluster_formations()
        _clustering_knowstr = self._pack_cluster_formations(_clustering, _form_typenames)
        _uavs_sgl_facts.append(_clustering_knowstr)

        return _uavs_sgl_facts
    
    def _pack_flying_speed(self, uav_id, speed_level, level_score):
        _fmt_str = "%.3f::flying_speed(%s, %s)." % (level_score, uav_id, speed_level)
        return _fmt_str
    
    def _pack_speed_up(self, uav_id, speed_up_score):
        _fmt_str = "%.3f::speed_up(%s). " % (speed_up_score, uav_id)
        return _fmt_str
    
    def _pack_slow_down(self, uav_id, slow_down_score):
        _fmt_str = "%.3f::slow_down(%s)." % (slow_down_score, uav_id)
        return _fmt_str
    
    def _pack_steady_speed(self, uav_id, steady_speed_score=0.8):
        _fmt_str = "%.3f::steady_speed(%s)." % (steady_speed_score, uav_id)
        return _fmt_str
    
    def _pack_orient_change(self, uav_id, change_bool, change_score):
        if change_bool:
            _fmt_str = "%.3f::change_direction(%s, large)." % (change_score, uav_id)
        else:
            _fmt_str = "0.8::change_direction(%s, small)." % (uav_id)
        
        return _fmt_str
    
    def _pack_turning_frequency(self, uav_id, high_freq_turn_bool, high_freq_turn_score):
        if high_freq_turn_bool:
            _fmt_str= "%.3f::direct_fluctuate(%s, high)." % (high_freq_turn_score, uav_id)
        else:
            _fmt_str = "0.8::direct_fluctuate(%s, low)." % (uav_id)
        
        return _fmt_str
    
    def _pack_probed_state(self, uav_id, probed_states):
        if len(probed_states) <= 0:
            return ""
        
        _comb_fmt_strs = []
        for _state in probed_states:
            _fmt_str = "0.9::probed_facility(%s, %s, %.3f)." % (uav_id, _state["facility"], _state["tstamp"])
            _comb_fmt_strs.append(_fmt_str)
        
        return "\n".join(_comb_fmt_strs)
    
    def _pack_directing_to_facilities(self, uav_id, directed_ts, directed_facs, directed_scores, last_n_direct=5):
        if np.sum([_fac is not None for _fac in directed_facs]) <= 0:
            return ""

        # import pdb; pdb.set_trace()
        # 找出最后N次朝向中包含的设施名称集合
        _last_n_direct_facs = [_facs if _facs is not None else [] for _facs in directed_facs[-last_n_direct:]]
        _last_n_direct_facs = np.unique(np.concatenate(_last_n_direct_facs))
        
        if len(_last_n_direct_facs) <= 0:
            return "0.9::targeting_facility(%s, none, %.3f)." % (uav_id, directed_ts[-1])
        
        _directed_knows_list = []
        for _fac in _last_n_direct_facs:
            for _iter in range(len(directed_ts)):
                if _fac in directed_facs[_iter]:
                    _fmt_str = "%.3f::targeting_facility(%s, %s, %.3f)."  \
                        % (directed_scores[_iter][directed_facs[_iter].index(_fac)], uav_id, _fac, directed_ts[_iter])
                    _directed_knows_list.append(_fmt_str)
                    break
        
        return "\n".join(_directed_knows_list)

    def _pack_closing_to_facilities(self, uav_id, facilities_names, closing_labels):
        _comb_fmt_strs = []
        
        for _label, _fac in zip(closing_labels, facilities_names):
            _fmt_str = "0.8::distance_to_facility(%s, %s, %s)." % (uav_id, _fac, _label)
            _comb_fmt_strs.append(_fmt_str)

        return "\n".join(_comb_fmt_strs)

    def _pack_arrival_at_same_facility(self, same_arrival_state, same_arrival_eps=10):
        _fmt_strs_list = []
        
        for _fac, _arrive_state in same_arrival_state.items():
            if len(_arrive_state['uav_ids']) >= 2:
                _arrival_times = _arrive_state['arrive_times']
                _arrival_timediffs = np.abs(np.array(_arrival_times).reshape(-1, 1) - np.array(_arrival_times).reshape(1, -1))
                
                if np.max(_arrival_timediffs) <= same_arrival_eps:
                    _fmt_str = "0.9::attack_same_facility([%s], [%s], %s, same_time)." % (', '.join(_arrive_state['uav_ids']), 
                                                                                      ', '.join([str(_t) for _t in _arrive_state['arrive_times']]),
                                                                                      _fac)
                else:
                    _fmt_str = "0.9::attack_same_facility([%s], [%s], %s, sequential)." % (', '.join(_arrive_state['uav_ids']), 
                                                                                       ', '.join([str(_t) for _t in _arrive_state['arrive_times']]),
                                                                                       _fac)

                _fmt_strs_list.append(_fmt_str)

        if len(_fmt_strs_list) > 0:
            return "\n".join(_fmt_strs_list)
        else:
            return "0.9::attack_same_facility([], [], none, none)."

    def _pack_cluster_formations(self, clustering_labels, formation_typenames):
        _uniq_cluster_labels, _clusters_counts = np.unique(clustering_labels, return_counts=True)
        _num_clusters = len(_uniq_cluster_labels)

        _fmt_strs_list = []
        if _num_clusters == 1:
            if _clusters_counts[0] >= 2:
                _in_cluster_eidxs = np.where(clustering_labels == _uniq_cluster_labels[0])[0]
                _fmt_strs = ["0.9::tight_fleet(%s)." % (self.tracks[_idx]) for _idx in _in_cluster_eidxs]

                if _clusters_counts[0] == 2:
                    _form_type = 'pair'
                else:
                    _form_type = formation_typenames
                _fmt_strs.append("0.9::in_group([%s], %s)." % (", ".join([self.tracks[_idx].id for _idx in _in_cluster_eidxs]), _form_type))

                _fmt_strs_list.extend(_fmt_strs)
        else:
            for _c_label, _c_count in zip(_uniq_cluster_labels, _clusters_counts):
                if _c_count >= 2:
                    _in_cluster_eidxs = np.where(clustering_labels == _c_label)[0]
                    _fmt_strs = ["0.9::tight_fleet(%s)." % (self.tracks[_idx]) for _idx in _in_cluster_eidxs]

                    if _c_count == 2:
                        _form_type = 'pair'
                    else:
                        _form_type = formation_typenames[_c_label]
                    _fmt_strs.append("0.9::in_group([%s], %s)." % (", ".join([self.tracks[_idx].id for _idx in _in_cluster_eidxs]), _form_type))

                    _fmt_strs_list.extend(_fmt_strs)
        
        if len(_fmt_strs_list) > 0:
            return "\n".join(_fmt_strs_list)
        else:
            return ""

class IntentionEvaluator:
    def __init__(self, situation_knows):
        self.config_parms = basic_units.GlobalConfigs()
        self.intent_infer_rules = self._load_intent_infer_rules(self.config_parms.INTENTION_RECOG_RULES_FILE)

        self.situation_knowledges = situation_knows
        self.infered_knows = self._infer_knows()

    def _load_intent_infer_rules(self, rules_fpath=None):
        """ 读取文本文件中的推理规则 """
        if rules_fpath is None:
            rules_fpath = self.config_parms.INTENTION_RECOG_RULES_FILE
        
        with open(rules_fpath, 'r') as _f:
            _rules_str = _f.read()
        
        return _rules_str
    
    def _infer_knows(self):
        _tic = time.time()

        _prolog_knowstr_combined = "\n".join(self.situation_knowledges) + "\n" + self.intent_infer_rules
        _problog_program = PrologString(_prolog_knowstr_combined)
        _result = get_evaluatable(name='sddx').create_from(_problog_program).evaluate()

        print("Rules based Intention inference time: %.3fsecs" % (time.time() - _tic))

        for query, probability in _result.items():
            if probability > 0.0:
                print(query, probability)

    def get_knows(self):
        return self.infered_knows

class ThreatEvaluator(MultiUavsBehavior):
    """ 通过敌方无人机的行为、状态，评估无人机的威胁程度 """
    def __init__(self, objs_tracks:list[basic_units.ObjTracks], facilities:basic_units.BasicFacilities, analyze_win=8):
        super().__init__(objs_tracks, analyze_win)
        self.config_parms = basic_units.GlobalConfigs()
        self.facilities = facilities
        
        # 获取敌方集群的基本分组情况，然分别进行基于行为的威胁评估
        self.uavs_tracks = objs_tracks
        self.uavs_behavs = [SingleUavBehavior(_uav, analyze_win=analyze_win) for _uav in self.uavs_tracks]
        self.uavs_clustering, self.clusters_uav_idxs = self.split_up_swarm()
    
    def _infer_arrive_at_times(self, objs_behavs:list[SingleUavBehavior]):
        """ 根据无人机到达各设施的时间评估威胁等级 """
        _arrive2facilities_times = []
        _facilities_names = [_key for _key in self.facilities.total_facilities.keys()]
        
        for _uav_behav in objs_behavs:
            _arrive_bools, _arrive_times = _uav_behav.estimate_arrive_time(self.facilities)
            if np.sum(_arrive_bools) > 0:
                _orig_arrive_times = np.array(_arrive_times)
                _orig_arrive_times[np.logical_not(_arrive_bools)] = np.inf
                
                _cur_arrive_idx = np.argmin(_orig_arrive_times)
                _cur_arrive_facility = _facilities_names[_cur_arrive_idx]
                _cur_arrive_time = _arrive_times[_cur_arrive_idx]

                _arrive2facilities_times.append({'facility':_cur_arrive_facility, 'time':_cur_arrive_time})

            else:
                _arrive2facilities_times.append(None)

        _valid_arrive_times = [_time['time'] for _time in _arrive2facilities_times if _time is not None]
        if len(_valid_arrive_times) <= 0:
            return 0.0
        elif len(_valid_arrive_times) == 1:
            _min_arrive_time = _valid_arrive_times[0]
        else:
            _min_arrive_time = np.min([_time['time'] for _time in _arrive2facilities_times if _time is not None])
        
        _min_seconds = 0; _min_secs_score = 1.0
        _max_seconds = 30 * 60; _max_secs_score = 0.0
        
        _levels_minmax_secs = np.stack([[_min_seconds] + self.config_parms.ATTACK_REMAIN_SECONDS, self.config_parms.ATTACK_REMAIN_SECONDS + [_max_seconds]], 1)   
        _levels_minmax_scores = np.stack([[_min_secs_score] + self.config_parms.THREAT_SCORE_BY_ATTACK_REMAIN_SECONDS, self.config_parms.THREAT_SCORE_BY_ATTACK_REMAIN_SECONDS + [_max_secs_score]], 1)
        
        _level_idx = np.sum(_min_arrive_time >= np.array(self.config_parms.ATTACK_REMAIN_SECONDS))
        _score= (_min_arrive_time - _levels_minmax_secs[_level_idx, 0]) / (_levels_minmax_secs[_level_idx, 1] - _levels_minmax_secs[_level_idx, 0]) \
            * (_levels_minmax_scores[_level_idx, 1] - _levels_minmax_scores[_level_idx, 0]) + _levels_minmax_scores[_level_idx, 0]
        
        return _score
    
    def _infer_dist_to_facilities(self, objs_behavs:list[SingleUavBehavior]):
        """ 基于无人机与设施的距离和距离变化快慢评估威胁等级 """
        _facilities_names = [_key for _key in self.facilities.total_facilities.keys()]
        _closing_facilities = []
        
        for _uav_behav in objs_behavs:
            _dist_labels, _dist2facilities, _dist_chg_ratios = _uav_behav.distance_to_facilities(self.facilities, True)
            
            _cur_closing_facility = None
            _cur_to_facility_dist = np.inf
            _cur_closing_ratio = None
            
            for _iter, (_dist_lbl, _dist2fac, _chg_ratio) in enumerate(zip(_dist_labels, _dist2facilities, _dist_chg_ratios)):
                if _dist_lbl == 'closing':
                    _cur_facility = _facilities_names[_iter]
                    if _iter == 0:
                        _cur_closing_facility = _cur_facility
                        _cur_to_facility_dist = _dist2fac
                        _cur_closing_ratio = _chg_ratio
                    else:
                        if _cur_to_facility_dist < _dist2fac:
                            _cur_closing_facility = _dist_lbl
                            _cur_to_facility_dist = _dist2fac
                            _cur_closing_ratio = _chg_ratio

            if _cur_closing_facility is not None:
                _closing_facilities.append({'facility': _cur_closing_facility, 'distance': _cur_to_facility_dist, 'close_ratio': _cur_closing_ratio})
            else:
                _closing_facilities.append(None)

        return _closing_facilities
    
    def _infer_cluster_quantity(self, objs_behavs:list[SingleUavBehavior]):
        """ 根据无人机集群的规模评估威胁等级 """
        _num_euavs = len(objs_behavs)
        _threat_quantities = np.array(self.config_parms.THREAT_LEVEL_BY_QUANTITIES)
        _level_idx = np.sum(_num_euavs >= _threat_quantities)
        
        _min_quant = 0; _min_score = 0.0
        _max_quant = 100; _max_score = 1.0
        _levels_min_maxs = np.stack([[_min_quant] + self.config_parms.THREAT_LEVEL_BY_QUANTITIES, self.config_parms.THREAT_LEVEL_BY_QUANTITIES + [_max_quant]], 1)
        _levels_min_max_scores = np.stack([[_min_score] + self.config_parms.THREAT_SCORE_BY_QUANTITIES, self.config_parms.THREAT_SCORE_BY_QUANTITIES + [_max_score]], 1)
        
        _score = (_num_euavs - _levels_min_maxs[_level_idx, 0]) / (_levels_min_maxs[_level_idx, 1] - _levels_min_maxs[_level_idx, 0]) \
            * (_levels_min_max_scores[_level_idx, 1] - _levels_min_max_scores[_level_idx, 0]) + _levels_min_max_scores[_level_idx, 0]

        return _score
            
    def _calculate_intercept_time(self, objs_tracks:list[basic_units.ObjTracks]):
        """ 计算敌方无人机被击毁的比例 """
        _facilities_names = self.facilities.total_facilities
        _airport_names = [_nm for _nm in _facilities_names if _nm.startswith('ua_')]
        _euav_names = [_trk.id for _trk in objs_tracks]
        
        _all_airports_intrcpt_times = []
        for _airport_nm in _airport_names:
            _cur_intrcpt_estimator = def_brch.DefenceTimeEstimate(self.facilities.total_facilities[_airport_nm], objs_tracks)
            _cur_intrcpt_infos = _cur_intrcpt_estimator.get_intercept_infos()
            
            _cur_intrcpt_times = [_info['time'] for _key, _info in _cur_intrcpt_infos.items()]
            _all_airports_intrcpt_times.append(_cur_intrcpt_times)
        
        _all_airports_intrcpt_times = np.stack(_all_airports_intrcpt_times, axis=1)
        _min_intrcpt_times = [np.min(_times) if not (None in _times) else None for _times in _all_airports_intrcpt_times]
        
        if None in _min_intrcpt_times:
            return 1.0, _euav_names[_min_intrcpt_times.index(None)], None
        else:
            _min_intrcpt_idx = np.argmin(_min_intrcpt_times)
            _min_intrcpt_time = _min_intrcpt_times[_min_intrcpt_idx]
            _min_intrcpt_euav = _euav_names[_min_intrcpt_idx]
            
            _min_seconds = 0; _min_secs_score = 1.0
            _max_seconds = 30 * 60; _max_secs_score = 0.0
            
            _levels_minmax_secs = np.stack([[_min_seconds] + self.config_parms.DEFEND_REMAIN_SECONDS, self.config_parms.DEFEND_REMAIN_SECONDS + [_max_seconds]], 1)   
            _levels_minmax_scores = np.stack([[_min_secs_score] + self.config_parms.THREAT_SCORE_BY_DEFEND_REMAIN_SECONDS, self.config_parms.THREAT_SCORE_BY_DEFEND_REMAIN_SECONDS + [_max_secs_score]], 1)
            
            _level_idx = np.sum(_min_intrcpt_time >= np.array(self.config_parms.DEFEND_REMAIN_SECONDS))
            _score= (_min_intrcpt_time - _levels_minmax_secs[_level_idx, 0]) / (_levels_minmax_secs[_level_idx, 1] - _levels_minmax_secs[_level_idx, 0]) \
                * (_levels_minmax_scores[_level_idx, 1] - _levels_minmax_scores[_level_idx, 0]) + _levels_minmax_scores[_level_idx, 0]

            return _score, _min_intrcpt_euav, _min_intrcpt_time

    def _estimate_cluster_threat(self, objs_tracks:list[basic_units.ObjTracks], objs_behavs:list[SingleUavBehavior]):
        """ 评估敌方无人机的威胁程度 （按照分组）"""
        _cur_arrive_score = self._infer_arrive_at_times(objs_behavs)
        _cur_quant_score = self._infer_cluster_quantity(objs_behavs)
        _cur_intrcpt_score, _max_threat_euav, _intrcpt_time = self._calculate_intercept_time(objs_tracks)

        # 使用反向联合威胁概率计算威胁程度
        _census_threat_score = 1 - (1 - _cur_arrive_score) * (1 - _cur_quant_score) * (1 - _cur_intrcpt_score)
        
        return _census_threat_score
    
    def estimate_threats(self, formated_output=False):
        """ 评估敌方无人机的威胁程度 """
        _enemy_clusters_threats = []
        _formated_output = ""
        
        for _cluster_i, _euav_idxs in enumerate(self.clusters_uav_idxs):
            _cur_euavs = [self.uavs_behavs[_idx] for _idx in _euav_idxs]
            _cur_euavs_tracks = [self.uavs_tracks[_idx] for _idx in _euav_idxs]
            
            _cur_threat_score = self._estimate_cluster_threat(_cur_euavs_tracks, _cur_euavs)
            
            if _cur_threat_score > self.config_parms.THREAT_SCORE_THRESHOLD:
                _cur_dist_info = self._infer_dist_to_facilities(_cur_euavs)
                
                _threated_facilities = [_fac for _fac in _cur_dist_info if _fac is not None and _fac['distance'] < self.config_parms.ENDANGER_DISTANCE]
            
            _enemy_clusters_threats.append({
                'cluster_idx': _cluster_i,
                'euav_ids': [_trk.id for _trk in _cur_euavs_tracks],
                'threat_score': _cur_threat_score,
                'threated_facilities': _threated_facilities,})
        
        if formated_output:
            return _enemy_clusters_threats, _formated_output
        else:
            return _enemy_clusters_threats