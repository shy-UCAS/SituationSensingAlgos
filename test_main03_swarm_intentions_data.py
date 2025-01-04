""" 对预先采集的敌方意图数据进行预处理，并生成动态加载的函数
    2024-12-13
"""
import os, os.path as osp
import re
import glob

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from formation_recognition import basic_units
from formation_recognition import intention_recognition as int_rec

class SwarmIntentExhibitor(object):
    def __init__(self, file_path, coord_scale=1, interp_scale=3, vis=False):
        self.file_path = file_path
        self.file_data = None

        self.uav_ids = None # 无人机的编号
        self.uav_xys = None # 无人机的轨迹坐标

        self.radar_ids = None # 雷达的编号
        self.radar_locs = None # 雷达的位置

        self.airport_ids = None # 机场的编号
        self.airport_locs = None # 机场的位置

        self.hq_ids = None # 指挥所的编号
        self.hq_locs = None # 指挥所的位置

        self.coord_scale = coord_scale
        self.interp_scale = interp_scale

        self._load_data(vis=vis)
    
    def _load_data(self, vis=False):
        self.file_data = pd.read_excel(self.file_path)

        # 获取无人机的轨迹坐标
        self.uav_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns 
                                  if re.match(r"uav\d+_(x|y|z)", _col_nm)])

        _uav_xs = np.array([self.file_data[_uav_id + "_x"] for _uav_id in self.uav_ids])
        _uav_ys = np.array([self.file_data[_uav_id + "_y"] for _uav_id in self.uav_ids])

        _uav_xys = np.stack((_uav_xs, _uav_ys), axis=2) * self.coord_scale
        
        # 对无人机轨迹进行插值和平滑
        _orig_trj_ts = np.arange(_uav_xys.shape[1])
        _interp_trj_ts = np.linspace(0, _orig_trj_ts[-1], int(_orig_trj_ts[-1] * self.interp_scale))

        self.uav_xys = np.zeros((_uav_xys.shape[0], _interp_trj_ts.shape[0], _uav_xys.shape[2]))

        for _uav_id in range(_uav_xys.shape[0]):
            for _dim in range(_uav_xys.shape[2]):
                _interp_func = interp1d(_orig_trj_ts, _uav_xys[_uav_id, :, _dim], kind="cubic")
                self.uav_xys[_uav_id, :, _dim] = gaussian_filter1d(_interp_func(_interp_trj_ts), sigma=1)
        
        # 获取雷达的编号和位置
        self.radar_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns
                                  if re.match(r"radar\d*_(x|y|z)", _col_nm)])

        _radar_xs = np.array([self.file_data[_radar_id + "_x"] for _radar_id in self.radar_ids])
        _radar_ys = np.array([self.file_data[_radar_id + "_y"] for _radar_id in self.radar_ids])
        self.radar_locs = np.concatenate((_radar_xs[:, :1], _radar_ys[:, :1]), axis=1) * self.coord_scale

        # 获取机场的编号和位置
        self.airport_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns
                                  if re.match(r"UavAirport\d*_(x|y|z)", _col_nm)])
        
        _airport_xs = np.array([self.file_data[_airport_id + "_x"] for _airport_id in self.airport_ids])
        _airport_ys = np.array([self.file_data[_airport_id + "_y"] for _airport_id in self.airport_ids])
        self.airport_locs = np.concatenate((_airport_xs[:, :1], _airport_ys[:, :1]), axis=1) * self.coord_scale

        # 获取无人机指挥所的位置
        self.hq_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns
                                  if re.match(r"HQ\d*_(x|y|z)", _col_nm)])

        _hq_xs = np.array([self.file_data[_hq_id + "_x"] for _hq_id in self.hq_ids])
        _hq_ys = np.array([self.file_data[_hq_id + "_y"] for _hq_id in self.hq_ids])
        self.hq_locs = np.concatenate((_hq_xs[:, :1], _hq_ys[:, :1]), axis=1) * self.coord_scale

        # 可视化
        if vis:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # 绘制无人机轨迹
            for _uav_i in range(len(self.uav_ids)):
                ax.plot(self.uav_xys[_uav_i, :, 0], self.uav_xys[_uav_i, :, 1])
            
            # 绘制雷达位置
            ax.scatter(self.radar_locs[:, 0], self.radar_locs[:, 1], c="r", marker="x")
            for _radar_id in range(len(self.radar_ids)):
                ax.text(self.radar_locs[_radar_id, 0] + 10, self.radar_locs[_radar_id, 1] + 10, self.radar_ids[_radar_id])
            
            # 绘制机场位置
            ax.scatter(self.airport_locs[:, 0], self.airport_locs[:, 1], c="g", marker="o")
            for _airport_id in range(len(self.airport_ids)):
                ax.text(self.airport_locs[_airport_id, 0] + 10, self.airport_locs[_airport_id, 1] + 10, self.airport_ids[_airport_id])

            # 绘制无人机指挥所位置
            ax.scatter(self.hq_locs[:, 0], self.hq_locs[:, 1], c="b", marker="^")
            for _hq_id in range(len(self.hq_ids)):
                ax.text(self.hq_locs[_hq_id, 0] + 10, self.hq_locs[_hq_id, 1] + 10, self.hq_ids[_hq_id])

            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.show()
    
    def pack_to_objtracks(self, lookback_start=1, lookback_len=10, vis=False):
        _num_objs = self.uav_xys.shape[0]

        if lookback_start > 0:
            _obj_tracks = [basic_units.ObjTracks(self.uav_xys[_o_i, -lookback_start-lookback_len:-lookback_start, 0], 
                                                self.uav_xys[_o_i, -lookback_start-lookback_len:-lookback_start, 1], 
                                                id='euav%d' % (_o_i)) 
                           for _o_i in range(_num_objs)]
            
        elif lookback_start <= 0:
            _obj_tracks = [basic_units.ObjTracks(self.uav_xys[_o_i, -lookback_len:, 0], 
                                                self.uav_xys[_o_i, -lookback_len:, 1], 
                                                id='euav%d' % (_o_i))
                           for _o_i in range(_num_objs)]
        
        if vis:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # 绘制主要设施的位置
            # HQ的位置
            ax.scatter(self.hq_locs[:, 0], self.hq_locs[:, 1], c="r", marker="*", s=100, label='head quartors')
            for _hq_i, _hq_id in enumerate(self.hq_ids):
                ax.text(self.hq_locs[_hq_i, 0] + 10, self.hq_locs[_hq_i, 1] + 10, _hq_id)

            # uav airport的位置
            ax.scatter(self.airport_locs[:, 0], self.airport_locs[:, 1], c="r", marker="^", s=100, label='uav airports')
            for _airport_i, _airport_id in enumerate(self.airport_ids):
                ax.text(self.airport_locs[_airport_i, 0] + 10, self.airport_locs[_airport_i, 1] + 10, _airport_id)

            # radar的位置
            ax.scatter(self.radar_locs[:, 0], self.radar_locs[:, 1], c="r", marker="x", s=100, label='radars')
            for _radar_i, _radar_id in enumerate(self.radar_ids):
                ax.text(self.radar_locs[_radar_i, 0] + 10, self.radar_locs[_radar_i, 1] + 10, _radar_id)


            for _obj_i, _obj_trk in enumerate(_obj_tracks):
                _ttl_xs, _ttl_ys = self.uav_xys[_obj_i, :, 0], self.uav_xys[_obj_i, :, 1]
                _obj_ts, _obj_xs, _obj_ys = _obj_trk.last_n_locations(lookback_len)

                ax.plot(_ttl_xs, _ttl_ys, c="green", linestyle='--', alpha=0.5)
                ax.plot(_obj_xs, _obj_ys, c="red", linestyle='-', alpha=0.9)

                ax.text(_obj_xs[-1], _obj_ys[-1], "euav%d" % (_obj_i), c="red", fontsize=10)
            
            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.legend()
            plt.show()

        return _obj_tracks

if __name__ == "__main__":
    swarm_intent_dir = r"data\manual_intention_recog"
    swarm_intent_file = osp.join(swarm_intent_dir, "fast_pass_through_no03.xlsx")
    # swarm_intent_file = osp.join(swarm_intent_dir, "ext_search_no01.xlsx")

    # 参数说明：
    # swarm_intent_file: 人工构建的无人机轨迹文件，格式为xlsx，包含无人机轨迹数据
    # interp_scale: 轨迹插值比例，表示将原始轨迹插值成interp_scale倍长度的轨迹点
    intent_exh = SwarmIntentExhibitor(swarm_intent_file, interp_scale=20, vis=False)

    # intent_exh.pack_to_objtracks 基于人工构建的轨迹，打包生成objtracks无人机轨迹对象
    # 参数说明：
    # lookback_start: 起始时间点，表示从输入轨迹的末尾，向前回溯lookback_start个轨迹点
    # lookback_len: 回溯长度，表示从lookback_start时间点开始，回溯lookback_len个轨迹点
    # vis: 是否可视化轨迹（完整轨迹点使用绿色虚线表示，pack_to_objtracks提取的轨迹点使用红色实线表示）
    test_objtracks = intent_exh.pack_to_objtracks(lookback_start=10, lookback_len=30, vis=True)
    print("Objects speeds: %s" % ([_obj.move_speed() for _obj in test_objtracks]))

    test_facilities = basic_units.BasicFacilities()

    test_sw = 3
    if test_sw == 1:
        # 功能类SingleUavBehavior，用于分析单个无人机的行为，包括加减速、转弯次数、方向变化、和建筑设施的距离等等
        # analyze_win: 分析窗口，表示从输入轨迹的末尾，向前回溯analyze_win个轨迹点，必须比轨迹本身短
        _test_objbehavs = [int_rec.SingleUavBehavior(_obj_trk, analyze_win=19) for _obj_trk in test_objtracks]
        
        print("[Speed Ups]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _acc_bool, _acc_score, _acc_ratio = _o_behav.speed_up(return_val=True) # 判断analyze_win部分轨迹是否加速
            print("Obj%d, speed-up: %s, acc-ratio: %.3f" % (_o_i, _acc_bool, _acc_ratio))
        
        print("[Slow Downs]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _dac_bool, _dac_score, _dac_ratio = _o_behav.slow_down(return_val=True) # 判断analyze_win部分轨迹是否减速
            print("Obj%d, slow-down: %s, dac-ratio: %.3f" % (_o_i, _dac_bool, _dac_ratio))
        
        print("[Orient Change]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _orc_bool, _orc_score, _orc_degrees = _o_behav.orient_change(return_val=True) # 判断analyze_win部分轨迹是否发生朝向角度变化
            print("Obj%d, orient-change: %s, diff-angle: %.3f" % (_o_i, _orc_bool, _orc_degrees))

        print("[Turning Frequency]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _tfr_bool, _tfr_score, _tfr_freq = _o_behav.turning_frequency(return_val=True) # 判断analyze_win部分轨迹的转向次数是否超过阈值
            print("Obj%d, turning-freq: %s, freq: %.3f" % (_o_i, _tfr_bool, _tfr_freq))
        
        print("[Directing Facilities]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            # 判断analyze_win部分轨迹是否指向建筑设施，每个轨迹点都给出指向的建筑设施名称（雷达、指挥中心、无人机机场）
            _direct_ts, _direct_facs, _direct_scores, _direct_angles = _o_behav.directing_facilities(test_facilities, return_val=True)
            print("time-stamps: %s, facilities: %s" % (_direct_ts, _direct_facs))
        
        print("[Closing Facilities]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            # 给出analyze_win部分轨迹中，每个轨迹点距离建筑设施的距离
            _distancing_stats, _closing_ratios = _o_behav.distance_to_facilities(test_facilities, return_val=True)
            print("closing to facilities: %s, close-ratios: %s" % (_distancing_stats, _closing_ratios))
        
        print("[Probing Facilities]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            # 根据预设的探测范围（1km），给出analyze_win部分轨迹中，每个轨迹点探测到的建筑设施名称
            _probing_stats = _o_behav.probed_facilities(test_facilities)
            print("probed facilities: %s" % (_probing_stats))
        
        print("[Shrinking Speed]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            # 根据analyze_win部分轨迹的最后三个点朝向，整个analyze_win部分的运动速度，估计无人机到达建筑设施的时间
            _arrive_bools, _arrive_times = _o_behav.estimate_arrive_time(test_facilities)
            print("arrive-bools: %s, arrive-times: %s" % (_arrive_bools, _arrive_times))
    
    elif test_sw == 2:
        # 测试涉及多个无人机的行为和特性评估方法
        # 参数：analyze_win表示分析窗口大小，表示从输入轨迹的末尾，向前回溯每个无人机轨迹的analyze_win个点，必须比轨迹本身短
        _test_objbehavs = int_rec.MultiUavsBehavior(test_objtracks, analyze_win=19)
        
        # 评估analyze_win部分轨迹中，无人机集群是否发生队形的收缩（间距缩小）
        _shrink_bool, _shrink_ratio = _test_objbehavs.shrink_fleet(return_val=True)
        print("Shrink-fleet: %s, shrink-ratio: %.3f" % (_shrink_bool, _shrink_ratio))

        # 评估analyze_win部分轨迹中，无人机集群是否发生队形的扩散（间距放大）
        _spread_bool, _spread_ratio = _test_objbehavs.spread_fleet(return_val=True)
        print("Spread-fleet: %s, spread-ratio: %.3f" % (_spread_bool, _spread_ratio))

        # 评估analyze_win部分轨迹中，被1个以上无人机朝向的建筑设施
        _targeting_targets_info = _test_objbehavs.targeting_same_facilities(test_facilities)
        print("Targeting-facilities: %s, arrive-uavs: %s" % ([_k for _k in _targeting_targets_info.keys()], 
                                                              [_targeting_targets_info[_k]['uav_ids'] for _k in _targeting_targets_info.keys()]))

        # 基于analyze_win部分轨迹的最后两个点位置和运动方向，给出无人集群的分群和每个分群的队形识别结果
        _cluster_labels, _form_types, _form_names = _test_objbehavs.infer_cluster_formations()
        print("Fleet-Formations: %s, formation-names: %s" % (_form_types, _form_names))
        
        # 结合analzye_win部分轨迹点的分群和队形识别结果，进行敌方集群的意图识别（结合基础分群和队形识别结果）
    
    elif test_sw == 3:
        # 对一组无人机的单机和多机行为特性进行综合分析
        int_extrctr = int_rec.IntentFactorExtractor(test_objtracks, test_facilities, analyze_win=18)
        factor_knows = int_extrctr.get_knows()
        
        intent_inferor = int_rec.IntentionEvaluator([_k for _k in factor_knows if _k != ''])
        intent_knows = intent_inferor.get_knows()

    elif test_sw == 4:
        pass
