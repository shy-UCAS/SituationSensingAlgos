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

        # import pdb; pdb.set_trace()
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

            for _obj_i, _obj_trk in enumerate(_obj_tracks):
                _ttl_xs, _ttl_ys = self.uav_xys[_obj_i, :, 0], self.uav_xys[_obj_i, :, 1]
                _obj_xs, _obj_ys = _obj_trk.last_n_locations(lookback_len)

                ax.plot(_ttl_xs, _ttl_ys, c="green", linestyle='--', alpha=0.5)
                ax.plot(_obj_xs, _obj_ys, c="red", linestyle='-', alpha=0.9)

                ax.text(_obj_xs[-1], _obj_ys[-1], "euav%d" % (_obj_i), c="red", fontsize=10)
            
            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.show()
        return _obj_tracks
        
if __name__ == "__main__":
    swarm_intent_dir = r"data\manual_intention_recog"
    # swarm_intent_file = osp.join(swarm_intent_dir, "fast_pass_through_no03.xlsx")
    swarm_intent_file = osp.join(swarm_intent_dir, "ext_search_no01.xlsx")

    intent_exh = SwarmIntentExhibitor(swarm_intent_file, vis=False)
    test_objtracks = intent_exh.pack_to_objtracks(lookback_start=0, lookback_len=12, vis=True)

    test_sw = 1
    if test_sw == 1:
        # 测试单个
        _test_objbehavs = [int_rec.SingleUavBehavior(_obj_trk, analyze_win=len(_obj_trk)) for _obj_trk in test_objtracks]
        
        print("[Speed Ups]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _acc_bool, _acc_ratio = _o_behav.speed_up(return_val=True)
            print("Obj%d, speed-up: %s, acc-ratio: %.3f" % (_o_i, _acc_bool, _acc_ratio))
        
        print("[Slow Downs]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _dac_bool, _dac_ratio = _o_behav.slow_down(return_val=True)
            print("Obj%d, slow-down: %s, dac-ratio: %.3f" % (_o_i, _dac_bool, _dac_ratio))
        
        print("[Orient Change]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _orc_bool, _orc_degrees = _o_behav.orient_change(return_val=True)
            print("Obj%d, orient-change: %s, diff-angle: %.3f" % (_o_i, _orc_bool, _orc_degrees))

        print("[Turning Frequency]:")
        for _o_i, _o_behav in enumerate(_test_objbehavs):
            _tfr_bool, _tfr_freq = _o_behav.turning_frequency(return_val=True)
            print("Obj%d, turning-freq: %s, freq: %.3f" % (_o_i, _tfr_bool, _tfr_freq))