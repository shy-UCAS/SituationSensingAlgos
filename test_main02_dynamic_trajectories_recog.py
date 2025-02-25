""" 测试脚本：动态加载人工设计的运行轨迹，以动态的方式进行合群/分群以及队形的识别
"""
import os, os.path as osp
import glob

import numpy as np
import pandas as pd
import json
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from formation_recognition import basic_units
from formation_recognition import clusters_recognition as clus_rec
from formation_recognition import formation_recognition as form_rec
from formation_recognition import defence_breach_analyze as brch_anz
from formation_recognition import defence_ability_analyze as def_eval
from formation_recognition import intention_recognition as int_rec

class TrajectoryExhibitor(object):
    def __init__(self, file_path, coord_scale=1, interp_scale=3, scale_time=True):
        """
        初始化类，读取 Excel 文件并存储轨迹数据。
        
        :param file_path: Excel 文件路径
        :param coord_scale: 坐标缩放比例
        """
        self.file_path = file_path
        self.coord_scale = coord_scale
        self.interp_scale = interp_scale
        self.scale_time = scale_time
        
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """
        加载 Excel 文件中的数据。
        假设 Excel 文件的表头为：时间，目标1_x，目标1_y，目标2_x，目标2_y，...
        """
        self.data = pd.read_excel(self.file_path)
        self.facilities = basic_units.BasicFacilities()
            
        self.time = self.data['time']  # 第一列为时间
        self.trajectories = self.data.iloc[:, 1:] # 后面的列为轨迹数据
        
        if not self.scale_time:
            _new_time = np.linspace(self.time.iloc[0], self.time.iloc[-1], len(self.time) * self.interp_scale)
        else:
            _old_interval = self.time.iloc[-1] - self.time.iloc[0]
            _new_interval = _old_interval * self.interp_scale
            
            # import pdb; pdb.set_trace()
            _new_time = np.linspace(self.time.iloc[0], self.time.iloc[0] + _new_interval, int(len(self.time) * self.interp_scale))
            
            _orig_time = self.time.to_numpy()
            _orig_to_start_difftimes = _orig_time - _orig_time[0]
            self.time = pd.Series(_orig_time[0] + _orig_to_start_difftimes * self.interp_scale)
        
        _interp_coords_comb = np.zeros((len(_new_time), self.trajectories.shape[1]))
        
        for _col_i in range(self.trajectories.shape[1]):
            _cur_coords = self.trajectories.iloc[:, _col_i]
            _interp_coords = interp1d(self.time, _cur_coords, kind='cubic')(_new_time)
            _interp_coords = gaussian_filter1d(_interp_coords, sigma=0.5)
            
            _interp_coords_comb[:, _col_i] = _interp_coords
        
        self.time = pd.Series(_new_time)
        self.trajectories = pd.DataFrame(_interp_coords_comb, columns=self.trajectories.columns)

    def get_points(self):
        """
        以 yield 的方式逐行返回轨迹点。
        每次返回一个时间步的所有目标的位置坐标。
        """
        print("nunm:",len(self.trajectories))
        for idx, row in self.trajectories.iterrows():
            time_step = self.time.iloc[idx]
            # 每个时间步目标的位置列表 (目标1_x, 目标1_y, 目标2_x, 目标2_y, ...)
            yield time_step, row.values.reshape(-1, 2)  # 每两个值为一个目标的 (x, y)
    
    def animate_trajectory(self, clusterings=None, formtypes=None, interval=30):
        """
        动画显示所有目标的运动轨迹。
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(self.trajectories.iloc[:, ::2].min().min() - 1, 
                    self.trajectories.iloc[:, ::2].max().max() + 1)
        ax.set_ylim(self.trajectories.iloc[:, 1::2].min().min() - 1, 
                    self.trajectories.iloc[:, 1::2].max().max() + 1)
        
        ax.set_title("Trajectory Animation")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)

        fig.tight_layout()
        
        # 补全输入的clustering聚类划分结果，在前面不足的部分补充None
        _clustering_pad_len = len(self.trajectories) - len(clusterings)
        _clustering_padded = [None] * _clustering_pad_len + clusterings
        
        # 补全输入的formtype队形结果，在前面不足的部分补充None
        _formtypes_pad_len = len(self.trajectories) - len(formtypes)
        _formtypes_padded = [None] * _formtypes_pad_len + formtypes
        
        # 初始化每个目标的轨迹线和当前位置标记
        num_targets = self.trajectories.shape[1] // 2
        
        lines = [ax.plot([], [], lw=2)[0] for _ in range(num_targets)]  # 轨迹线
        points = [ax.plot([], [], 'o')[0] for _ in range(num_targets)]  # 当前位置点
        
        clust_connect_lines = []
        clust_convex_hulls = []
        clust_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']  # 颜色列表，支持多个簇
        formtype_labels = []

        # 更新函数
        def update(frame):
            clust_connect_lines.clear()
            clust_convex_hulls.clear()
            formtype_labels.clear()
            
            time_step, points_data = frame

            # 绘制主要设施和防御圈多边形
            ax.fill(self.facilities.RING2_XYS[:, 0], self.facilities.RING2_XYS[:, 1], color='yellow', alpha=0.3)
            ax.fill(self.facilities.RING1_XYS[:, 0], self.facilities.RING1_XYS[:, 1], color='red', alpha=0.3)
            
            for i, (line, point) in enumerate(zip(lines, points)):
                # 提取当前目标的轨迹和位置
                x_data = self.trajectories.iloc[:frame[0]+1, i*2].values
                y_data = self.trajectories.iloc[:frame[0]+1, i*2+1].values
                
                line.set_data(x_data, y_data)
                point.set_data([points_data[1][i, 0]], [points_data[1][i, 1]])
            
            # print(f"time step: {time_step}")
            _cur_cluster_labels = _clustering_padded[time_step]
            _cur_cluster_formtype = _formtypes_padded[time_step]

            if _cur_cluster_labels is not None:
                _uniq_labels = np.unique(_cur_cluster_labels)
                
                for _lbl in _uniq_labels:
                    _cur_lbl_bools = _cur_cluster_labels == _lbl
                    
                    if np.sum(_cur_lbl_bools) == 2:
                        _conn_line = ax.plot(points_data[1][_cur_lbl_bools, 0], points_data[1][_cur_lbl_bools, 1], color=clust_colors[_lbl], linewidth=2, zorder=3)[0]
                        clust_connect_lines.append(_conn_line)
                        
                    elif np.sum(_cur_lbl_bools) > 2:
                        _hull = ConvexHull(points_data[1][_cur_lbl_bools, :])
                        _hull_points = points_data[1][_cur_lbl_bools, :][_hull.vertices, :]
                        _hull_poly = ax.fill(_hull_points[:, 0], _hull_points[:, 1], color=clust_colors[_lbl], linewidth=2, zorder=1)[0]
                        clust_convex_hulls.append(_hull_poly)

                        # 在当前的集群坐标旁边标记队形类型信息
                        _formtype = _cur_cluster_formtype[_lbl]
                        if _formtype is not None:
                            _formtype_label = ax.text(_hull_points.mean(axis=0)[0], _hull_points.mean(axis=0)[1], f"{_formtype}", color='black', fontsize=12, zorder=5)
                            formtype_labels.append(_formtype_label)

            return lines + points + clust_connect_lines + clust_convex_hulls + formtype_labels

        # 动画帧
        frames = list(self.get_points())

        # ani = FuncAnimation(fig, update, frames=enumerate(frames), interval=interval, blit=True)
        ani = FuncAnimation(fig, update, frames=enumerate(frames), interval=interval, blit=True, save_count=100)

        plt.show()

# 使用示例
if __name__ == "__main__":
    # 假设文件名为 "trajectory.xlsx"
    _root_dir = osp.dirname(osp.abspath(__file__))
    _man_trajs_dir = osp.join(_root_dir, 'data', 'manual_formation_recog')

    _man_trajs_infos = [{'filename': 'fleet_form_trj01_shrink1.0.xlsx', 'scale': 25.0},
                        {'filename': 'fleet_form_trj02_shrink1.5.xlsx', 'scale': 35.0},
                        {'filename': 'fleet_form_trj03_shrink1.2.xlsx', 'scale': 40.0},]
    
    _test_idx = 2
    processor = TrajectoryExhibitor(osp.join(_man_trajs_dir, _man_trajs_infos[_test_idx]['filename']), interp_scale=_man_trajs_infos[_test_idx]['scale'])
    # # 显示processor中的轨迹
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_xlabel("X Position")
    # ax.set_ylabel("Y Position")
    # ax.set_title("UAV Trajectories")
    #
    # # 获取每个时间步的无人机轨迹
    # for time_step, positions in processor.get_points():
    #     # 假设每个时间步包含多个目标的位置 (positions为一个2D数组，每一行是一个目标的(x, y)位置)
    #     for pos in positions:
    #         ax.plot(pos[0], pos[1], 'o', markersize=5)  # 绘制每个时间步的目标位置
    # # 显示图形
    # plt.show()

    # 遍历轨迹点
    _trj_counter = 0
    _trj_objs_list = []
    
    _clustering_lists = []
    _formtypes_lists = []
    _formtype_names_lists = []
    
    workspace_dir = osp.dirname(osp.abspath(__file__))
    weight_fpath = osp.join(workspace_dir, 'pretrained_weights', 'formation_recognition', 'form_recog_model_192000.pth')
    
    form_types = ['vertical', 'horizontal', 'echelon', 'wedge', 'circular', 'random']
    _formtype_rec = form_rec.FormationRecognizer(form_types=form_types, num_layers=3, hidden_size=64, pretrained_weights=weight_fpath)
    
    _ring_breacher = brch_anz.DefRingBreach()
    _def_evaluator = def_eval.DefenseEvaluator()
    _basic_facilities = basic_units.BasicFacilities()

    _prev_positions = None
    
    for time_step, positions in processor.get_points():
        print("\n")
        print(f"Time: {time_step}, Positions: {positions}")

        if _trj_counter <= 0:
            _num_objs = len(positions)
            _trj_objs_list = [basic_units.ObjTracks([_pos[0]], [_pos[1]], ts=[time_step], id="euav%02d" % (_iter + 1)) for _iter, _pos in enumerate(positions)]
            
            _prev_positions = positions
            _trj_counter = _trj_counter + 1
            
            continue
        
        # add new positions to existing trajectories
        for _p_iter, _pos in enumerate(positions):
            _trj_objs_list[_p_iter].append_location(_pos[0], _pos[1], t=time_step)

        # 通过聚类分析计算编队分组
        _cur_clust_split = clus_rec.SplitClusters(_trj_objs_list, spatial_scale=_man_trajs_infos[_test_idx]['scale'])
        _clustering_lists.append(_cur_clust_split.last_clustering())
        
        # predict the formtype of clusters
        _vis_formtype = False
        _clust_formtypes, _clust_formtype_names = _formtype_rec.infer_movements(_prev_positions, positions, _cur_clust_split.last_clustering(), vis=_vis_formtype)
        _formated_formtypes = _formtype_rec.formated_formtype_result(_cur_clust_split.last_clustering(), _clust_formtypes, _clust_formtype_names)

        # analyze the defence ring breaching status
        _r1_clust_idxs, _r2_clust_idxs, _brch_frmt_str = _ring_breacher.infer_rings_breach_groups(_trj_objs_list, _cur_clust_split.last_clustering(), formated_output=True)
        
        if len(_trj_objs_list[0]) > 10:
            _threat_evaluator = int_rec.ThreatEvaluator(_trj_objs_list, _basic_facilities)
            _cur_clusters_threats = _threat_evaluator.estimate_threats()
            # print("Num threating clusters: %d, threating scores: %s , uav_id:%s " % (len(_cur_clusters_threats), [_th['threat_score'] for _th in _cur_clusters_threats],[_id['euav_ids'] for _id in _cur_clusters_threats]))
            print("_cur_clusters_threats", json.dumps(_cur_clusters_threats, indent=4, ensure_ascii=False))
        if len(_trj_objs_list[0]) > 18:
            _int_extrctr = int_rec.IntentFactorExtractor(_trj_objs_list, _basic_facilities, analyze_win=18)
            _int_factor_knows = _int_extrctr.get_knows()

            _intent_inferior = int_rec.IntentionEvaluator([_k for _k in _int_factor_knows if _k != ''])
            _intent_knows = _intent_inferior.get_knows()
        #
        # if len(_brch_frmt_str) > 0:
        #     # import pdb; pdb.set_trace()
        #     _def_results = _def_evaluator.evaluate_defense(_brch_frmt_str)

        _formtypes_lists.append(_clust_formtypes)
        _formtype_names_lists.append(_clust_formtype_names)
        
        print("Timestep: %d, clustering: %s" % (time_step, _cur_clust_split.last_clustering()))
        
        _prev_positions = positions
        _trj_counter = _trj_counter + 1

    # for obj in _trj_objs_list:
    #     print(f"ID: {obj.id}, X: {obj.xs}, Y: {obj.ys}")

    # 播放动画
    processor.animate_trajectory(_clustering_lists, _formtype_names_lists)