import numpy as np
import time
import json

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from formation_recognition import basic_units

class SplitClusters(object):
    """ 基于给定的目标运动轨迹，进行聚类划分
    """
    def __init__(self, swarm_objs:list[basic_units.ObjTracks], spatial_scale=None, memory_len:int=3):
        self.glb_cfgs = basic_units.GlobalConfigs()

        self.swarm_objs = swarm_objs
        self.num_objs = len(swarm_objs)
        
        self.anlz_trajs = []
        self.clusters_list = []
        self.spatial_scale = spatial_scale
        self.memory_len = memory_len
        
        self._make_clustering()

    def normalize_locs_dists(self, swarm_locs):     
        _swarm_locs = np.array(swarm_locs).reshape(self.num_objs, -1)
        _mutual_dists = np.sqrt(np.sum((swarm_locs[:, np.newaxis, :] - swarm_locs[np.newaxis, :, :])**2, axis=2))
        _mutual_dists = _mutual_dists[np.triu_indices(self.num_objs, k=1)]

        # 找出与集群基本距离接近的距离参数
        if self.spatial_scale is not None:
            spatial_scale = self.spatial_scale
        else:
            spatial_scale = 1.0

        _suitable_dist_min = self.glb_cfgs.SWARM_MUTUAL_DISTANCE * spatial_scale
        _suitable_dist_max = self.glb_cfgs.SWARM_MUTUAL_DISTANCE * 2.0 * spatial_scale

        _inrange_dists = _mutual_dists[np.logical_and(_mutual_dists >= _suitable_dist_min, _mutual_dists <= _suitable_dist_max)]
        
        if len(_inrange_dists) <= 0:
            _inrange_dists_mean = self.glb_cfgs.SWARM_MUTUAL_DISTANCE * spatial_scale
        else:
            _inrange_dists_mean = np.mean(_inrange_dists)
        
        return swarm_locs / _inrange_dists_mean
    
    def normalize_direct_angles(self, direct_xys):
        _near_angle_rad = self.glb_cfgs.SWARM_NEAR_ANGLE_DEGREES / 180 * np.pi
        _direct_vec_scale = 1 / np.sqrt(2 * (1 - np.cos(_near_angle_rad)))

        # 归一化方向向量
        return direct_xys * _direct_vec_scale

    def normalize_speeds(self, swarm_speeds):
        _suitable_speed_min = self.glb_cfgs.SWARM_AVERAGE_SPEED * 0.1
        _suitable_speed_max = self.glb_cfgs.SWARM_AVERAGE_SPEED * 1.5

        _inrange_speeds = swarm_speeds[np.logical_and(swarm_speeds >= _suitable_speed_min, swarm_speeds <= _suitable_speed_max)]
        if len(_inrange_speeds) <= 0:
            _inrange_speeds_mean = self.glb_cfgs.SWARM_AVERAGE_SPEED
        else:
            _inrange_speeds_mean = np.mean(_inrange_speeds)

        return swarm_speeds / _inrange_speeds_mean
    
    def contiguous_split(self, swarm_locs):
        self.anlz_trajs.append(swarm_locs)
        
        if len(self.anlz_trajs) > self.memory_len:
            self.anlz_trajs = self.anlz_trajs[-self.anlz_trajs_len:]

    def _make_clustering(self):
        _tic = time.time()

        _swarm_feats = []
        
        # 计算所有对象的位置
        _swarm_locs = np.array([_obj.last_location() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)

        _swarm_locs_norm = self.normalize_locs_dists(_swarm_locs)
        _swarm_feats.append(_swarm_locs_norm)
        
        # 计算所有对象的方向
        _swarm_directs = np.array([_obj.move_direction() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)
        # _swarm_directs_dims = _swarm_directs.shape[1]
        _swarm_directs_norm = self.normalize_direct_angles(_swarm_directs)
        _swarm_feats.append(_swarm_directs_norm)
        
        # 计算所有对象的速度
        _swarm_speeds = np.array([_obj.move_speed() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)
        _swarm_speeds_norm = self.normalize_speeds(_swarm_speeds)
        _swarm_feats.append(_swarm_speeds_norm)
        
        _swarm_comb_feats = np.concatenate(_swarm_feats, axis=1)
        
        _dbscan = DBSCAN(eps=1.5, min_samples=1)
        _clusters = _dbscan.fit(_swarm_comb_feats)

        self.clusters_list.append(_clusters)
        
        if len(self.clusters_list) > self.memory_len:
            self.clusters_list = self.clusters_list[-self.memory_len:]
        
        _calcu_time = time.time() - _tic

        print("Clustering time: {:.3f}s".format(_calcu_time))
    
    def last_clustering(self):
        return self.clusters_list[-1].labels_
    
    def formated_cluster_result(self):
        """
        返回格式化后的集群分组结果
        数据格式: {eSwarm1: [eUav1, eUav2, …], eSwarm2: [eUav5, eUav7, …]}
        """
        clusters = self.last_clustering()
        enemy_uavs_clusters = {}

        for idx, label in enumerate(clusters):
            swarm_key = f"eSwarm{label + 1}"

            _cur_swarm_obj = self.swarm_objs[idx]
            if _cur_swarm_obj.id is None:
                enemy_uavs_clusters.setdefault(swarm_key, []).append(f"eUav{idx + 1}")
            else:
                enemy_uavs_clusters.setdefault(swarm_key, []).append(_cur_swarm_obj.id)

        _result_json_str = json.dumps(enemy_uavs_clusters, ensure_ascii=False)
        return _result_json_str
    
    def show_clusters(self):
        _clustering_labels = self.clusters_list[-1].labels_

        unique_labels = np.unique(_clustering_labels)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'brown', 'pink', 'gray', 'olive']  # 颜色列表，支持多个簇
        
        _start_locs = np.array([_obj.start_location() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)
        _stop_locs = np.array([_obj.last_location() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)

        # 创建绘图
        plt.figure(figsize=(8, 6))

        for label in unique_labels:
            # 获取属于当前聚类的点
            cluster_points = _stop_locs[_clustering_labels == label]
            
            # 绘制聚类点
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        label=f"Cluster {label}", color=colors[label], s=100, edgecolor='black', zorder=2)
            
            for _start_x, _start_y, _stop_x, _stop_y in np.concatenate([_start_locs, _stop_locs], axis=1):
                plt.arrow(_start_x, _start_y, _stop_x - _start_x, _stop_y - _start_y, head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            if len(cluster_points) == 2:
                # 如果聚类点数等于2，绘制连接这两个点的线段
                plt.plot(cluster_points[:, 0], cluster_points[:, 1], 
                        color=colors[label], linewidth=2, zorder=3)
            
            elif len(cluster_points) > 2:
                # 如果聚类点数大于2，绘制外部包络曲线（凸包）
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                # 绘制凸包曲线
                plt.fill(hull_points[:, 0], hull_points[:, 1], 
                        color=colors[label], alpha=0.3, edgecolor=colors[label], zorder=1)
        
        # 图形美化
        plt.title("Cluster Visualization with Convex Hulls", fontsize=14)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.axhline(0, color='black', linewidth=0.5, linestyle="--")  # x轴参考线
        plt.axvline(0, color='black', linewidth=0.5, linestyle="--")  # y轴参考线
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.show()
