""" 给出我方防御圈I和防御II里面有哪些敌方的集群编组
"""
import numpy as np
import matplotlib.path as mpltPath

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
