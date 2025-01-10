import os, os.path as osp
import glob
import json
import time
from enum import member

import numpy as np
import re
from collections import defaultdict

from Cython.Shadow import nonecheck
from matplotlib.font_manager import json_dump

from formation_recognition import basic_units
from formation_recognition import clusters_recognition as clus_rec
from formation_recognition import formation_recognition as form_rec
from formation_recognition import defence_breach_analyze as def_breach
from formation_recognition import intention_recognition as int_rec
from formation_recognition.basic_units import ObjTracks
import matplotlib.pyplot as plt

WORKSPACE_DIR = osp.dirname(osp.abspath(__file__))
FORM_WEIGHT_FPATH = osp.join(WORKSPACE_DIR, 'pretrained_weights', 'formation_recognition', 'form_recog_model_192000.pth')

def get_uavs_clusters(uav_coords_str):
    """ 获取无人机集群分组情况 """
    uav_coords_dict = json.loads(uav_coords_str)
    uavs_tracks = []
    for uav_id, uav_coords in uav_coords_dict.items():
        _uav_track = basic_units.ObjTracksJson(uav_coords, id=uav_id)
        uavs_tracks.append(_uav_track)

    _split_clusters = clus_rec.SplitClusters(uavs_tracks)

    return _split_clusters.formated_cluster_result()

def recog_fleet_formtype(uav_coords_str, clustering_str):
    """ 识别各个无人机分组队形 """
    global FORM_WEIGHT_FPATH

    uav_coords_dict = json.loads(uav_coords_str)
    clustering_dict = json.loads(clustering_str) # {"eSwarm1": ["euav1", "euav2", "euav3"], "eSwarm2": ["euav4"], "eSwarm3": ["euav5"]}

    recog_counter = 0
    clusters_formtypes = []
    clusters_uavs_coords = []

    for swarm_id, uav_ids in clustering_dict.items():
        if len(uav_ids) <= 2:
            clusters_formtypes.append('none')
            clusters_uavs_coords.append(None)
            continue
        
        recog_counter += 1

        _cur_cluster_prvlocs = []
        _cur_cluster_curlocs = []
        
        for _uav_id in uav_ids:
            _cur_cluster_prvlocs.append([uav_coords_dict[_uav_id]['xs'][-2], uav_coords_dict[_uav_id]['ys'][-2]])
            _cur_cluster_curlocs.append([uav_coords_dict[_uav_id]['xs'][-1], uav_coords_dict[_uav_id]['ys'][-1]])

        _cur_cluster_prvlocs = np.array(_cur_cluster_prvlocs)
        _cur_cluster_curlocs = np.array(_cur_cluster_curlocs)

        clusters_formtypes.append('unknown')
        clusters_uavs_coords.append((_cur_cluster_prvlocs, _cur_cluster_curlocs))

    if recog_counter == 0:
        return clusters_formtypes

    form_types = ['vertical', 'horizontal', 'echelon', 'wedge', 'circular', 'random']
    _formtype_rec = form_rec.FormationRecognizer(form_types=form_types, num_layers=3, hidden_size=64, pretrained_weights=FORM_WEIGHT_FPATH)
    
    for _cluster_i, _formtype in enumerate(clusters_formtypes):
        if _formtype == 'none':
            continue
        
        _prv_positions, _cur_positions = clusters_uavs_coords[_cluster_i]
        _formtype_id, _formtype_name = _formtype_rec.infer_movements(_prv_positions, _cur_positions, None, vis=False)
        clusters_formtypes[_cluster_i] = _formtype_name

    # 格式化输出
    uav_ids = [_id for _id in uav_coords_dict.keys()]
    formtypes_result = {'timestamp': uav_coords_dict[uav_ids[0]]['ts'][-1],
                        'swarms': []}
    
    for _iter, (_swarm_id, _uav_ids) in enumerate(clustering_dict.items()):
        formtypes_result['swarms'].append({'swarm_no': _swarm_id,
                                           'members': _uav_ids,
                                           'formation': clusters_formtypes[_iter]})
    return json.dumps(formtypes_result)

def ring_breach_infer(uav_coords_str, clustering_str):
    """ 基于输入的无人机轨迹数据、无人机分群情况，给出各轨迹点上面无人机的突破情况 """
    _breacher = def_breach.DefRingBreach()

    uav_coords_dict = json.loads(uav_coords_str)
    clustering_dict = json.loads(clustering_str) # {"eSwarm1": ["euav1", "euav2", "euav3"], "eSwarm2": ["euav4"], "eSwarm3": ["euav5"]}

    uavs_tracks = []
    for uav_id, uav_coords in uav_coords_dict.items():
        _uav_track = basic_units.ObjTracksJson(uav_coords, id=uav_id)
        uavs_tracks.append(_uav_track)
    
    _clustering_labels = np.zeros((len(uavs_tracks), 1), dtype=int)
    _uavs_ids = [_trk.id for _trk in uavs_tracks]
    for _swrm_i, _swrm_nm in enumerate(clustering_dict):
        _cur_swarm_uav_ids = clustering_dict[_swrm_nm]
        _cur_uav_idxs = [_uavs_ids.index(_uid) for _uid in _cur_swarm_uav_ids]
        _clustering_labels[_cur_uav_idxs] = _swrm_i
    
    _r1_uidxs, _r2_uidxs, _formated_output = _breacher.infer_rings_breach_groups(uavs_tracks, _clustering_labels, formated_output=True)

    return _formated_output



def infer_enemy_intention(uav_coords_str, facilities_str):
    """ 基于输入的无人机轨迹数据、无人机分群情况、我方设施位置分布，给出敌方无人机集群中各分组的意图信息 """
    #json格式字符串转换
    uav_coords_dict = json.loads(uav_coords_str)
    obj_tracks = []
    for uav_id, uav_data in uav_coords_dict.items():
        # 从字典中提取轨迹数据
        xs = uav_data["xs"]
        ys = uav_data["ys"]
        ts = uav_data.get("ts", None)  # 如果没有 'ts' 键，则为 None
        zs = uav_data.get("zs", None)  # 如果没有 'zs' 键，则为 None
        obj_track = ObjTracks(xs=xs, ys=ys, ts=ts, zs=zs, id=uav_id)
        # 将 ObjTracks 对象添加到列表
        obj_tracks.append(obj_track)

    int_extrctr = int_rec.IntentFactorExtractor(obj_tracks, facilities_str, analyze_win=18)
    factor_knows = int_extrctr.get_knows()

    intent_inferor = int_rec.IntentionEvaluator([_k for _k in factor_knows if _k != ''])
    intent_knows = intent_inferor.get_knows()

    factor_knows_clustering = []

    for know in factor_knows:
        if 'in_group' in know or 'tight_fleet' in know:
            factor_knows_clustering.append(know)

    for clustering_str in factor_knows_clustering:
        lines = clustering_str.split('\n')
        clusters_info = {}
        group_counter = 1
        # 遍历每行信息，提取每个聚类的情况
        for line in lines:
            if "in_group" in line:
                match = re.search(r"in_group\(\[(.*?)\],", line)
                if match:
                    in_group_cleaned = match.group(1)  # 提取匹配的组内容
                    group_key = f"group{group_counter}"  # 动态生成 group1, group2, ...
                    clusters_info[group_key] = in_group_cleaned  # 存储每个分组的成员
                    group_counter += 1  # 增加 group 编号
        print(f"Clusters result: {clusters_info}")


    cluster_intentions = {key: [] for key in clusters_info}
    # 遍历意图和概率
    for intent, probability in intent_knows.items():
        # print("intent, probability",str(intent),probability)
        match = re.match(r"(\w+_\w+)\(([^)]+)\)", str(intent))
        if match:
            function_name  = match.group(1)
            argument  = match.group(2)
            first_quantity = argument.split(',')[0].strip()
            for swarm, members_list in clusters_info.items():
                members_list = members_list.split(", ")
                for member in members_list:
                    if  first_quantity in member:  # 如果目标是该分群的成员
                        cluster_intentions[swarm].append((function_name, argument, probability))
    print("cluster_intentions",cluster_intentions)

    _enemy_clusters_intention = []
    # 统计每个分群中的意图和概率
    for swarm, intentions in cluster_intentions.items():
        # 对每个分群中的意图按概率排序，并输出
        intention_probs = defaultdict(list)
        for action, target, prob in intentions:
            intention_probs[action].append((target, prob))
        action_max_probs = {}
        # 计算每个 action 的最高概率并存储
        for action, targets in intention_probs.items():
            sorted_targets = sorted(targets, key=lambda x: x[1], reverse=True)
            highest_prob = sorted_targets[0][1]
            action_max_probs[action] = highest_prob
        # 对所有 action 按最高概率降序排列
        sorted_actions = sorted(action_max_probs.items(), key=lambda x: x[1], reverse=True)

        _enemy_clusters_intention.append({
            'cluster_idx': swarm,
            'euav_ids': [clusters_info.get(swarm, [])],
            'intentions':[action[0] for action in sorted_actions if sorted_actions],
            'probabilities':[action[1] for action in sorted_actions if sorted_actions],
        })
    _formated_output = json.dumps(_enemy_clusters_intention, indent=4, ensure_ascii=False)
    return _formated_output

def infer_enemy_threats(uav_coords_str, facilities_str):
    """ 基于输入的无人机轨迹数据、无人机分群情况、我方的设施位置、发生攻击的时间，给出敌方无人机集群中各分组的威胁信息 """
    #json格式字符串转换
    uav_coords_dict = json.loads(uav_coords_str)
    obj_tracks = []
    for uav_id, uav_data in uav_coords_dict.items():
        # 从字典中提取轨迹数据
        xs = uav_data["xs"]
        ys = uav_data["ys"]
        ts = uav_data.get("ts", None)  # 如果没有 'ts' 键，则为 None
        zs = uav_data.get("zs", None)  # 如果没有 'zs' 键，则为 None
        obj_track = ObjTracks(xs=xs, ys=ys, ts=ts, zs=zs, id=uav_id)
        # 将 ObjTracks 对象添加到列表
        obj_tracks.append(obj_track)
    _clustering_lists = []
    _formtypes_lists = []
    _formtype_names_lists = []
    _formated_output = []
    _cur_clust_split = clus_rec.SplitClusters(obj_tracks, spatial_scale=_man_trajs_infos[_test_idx]['scale'])
    _clustering_lists.append(_cur_clust_split.last_clustering())

    if len(obj_tracks[0]) > 10:
        _threat_evaluator = int_rec.ThreatEvaluator(obj_tracks, facilities_str)
        _cur_clusters_threats = _threat_evaluator.estimate_threats()
        _formated_output = _cur_clusters_threats

    return _formated_output

def infer_defend_status(uav_coords_str, clustering_str, facilities_str, total_uavs_num=50):
    """ 基于输入的敌方无人机轨迹、分组情况，以及、我方的防御圈位置，给出我方对敌防御的情况 """
    _formated_output = ""
    return _formated_output



if __name__ == '__main__':
    from test_main02_dynamic_trajectories_recog import TrajectoryExhibitor

    test_facilities = basic_units.BasicFacilities()
    _root_dir = osp.dirname(osp.abspath(__file__))
    _man_trajs_dir = osp.join(_root_dir, 'data', 'manual_formation_recog')

    # 读取数据
    _man_trajs_infos = [{'filename': 'fleet_form_trj01_shrink1.0.xlsx', 'scale': 25.0},
                        {'filename': 'fleet_form_trj02_shrink1.5.xlsx', 'scale': 35.0},
                        {'filename': 'fleet_form_trj03_shrink1.2.xlsx', 'scale': 40.0},]
    _test_idx = 0
    # processor = TrajectoryExhibitor(osp.join(_man_trajs_dir, _man_trajs_infos[_test_idx]['filename']), _man_trajs_infos[_test_idx]['scale'], interp_scale=_man_trajs_infos[_test_idx]['scale'])
    processor = TrajectoryExhibitor(osp.join(_man_trajs_dir, _man_trajs_infos[_test_idx]['filename']), interp_scale=_man_trajs_infos[_test_idx]['scale'])
    test_sw = 1





    if test_sw == 1:
         # 测试无人机集群分组方法和队形识别方法
        euavs_trjs_json_dict = {'euav1': {'xs': processor.trajectories['uav1_x'].tolist(),
                                     'ys': processor.trajectories['uav1_y'].tolist(),
                                     'ts': np.arange(len(processor.trajectories['uav1_x'])).tolist()},
                           'euav2': {'xs': processor.trajectories['uav2_x'].tolist(),
                                     'ys': processor.trajectories['uav2_y'].tolist(),
                                     'ts': np.arange(len(processor.trajectories['uav2_x'])).tolist()},
                           'euav3': {'xs': processor.trajectories['uav3_x'].tolist(),
                                     'ys': processor.trajectories['uav3_y'].tolist(),
                                     'ts': np.arange(len(processor.trajectories['uav3_x'])).tolist()},
                           'euav4': {'xs': processor.trajectories['uav4_x'].tolist(),
                                     'ys': processor.trajectories['uav4_y'].tolist(),
                                     'ts': np.arange(len(processor.trajectories['uav4_x'])).tolist()},
                           'euav5': {'xs': processor.trajectories['uav5_x'].tolist(),
                                     'ys': processor.trajectories['uav5_y'].tolist(),
                                     'ts': np.arange(len(processor.trajectories['uav5_x'])).tolist()},
                                }

        enemy_trajectories_str = json.dumps(euavs_trjs_json_dict, indent=4) # 包含无人机轨迹的json字符串
        # 无人机轨迹
        # print("enemy_trajectories_str:\n",enemy_trajectories_str)
        #这个聚类和IntentFactorExtractor的_clustering_knowstr区别在哪里
        clustering_result = get_uavs_clusters(enemy_trajectories_str)
        # print("clustering_result:",clustering_result)

        #识别无人机队形
        formation_result = recog_fleet_formtype(enemy_trajectories_str, clustering_result)
        # print("formation_result:",formation_result)

        # 基于无人机轨迹、分组情况，给出各无人机的突破情况
        formated_breaches = ring_breach_infer(enemy_trajectories_str, clustering_result)
        # print("formated_breaches:",json.dumps(formated_breaches, indent=4, ensure_ascii=False))

        # 基于无人机轨迹、分组情况，给出敌方各个分组的意图信息
        formated_intent = infer_enemy_intention(enemy_trajectories_str,test_facilities)
        print(formated_intent)

        # # 基于无人机轨迹、分组情况、我方设施位置，给出敌方各个分组的威胁信息（威胁程度、威胁设施、预计攻击发生时间）
        # formated_threats = infer_enemy_threats(enemy_trajectories_str,test_facilities)
        # print("formated_threats\n", json.dumps(formated_threats, indent=4, ensure_ascii=False))
        #
        # # 基于敌方无人机轨迹、分组情况、我方防御圈分布，给出我方防御情况（圈层防御比例、防御消耗无人机数量）
        # formated_defstatus = infer_defend_status(enemy_trajectories_str, clustering_result,test_facilities)
        # print(formated_defstatus)





