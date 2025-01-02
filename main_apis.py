import os, os.path as osp
import glob
import json
import time
import numpy as np

from formation_recognition import basic_units
from formation_recognition import clusters_recognition as clus_rec
from formation_recognition import formation_recognition as form_rec
from formation_recognition import defence_breach_analyze as def_breach

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

if __name__ == '__main__':
    from test_main02_dynamic_trajectories_recog import TrajectoryExhibitor

    _root_dir = osp.dirname(osp.abspath(__file__))
    _man_trajs_dir = osp.join(_root_dir, 'data', 'manual_formation_recog')

    # 读取数据
    _man_trajs_infos = [{'filename': 'fleet_form_trj01_shrink1.0.xlsx', 'scale': 7.0},
                        {'filename': 'fleet_form_trj02_shrink1.5.xlsx', 'scale': 12.0},
                        {'filename': 'fleet_form_trj03_shrink1.2.xlsx', 'scale': 14.0},]
    _test_idx = 0
    processor = TrajectoryExhibitor(osp.join(_man_trajs_dir, _man_trajs_infos[_test_idx]['filename']), _man_trajs_infos[_test_idx]['scale'])
    
    test_sw = 1

    if test_sw == 1:
         # 测试无人机集群分组方法和队形识别方法
        input_json_dict = {'euav1': {'xs': processor.trajectories['uav1_x'].tolist(),
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
                                     'ts': np.arange(len(processor.trajectories['uav5_x'])).tolist()},}
        
        input_json_str = json.dumps(input_json_dict) # 包含无人机轨迹的json字符串

        print(input_json_str)
        clustering_result = get_uavs_clusters(input_json_str)
        print(clustering_result)

        formation_result = recog_fleet_formtype(input_json_str, clustering_result)
        print(formation_result)

        # 基于无人机轨迹、分组情况，给出各无人机的突破情况
        formated_breaches = ring_breach_infer(input_json_str, clustering_result)
        print(formated_breaches)
