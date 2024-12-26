import os.path as osp

import math
import numpy as np
import configparser

""" 一些和无人机集群运动、轨迹相关的常量，例如：
    一般的运动速度：10m/s
    一般的相互距离：10m左右
"""

# 构建指向上一级目录的路径来读取config.ini文件
DEFAULT_CONFIG_FILE = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'config.ini')
DEFAULT_FACILITIES_FILE = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'facilities.ini')

class GlobalConfigs(object):
    def __init__(self, cfg_file=DEFAULT_CONFIG_FILE):
        self.SWARM_MUTUAL_DISTANCE = None
        self.SWARM_NEAR_ANGLE_DEGREES = None
        self.SWARM_AVERAGE_SPEED = None
        self.DBSCAN_EPS = None
        
        # UAV特性分析参数
        self.SPEED_CHANGE_THRESHOLD = 0.5
        self.ORIENT_CHANGE_THRESHOLD = None
        self.ORIENT_CHANGE_FREQ_THRESHOLD = None
        self.DIRECTING_ANGLE_EPS = None
        self.OPTICAL_OBSERVE_RANGE = 1500
        self.ARRIVE_AT_EPS = 10
        
        # UAV集群特性分析
        self.DIST_CHANGE_RATIO_THRESHOLD = None

        self._load_basic_cfgs()
    
    def _load_basic_cfgs(self, cfg_file=None):
        if cfg_file is None:
            cfg_file = DEFAULT_CONFIG_FILE
        
        _config = configparser.ConfigParser()

        try:
            _config.read(cfg_file)
            self.SWARM_MUTUAL_DISTANCE = float(_config['DEFAULT']['SWARM_MUTUAL_DISTANCE'])
            self.SWARM_NEAR_ANGLE_DEGREES = float(_config['DEFAULT']['SWARM_NEAR_ANGLE_DEGREES'])
            self.SWARM_AVERAGE_SPEED = float(_config['DEFAULT']['SWARM_AVERAGE_SPEED'])
            self.DBSCAN_EPS = float(_config['DEFAULT']['DBSCAN_EPS'])
            
            self.SPEED_CHANGE_THRESHOLD = float(_config['SINGLE_UAV_BEHAVIOR']['SPEED_CHANGE_THRESHOLD'])
            self.ORIENT_CHANGE_THRESHOLD = float(_config['SINGLE_UAV_BEHAVIOR']['ORIENT_CHANGE_THRESHOLD'])
            self.ORIENT_CHANGE_FREQ_THRESHOLD = float(_config['SINGLE_UAV_BEHAVIOR']['ORIENT_CHANGE_FREQ_THRESHOLD'])
            self.DIRECTING_ANGLE_EPS = float(_config['SINGLE_UAV_BEHAVIOR']['DIRECTING_ANGLE_EPS'])
            self.OPTICAL_OBSERVE_RANGE = float(_config['SINGLE_UAV_BEHAVIOR']['OPTICAL_OBSERVE_RANGE'])
            self.ARRIVE_AT_EPS = float(_config['SINGLE_UAV_BEHAVIOR']['ARRIVE_AT_EPS'])
            
            self.DIST_CHANGE_RATIO_THRESHOLD = float(_config['MULTI_UAVS_BEHAVIOR']['DIST_CHANGE_RATIO_THRESHOLD'])
        
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            print(f"Error reading configuration file: {e}")
            exit(1)
        
        except ValueError as e:
            print(f"Error converting configuration values to float: {e}")
            exit(1)
    
    def reload(self):
        self._load_basic_cfgs()

class BasicFacilities(object):
    def __init__(self, cfg_file=DEFAULT_FACILITIES_FILE):
        self.cfg_file = cfg_file
        
        self.RING1_XYS = None
        self.RING2_XYS = None

        self.RADARS_XYS = []
        self.HEADQUARTERS_XYS = []
        self.UAV_AIRPORTS_XYS = []

        self.total_facilities = None

        self._load_rings_coords()
        self.refresh_categories()
    
    def _load_rings_coords(self, cfg_file=None):
        if cfg_file is None:
            self.cfg_file = DEFAULT_FACILITIES_FILE

        _config = configparser.ConfigParser()

        try:
            _config.read(self.cfg_file)

            _ring1_xs = [float(_str) for _str in _config['DEFENCE_RING1']['BORDER_XS'].split(',')]
            _ring1_ys = [float(_str) for _str in _config['DEFENCE_RING1']['BORDER_YS'].split(',')]
            self.RING1_XYS = np.stack([np.array(_ring1_xs), np.array(_ring1_ys)], axis=1)

            _ring2_xs = [float(_str) for _str in _config['DEFENCE_RING2']['BORDER_XS'].split(',')]
            _ring2_ys = [float(_str) for _str in _config['DEFENCE_RING2']['BORDER_YS'].split(',')]
            self.RING2_XYS = np.stack([np.array(_ring2_xs), np.array(_ring2_ys)], axis=1)

            _radars_xs = [float(_str) for _str in _config['RADARS']['RADARS_XS'].split(',')]
            _radars_ys = [float(_str) for _str in _config['RADARS']['RADARS_YS'].split(',')]
            self.RADARS_XYS = np.stack([np.array(_radars_xs), np.array(_radars_ys)], axis=1).reshape(-1, 2)
            
            _hq_xs = [float(_str) for _str in _config['HEADQUARTERS']['HQ_XS'].split(',')]
            _hq_ys = [float(_str) for _str in _config['HEADQUARTERS']['HQ_YS'].split(',')]
            self.HEADQUARTERS_XYS = np.stack([np.array(_hq_xs), np.array(_hq_ys)], axis=1).reshape(-1, 2)

            _ua_xs = [float(_str) for _str in _config['UAV_AIRPORTS']['UA_XS'].split(',')]
            _ua_ys = [float(_str) for _str in _config['UAV_AIRPORTS']['UA_YS'].split(',')]
            self.UAV_AIRPORTS_XYS = np.stack([np.array(_ua_xs), np.array(_ua_ys)], axis=1).reshape(-1, 2)
        
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            print(f"Error reading defence ring file: {e}")
            exit(1)
        
        except ValueError as e:
            print(f"Error converting defence ring coords to float: {e}")
            exit(1)
    
    def refresh_categories(self):
        self.total_facilities = {}

        for _iter in range(len(self.RADARS_XYS)):
            self.total_facilities[f"radar_{_iter+1}"] = self.RADARS_XYS[_iter]
        
        for _iter in range(len(self.HEADQUARTERS_XYS)):
            self.total_facilities[f"hq_{_iter+1}"] = self.HEADQUARTERS_XYS[_iter]

        for _iter in range(len(self.UAV_AIRPORTS_XYS)):
            self.total_facilities[f"ua_{_iter+1}"] = self.UAV_AIRPORTS_XYS[_iter]
    
    def subset_facilities(self, facility_names):
        _subset_facilities = {}

        for _name in facility_names:
            _subset_facilities[_name] = self.total_facilities[_name]

        return _subset_facilities

class BasicFacilitiesSubset(BasicFacilities):
    def __init__(self, facilities:BasicFacilities, facility_names:list[str]):
        super().__init__(facilities.cfg_file)
        
        self.total_facilities = facilities.subset_facilities(facility_names)
        self.subset_facilities_coordinates()
    
    def subset_facilities_coordinates(self):
        _radar_coordinates = [self.total_facilities[_name].reshape(-1, 2) for _name in self.total_facilities if _name.startswith("radar_")]
        if len(_radar_coordinates) > 0:
            self.RADARS_XYS = np.stack(_radar_coordinates, axis=0)
        else:
            self.RADARS_XYS = []
        
        _hq_coordinates = [self.total_facilities[_name].reshape(-1, 2) for _name in self.total_facilities if _name.startswith("hq_")]
        if len(_hq_coordinates) > 0:
            self.HEADQUARTERS_XYS = np.stack(_hq_coordinates, axis=0)
        else:
            self.HEADQUARTERS_XYS = []
        
        _uav_coordinates = [self.total_facilities[_name].reshape(-1, 2) for _name in self.total_facilities if _name.startswith("ua_")]
        if len(_uav_coordinates) > 0:
            self.UAV_AIRPORTS_XYS = np.stack(_uav_coordinates, axis=0)
        else:
            self.UAV_AIRPORTS_XYS = []

class ScaleSimMovements(object):
    def __init__(self, movements, avg_speed=None, min_distance=None):
        self.glb_cfgs = GlobalConfigs()

        self.movements = movements
        self.avg_speed = self.glb_cfgs.SWARM_AVERAGE_SPEED if avg_speed is None else avg_speed
        self.min_distance = self.glb_cfgs.SWARM_MUTUAL_DISTANCE if min_distance is None else min_distance
    
    def __len__(self):
        return len(self.movements)
    
    def scale_factor_bylocs(self):
        _start_xys = np.array([_move[0:2] for _move in self.movements])
        _stop_xys = np.array([_move[2:4] for _move in self.movements])

        _start_mutual_dists = np.sqrt(np.sum((_start_xys[:, np.newaxis, :] - _start_xys[np.newaxis, :, :])**2, axis=2))
        _start_min_dist = _start_mutual_dists[np.triu_indices(len(_start_xys), k=1)].min()
        _start_scale_factor = self.min_distance / _start_min_dist

        _stop_mutual_dists = np.sqrt(np.sum((_stop_xys[:, np.newaxis, :] - _stop_xys[np.newaxis, :, :])**2, axis=2))
        _stop_min_dist = _stop_mutual_dists[np.triu_indices(len(_stop_xys), k=1)].min()
        _stop_scale_factor = self.min_distance / _stop_min_dist

        return (_start_scale_factor + _stop_scale_factor) / 2
    
    def scale_factor_bymovements(self):
        # 获取一组x,y坐标之间的所有相对距离
        _start_xys = np.array([_move[0:2] for _move in self.movements])
        _stop_xys = np.array([_move[2:4] for _move in self.movements])

        _start2stop_dists = np.sqrt(np.sum((_stop_xys - _start_xys)**2, axis=1))
        _start2stop_scale_factor = self.avg_speed / _start2stop_dists.mean()

        return _start2stop_scale_factor

class ObjTracks(object):
    def __init__(self, xs, ys, zs=None, ts=None, id=None):
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        
        self.zs = np.array(zs) if zs is not None else None
        self.ts = np.array(ts) if ts is not None else np.arange(len(xs))
        self.id = id
        
        assert len(self.xs) == len(self.ys), "[Erorr] Length of x and y coordinates are not equal"
    
    def __len__(self):
        return len(self.xs)
    
    def append_location(self, x, y, z=None, t=None):
        self.xs = np.append(self.xs, x)
        self.ys = np.append(self.ys, y)
        
        self.zs = np.append(self.zs, z) if (self.zs is not None and z is not None) else None
        self.ts = np.append(self.ts, t) if (self.ts is not None and t is not None) else None
    
    def total_locations(self):
        if self.zs is None:
            return self.xs, self.ys
        else:
            return self.xs, self.ys, self.zs

    def last_n_locations(self, lookback=10):
        if self.zs is None:
            return self.ts[-lookback:], self.xs[-lookback:], self.ys[-lookback:]
        else:
            return self.ts[-lookback:], self.xs[-lookback:], self.ys[-lookback:], self.zs[-lookback:]
    
    def last_location(self, lookback=1):
        if self.zs is not None:
            return np.mean(self.xs[-lookback:]), np.mean(self.ys[-lookback:]), np.mean(self.zs[-lookback:])
        else:  
            return np.mean(self.xs[-lookback:]), np.mean(self.ys[-lookback:])
    
    def start_location(self):
        if self.zs is not None:
            return self.xs[0], self.ys[0], self.zs[0]
        else:  
            return self.xs[0], self.ys[0]

    def move_direction(self, lookback=1):
        if len(self.xs) < lookback:
            return None
        
        dx = np.mean(self.xs[-lookback:] - self.xs[-1-lookback:-1])
        dy = np.mean(self.ys[-lookback:] - self.ys[-1-lookback:-1])
        
        if self.zs is not None:
            dz = np.mean(self.zs[-lookback:] - self.zs[-1-lookback:-1])
            
            return np.array([dx, dy, dz]) / np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            return np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)
    
    def move_direct_angles(self, lookback=1):
        if len(self.xs) < lookback:
            return None
        
        mts = self.ts[-lookback:]
        dxs = self.xs[-lookback:] - self.xs[-1-lookback:-1]
        dys = self.ys[-lookback:] - self.ys[-1-lookback:-1]
        
        # 目前角度计算不考虑z轴上面的数值变化
        _orient_rads = [math.atan2(_dx, _dy) for _dx, _dy in zip(dxs, dys)]
        _orient_degs = [math.degrees(_orient_rad) for _orient_rad in _orient_rads]

        return mts, _orient_degs
    
    def to_position_angles(self, position, lookback=1):
        if len(self.xs) < lookback:
            return None

        to_pos_dxs = position[0] - self.xs[-lookback:]
        to_pos_dys = position[1] - self.ys[-lookback:]

        # 目前角度计算不考虑z轴上面的数值变化
        to_pos_rads = [math.atan2(_dx, _dy) for _dx, _dy in zip(to_pos_dxs, to_pos_dys)]
        to_pos_degs = [math.degrees(_orient_rad) for _orient_rad in to_pos_rads]

        return to_pos_degs
    
    def move_speed(self, lookback=1):
        if len(self.xs) < lookback:
            return None

        dx = np.mean(self.xs[-lookback:] - self.xs[-1-lookback:-1])
        dy = np.mean(self.ys[-lookback:] - self.ys[-1-lookback:-1])

        return np.sqrt(dx**2 + dy**2)
    
    def move_speeds(self, lookback=1):
        if len(self.xs) < lookback:
            return None

        dx = self.xs[-lookback:] - self.xs[-1-lookback:-1]
        dy = self.ys[-lookback:] - self.ys[-1-lookback:-1]

        return np.sqrt(dx**2 + dy**2)

class ObjTracksJson(ObjTracks):
    def __init__(self, obj_json, id=None):
        super().__init__(obj_json['xs'], obj_json['ys'])

        self.zs = obj_json['zs'] if 'zs' in obj_json else None
        self.ts = obj_json['ts'] if 'ts' in obj_json else None
        self.id = id
