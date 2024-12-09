import numpy as np

""" 一些和无人机集群运动、轨迹相关的常量，例如：
    一般的运动速度：10m/s
    一般的相互距离：10m左右
"""

SWARM_AVERAGE_SPEED = 10
SWARM_MUTUAL_DISTANCE = 10 # 一个空中集群基本的相互距离（避障、通信等相关）
SWARM_NEAR_ANGLE_DEGREES = 20 # 可以判定为相似角度的最大差值阈值

class ScaleSimMovements(object):
    def __init__(self, movements, avg_speed=SWARM_AVERAGE_SPEED, min_distance=SWARM_MUTUAL_DISTANCE):
        self.movements = movements
        self.avg_speed = avg_speed
        self.min_distance = min_distance
    
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
        self.ts = np.array(ts) if ts is not None else None
        self.id = id
        
        assert len(self.xs) == len(self.ys), "[Erorr] Length of x and y coordinates are not equal"
    
    def __len__(self):
        return len(self.xs)
    
    def append_location(self, x, y, z=None, t=None):
        self.xs = np.append(self.xs, x)
        self.ys = np.append(self.ys, y)
        
        self.zs = np.append(self.zs, z) if (self.zs is not None and z is not None) else None
        self.ts = np.append(self.ts, t) if (self.ts is not None and t is not None) else None
    
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
    
    def move_speed(self, lookback=1):
        if len(self.xs) < lookback:
            return None

        dx = np.mean(self.xs[-lookback:] - self.xs[-1-lookback:-1])
        dy = np.mean(self.ys[-lookback:] - self.ys[-1-lookback:-1])

        return np.sqrt(dx**2 + dy**2)

