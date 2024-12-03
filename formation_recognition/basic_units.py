import numpy as np
from sklearn import cluster

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
    
    def last_location(self, lookback=1):
        if self.zs is not None:
            return np.mean(self.xs[-lookback:]), np.mean(self.ys[-lookback:]), np.mean(self.zs[-lookback:])
        else:  
            return np.mean(self.xs[-lookback:]), np.mean(self.ys[-lookback:])
    
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

