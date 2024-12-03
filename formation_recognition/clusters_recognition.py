import numpy as np
from formation_recognition import basic_units

class SplitClusters(object):
    """ 基于给定的目标运动轨迹，进行聚类划分
    """
    def __init__(self, swarm_objs:basic_units.ObjTracks):
        self.swarm_objs = swarm_objs
        self.num_objs = len(swarm_objs)
        self.clusters = []
        
        self._make_spat_features()

    def _make_spat_features(self):
        _swarm_feats = []
        
        _swarm_locs = np.array([_obj.last_location() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)
        _swram_locs_dims = _swarm_locs.shape[1]
        
        _swarm_directs = np.array([_obj.move_direction() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)
        _swarm_directs_dims = _swarm_directs.shape[1]
        
        _swarm_speeds = np.array([_obj.move_speed() for _obj in self.swarm_objs]).reshape(self.num_objs, -1)
        _swarm_speeds_dims = _swarm_speeds.shape[1]

        import pdb; pdb.set_trace()