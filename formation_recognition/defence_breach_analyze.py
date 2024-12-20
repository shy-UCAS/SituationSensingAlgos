""" 给出我方防御圈I和防御II里面有哪些敌方的集群编组
"""
from formation_recognition import basic_units

class DefRingBreach(object):
    def __init__(self, rings_cfg_file=None):
        if rings_cfg_file is not None:
            self.rings_cfg = basic_units.BasicFacilities(rings_cfg_file)
        else:
            self.rings_cfg = basic_units.BasicFacilities()
        
        self.ring1_coords = self.rings_cfg.RING1_XYS
        self.ring2_coords = self.rings_cfg.RING2_XYS
    
    def infer_rings_breach_groups(self, objs_trjs:list[basic_units.ObjTracks], cluster_labels, formated_output=True):
        objs_xys = [_obj.last_location() for _obj in objs_trjs]
        
        return ring1_objs_idxs, ring2_objs_idxs, formated_result_jsonstr
