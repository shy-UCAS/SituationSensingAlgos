from formation_recognition import basic_units

class TotalInterceptCapability:
    def __init__(self, enemy_swarm:list[basic_units.ObjTracks]):
        pass


class InterceptRateCalculator:
    """ 基于给定的敌机数量，分析所需的拦截无人机数量和拦截率 """
    def __init__(self, enemy_num:int, single_intercept_rate:float=None):
        
        if single_intercept_rate is None:
            self.single_intercept_rate = single_intercept_rate
        else:
            self.single_intercept_rate = 0.5