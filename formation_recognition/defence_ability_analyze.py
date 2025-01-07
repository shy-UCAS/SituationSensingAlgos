""" 防御能力评估模块，根据敌方无人机数量和所在防御圈的位置，分配己方无人机并计算拦截成功率
"""
import os.path as osp
import numpy as np
import json
import logging
import configparser

# from formation_recognition.defence_breach_analyze import DefRingBreach
from formation_recognition import basic_units

class DefenseEvaluator(object):
    def __init__(self, success_rate_p1=0.8, config_file=None):
        """
        初始化防御评估器
        
        :param success_rate_p1: 我方一敌一拦截成功率，默认80%
        :param config_file: 配置文件路径，默认None使用默认路径
        """
        self.P1 = success_rate_p1
        
        # 更新拦截率要求
        self.success_rates = {
            'C1': 0.9,      # 圈层1的拦截率要求为90%
            'C2': 0.6,      # 圈层2的拦截率要求为60%
            'Others': 0.3   # 不在圈层1、2内的拦截率要求为30%
        }
        
        # 从配置文件中读取总友方无人机数量
        cfg = configparser.ConfigParser()
        if config_file is None:
            # 假设 facilities.ini 位于 defence_evaluation.py 的上一级目录
            config_file = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'facilities.ini')
        cfg.read(config_file)
        try:
            self.total_allocated_friendly_uavs = cfg.getint('DEFAULT', 'TOTAL_ALLOCATED_FRIENDLY_UAVS')
            logging.info(f"从配置文件读取到的总友方无人机数量: {self.total_allocated_friendly_uavs}")
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError) as e:
            logging.error(f"读取 TOTAL_ALLOCATED_FRIENDLY_UAVS 失败: {e}")
            self.total_allocated_friendly_uavs = 10  # 默认值
            logging.info(f"使用默认的总友方无人机数量: {self.total_allocated_friendly_uavs}")
    
    def assign_and_estimate_friendly_uavs(self, enemy_clusters, total_friendly_uavs=None):
        """
        根据敌方集群分配友方无人机，并估计所需的无人机数量
        
        :param enemy_clusters: 敌方集群列表，每个集群是字典包含 'swarm_no', 'members', 'breach_circle'
        :param total_friendly_uavs: 总友方无人机数量，默认使用配置文件中的值
        :return: 分配结果字典，key为集群编号，value为分配的友方无人机数量
        """
        if total_friendly_uavs is None:
            total_friendly_uavs = self.total_allocated_friendly_uavs
        assignment = {}
        if not enemy_clusters:
            return assignment
        
        # 加载友方机库位置
        facilities = basic_units.BasicFacilities()
        friendly_depots = facilities.UAV_AIRPORTS_XYS  # 使用 UAV_AIRPORTS 作为机库位置
        
        # 为每个集群计算到最近机库的距离
        for cluster in enemy_clusters:
            if 'location' in cluster:
                cluster['min_distance'] = min(
                    np.linalg.norm(np.array(cluster['location']) - depot) 
                    for depot in friendly_depots
                )
            else:
                # 如果没有 location 信息，默认距离为无穷大
                cluster['min_distance'] = float('inf')
        
        # 按照圈层优先级和距离排序：C1 > C2 > Others，并在同圈层内优先分配距离较近的集群
        sorted_clusters = sorted(
            enemy_clusters,
            key=lambda x: (
                1 if x['breach_circle'] == 'C1' else 
                2 if x['breach_circle'] == 'C2' else 
                3,
                x['min_distance']
            )
        )
        
        # 初步分配以满足拦截率要求，同圈层内优先考虑距离较近的集群
        for cluster in sorted_clusters:
            swarm_no = cluster['swarm_no']
            breach = cluster['breach_circle']
            enemy_num = len(cluster['members'])
            required_rate = self.success_rates.get(breach, self.success_rates['Others'])
            
            # 逐步分配友方无人机，直到满足拦截率要求或无人机用尽
            allocated = 0
            max_allocation = min(total_friendly_uavs, enemy_num * 2)  # 设置合理的上限，避免无限循环
            while allocated <= max_allocation:
                computed_rate = self.compute_interception_success_rate(allocated, enemy_num, swarm_no)
                if computed_rate >= required_rate:
                    break
                allocated += 1
            
            if allocated > total_friendly_uavs:
                allocated = total_friendly_uavs
            
            if allocated > 0:
                assignment[swarm_no] = allocated
                total_friendly_uavs -= allocated
        
        # 分配剩余的友方无人机
        if total_friendly_uavs > 0:
            # 按照圈层优先级和距离排序
            remaining_sorted = sorted(
                sorted_clusters,
                key=lambda x: (
                    1 if x['breach_circle'] == 'C1' else 
                    2 if x['breach_circle'] == 'C2' else 
                    3,
                    x['min_distance']
                )
            )
            
            swarm_count = len(remaining_sorted)
            if total_friendly_uavs > swarm_count:
                # 计算每个蜂群分配的基本数量
                base_allocation = total_friendly_uavs // swarm_count
                remainder = total_friendly_uavs % swarm_count

                # 均分分配
                for cluster in remaining_sorted:
                    swarm_no = cluster['swarm_no']
                    allocation = base_allocation
                    assignment[swarm_no] = assignment.get(swarm_no, 0) + allocation
                    total_friendly_uavs -= allocation

                # 根据圈层优先级和距离分配剩余无人机
                for cluster in remaining_sorted:
                    if remainder <= 0:
                        break
                    swarm_no = cluster['swarm_no']
                    assignment[swarm_no] += 1
                    remainder -= 1
                    total_friendly_uavs -= 1
        
        return assignment
    
    def compute_interception_success_rate(self, N, M, swarm_no):
        """
        计算拦截成功率
        
        :param N: 分配给该集群的友方无人机数量
        :param M: 敌方无人机数量
        :param swarm_no: 集群编号
        :return: 拦截成功率（0-1）
        """
        if M == 0:
            return 1.0
        if N == 1 and M == 1:
            return self.P1
        elif N >=1 and M == 1:
            P2 = 1 - (1 - self.P1) ** N
            return P2
        elif N >= M:
            P3 = (1 - (1 - self.P1) ** (N / M)) ** (N / M)#在evaluate里结果会被取整
            return P3
        else:
            return 0.0
    
    def evaluate_defense(self, enemy_clusters, total_friendly_uavs=None):
        """
        评估防御能力，分配友方无人机并计算每个集群的拦截成功率
        
        :param enemy_clusters: 敌方集群列表，每个集群是字典包含 'swarm_no', 'members', 'breach_circle'
        :param total_friendly_uavs: 总友方无人机数量，默认使用配置文件中的值
        :return: 评估结果字典，包含每个集群的详细信息和总体剩余友方无人机数
        """
        if total_friendly_uavs is None:
            total_friendly_uavs = self.total_allocated_friendly_uavs
        self.enemy_clusters = enemy_clusters  # 保存集群信息以供其它方法使用
        assignment = self.assign_and_estimate_friendly_uavs(enemy_clusters, total_friendly_uavs)
        self.assignment = assignment  # 保存分配结果以便其他方法调用
        results = []
        total_allocated = 0

        for cluster in enemy_clusters:
            swarm_no = cluster['swarm_no']
            breach_circle = cluster.get('breach_circle', 'Unknown')
            enemy_uav_num = len(cluster['members'])
            friendly_uav_num = assignment.get(swarm_no, 0)
            total_allocated += friendly_uav_num
            success_rate = self.compute_interception_success_rate(
                N=friendly_uav_num, 
                M=enemy_uav_num, 
                swarm_no=swarm_no
            )
            comparison = {
                'computed_rate': round(success_rate * 100, 2),
                'target_rate': round(self.success_rates.get(breach_circle, self.success_rates['Others']) * 100, 2),
                'is_met': success_rate >= self.success_rates.get(breach_circle, self.success_rates['Others'])
            }
            results.append({
                'swarm_no': swarm_no,
                'breach_circle': breach_circle,
                'enemy_uav_num': enemy_uav_num,
                'friendly_uav_num': friendly_uav_num,
                'success_rate': success_rate,
                'comparison': comparison
            })
        
        remaining_friendly_uavs = total_friendly_uavs - total_allocated
        evaluation = {
            'total_friendly_uavs': total_friendly_uavs,
            'total_allocated_friendly_uavs': total_allocated,
            'remaining_friendly_uavs': remaining_friendly_uavs,
            'clusters': results
        }
        
        return evaluation
    
    