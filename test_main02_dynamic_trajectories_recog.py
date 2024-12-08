""" 测试脚本：动态加载人工设计的运行轨迹，以动态的方式进行合群/分群以及队形的识别
"""
import os, os.path as osp
import glob

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TrajectoryExhibitor(object):
    def __init__(self, file_path, coord_scale, interp_scale=3):
        """
        初始化类，读取 Excel 文件并存储轨迹数据。
        
        :param file_path: Excel 文件路径
        :param coord_scale: 坐标缩放比例
        """
        self.file_path = file_path
        self.coord_scale = coord_scale
        self.interp_scale = interp_scale
        
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """
        加载 Excel 文件中的数据。
        假设 Excel 文件的表头为：时间，目标1_x，目标1_y，目标2_x，目标2_y，...
        """
        self.data = pd.read_excel(self.file_path)
            
        self.time = self.data['time']  # 第一列为时间
        self.trajectories = self.data.iloc[:, 1:] / self.coord_scale  # 后面的列为轨迹数据
        # import pdb; pdb.set_trace()
        
        _new_time = np.linspace(self.time.iloc[0], self.time.iloc[-1], len(self.time) * self.interp_scale)
        _interp_coords_comb = np.zeros((len(_new_time), self.trajectories.shape[1]))
        
        for _col_i in range(self.trajectories.shape[1]):
            _cur_coords = self.trajectories.iloc[:, _col_i]
            _interp_coords = interp1d(self.time, _cur_coords, kind='cubic')(_new_time)
            _interp_coords = gaussian_filter1d(_interp_coords, sigma=0.5)
            
            # import pdb; pdb.set_trace()
            _interp_coords_comb[:, _col_i] = _interp_coords
        
        self.time = pd.Series(_new_time)
        self.trajectories = pd.DataFrame(_interp_coords_comb, columns=self.trajectories.columns)

    def get_points(self):
        """
        以 yield 的方式逐行返回轨迹点。
        每次返回一个时间步的所有目标的位置坐标。
        """
        for idx, row in self.trajectories.iterrows():
            time_step = self.time.iloc[idx]
            # 每个时间步目标的位置列表 (目标1_x, 目标1_y, 目标2_x, 目标2_y, ...)
            yield time_step, row.values.reshape(-1, 2)  # 每两个值为一个目标的 (x, y)
    
    def animate_trajectory(self):
        """
        动画显示所有目标的运动轨迹。
        """
        fig, ax = plt.subplots()
        ax.set_xlim(self.trajectories.iloc[:, ::2].min().min() - 1, 
                    self.trajectories.iloc[:, ::2].max().max() + 1)
        ax.set_ylim(self.trajectories.iloc[:, 1::2].min().min() - 1, 
                    self.trajectories.iloc[:, 1::2].max().max() + 1)
        ax.set_title("Trajectory Animation")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
        
        # 初始化每个目标的轨迹线和当前位置标记
        num_targets = self.trajectories.shape[1] // 2
        
        lines = [ax.plot([], [], lw=2)[0] for _ in range(num_targets)]  # 轨迹线
        points = [ax.plot([], [], 'o')[0] for _ in range(num_targets)]  # 当前位置点

        # 更新函数
        def update(frame):
            time_step, points_data = frame
            
            for i, (line, point) in enumerate(zip(lines, points)):
                # 提取当前目标的轨迹和位置
                # import pdb; pdb.set_trace()
                x_data = self.trajectories.iloc[:frame[0]+1, i*2].values
                y_data = self.trajectories.iloc[:frame[0]+1, i*2+1].values
                
                line.set_data(x_data, y_data)
                point.set_data([points_data[1][i, 0]], [points_data[1][i, 1]])
            return lines + points

        # 动画帧
        frames = list(self.get_points())

        ani = FuncAnimation(fig, update, frames=enumerate(frames), interval=500, blit=True)
        plt.show()
    
# 使用示例
if __name__ == "__main__":
    # 假设文件名为 "trajectory.xlsx"
    _root_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    _man_trajs_dir = osp.join(_root_dir, 'data', 'manual_formation_recog')
    _man_trajs_infos = [{'filename': 'fleet_form_trj01_shrink1.0.xlsx', 'scale': 1.0},
                        {'filename': 'fleet_form_trj02_shrink1.5.xlsx', 'scale': 1.5},
                        {'filename': 'fleet_form_trj03_shrink1.2.xlsx', 'scale': 1.2},]
    
    _test_idx = 2
    processor = TrajectoryExhibitor(osp.join(_man_trajs_dir, _man_trajs_infos[_test_idx]['filename']), _man_trajs_infos[_test_idx]['scale'])
    
    # 遍历轨迹点
    for time_step, positions in processor.get_points():
        print(f"Time: {time_step}, Positions: {positions}")
    
    # 播放动画
    processor.animate_trajectory()