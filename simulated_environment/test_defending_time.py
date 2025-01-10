import os, os.path as osp
import sys

import numpy as np
import glob
import matplotlib.pyplot as plt

ws_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if not ws_root in sys.path:
    sys.path.append(ws_root)

from formation_recognition import basic_units as b_units
from formation_recognition import defence_breach_analyze as def_brch

# 初始化一个空列表，用于存储鼠标点击的位置
clicked_movements = []

click_state = 0
cur_movement = []

# 定义鼠标点击事件的回调函数
def on_mouse_click(event):
    global click_state
    global cur_movement

    # 检查是否是鼠标左键点击事件
    if event.button == 1:  # 左键点击
        # 获取点击的 x 和 y 坐标
        x, y = event.xdata, event.ydata
        
        # 如果点击在绘图区外，event.xdata 和 event.ydata 为 None
        if x is None or y is None:
            return

        # 将点击的位置添加到列表中
        if click_state == 0:
            cur_movement = []
            cur_movement.append((x, y))

            # 在图中标记点击的位置
            plt.scatter(x, y, color='red', s=50)  # 标记为红点
            plt.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points", xytext=(10, 10), fontsize=8)

            click_state = 1

        elif click_state == 1:
            cur_movement.append((x, y))
            clicked_movements.append(cur_movement)

            # 在图中标记点击的位置
            plt.arrow(cur_movement[0][0], cur_movement[0][1], x - cur_movement[0][0], y - cur_movement[0][1], color='black', head_width=0.1, head_length=0.1)
            plt.scatter(x, y, color='blue', s=50)  # 标记为蓝色
            plt.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points", xytext=(10, 10), fontsize=8)

            click_state = 0
        
        print(f"Clicked at: x={x:.2f}, y={y:.2f}")

        # 更新图形显示
        plt.draw()

# 创建图形窗口
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.set_title("Click on the plot to mark points")
ax.set_xlim(-15, 15)  # 设置x轴范围
ax.set_ylim(-15, 15)  # 设置y轴范围
ax.grid(True)

# 连接鼠标点击事件监听器
cid = fig.canvas.mpl_connect('button_press_event', on_mouse_click)
# 显示图形
plt.show()

# 基于提取的movement，获取到敌方的运动方向、速度等信息
objs_trajectories = [b_units.ObjTracks([_loc[0] for _loc in _mov], 
                                       [_loc[1] for _loc in _mov],
                                       ts = [0, 1], id="euav%d"%(_iter+1)) for _iter, _mov in enumerate(clicked_movements)]
uav_airport_location = np.array([0, 0])

deftime_estimator = def_brch.DefenceTimeEstimate(uav_airport_location, objs_trajectories, speed=10)
deftime_estimator.get_intercept_infos(vis=True)
# import pdb; pdb.set_trace()