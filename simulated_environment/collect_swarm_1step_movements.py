import os, os.path as osp
import numpy as np
import glob
import matplotlib.pyplot as plt

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
fig, ax = plt.subplots()
ax.set_title("Click on the plot to mark points")
ax.set_xlim(0, 10)  # 设置x轴范围
ax.set_ylim(0, 10)  # 设置y轴范围
ax.grid(True)

# 连接鼠标点击事件监听器
cid = fig.canvas.mpl_connect('button_press_event', on_mouse_click)
# 显示图形
plt.show()

if len(clicked_movements) > 0:
    # 输出记录的点击点
    saved_points_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data', 'manual_clusters')
    saved_ptrs_prefix = 'manual_clusters_'
    exists_ptrs_files = glob.glob(osp.join(saved_points_dir, saved_ptrs_prefix + '*.txt'))

    exists_ptrs_barenames = [osp.splitext(osp.basename(_fpath))[0] for _fpath in exists_ptrs_files]
    exists_ptrs_barenames = np.sort(exists_ptrs_barenames)
    exists_ptrs_namenums = [int(_name.split('_')[-1]) for _name in exists_ptrs_barenames]

    # import pdb; pdb.set_trace()

    if len(exists_ptrs_namenums) > 0:
        next_ptr_num = max(exists_ptrs_namenums) + 1
    else:
        next_ptr_num = 0

    save_ptrs_filepath = osp.join(saved_points_dir, "%s%03d.txt" % (saved_ptrs_prefix, next_ptr_num))

    with open(save_ptrs_filepath, 'wt') as wfid:
        for _cm in clicked_movements:
            wfid.write(f"{_cm[0][0]}, {_cm[0][1]} -> {_cm[1][0]}, {_cm[1][1]}\n")

print("All clicked movements:", clicked_movements)
