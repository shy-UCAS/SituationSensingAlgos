import os, os.path as osp
import glob
import numpy as np
import matplotlib.pyplot as plt
from formation_recognition import basic_units

def load_coordinates_from_txt(file_path):
    """
    从txt文件中加载坐标点。
    
    :param file_path: txt文件路径
    :return: 坐标点列表
    """

    # 初始化存储坐标点的列表
    _start_xs = []; _stop_xs = []
    _start_ys = []; _stop_ys = []
    
    # 读取 txt 文件中的坐标点
    file_name = osp.basename(file_path)
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 按空格或逗号分隔 (支持 "x y" 或 "x,y" 格式)
                _from_to_parts = line.split('->')

                if len(_from_to_parts) >= 2:
                    _from_str, _to_str = _from_to_parts[0:2]

                    _from_parts = [float(_part.strip()) for _part in _from_str.split(',')]
                    _to_parts = [float(_part.strip()) for _part in _to_str.split(',')]

                    _start_x = _from_parts[0]; _start_y = _from_parts[1]
                    _stop_x = _to_parts[0]; _stop_y = _to_parts[1]

                    _start_xs.append(_start_x); _stop_xs.append(_stop_x)
                    _start_ys.append(_start_y); _stop_ys.append(_stop_y)
        
        _movements = [(_start_x, _start_y, _stop_x, _stop_y) for _start_x, _start_y, _stop_x, _stop_y 
                      in zip(_start_xs, _start_ys, _stop_xs, _stop_ys)]
        
        _locs_scaler = basic_units.ScaleSimMovements(_movements)

        _locs_scale = _locs_scaler.scale_factor_bylocs()
        _movs_scale = _locs_scaler.scale_factor_bymovements()
        # import pdb; pdb.set_trace()

        return np.array(_movements) * (_locs_scale + _movs_scale) / 2
    
    except Exception as e:
        print(f"读取文件 {file_name} 时出错: {e}")
        return None

def load_swarm_from_txt(file_path):
    """
    从给定路径的txt文件中加载群体数据。
    
    :param file_path: txt文件路径
    :return: 返回群体数据
    """
    _movements = load_coordinates_from_txt(file_path)
    _move_objs = [basic_units.ObjTracks([_move[0], _move[2]], [_move[1], _move[3]]) for _move in _movements]
    # import pdb; pdb.set_trace()
    return _move_objs

def plot_coordinates_from_txt(folder_path):
    """
    遍历给定文件夹中的所有txt文件，读取坐标点并绘制散点图，保存为jpg文件。
    
    :param folder_path: 包含txt文件的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在，请检查路径")
        return
    
    # 遍历文件夹中的所有文件
    _movement_files = glob.glob(os.path.join(folder_path, 'manual_clusters_*.txt'))
    # import pdb; pdb.set_trace()

    for file_path in _movement_files:
        # 只处理 .txt 文件
        if file_path.endswith('.txt'):
            # file_path = os.path.join(folder_path, file_name)
            file_name = osp.basename(file_path)
            movements_vectors = load_coordinates_from_txt(file_path)
            
            # 如果文件中没有有效的坐标点，跳过绘制
            if movements_vectors is None or len(movements_vectors) <= 0:
                print(f"文件 {file_name} 不包含有效的坐标点，跳过处理")
                continue
            
            _start_xs = [_vec[0] for _vec in movements_vectors]
            _start_ys = [_vec[1] for _vec in movements_vectors]
            
            _stop_xs = [_vec[2] for _vec in movements_vectors]
            _stop_ys = [_vec[3] for _vec in movements_vectors]
            
            # 绘制散点图
            plt.figure(figsize=(8, 6))  # 设置图像大小

            plt.scatter(_start_xs, _start_ys, color='red', s=50, alpha=0.7)  # 绘制散点图
            plt.scatter(_stop_xs, _stop_ys, color='blue', s=50, alpha=0.7)  # 绘制散点图

            for _start_x, _start_y, _stop_x, _stop_y in zip(_start_xs, _start_ys, _stop_xs, _stop_ys):
                plt.arrow(_start_x, _start_y, _stop_x - _start_x, _stop_y - _start_y, head_width=0.1, head_length=0.1, fc='black', ec='black')

            plt.title(f"Scatter Plot of {file_name}", fontsize=14)

            plt.xlabel("X", fontsize=12)
            plt.ylabel("Y", fontsize=12)

            _ttl_xs = _start_xs + _stop_xs
            plt.xlim(min(_ttl_xs) - 10, max(_ttl_xs) + 10)
            _ttl_ys = _start_ys + _stop_ys
            plt.ylim(min(_ttl_ys) - 10, max(_ttl_ys) + 10)

            plt.grid(True)  # 显示网格
            
            # 保存为 jpg 文件，与 txt 文件同名
            output_file_name = os.path.splitext(file_name)[0] + '.jpg'
            output_file_path = os.path.join(folder_path, output_file_name)
            try:
                plt.savefig(output_file_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率 jpg 文件
                print(f"图像已保存: {output_file_path}")
            except Exception as e:
                print(f"保存图像 {output_file_name} 时出错: {e}")
            
            # 关闭当前绘图，释放内存
            plt.close()

def plot_coordinates_from_txt_withparms(folder_path):
    """ 从指定文件夹中读取目标位置和运动方向信息，同时绘制必要的参数
    """
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在，请检查路径")
        return
    
    # 遍历文件夹中的所有文件
    _movement_files = glob.glob(os.path.join(folder_path, 'manual_clusters_*.txt'))

    for file_path in _movement_files:
        # 只处理 .txt 文件
        if file_path.endswith('.txt'):
            file_name = osp.basename(file_path)
            _swarm_objs = load_swarm_from_txt(file_path)

            if _swarm_objs is None or len(_swarm_objs) <= 0:
                print(f"文件 {file_name} 不包含有效的坐标点，跳过处理")
                continue
            
            _num_objs = len(_swarm_objs)

            _swarm_locs = np.array([_obj.last_location() for _obj in _swarm_objs]).reshape(_num_objs, -1)
            _swarm_dists = np.sqrt(np.sum((_swarm_locs[:, np.newaxis, :] - _swarm_locs[np.newaxis, :, :]) ** 2, axis=2))
            _swarm_dists = _swarm_dists[np.triu_indices(_num_objs, k=1)]

            _swarm_directs = np.array([_obj.move_direction() for _obj in _swarm_objs]).reshape(_num_objs, -1)
            _swarm_direct_angles = np.arctan2(_swarm_directs[:, 1], _swarm_directs[:, 0])
            _swarm_direct_angles = np.degrees(_swarm_direct_angles)
            _swarm_direct_angles[_swarm_direct_angles < 0] += 360

            _swarm_speeds = np.array([_obj.move_speed() for _obj in _swarm_objs]).reshape(_num_objs, -1)

            # 创建图表
            fig = plt.figure(figsize=(12, 6))  # 设置整个图的大小

            # 左侧：散点图
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.scatter(_swarm_locs[:, 0], _swarm_locs[:, 1], c='blue', s=50, label="Points")  # 绘制坐标点

            _draw_speed_scale = 4.0
            for i, (x, y) in enumerate(_swarm_locs):
                vx, vy = _swarm_directs[i] * _draw_speed_scale
                ax1.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7)  # 绘制方向向量
                ax1.text(x, y, f"P{i+1}", fontsize=9, ha='right')  # 标记点编号

            ax1.set_title("Scatterplot with Directions", fontsize=14)
            ax1.set_xlabel("X", fontsize=12)
            ax1.set_ylabel("Y", fontsize=12)
            ax1.grid(True)
            ax1.axis('equal')
            ax1.legend()

            # 右侧：三个子图
            # 距离直方图
            ax2 = fig.add_subplot(3, 2, 2)
            ax2.hist(_swarm_dists, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.set_title("Distance Histogram", fontsize=12)
            ax2.set_xlabel("Distance", fontsize=10)
            ax2.set_ylabel("Frequency", fontsize=10)

            # 角度直方图
            ax3 = fig.add_subplot(3, 2, 4)
            ax3.hist(_swarm_direct_angles, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
            ax3.set_title("Angle Histogram", fontsize=12)
            ax3.set_xlabel("Angle (degrees)", fontsize=10)
            ax3.set_ylabel("Frequency", fontsize=10)

            # 速度直方图
            ax4 = fig.add_subplot(3, 2, 6)
            ax4.hist(_swarm_speeds, bins=10, color='salmon', edgecolor='black', alpha=0.7)
            ax4.set_title("Speed Histogram", fontsize=12)
            ax4.set_xlabel("Speed", fontsize=10)
            ax4.set_ylabel("Frequency", fontsize=10)

            # 调整布局
            plt.tight_layout()
            # 保存为 jpg 文件，与 txt 文件同名
            output_file_name = os.path.splitext(file_name)[0] + '.jpg'
            output_file_path = os.path.join(folder_path, output_file_name)
            try:
                plt.savefig(output_file_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率 jpg 文件
                print(f"图像已保存: {output_file_path}")
            except Exception as e:
                print(f"保存图像 {output_file_name} 时出错: {e}")
            
            # 关闭当前绘图，释放内存
            plt.close()