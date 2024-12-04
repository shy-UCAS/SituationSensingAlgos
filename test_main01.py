import os, os.path as osp

from simulated_environment import draw_swarm_1step_movements as draw_movs
from formation_recognition import clusters_recognition as clus_rec

if __name__ == "__main__":
    func_sw = 2
    
    if func_sw == 1:
        # 示例使用
        # 替换为你的txt文件所在的文件夹路径
        folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'manual_clusters')
        draw_movs.plot_coordinates_from_txt(folder_path)
    
    elif func_sw == 2:
        folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'manual_clusters')
        draw_movs.plot_coordinates_from_txt_withparms(folder_path)

    elif func_sw == 3:
        folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'manual_clusters')
        test_trajs_file = osp.join(folder_path, 'manual_clusters_000.txt')
        
        swarm_objs = draw_movs.load_swarm_from_txt(test_trajs_file)
        clus_rec.SplitClusters(swarm_objs)