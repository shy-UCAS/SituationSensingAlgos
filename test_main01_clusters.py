import os, os.path as osp
import glob

from simulated_environment import draw_swarm_1step_movements as draw_movs
from formation_recognition import clusters_recognition as clus_rec
from simulated_environment import sim_swarm_formation_generate as sim_form

if __name__ == "__main__":
    func_sw = 5
    
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
        # test_trajs_file = osp.join(folder_path, 'manual_clusters_000.txt')
        test_trajs_files = glob.glob(osp.join(folder_path, 'manual_clusters_*.txt'))

        for _trajs_file in test_trajs_files:
            swarm_objs = draw_movs.load_swarm_from_txt(_trajs_file)
            swarm_cluster = clus_rec.SplitClusters(swarm_objs)
            swarm_cluster.show_clusters()
    
    elif func_sw == 4:
        # 对集群进行划分，并识别其中各个分组的队形样式
        folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'auto_formation_recog')
        form_locs_files = glob.glob(osp.join(folder_path, 'manual_formation_*.txt'))
        
        sim_former = sim_form.SwarmFormationGenerate(3, 'horizontal')
        # sim_former = sim_form.SwarmFormationGenerate(3, 'echelon')
        sim_former.show_formation()
    
    elif func_sw == 5:
        # 利用上面的队形生成代码，生成一组包含典型队形的测试数据
        gene_fleet_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        gene_fleet_forms = ['horizontal', 'echelon', 'vertical', 'wedge']

        folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'auto_formation_recog')
        gene_data_file = osp.join(folder_path, 'gene_formation_data.txt')

        form_generator = sim_form.MaskSwarmFormationDataset(gene_fleet_sizes, gene_fleet_forms, folder_path, num_samples=4000)
        form_generator.generate_filewise(folder_path)

    elif func_sw == 6:
        # 基于生成的队形数据，训练基于RNN的队形识别模型
        pass