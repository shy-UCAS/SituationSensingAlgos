import os, os.path as osp
import glob
import time
import numpy as np

from simulated_environment import draw_swarm_1step_movements as draw_movs
from formation_recognition import clusters_recognition as clus_rec
from formation_recognition import formation_recognition as form_rec
from simulated_environment import sim_swarm_formation_generate as sim_form

if __name__ == "__main__":
    func_sw = 6
    
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
        
        # sim_former = sim_form.SwarmFormationGenerate(7, 'horizontal')
        # sim_former = sim_form.SwarmFormationGenerate(7, 'vertical')
        # sim_former = sim_form.SwarmFormationGenerate(7, 'echelon')
        # sim_former = sim_form.SwarmFormationGenerate(7, 'wedge')
        # sim_former = sim_form.SwarmFormationGenerate(7, 'circular')
        sim_former = sim_form.SwarmFormationGenerate(5, 'random')
        sim_former.show_formation()
    
    elif func_sw == 5:
        # 利用上面的队形生成代码，生成一组包含典型队形的测试数据
        gene_fleet_sizes = [3, 4, 5, 6, 7, 8, 9, 10]
        gene_fleet_forms = ['horizontal', 'echelon', 'vertical', 'wedge', 'circular', 'random']

        folder_path = osp.join(osp.dirname(osp.abspath(__file__)), 'data', 'auto_formation_recog')
        gene_data_file = osp.join(folder_path, 'gene_formation_data.txt')

        form_generator = sim_form.MaskSwarmFormationDataset(gene_fleet_sizes, gene_fleet_forms, folder_path, num_samples=12000)
        form_generator.generate_filewise(folder_path)

    elif func_sw == 6:
        # orig_coords = np.random.uniform(-30, 30, size=(5, 2))
        # direct_vec = np.random.uniform(-1, 1, size=(2,))

        # _tic = time.time()
        # spat_conv = form_rec.SpatialFeatConv(orig_coords, direction=direct_vec)
        # print("[SpatialFeatExtract] %d locations in %.3fsecs, shape is %s" % (orig_coords.shape[0], time.time() - _tic, spat_conv.fleet_locs.shape))

        # 基于生成的队形数据，训练基于RNN的队形识别模型
        workspace_dir = osp.dirname(osp.abspath(__file__))
        folder_path = osp.join(workspace_dir, 'data', 'auto_formation_recog')
        
        #data_fpath = osp.join(folder_path, 'swarm_formations_4000.txt')
        data_fpath = osp.join(folder_path, 'swarm_formations_12000.txt')
        eval_fpath = osp.join(folder_path, 'swarm_formations_1000.txt')
        form_types = ['vertical', 'horizontal', 'echelon', 'wedge', 'circular', 'random']
        # form_dataset = form_rec.FormationDataset(form_types, data_fpath)

        form_recog = form_rec.FormationRecognizer(form_types=form_types, num_layers=3)
        save_model_prefix = osp.join(workspace_dir, 'pretrained_weights', 'formation_recognition', 'form_recog_model')
        form_recog.fit_on_data(data_fpath, eval_fpath, save_model_prefix, epochs=[6, 10, 14, 100], lrs=[1e-3, 1e-4, 1e-5, 1e-6])
