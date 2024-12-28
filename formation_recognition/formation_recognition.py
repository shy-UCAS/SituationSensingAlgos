import os, os.path as osp

import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from formation_recognition import basic_units

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层，用于分类
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # pack输入的数据序列
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # import pdb; pdb.set_trace()
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # RNN 前向传播
        packed_out, _ = self.rnn(packed_x, h0)

        # 解包序列
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # 取 RNN 的最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层
        out = self.fc(out)  # (batch_size, output_size)
        return out

class SpatialFeatConv(object):
    """ 将位置数据转换为空间特征（局部相对位置，归一化等等）
        批注：针对2D空间中的坐标
    """
    def __init__(self, fleet_locs, direction=None, vis=False):
        self.fleet_locs = np.array(fleet_locs, dtype=float)
        self.direct_vec = None
        self.vis = vis
        # import pdb; pdb.set_trace()

        # 根据输入的方向向量，对编队的坐标进行旋转，使得新的坐标系中方向向量箭头冲下
        if direction is not None:
            _direct_norm = np.linalg.norm(direction)
            if _direct_norm > 0:
                self.direct_vec = direction / _direct_norm
                self.fleet_locs = self._align_with_direct(self.fleet_locs, self.direct_vec, vis=False)
        
        # 将旋转后得到的新坐标点进行平移，以其中y轴最小的点作为原点
        self.fleet_locs = self._shift_to_origin(self.fleet_locs)

        # 使用新的坐标点中x、y的最大值进行坐标归一化
        self.fleet_locs = self._normalize_locs(self.fleet_locs)

        # 进一步提取其中坐标点之间的相对距离信息
        _extra_spat_feats = self._extra_relv_feats(self.fleet_locs)

        self.fleet_locs = np.concatenate([self.fleet_locs, _extra_spat_feats], axis=1)
    
    def _shift_to_origin(self, locs=None):
        if locs is None:
            locs = self.fleet_locs

        _argmin_y = np.argmin(locs[:, 1])
        _min_x = locs[_argmin_y, 0]
        _min_y = locs[_argmin_y, 1]
        
        return locs - np.array([_min_x, _min_y])

    def _align_with_direct(self, locs=None, direct=None, vis=None):
        if locs is None:
            locs = self.fleet_locs
        
        if vis is None:
            vis = self.vis
        
        if direct is None:
            direct = self.direct_vec

        _tgt_direct = np.array([0, -1])

        # 计算旋转角度 theta（顺时针为正，逆时针为负）
        _theta = np.arctan2(_tgt_direct[1], _tgt_direct[0]) - np.arctan2(direct[1], direct[0])
        
        # 构造旋转矩阵
        _cos_theta = np.cos(_theta)
        _sin_theta = np.sin(_theta)
        _rotation_matrix = np.array([
            [_cos_theta, -_sin_theta],
            [_sin_theta, _cos_theta]
        ])

        _aligned_locs = (_rotation_matrix @ self.fleet_locs.T).T
        # _aligned_locs = self._shift_to_origin(_aligned_locs)

        if vis:
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))

            axes[0].scatter(self.fleet_locs[:, 0], self.fleet_locs[:, 1], color='red', s=50, alpha=0.7)
            axes[0].arrow(0, 0, self.direct_vec[0] * 5, self.direct_vec[1] * 5, head_width=0.1, head_length=0.1, fc='black', ec='black')
            axes[0].set_aspect('equal', 'box')
            axes[0].grid(True)

            axes[1].scatter(_aligned_locs[:, 0], _aligned_locs[:, 1], color='red', s=50, alpha=0.7)
            axes[1].set_aspect('equal', 'box')
            axes[1].grid(True)

            plt.show()

        return _aligned_locs

    def _normalize_locs(self, locs=None):
        if locs is None:
            locs = self.fleet_locs
        
        _max_y = np.max(np.abs(locs[:, 1]))
        _max_x = np.max(np.abs(locs[:, 0]))
        _max_edge = max([_max_x, _max_y])

        if _max_edge > 1e-3:
            locs[:, 1] = locs[:, 1] / _max_edge
            locs[:, 0] = locs[:, 0] / _max_edge

        return locs
    
    def _extra_relv_feats(self, locs=None):
        if locs is None:
            locs = self.fleet_locs

        # 按照x坐标由低到高的顺序，获取每个位置点相对于前面位置点的相对距离
        _x_argsort_idxs = locs[:, 0].argsort()
        _relv_feats = []

        for _iter, _x_idx in enumerate(_x_argsort_idxs):
            if _iter <= 0:
                _relv_feats.append(np.array([0, 0]))
            else:
                _prv_idx, _cur_idx = _x_argsort_idxs[_iter - 1], _x_argsort_idxs[_iter]
                _relv_feats.append(locs[_cur_idx] - locs[_prv_idx])
        
        _relv_feats_mat = np.stack(_relv_feats, 0)

        return _relv_feats_mat

class FormationDataset(Dataset):
    def __init__(self, form_types, data_file):
        self.form_types = form_types

        self.data_file = data_file
        self.feats, self.labels = self._load_data()
    
    def _load_data(self, data_file=None):
        if data_file is None:
            data_file = self.data_file
        
        with open(data_file, 'rt') as rf:
            _data = json.load(rf)

        # 将位置数据转换为空间特征（局部相对位置，归一化等等）
        _tic = time.time()
        _feats = []; _labels = []

        for _iter, _d in enumerate(_data):
            _form_type = _d['formtype']
            _locs_json = _d['fleet_locs']

            _locs_xys = np.array([[_ptr['x'], _ptr['y']] for _ptr in _locs_json])
            _form_rec = SpatialFeatConv(_locs_xys)

            _feats.append(_form_rec.fleet_locs)
            _labels.append(self.form_types.index(_form_type))

            if (_iter + 1) % 500 == 0:
                print("[Conv2Feats] %d/%d locs processed in %.3fsecs" % (_iter + 1, len(_data), time.time() - _tic))
        
        return _feats, _labels
    
    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

def fleet_locs_collate_fn(batch):
    # 按照序列长度对输入的训练batch进行排序
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # 分离数据和标签
    sequences, labels = zip(*batch)
    sequences = [torch.tensor(_s, dtype=torch.float) for _s in sequences]
    # import pdb; pdb.set_trace()

    # 真实的序列长度
    lengths = torch.tensor([seq.shape[0] for seq in sequences])

    # 对序列进行填充，填充后的形状为 (batch_size, max_seq_length, input_size)
    padded_data = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # 转换标签为张量
    labels = torch.tensor(labels)
    # import pdb; pdb.set_trace()

    return padded_data, lengths, labels

class FormationRecognizer(object):
    def __init__(self, form_types=['vertical', 'horizontal', 'echelon', 'wedge'], 
                 hidden_size=64, num_layers=3, pretrained_weights=None):
        self.form_types = form_types
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self._loc_dims = 4
        self._output_size = len(self.form_types)
        self._learning_rate = 1e-3
        self.model = RNNClassifier(input_size=self._loc_dims,
                                   hidden_size=self.hidden_size,
                                   output_size=self._output_size,
                                   num_layers=self.num_layers)

        if not (pretrained_weights is None):
            self.model_weights = pretrained_weights
            self.model.load_state_dict(torch.load(self.model_weights))
        else:
            self.model_weights = None

    def infer_formtype(self, fleet_locs, direct_vec=None, vis=False):
        _tic = time.time()
        
        if direct_vec is None:
            direct_vec = np.array([0, -1])
        
        _sfeat_conv = SpatialFeatConv(fleet_locs, direct_vec, vis=vis)
        _locs_feat = _sfeat_conv.fleet_locs
        
        _pred_outputs = self.model(torch.tensor(_locs_feat[np.newaxis, ...], dtype=torch.float32), torch.tensor([len(_locs_feat)]))
        _pred_cls = torch.argmax(_pred_outputs, axis=1).numpy()
        
        print("Formtype infer in %.3fsecs, type: %s" % (time.time() - _tic, self.form_types[_pred_cls[0]]))
        if vis:
            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            axes[0].scatter(fleet_locs[:, 0], fleet_locs[:, 1], c='blue', s=50, label="Origin")
            axes[0].quiver(fleet_locs[:, 0], fleet_locs[:, 1], direct_vec[0], direct_vec[1], color='red', label="Direction")
            axes[0].set_xlabel("X", fontsize=12)
            axes[0].set_ylabel("Y", fontsize=12)
            axes[0].legend(fontsize=12)
            axes[0].axis('equal')

            axes[1].scatter(_locs_feat[:, 0], _locs_feat[:, 1], c='blue', s=50, label="Featured")
            axes[1].set_xlabel("X", fontsize=12)
            axes[1].set_ylabel("Y", fontsize=12)
            axes[1].legend(fontsize=12)
            axes[1].axis('equal')

            plt.tight_layout()
            plt.show()

        return _pred_cls[0], self.form_types[_pred_cls[0]]
    
    def _get_direct_vec(self, prev_locs, cur_locs):
        _pre2cur_movements = cur_locs - prev_locs
        _mean_direct_vec = np.mean(_pre2cur_movements, axis=0)
        _norm_dir = _mean_direct_vec / np.linalg.norm(_mean_direct_vec)
        return _norm_dir
    
    def infer_movements(self, prev_locs, cur_locs, cluster_labels=None, vis=False):
        if cluster_labels is None:
            _norm_direct_vec = self._get_direct_vec(prev_locs, cur_locs)
            _formtype, _formtype_name = self.infer_formtype(cur_locs, _norm_direct_vec, vis=vis)
            return _formtype, _formtype_name
        
        else:
            _uniq_labels = np.unique(cluster_labels)
            _num_clusters = len(_uniq_labels)
            
            _clusters_formtypes = []
            _clusters_formtype_names = []

            for _c_i in range(_num_clusters):
                _c_bools = cluster_labels == _uniq_labels[_c_i]
                
                if np.sum(_c_bools) <= 2:
                    _clusters_formtypes.append(None)
                    _clusters_formtype_names.append(None)
                else:
                    _cur_clust_locs = cur_locs[_c_bools]
                    _prev_clust_locs = prev_locs[_c_bools]
                    
                    _cur_direct_vec = self._get_direct_vec(_prev_clust_locs, _cur_clust_locs)
                    _cur_formtype_clsidx, _cur_formtype_name = self.infer_formtype(_cur_clust_locs, _cur_direct_vec, vis=vis)

                    _clusters_formtypes.append(_cur_formtype_clsidx)
                    _clusters_formtype_names.append(_cur_formtype_name)

            return _clusters_formtypes, _clusters_formtype_names
    
    def infer_swarm_formations(self, swarm_objs:list[basic_units.ObjTracks], cluster_labels=None, vis=False):
        # 首先获取输入轨迹的运动方向（prev_location, cur_location)
        _prv_locs = np.array([[_obj.xs[-2], _obj.ys[-2]] for _obj in swarm_objs])
        _cur_locs = np.array([[_obj.xs[-1], _obj.ys[-1]] for _obj in swarm_objs])

        # 然后提取cluster中的主要目标成员，分别计算每个编组的队型
        _uniq_clust_ids = np.unique(cluster_labels)
        _num_clusters = len(_uniq_clust_ids)

        if (cluster_labels is None) or _num_clusters <= 1:
            return self.infer_movements(_prv_locs, _cur_locs, vis=vis)
        else:
            return self.infer_movements(_prv_locs, _cur_locs, cluster_labels, vis=vis)

    def formated_formtype_result(self, clusters_labels, clusters_formtypes, clusters_formtype_names):
        """
        格式化队形结果，生成敌方每一次队形调整时各个群组的队形类型信息。
        """
        swarms = []
        unique_clusters = np.unique(clusters_labels)
        
        for idx, cluster_label in enumerate(unique_clusters):
            swarm = {}
            swarm['swarm_no'] = f'swarm{idx + 1}'
            
            # 假设 eUav 的编号从 1 开始，与 clusters_labels 的索引对应
            members = [f'eUav{uav_idx + 1}' for uav_idx, label in enumerate(clusters_labels) if label == cluster_label]
            swarm['members'] = members
            
            # 获取对应的队形类型，如果不存在则为 'none'
            if clusters_formtype_names and cluster_label < len(clusters_formtype_names):
                formtype = clusters_formtype_names[cluster_label]
                swarm['formation'] = formtype if formtype else 'none'
            else:
                swarm['formation'] = 'none'
            
            swarms.append(swarm)
        
        return swarms

    def eval_accuracy(self, model, criterion, eval_loader):
        # 一轮迭代之后，在eval数据集上面测试一下
        _eval_loss_census = 0
        
        # import pdb; pdb.set_trace()
        _orig_labels = eval_loader.dataset.labels
        _pred_outputs = []
        _pred_labels = []
        
        for _batch_i, (_padded_data, _lens, _labels) in enumerate(eval_loader):
            _outputs = model(_padded_data, _lens)
            _pred_outputs.append(_outputs)
            
            _loss = criterion(_outputs, _labels)
            _eval_loss_census += _loss.item()
        
        _pred_outputs = torch.cat(_pred_outputs, axis=0)
        _pred_labels = torch.argmax(_pred_outputs, axis=1).numpy()

        _pred_report = classification_report(_orig_labels, _pred_labels, target_names=self.form_types, output_dict=True)
        # import pdb; pdb.set_trace()
        _conf_matrix = confusion_matrix(_orig_labels, _pred_labels, labels=np.arange(len(self.form_types)))
        
        _epoch_eval_loss = _eval_loss_census / len(eval_loader)
        
        return _epoch_eval_loss, _pred_report, _conf_matrix
    
    def fit_on_data(self, train_file, eval_file, weight_save_prefix, batch_size=1, epochs=[4, 8, 12], lrs=[1e-3, 1e-5, 1e-6]):
        _train_dataset = FormationDataset(self.form_types, train_file)
        _eval_dataset = FormationDataset(self.form_types, eval_file)

        _criterion = torch.nn.CrossEntropyLoss()
        _optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        _schedular = ReduceLROnPlateau(_optimizer, mode='min', factor=0.1, patience=4, verbose=True)

        _iter_counter = 0

        _census_period = 3000
        _census_loss = 0

        self.model.train()

        for _epoch_i in range(max(epochs)):
            _train_loader = DataLoader(_train_dataset, batch_size=batch_size, collate_fn=fleet_locs_collate_fn, shuffle=True)
            _eval_loader = DataLoader(_eval_dataset, batch_size=batch_size, collate_fn=fleet_locs_collate_fn, shuffle=False)

            _cur_lr = lrs[np.sum(_epoch_i > np.array(epochs))]
            
            for _batch_i, (_padded_data, _lens, _labels) in enumerate(_train_loader):
                _outputs = self.model(_padded_data, _lens)
                _loss = _criterion(_outputs, _labels)
                _census_loss += _loss.item()

                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

                # print(f"Batch {_batch_i + 1}")
                # print(f"Padded data shape: {_padded_data.shape}")
                # print(f"Lengths: {_lens}")
                # print(f"Labels: {_labels}")

                _iter_counter = _iter_counter + 1
                if _iter_counter % _census_period == 0:
                    print(f"Epoch: {_epoch_i + 1}/{max(epochs)},Iter: {_iter_counter}, AvgLoss: {_census_loss/_census_period}")
                    _census_loss = 0
            
            _epoch_eval_loss, _eval_acc_report, _eval_cm = self.eval_accuracy(self.model, _criterion, _eval_loader)
            for _class_name, _metrics in _eval_acc_report.items():
                print(f"{_class_name}: {_metrics}")
            
            print(_eval_cm) # 打印展示各类分类结果的混淆矩阵
            
            if _eval_acc_report['accuracy'] >= 0.99:
                break               
            
            _schedular.step(_epoch_eval_loss)
            print("[Epoch %d] eval loss: %g, learning rate set to %g" 
                  % (_epoch_i + 1, _epoch_eval_loss, _optimizer.param_groups[0]['lr']))
        
        # 保存训练好的模型
        self.model_weights = self.model.state_dict()
        
        _weights_save_path = f"{weight_save_prefix}_{_iter_counter}.pth"
        torch.save(self.model_weights, _weights_save_path)
