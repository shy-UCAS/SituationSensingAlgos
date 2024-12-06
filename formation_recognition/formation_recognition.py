import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    def __init__(self, fleet_locs, direction=None):
        self.fleet_locs = np.array(fleet_locs, dtype=float)
        self.direct_vec = None
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

    def _align_with_direct(self, locs=None, direct=None, vis=False):
        if locs is None:
            locs = self.fleet_locs
        
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
        if np.abs(_max_y) > 1e-3:
            locs[:, 1] = locs[:, 1] / _max_y

        _max_x = np.max(np.abs(locs[:, 0]))
        if np.abs(_max_x) > 1e-3:
            locs[:, 0] = locs[:, 0] / _max_x

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

    def fit_on_data(self, data_file, batch_size=64, epochs=10):
        _train_dataset = FormationDataset(self.form_types, data_file)

        _criterion = torch.nn.CrossEntropyLoss()
        _optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        _load_num = 1e5
        self.model.train()

        for _epoch_i in range(epochs):
            _train_loader = DataLoader(_train_dataset, batch_size=batch_size, collate_fn=fleet_locs_collate_fn, shuffle=True)

            for _batch_i, (_padded_data, _lens, _labels) in enumerate(_train_loader):
                _outputs = self.model(_padded_data, _lens)
                _loss = _criterion(_outputs, _labels)

                _optimizer.zero_grad()
                _loss.backward()
                _optimizer.step()

                # print(f"Batch {_batch_i + 1}")
                # print(f"Padded data shape: {_padded_data.shape}")
                # print(f"Lengths: {_lens}")
                # print(f"Labels: {_labels}")
                print(f"Iter: {_batch_i}, Loss: {_loss.item()}")

                # _load_num = _load_num - 1
                # if _load_num <= 0:
                #     break
        
        import pdb; pdb.set_trace()
