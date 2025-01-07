import sys
import os.path as osp
from datetime import datetime
import re
import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLineEdit, QLabel, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QFileDialog, QGroupBox, QGridLayout,
    QListWidget, QAbstractItemView, QInputDialog, QComboBox
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure

ws_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if not ws_root in sys.path:
    sys.path.append(ws_root)
from formation_recognition import basic_units

class SwarmIntentExhibitor(object):
    def __init__(self, file_path, coord_scale=1, interp_scale=3, vis=False):
        self.file_path = file_path
        self.file_data = None

        self.uav_ids = None  # 无人机的编号
        self.uav_xys = None  # 无人机的轨迹坐标

        self.radar_ids = None  # 雷达的编号
        self.radar_locs = None  # 雷达的位置

        self.airport_ids = None  # 机场的编号
        self.airport_locs = None  # 机场的位置

        self.hq_ids = None  # 指挥所的编号
        self.hq_locs = None  # 指挥所的位置

        self.coord_scale = coord_scale
        self.interp_scale = interp_scale

        self._load_data(vis=vis)
        self.uavs_tracks = self.pack_to_objtracks()
    
    def _load_data(self, vis=False):
        if self.file_path.endswith('.xlsx'):
            self.file_data = pd.read_excel(self.file_path)
        elif self.file_path.endswith('.json'):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.file_data = json.load(f)
        else:
            raise ValueError("Unsupported file format")

        # 获取无人机的轨迹坐标
        self.uav_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns 
                                  if re.match(r"uav\d+_(x|y|z)", _col_nm)])

        _uav_xs = np.array([self.file_data[_uav_id + "_x"] for _uav_id in self.uav_ids])
        _uav_ys = np.array([self.file_data[_uav_id + "_y"] for _uav_id in self.uav_ids])

        _uav_xys = np.stack((_uav_xs, _uav_ys), axis=2) * self.coord_scale
        
        # 对无人机轨迹进行插值和平滑
        _orig_trj_ts = np.arange(_uav_xys.shape[1])
        _interp_trj_ts = np.linspace(0, _orig_trj_ts[-1], int(_orig_trj_ts[-1] * self.interp_scale))

        self.ts = _interp_trj_ts
        self.uav_xys = np.zeros((_uav_xys.shape[0], _interp_trj_ts.shape[0], _uav_xys.shape[2]))

        for _uav_id in range(_uav_xys.shape[0]):
            for _dim in range(_uav_xys.shape[2]):
                _interp_func = interp1d(_orig_trj_ts, _uav_xys[_uav_id, :, _dim], kind="cubic")
                self.uav_xys[_uav_id, :, _dim] = gaussian_filter1d(_interp_func(_interp_trj_ts), sigma=1)
        
        # 获取雷达的编号和位置
        self.radar_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns
                                  if re.match(r"radar\d*_(x|y|z)", _col_nm)])

        _radar_xs = np.array([self.file_data[_radar_id + "_x"] for _radar_id in self.radar_ids])
        _radar_ys = np.array([self.file_data[_radar_id + "_y"] for _radar_id in self.radar_ids])
        self.radar_locs = np.concatenate((_radar_xs[:, :1], _radar_ys[:, :1]), axis=1) * self.coord_scale

        # 获取机场的编号和位置
        self.airport_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns
                                  if re.match(r"UavAirport\d*_(x|y|z)", _col_nm)])
        
        _airport_xs = np.array([self.file_data[_airport_id + "_x"] for _airport_id in self.airport_ids])
        _airport_ys = np.array([self.file_data[_airport_id + "_y"] for _airport_id in self.airport_ids])
        self.airport_locs = np.concatenate((_airport_xs[:, :1], _airport_ys[:, :1]), axis=1) * self.coord_scale

        # 获取无人机指挥所的位置
        self.hq_ids = np.unique([_col_nm.split('_')[0] for _col_nm in self.file_data.columns
                                  if re.match(r"HQ\d*_(x|y|z)", _col_nm)])

        _hq_xs = np.array([self.file_data[_hq_id + "_x"] for _hq_id in self.hq_ids])
        _hq_ys = np.array([self.file_data[_hq_id + "_y"] for _hq_id in self.hq_ids])
        self.hq_locs = np.concatenate((_hq_xs[:, :1], _hq_ys[:, :1]), axis=1) * self.coord_scale

        # 加载意图类型（如果存在）
        if self.file_path.endswith('.json') and 'unit_types' in self.file_data:
            self.unit_types = self.file_data['unit_types']
        else:
            self.unit_types = {}
    
    def time_range(self):
        return self.ts[0], self.ts[-1]
    
    def pack_to_objtracks(self, lookback_start=1, lookback_len=10, vis=False):
        _num_objs = self.uav_xys.shape[0]

        if lookback_start > 0:
            _obj_tracks = [basic_units.ObjTracks(self.uav_xys[_o_i, -lookback_start-lookback_len:-lookback_start, 0], 
                                                self.uav_xys[_o_i, -lookback_start-lookback_len:-lookback_start, 1], 
                                                id='euav%d' % (_o_i)) 
                           for _o_i in range(_num_objs)]
            
        elif lookback_start <= 0:
            _obj_tracks = [basic_units.ObjTracks(self.uav_xys[_o_i, -lookback_len:, 0], 
                                                self.uav_xys[_o_i, -lookback_len:, 1], 
                                                id='euav%d' % (_o_i))
                           for _o_i in range(_num_objs)]
        
        return _obj_tracks
    
    def pack_to_jsondict(self):
        _json_dict = {'hqs': {'ids': [_id for _id in self.hq_ids], 'xs': self.hq_locs[:, 0].tolist(), 'ys': self.hq_locs[:, 1].tolist()},
                      'airports': {'ids': [_id for _id in self.airport_ids], 'xs': self.airport_locs[:, 0].tolist(), 'ys': self.airport_locs[:, 1].tolist()},
                      'radars': {'ids': [_id for _id in self.radar_ids], 'xs': self.radar_locs[:, 0].tolist(), 'ys': self.radar_locs[:, 1].tolist()},
                      'uavs': [
                          {'id': _trk.id, 'xs': _trk.xs.tolist(), 'ys': _trk.ys.tolist(), 'ts': _trk.ts.tolist()} for _trk in self.uavs_tracks
                      ]}
        # 添加无人机意图类型到JSON
        if hasattr(self, 'unit_types') and self.unit_types:
            _json_dict['unit_types'] = self.unit_types
        return _json_dict
    
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, exhibitor:SwarmIntentExhibitor, start=0, end=10):
        self.ax.clear()
        
        # HQ的位置
        self.ax.scatter(exhibitor.hq_locs[:, 0], exhibitor.hq_locs[:, 1], c="r", marker="*", s=100, label='head quartors')
        for _hq_i, _hq_id in enumerate(exhibitor.hq_ids):
            self.ax.text(exhibitor.hq_locs[_hq_i, 0] + 10, exhibitor.hq_locs[_hq_i, 1] + 10, _hq_id)
        
        # uav airport的位置
        self.ax.scatter(exhibitor.airport_locs[:, 0], exhibitor.airport_locs[:, 1], c="r", marker="^", s=100, label='uav airports')
        for _airport_i, _airport_id in enumerate(exhibitor.airport_ids):
            self.ax.text(exhibitor.airport_locs[_airport_i, 0] + 10, exhibitor.airport_locs[_airport_i, 1] + 10, _airport_id)
        
        # radar的位置
        self.ax.scatter(exhibitor.radar_locs[:, 0], exhibitor.radar_locs[:, 1], c="r", marker="x", s=100, label='radars')
        for _radar_i, _radar_id in enumerate(exhibitor.radar_ids):
            self.ax.text(exhibitor.radar_locs[_radar_i, 0] + 10, exhibitor.radar_locs[_radar_i, 1] + 10, _radar_id)
        
        # import pdb; pdb.set_trace()
        for _obj_i in range(exhibitor.uav_xys.shape[0]):
            _ttl_xs, _ttl_ys = exhibitor.uav_xys[_obj_i, :, 0], exhibitor.uav_xys[_obj_i, :, 1]
            
            _start_i = max(np.sum(exhibitor.ts <= start) - 1, 0)
            _end_i = min(np.sum(exhibitor.ts <= end), len(exhibitor.ts))

            _obj_xs, _obj_ys = exhibitor.uav_xys[_obj_i, _start_i:_end_i, 0], exhibitor.uav_xys[_obj_i, _start_i:_end_i, 1]

            self.ax.plot(_ttl_xs, _ttl_ys, c="green", linestyle='--', alpha=0.5)
            self.ax.plot(_obj_xs, _obj_ys, c="red", linestyle='-', alpha=0.9)

            self.ax.text(_obj_xs[-1], _obj_ys[-1], "euav%d" % (_obj_i + 1), c="red", fontsize=10)

        self.ax.grid(True, linestyle='--', linewidth=0.5)
        self.ax.legend()

        self.draw()
    
    def clear_canvas(self):
        self.ax.clear()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Trajectory Plotter")
        self.setGeometry(100, 100, 1200, 600)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Plot area
        self.plot_canvas = PlotCanvas(self)
        main_layout.addWidget(self.plot_canvas)

        # Right panel layout
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)

        # 添加单位选择列表（放在上方）
        unit_group = QGroupBox("选择无人机单位")
        unit_layout = QVBoxLayout()
        unit_group.setLayout(unit_layout)
        
        self.unit_list_widget = QListWidget()
        self.unit_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        unit_layout.addWidget(self.unit_list_widget)
        right_layout.addWidget(unit_group)

        # 添加意图类型选择组（放在下方）
        type_selection_group = QGroupBox("设置意图类型") 
        type_selection_layout = QVBoxLayout()
        type_selection_group.setLayout(type_selection_layout)

        self.type_combo_box = QComboBox()
        self.type_combo_box.addItems([
            "突破（Penetration）",
            "迂回（Detouring）", 
            "侦察（Reconnaissance）",
            "侦察打击（Search and Strike）",
            "快速通过（Fast Pass）",
            "接力打击（Sequential Attack）",
            "协同打击（Salvo Attack）"
        ])
        type_selection_layout.addWidget(QLabel("选择意图类型:"))
        type_selection_layout.addWidget(self.type_combo_box)

        self.apply_type_button = QPushButton("应用到选中单位")
        self.apply_type_button.clicked.connect(self.apply_type_to_selected)
        type_selection_layout.addWidget(self.apply_type_button)

        self.apply_all_type_button = QPushButton("应用到所有单位")
        self.apply_all_type_button.clicked.connect(self.apply_type_to_all)
        type_selection_layout.addWidget(self.apply_all_type_button)

        right_layout.addWidget(type_selection_group)

        # Time range group
        time_group = QGroupBox("设置时间范围")
        time_layout = QHBoxLayout()
        time_group.setLayout(time_layout)

        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setRange(0, 100)
        self.start_spinbox.setValue(0)
        self.start_spinbox.setSingleStep(0.1)
        self.start_spinbox.setSuffix(" 秒")
        time_layout.addWidget(QLabel("起始:"))
        time_layout.addWidget(self.start_spinbox)

        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setRange(0, 100)
        self.end_spinbox.setValue(10)
        self.end_spinbox.setSingleStep(0.1)
        self.end_spinbox.setSuffix(" 秒")
        time_layout.addWidget(QLabel("结束:"))
        time_layout.addWidget(self.end_spinbox)

        self.update_plot_button = QPushButton("绘制图表(&D)")
        self.update_plot_button.clicked.connect(self.update_plot)
        time_layout.addWidget(self.update_plot_button)
        right_layout.addWidget(time_group)

        # File selection layout
        file_layout = QHBoxLayout()
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setReadOnly(False)
        file_layout.addWidget(self.file_line_edit)

        self.open_file_button = QPushButton("打开文件(&F)")
        self.open_file_button.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_file_button)
        right_layout.addLayout(file_layout)

        # 保存标记数据的按钮
        save_layout = QHBoxLayout()
        self.save_line_edit = QLineEdit()
        self.save_line_edit.setReadOnly(False)
        save_layout.addWidget(self.save_line_edit)

        self.save_button = QPushButton("保存数据(&S)")
        self.save_button.clicked.connect(self.save_data)
        save_layout.addWidget(self.save_button)
        right_layout.addLayout(save_layout)

        self.intent_exhibitor = None
        self.trajectories = None

        self.default_data_file = osp.join("..", "data", "manual_intention_recog", "ext_search_no01.xlsx")
        self.load_trajectories(self.default_data_file)
        self.file_line_edit.setText(self.default_data_file)

        self.default_save_file = osp.join("..", "data", "manual_intention_reocg", "ext_search_no01_labeling01.json")
    
    def load_trajectories(self, file_path):
        # Load uavs trajectories from data excel file
        if (file_path is None) or (not osp.exists(file_path)):
            self.plot_canvas.clear_canvas()
        
        self.intent_exhibitor = SwarmIntentExhibitor(file_path, interp_scale=20)

        _start_t, _stop_t = self.intent_exhibitor.time_range()
        self.start_spinbox.setValue(_start_t)
        self.end_spinbox.setValue(_stop_t)

        self.plot_canvas.plot(self.intent_exhibitor, start=_start_t, end=_stop_t)

        # 填充无人机列表
        self.populate_unit_list()

    def populate_unit_list(self):
        """填充无人机列表到界面中"""
        self.unit_list_widget.clear()
        for uav_id in self.intent_exhibitor.uav_ids:
            # 检查无人机是否已有意图类型
            intent_type = self.intent_exhibitor.unit_types.get(uav_id, "")
            if intent_type:
                display_text = f"{uav_id} ({intent_type})"
            else:
                display_text = uav_id
            self.unit_list_widget.addItem(display_text)

    def update_plot(self):
        _trjs_start_t, _trjs_end_t = self.intent_exhibitor.time_range()

        start = self.start_spinbox.value()
        start = min(max(start, _trjs_start_t), _trjs_end_t)

        end = self.end_spinbox.value()
        end = min(max(end, _trjs_start_t), _trjs_end_t)

        if start < end:
            self.plot_canvas.plot(self.intent_exhibitor, start=start, end=end)
        else:
            self.statusBar().showMessage("开始时间必须小于结束时间", 5000)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开文件", osp.dirname(self.default_data_file), "All Files (*.*)", )

        if file_path:
            self.file_line_edit.setText(file_path)
        
        self.intent_exhibitor = SwarmIntentExhibitor(file_path, interp_scale=20)

        _start_t, _stop_t = self.intent_exhibitor.time_range()
        self.start_spinbox.setValue(_start_t)
        self.end_spinbox.setValue(_stop_t)

        self.plot_canvas.plot(self.intent_exhibitor, start=_start_t, end=_stop_t)

        # 重新填充无人机列表
        self.populate_unit_list()

    def save_data(self):
        # 这里使用json字符串的方式进行保存
        _base_json = self.intent_exhibitor.pack_to_jsondict()
    
        _label_start_time = self.start_spinbox.value()
        _label_end_time = self.end_spinbox.value()
    
        # 添加无人机意图类型到JSON
        if hasattr(self.intent_exhibitor, 'unit_types') and self.intent_exhibitor.unit_types:
            _base_json['unit_types'] = self.intent_exhibitor.unit_types
        # QtCore.pyqtRemoveInputHook()
        # import pdb; pdb.set_trace()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文件", osp.dirname(self.default_save_file), "Json files (*.json)")
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(_base_json, f, ensure_ascii=False, indent=4)
            self.update_status_bar(f"数据已保存到 {file_path}")
    
    def update_status_bar(self, message:str):
        _cur_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]  # 只保留前两位微秒
        self.statusBar().showMessage(f"{_cur_datetime}: {message}", 5000)

    def apply_type_to_selected(self):
        """将选择的类型应用到选中的无人机"""
        selected_items = self.unit_list_widget.selectedItems()
        if not selected_items:
            self.update_status_bar("请先选择一个无人机单位")
            return
        
        selected_item = selected_items[0]
        selected_text = selected_item.text()
        # 提取无人机ID
        uav_id = selected_text.split(' ')[0]
        selected_type = self.type_combo_box.currentText()
        
        if not hasattr(self.intent_exhibitor, 'unit_types'):
            self.intent_exhibitor.unit_types = {}
        
        self.intent_exhibitor.unit_types[uav_id] = selected_type
        # 更新列表项显示
        display_text = f"{uav_id} ({selected_type})"
        selected_item.setText(display_text)
        self.update_status_bar(f"已为无人机 {uav_id} 设置类型为 {selected_type}")
    
    def apply_type_to_all(self):
        """将选择的类型应用到所有无人机"""
        selected_type = self.type_combo_box.currentText()
        
        if not hasattr(self.intent_exhibitor, 'unit_types'):
            self.intent_exhibitor.unit_types = {}
        
        for index in range(self.unit_list_widget.count()):
            item = self.unit_list_widget.item(index)
            uav_id = item.text().split(' ')[0]
            self.intent_exhibitor.unit_types[uav_id] = selected_type
            # 更新列表项显示
            display_text = f"{uav_id} ({selected_type})"
            item.setText(display_text)
        
        self.update_status_bar(f"已为所有无人机设置类型为 {selected_type}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())