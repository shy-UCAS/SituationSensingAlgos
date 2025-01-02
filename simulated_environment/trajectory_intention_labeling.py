import sys


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLineEdit, QLabel, QDoubleSpinBox, QRadioButton,
    QButtonGroup, QFileDialog, QGroupBox, QGridLayout
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure
import numpy as np

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, start=0, end=10):
        self.ax.clear()
        t = np.linspace(start, end, 500)
        self.ax.plot(t, np.sin(t))
        self.ax.set_title("Trajectory Plot")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
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

        # Time range group
        time_group = QGroupBox("Set Time Range")
        time_layout = QHBoxLayout()
        time_group.setLayout(time_layout)

        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setRange(0, 100)
        self.start_spinbox.setValue(0)
        self.start_spinbox.setSuffix(" s")
        time_layout.addWidget(QLabel("Start:"))
        time_layout.addWidget(self.start_spinbox)

        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setRange(0, 100)
        self.end_spinbox.setValue(10)
        self.end_spinbox.setSuffix(" s")
        time_layout.addWidget(QLabel("End:"))
        time_layout.addWidget(self.end_spinbox)

        self.update_plot_button = QPushButton("Draw Plot")
        self.update_plot_button.clicked.connect(self.update_plot)
        time_layout.addWidget(self.update_plot_button)
        right_layout.addWidget(time_group)

        # Radio button group
        type_group = QGroupBox("Select Type")
        type_layout = QVBoxLayout()
        type_group.setLayout(type_layout)

        self.radio_group = QButtonGroup()
        for i in range(1, 8):
            radio_button = QRadioButton(f"Type {i}")
            self.radio_group.addButton(radio_button, i)
            type_layout.addWidget(radio_button)
        self.radio_group.buttons()[0].setChecked(True)
        right_layout.addWidget(type_group)

        # File selection layout
        file_layout = QHBoxLayout()
        self.file_line_edit = QLineEdit()
        self.file_line_edit.setReadOnly(True)
        file_layout.addWidget(self.file_line_edit)

        self.open_file_button = QPushButton("Open File")
        self.open_file_button.clicked.connect(self.open_file)
        file_layout.addWidget(self.open_file_button)
        right_layout.addLayout(file_layout)

        # Initialize default plot
        self.plot_canvas.plot()

    def update_plot(self):
        start = self.start_spinbox.value()
        end = self.end_spinbox.value()
        if start < end:
            self.plot_canvas.plot(start=start, end=end)
        else:
            self.statusBar().showMessage("Start time must be less than end time", 5000)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*.*)")
        if file_path:
            self.file_line_edit.setText(file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())