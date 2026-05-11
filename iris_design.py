import sys
import math
from typing import Optional, List, Tuple
import datetime
import numpy as np
import pandas as pd
from PIL import Image

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSlider, QLabel, QApplication,
    QHBoxLayout, QPushButton, QCheckBox, QTabWidget,
    QComboBox, QLineEdit, QScrollArea, QFrame, QSizePolicy,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QDoubleValidator

from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
import matplotlib.patches as patches


# =============================================================================
# 유틸 함수
# =============================================================================

def rotMat(base_coordinate, moved_coordinate, theta):
    rad = np.deg2rad(theta)
    rotmatrix = np.matrix([[np.cos(rad), -np.sin(rad)],
                           [np.sin(rad),  np.cos(rad)]])
    return base_coordinate + rotmatrix * moved_coordinate

def rotMat2(base_pos, moved_pos, theta):
    rad = np.deg2rad(theta)
    return [
        base_pos[0] + np.cos(rad) * moved_pos[0] - np.sin(rad) * moved_pos[1],
        base_pos[1] + np.sin(rad) * moved_pos[0] + np.cos(rad) * moved_pos[1]
    ]

def rotMat3(base_pos, moved_pos, theta):
    rad = np.deg2rad(theta)
    return [
        base_pos[0] + np.cos(rad) * (moved_pos[0] - base_pos[0]) - np.sin(rad) * (moved_pos[1] - base_pos[1]),
        base_pos[1] + np.sin(rad) * (moved_pos[0] - base_pos[0]) + np.cos(rad) * (moved_pos[1] - base_pos[1])
    ]

def line(a1x, a1y, a2x, a2y):
    A = (a1y - a2y)
    B = (a2x - a1x)
    C = (a1x * a2y - a1y * a2x)
    return A, B, -C

def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    return False

def in_segment(p, seg):
    return (min(seg[0], seg[2]) <= p[0] <= max(seg[0], seg[2]) and
            min(seg[1], seg[3]) <= p[1] <= max(seg[1], seg[3]))

def find_nearlist_index(lst, target):
    return min(range(len(lst)), key=lambda t: abs(lst[t] - target))

def distance_from_origin(x1, y1, x2, y2):
    distance_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    dot_product = (x2 - x1) * (-x1) + (y2 - y1) * (-y1)
    t = max(0, min(1, dot_product / distance_squared))
    closest_point = [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]
    return math.sqrt(closest_point[0] ** 2 + closest_point[1] ** 2)

def rotate_point(xs, ys, angle_degrees):
    x = np.array(xs)
    y = np.array(ys)
    angle_radians = np.radians(angle_degrees)
    x_rotated = x * np.cos(angle_radians) - y * np.sin(angle_radians)
    y_rotated = x * np.sin(angle_radians) + y * np.cos(angle_radians)
    return x_rotated, y_rotated

def line_intersection(a1, b1, a2, b2):
    for i in range(len(a1) - 1):
        for j in range(len(a2) - 1):
            x1, y1 = a1[i], b1[i]
            x2, y2 = a1[i + 1], b1[i + 1]
            x3, y3 = a2[j], b2[j]
            x4, y4 = a2[j + 1], b2[j + 1]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            if (min(x1, x2) <= intersect_x <= max(x1, x2) and
                min(y1, y2) <= intersect_y <= max(y1, y2) and
                min(x3, x4) <= intersect_x <= max(x3, x4) and
                min(y3, y4) <= intersect_y <= max(y3, y4)):
                return (intersect_x, intersect_y)
    return None

def circumcircle(points):
    point_distances = [math.sqrt(x ** 2 + y ** 2) for x, y in points]
    return max(point_distances)

def point_to_line_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(x1, y1)
    t = max(0.0, min(1.0, (-(x1) * dx + (-(y1)) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(proj_x, proj_y)

def point_to_segment_distance(x1, y1, x2, y2, a, b):
    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx ** 2 + dy ** 2
    if length_squared == 0:
        return math.sqrt((a - x1) ** 2 + (b - y1) ** 2)
    t = ((a - x1) * dx + (b - y1) * dy) / length_squared
    t = max(0, min(1, t))
    closest_x, closest_y = x1 + t * dx, y1 + t * dy
    distance = math.sqrt((a - closest_x) ** 2 + (b - closest_y) ** 2)
    return distance

def point_to_arc_distance(px, py, cx, cy, r, theta1, theta2):
    dx = px - cx
    dy = py - cy
    distance_to_center = math.sqrt(dx ** 2 + dy ** 2)
    angle_to_point = math.degrees(math.atan2(dy, dx)) % 360
    if theta1 <= angle_to_point <= theta2:
        return abs(distance_to_center - r)
    else:
        arc_start_x = cx + r * math.cos(math.radians(theta1))
        arc_start_y = cy + r * math.sin(math.radians(theta1))
        arc_end_x = cx + r * math.cos(math.radians(theta2))
        arc_end_y = cy + r * math.sin(math.radians(theta2))
        distance_to_start = math.sqrt((px - arc_start_x) ** 2 + (py - arc_start_y) ** 2)
        distance_to_end = math.sqrt((px - arc_end_x) ** 2 + (py - arc_end_y) ** 2)
        return min(distance_to_start, distance_to_end)


# =============================================================================
# 메인 위젯
# =============================================================================

class DesignWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 기본값
        self.defaults = {
            "a": -410, "b": 9, "c": 690, "d": 580, "e": -300, "f": -70,
            "g": 100, "h": 0, "i": 3, "j": 812, "k": 1930, "m": 782,
            "reverse_slot": False,
            "measure_textset_count": 1,
            "measure_set_rotation": 0
        }

        # 공통 스윕/탐색 범위 (angle2 >= angle1 조건을 적용해 사용)
        self.range_angle2 = (-50, -5, 5)   # (start, end, step)
        self.range_angle1 = (-50, -5, 5)   # (start, end, step)
        self.range_extlen = (1.0, 2.0, 0.5)

        # Figures/canvases
        self.figure = Figure(); self.canvas = FigureCanvas(self.figure); self.toolbar = NavigationToolbar(self.canvas, self)
        self.figure2 = Figure(); self.canvas2 = FigureCanvas(self.figure2); self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.figure3 = Figure(); self.canvas3 = FigureCanvas(self.figure3); self.toolbar3 = NavigationToolbar(self.canvas3, self)
        self.figure4 = Figure(); self.canvas4 = FigureCanvas(self.figure4); self.toolbar4 = NavigationToolbar(self.canvas4, self)
        self.figure5 = Figure(); self.canvas5 = FigureCanvas(self.figure5); self.toolbar5 = NavigationToolbar(self.canvas5, self)
        self.figure6 = Figure(); self.canvas6 = FigureCanvas(self.figure6); self.toolbar6 = NavigationToolbar(self.canvas6, self)

        # Root tabs
        root_vbox = QVBoxLayout(self)
        self.root_tabs = QTabWidget()
        root_vbox.addWidget(self.root_tabs)

        # ---------------------------------------
        # Tab 1: Design
        # ---------------------------------------
        iris_tab = QWidget()
        iris_hbox = QHBoxLayout(iris_tab)

        # Left controls
        left_vbox = QVBoxLayout()
        self.label_title1 = QLabel("Settings")
        self.label_title1.setFixedSize(200, 20)
        left_vbox.addWidget(self.label_title1)

        controls_vbox = left_vbox
        self.centralwidget = QWidget()
        self.centralwidget.setObjectName(u"centralwidget")

        # Buttons
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QRect(30, 30, 120, 30))
        self.pushButton.clicked.connect(lambda: self.button_clicked('Button 1'))
        self.pushButton.setText("Blade Parameter")
        controls_vbox.addWidget(self.pushButton)

        self.resetButton = QPushButton(self.centralwidget)
        self.resetButton.setText("Reset Parameters")
        self.resetButton.clicked.connect(self.reset_parameters)
        controls_vbox.addWidget(self.resetButton)

        # Checkbox
        self.checkbox = QCheckBox("Reverse slot")
        self.checkbox.setChecked(self.defaults["reverse_slot"])
        self.checkbox.stateChanged.connect(self.update_graph)
        controls_vbox.addWidget(self.checkbox)

        # Sliders
        self.a = self.defaults["a"]
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-500); self.slider.setMaximum(500); self.slider.setValue(self.a)
        self.slider.setTickInterval(100); self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider)
        self.label = QLabel(f"Slot Angle : {self.a/10:.1f}"); self.label.setFixedSize(200, 20); controls_vbox.addWidget(self.label)

        self.b = self.defaults["b"]
        self.slider_2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_2.setMinimum(4); self.slider_2.setMaximum(10); self.slider_2.setValue(self.b)
        self.slider_2.setTickInterval(1); self.slider_2.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_2.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_2)
        self.label_2 = QLabel(f"Blade N : {self.b}"); self.label_2.setFixedSize(200, 20); controls_vbox.addWidget(self.label_2)

        self.c = self.defaults["c"]
        self.slider_3 = QSlider(Qt.Orientation.Horizontal)
        self.slider_3.setMinimum(510); self.slider_3.setMaximum(1000); self.slider_3.setValue(self.c)
        self.slider_3.setTickInterval(10); self.slider_3.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_3.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_3)
        self.label_3 = QLabel(f"Radius Stator : {self.c/100:.2f}"); self.label_3.setFixedSize(200, 20); controls_vbox.addWidget(self.label_3)

        self.d = self.defaults["d"]
        self.slider_4 = QSlider(Qt.Orientation.Horizontal)
        self.slider_4.setMinimum(400); self.slider_4.setMaximum(950); self.slider_4.setValue(self.d)
        self.slider_4.setTickInterval(10); self.slider_4.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_4.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_4)
        self.label_4 = QLabel(f"Radius Rotor : {self.d/100:.2f}"); self.label_4.setFixedSize(200, 20); controls_vbox.addWidget(self.label_4)

        self.e = self.defaults["e"]
        self.slider_5 = QSlider(Qt.Orientation.Horizontal)
        self.slider_5.setMinimum(-800); self.slider_5.setMaximum(200); self.slider_5.setValue(self.e)
        self.slider_5.setTickInterval(100); self.slider_5.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_5.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_5)
        self.label_5 = QLabel(f"Blade Angle2 : {self.e/10:.1f}"); self.label_5.setFixedSize(200, 20); controls_vbox.addWidget(self.label_5)

        self.f = self.defaults["f"]
        self.slider_6 = QSlider(Qt.Orientation.Horizontal)
        self.slider_6.setMinimum(-600); self.slider_6.setMaximum(300); self.slider_6.setValue(self.f)
        self.slider_6.setTickInterval(100); self.slider_6.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_6.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_6)
        self.label_6 = QLabel(f"Blade Angle1 : {self.f/10:.1f}"); self.label_6.setFixedSize(200, 20); controls_vbox.addWidget(self.label_6)

        self.g = self.defaults["g"]
        self.slider_7 = QSlider(Qt.Orientation.Horizontal)
        self.slider_7.setMinimum(0); self.slider_7.setMaximum(300); self.slider_7.setValue(self.g)
        self.slider_7.setTickInterval(100); self.slider_7.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_7.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_7)
        self.label_7 = QLabel(f"extention length : {self.g/100:.2f}"); self.label_7.setFixedSize(200, 20); controls_vbox.addWidget(self.label_7)

        self.h = self.defaults["h"]
        self.slider_8 = QSlider(Qt.Orientation.Horizontal)
        self.slider_8.setMinimum(0); self.slider_8.setMaximum(100); self.slider_8.setValue(self.h)
        self.slider_8.setTickInterval(10); self.slider_8.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_8.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_8)
        self.label_8 = QLabel(f"Rotation Rotor : {self.h/10:.1f}"); self.label_8.setFixedSize(200, 20); controls_vbox.addWidget(self.label_8)

        self.i = self.defaults["i"]
        self.slider_9 = QSlider(Qt.Orientation.Horizontal)
        self.slider_9.setMinimum(2); self.slider_9.setMaximum(4); self.slider_9.setValue(self.i)
        self.slider_9.setTickInterval(1); self.slider_9.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_9.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_9)
        self.label_9 = QLabel(f"Layer : {self.i}"); self.label_9.setFixedSize(200, 20); controls_vbox.addWidget(self.label_9)

        self.j = self.defaults["j"]
        self.slider_10 = QSlider(Qt.Orientation.Horizontal)
        self.slider_10.setMinimum(400); self.slider_10.setMaximum(900); self.slider_10.setValue(self.j)
        self.slider_10.setTickInterval(100); self.slider_10.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_10.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_10)
        self.label_10 = QLabel(f"Diameter_Aperture_Max : {self.j/100:.2f}"); self.label_10.setFixedSize(200, 20); controls_vbox.addWidget(self.label_10)

        self.k = self.defaults["k"]
        self.slider_11 = QSlider(Qt.Orientation.Horizontal)
        self.slider_11.setMinimum(1000); self.slider_11.setMaximum(3000); self.slider_11.setValue(self.k)
        self.slider_11.setTickInterval(200); self.slider_11.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_11.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_11)
        self.label_11 = QLabel(f"Diameter_Aperture_Min : {self.k/1000:.3f}"); self.label_11.setFixedSize(200, 20); controls_vbox.addWidget(self.label_11)

        self.m = self.defaults["m"]
        self.slider_12 = QSlider(Qt.Orientation.Horizontal)
        self.slider_12.setMinimum(500); self.slider_12.setMaximum(800); self.slider_12.setValue(self.m)
        self.slider_12.setTickInterval(10); self.slider_12.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_12.valueChanged.connect(self.update_graph); controls_vbox.addWidget(self.slider_12)
        self.label_12 = QLabel(f"Soma : {self.m/100:.2f}"); self.label_12.setFixedSize(200, 20); controls_vbox.addWidget(self.label_12)

        left_vbox.addStretch(1)

        # Center panel
        center_vbox = QVBoxLayout()
        self.center_tabs = QTabWidget()
        self.center_tabs.setTabPosition(QTabWidget.North)

        iris_center = QWidget()
        iris_center_vbox = QVBoxLayout(iris_center)
        self.label_title2 = QLabel("Iris")
        self.label_title2.setFixedSize(200, 20)
        iris_center_vbox.addWidget(self.label_title2)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        iris_center_vbox.addWidget(self.canvas, stretch=1)
        iris_center_vbox.addWidget(self.toolbar)
        self.center_tabs.addTab(iris_center, "Iris")

        pressure_center = QWidget()
        pressure_center_vbox = QVBoxLayout(pressure_center)
        self.label_title2_pressure_center = QLabel("pressure_angle")
        self.label_title2_pressure_center.setFixedSize(200, 20)
        pressure_center_vbox.addWidget(self.label_title2_pressure_center)
        self.canvas6.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pressure_center_vbox.addWidget(self.canvas6, stretch=1)
        pressure_center_vbox.addWidget(self.toolbar6)
        self.center_tabs.addTab(pressure_center, "Pressure_angle")

        round_diam_tab = QWidget()
        round_diam_vbox = QVBoxLayout(round_diam_tab)
        title_rd = QLabel('Roundness@Diameter'); title_rd.setFixedHeight(20)
        round_diam_vbox.addWidget(title_rd)

        input_container = QWidget()
        input_vbox = QVBoxLayout(input_container)
        self.diam_inputs = []
        validator = QDoubleValidator(); validator.setNotation(QDoubleValidator.StandardNotation)
        for idx in range(4):
            row = QHBoxLayout()
            lbl = QLabel(f"Diameter {idx+1}"); lbl.setFixedWidth(90)
            edit = QLineEdit(); edit.setPlaceholderText("예: 3.50"); edit.setValidator(validator)
            edit.textChanged.connect(self._on_roundness_diameter_changed)
            self.diam_inputs.append(edit)
            row.addWidget(lbl); row.addWidget(edit); input_vbox.addLayout(row)
        round_diam_vbox.addWidget(input_container)

        self.round_table = QTableWidget(0, 6)
        self.round_table.setHorizontalHeaderLabels([
            "Blade Angle2", "Blade Angle1", "Diameter 1", "Diameter 2", "Diameter 3", "Diameter 4"
        ])
        self.round_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.round_table.cellClicked.connect(self._on_round_table_row_clicked)
        round_diam_vbox.addWidget(self.round_table, stretch=1)
        self.center_tabs.addTab(round_diam_tab, "Roundness@Diameter")
        center_vbox.addWidget(self.center_tabs)

        # Right panel
        right_vbox = QVBoxLayout()
        self.label_title3 = QLabel("Blade"); self.label_title3.setFixedSize(200, 20); right_vbox.addWidget(self.label_title3)
        right_vbox.addWidget(self.canvas2); right_vbox.addWidget(self.toolbar2)

        self.label_title4 = QLabel("Aperture"); self.label_title4.setFixedSize(200, 20); right_vbox.addWidget(self.canvas3); right_vbox.addWidget(self.toolbar3)

        self.label_title5 = QLabel("Roundness"); self.label_title5.setFixedSize(200, 20); right_vbox.addWidget(self.canvas4); right_vbox.addWidget(self.toolbar4)

        self.pushButton2 = QPushButton(self.centralwidget)
        self.pushButton2.clicked.connect(lambda: self.button_clicked('Button 2'))
        self.pushButton2.setText("Export roundness Data to Excel")
        right_vbox.addWidget(self.pushButton2)

        iris_hbox.addLayout(left_vbox); iris_hbox.addLayout(center_vbox); iris_hbox.addLayout(right_vbox)
        iris_hbox.setStretch(0, 1); iris_hbox.setStretch(1, 2); iris_hbox.setStretch(2, 1)
        self.root_tabs.addTab(iris_tab, "Design")

        # ---------------------------------------
        # Tab 2: measure
        # ---------------------------------------
        measure_tab = QWidget()
        measure_hbox = QHBoxLayout(measure_tab)

        measure_left_panel = QVBoxLayout()
        self.label_title2_measure = QLabel("Measure"); self.label_title2_measure.setFixedSize(200, 20)
        measure_left_panel.addWidget(self.label_title2_measure)

        upload_row = QHBoxLayout()
        self.btn_upload_measure = QPushButton("엑셀 업로드")
        self.btn_upload_measure.clicked.connect(self._upload_measure_sets_from_excel)
        upload_row.addWidget(self.btn_upload_measure)
        measure_left_panel.addLayout(upload_row)

        self.measure_status_label = QLabel("")
        self.measure_status_label.setWordWrap(True)
        self.measure_status_label.setStyleSheet("color: #555;")
        measure_left_panel.addWidget(self.measure_status_label)

        combo_row = QHBoxLayout()
        combo_label = QLabel("세트 개수"); combo_label.setFixedWidth(64)
        self.measure_combo = QComboBox()
        for n in range(1, 11):
            self.measure_combo.addItem(str(n), userData=n)
        init_count = self.defaults["measure_textset_count"]
        idx = self.measure_combo.findData(init_count)
        if idx >= 0:
            self.measure_combo.setCurrentIndex(idx)
        combo_row.addWidget(combo_label); combo_row.addWidget(self.measure_combo)
        measure_left_panel.addLayout(combo_row)

        self.measure_scroll = QScrollArea(); self.measure_scroll.setWidgetResizable(True)
        measure_container = QWidget()
        self.measure_set_layout = QVBoxLayout(measure_container)
        self.measure_scroll.setWidget(measure_container)
        measure_left_panel.addWidget(self.measure_scroll, stretch=1)

        rotation_row = QHBoxLayout()
        rotation_label = QLabel("회전(도)"); rotation_label.setFixedWidth(64)
        self.measure_rotation_edit = QLineEdit(); self.measure_rotation_edit.setPlaceholderText("예: 0")
        rot_validator = QDoubleValidator(); rot_validator.setNotation(QDoubleValidator.StandardNotation)
        self.measure_rotation_edit.setValidator(rot_validator)
        self.measure_set_rotation = self.defaults.get("measure_set_rotation", 0)
        self.measure_rotation_edit.setText(str(self.measure_set_rotation))
        self.measure_rotation_edit.textChanged.connect(self._on_measure_set_changed)
        rotation_row.addWidget(rotation_label); rotation_row.addWidget(self.measure_rotation_edit)
        measure_left_panel.addLayout(rotation_row)

        self.measure_sets = []

        measure_plot_panel = QVBoxLayout()
        self.canvas5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        measure_plot_panel.addWidget(self.canvas5, stretch=1)
        measure_plot_panel.addWidget(self.toolbar5)

        measure_hbox.addLayout(measure_left_panel, stretch=0)
        measure_hbox.addLayout(measure_plot_panel, stretch=1)
        self.root_tabs.addTab(measure_tab, "Measure")

        # Axes
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.figure2.add_subplot(111)
        self.ax3 = self.figure3.add_subplot(111)
        self.ax4 = self.figure4.add_subplot(111)
        self.ax5 = self.figure5.add_subplot(111)
        self.ax6 = self.figure6.add_subplot(111)

        self.xlim_init = (-8, 8); self.ylim_init = (-8, 8)
        self.x2lim_init = (-4, 2); self.y2lim_init = (-4, 2)
        self.x3lim_init = (-4, 4); self.y3lim_init = (-4, 4)
        self.x4lim_init = (1, 8); self.y4lim_init = (0.85, 1)
        self.x5lim_init = (-7, 7); self.y5lim_init = (-7, 7)
        self.x6lim_init = (-7, 1); self.y6lim_init = (-7, 1)

        self.xlim_current = self.xlim_init; self.ylim_current = self.ylim_init
        self.x2lim_current = self.x2lim_init; self.y2lim_current = self.y2lim_init
        self.x3lim_current = self.x3lim_init; self.y3lim_current = self.y3lim_init
        self.x4lim_current = self.x4lim_init; self.y4lim_current = self.y4lim_init
        self.x5lim_current = self.x5lim_init; self.y5lim_current = self.y5lim_init
        self.x6lim_current = self.x6lim_init; self.y6lim_current = self.y6lim_init

        self.df = pd.DataFrame()

        self.measure_combo.currentIndexChanged.connect(self._rebuild_measure_sets)
        self._rebuild_measure_sets()

        self.root_tabs.currentChanged.connect(self._on_root_tab_changed)
        self.center_tabs.currentChanged.connect(self._on_center_tab_changed)

        self.draw_graph()

    # ---------------------------
    # angle2 >= angle1 조합 생성기
    # ---------------------------
    def _iter_angle_pairs_filtered(self):
        a2s = range(self.range_angle2[0], self.range_angle2[1] + 1, self.range_angle2[2])
        a1s = range(self.range_angle1[0], self.range_angle1[1] + 1, self.range_angle1[2])
        for a2 in a2s:
            for a1 in a1s:
                if a2 <= a1:
                    yield a2, a1

    # ---------------------------
    # Tab switch handlers
    # ---------------------------
    def _on_center_tab_changed(self, idx: int):
        tab_text = self.center_tabs.tabText(idx)
        if tab_text == "pressure_angle":
            self._draw_pressure_angle(self.ax6)
        elif tab_text == "Iris":
            self._draw_iris(self.ax)
        elif tab_text == "Roundness@Diameter":
            self._update_roundness_diameter_table()
        else:
            self._draw_iris(self.ax)

    def _on_root_tab_changed(self, idx: int):
        tab_text = self.root_tabs.tabText(idx)
        if tab_text == "measure":
            self._draw_measure(self.ax5)
        else:
            cidx = self.center_tabs.currentIndex()
            ctab = self.center_tabs.tabText(cidx)
            if ctab == "pressure_angle":
                self._draw_pressure_angle(self.ax6)
            elif ctab == "Roundness@Diameter":
                self._update_roundness_diameter_table()
            else:
                self._draw_iris(self.ax)
            self._draw_blade_and_gaps(self.ax2)
            self._draw_aperture_shape(self.ax3)
            self._draw_roundness_curve(self.ax4)

    # ===== measure UI =====
    def _rebuild_measure_sets(self):
        while self.measure_set_layout.count():
            self.measure_set_layout.takeAt(0)
        self.measure_sets = []

        count = self.measure_combo.currentData() or self.defaults["measure_textset_count"]
        for i in range(count):
            frame = QFrame(); frame.setFrameShape(QFrame.StyledPanel)
            row = QHBoxLayout(frame); row.setContentsMargins(6, 6, 6, 6); row.setSpacing(6)

            lbl = QLabel(f"Set {i+1}"); lbl.setFixedWidth(56)
            validator = QDoubleValidator(); validator.setNotation(QDoubleValidator.StandardNotation)

            sx = QLineEdit(); sx.setPlaceholderText(f"sx{i+1}="); sx.setValidator(validator)
            sy = QLineEdit(); sy.setPlaceholderText(f"sy{i+1}="); sy.setValidator(validator)
            sr = QLineEdit(); sr.setPlaceholderText(f"sr{i+1}="); sr.setValidator(validator)

            rx = QLineEdit(); rx.setPlaceholderText(f"rx{i+1}="); rx.setValidator(validator)
            ry = QLineEdit(); ry.setPlaceholderText(f"ry{i+1}="); ry.setValidator(validator)
            rr = QLineEdit(); rr.setPlaceholderText(f"rr{i+1}="); rr.setValidator(validator)

            sx.textChanged.connect(self._on_measure_set_changed)
            sy.textChanged.connect(self._on_measure_set_changed)
            sr.textChanged.connect(self._on_measure_set_changed)
            rx.textChanged.connect(self._on_measure_set_changed)
            ry.textChanged.connect(self._on_measure_set_changed)
            rr.textChanged.connect(self._on_measure_set_changed)

            row.addWidget(lbl)
            row.addWidget(sx); row.addWidget(sy); row.addWidget(sr)
            row.addWidget(rx); row.addWidget(ry); row.addWidget(rr)

            self.measure_set_layout.addWidget(frame)

            self.measure_sets.append({
                "sx": None, "sy": None, "sr": None,
                "rx": None, "ry": None, "rr": None,
                "edits": (sx, sy, sr, rx, ry, rr)
            })

        self.measure_set_layout.addStretch()
        self._on_measure_set_changed()
        self._set_measure_status("세트 UI가 재구성되었습니다.", ok=True)

    def _on_measure_set_changed(self, _text=None):
        for s in self.measure_sets:
            sx, sy, sr, rx, ry, rr = s["edits"]
            s["sx"] = self._parse_float(sx.text()); s["sy"] = self._parse_float(sy.text()); s["sr"] = self._parse_float(sr.text())
            s["rx"] = self._parse_float(rx.text()); s["ry"] = self._parse_float(ry.text()); s["rr"] = self._parse_float(rr.text())
        self._draw_measure(self.ax5)

    def _parse_float(self, txt):
        try:
            if txt is None or txt.strip() == "":
                return None
            return float(txt)
        except ValueError:
            return None

    def _set_measure_status(self, msg: str, ok: bool = True):
        self.measure_status_label.setText(msg)
        self.measure_status_label.setStyleSheet(f"color: {'#2e7d32' if ok else '#c62828'};")

    # ===== 엑셀 업로드 =====
    def _upload_measure_sets_from_excel(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "엑셀 파일 선택", "", "Excel Files (*.xlsx *.xls)")
            if not file_path:
                self._set_measure_status("업로드가 취소되었습니다.", ok=False)
                return

            df = pd.read_excel(file_path)
            if df is None or df.empty:
                self._set_measure_status("엑셀에 데이터가 없습니다.", ok=False)
                print("엑셀에 데이터가 없습니다.")
                return

            normalized_map = self._normalize_measure_columns(df.columns)
            required = ["sx", "sy", "sr", "rx", "ry", "rr"]
            missing = [c for c in required if c not in normalized_map]
            if missing:
                msg = f"필요 컬럼 누락: {missing}. 허용 컬럼: {required}"
                self._set_measure_status(msg, ok=False)
                print(msg)
                return

            n_rows = len(df)
            if n_rows <= 0:
                self._set_measure_status("업로드할 행이 없습니다.", ok=False)
                print("업로드할 행이 없습니다.")
                return

            target_count = min(n_rows, self.measure_combo.itemData(self.measure_combo.count() - 1))
            idx = self.measure_combo.findData(target_count)
            if idx >= 0:
                self.measure_combo.setCurrentIndex(idx)
            else:
                self.measure_combo.setCurrentIndex(self.measure_combo.count() - 1)

            rows_to_fill = min(n_rows, len(self.measure_sets))

            def safe_val(v):
                try:
                    if pd.isna(v):
                        return ""
                    return str(float(v))
                except Exception:
                    return ""

            for i in range(rows_to_fill):
                srow = df.iloc[i]
                sx_val = safe_val(srow[normalized_map["sx"]])
                sy_val = safe_val(srow[normalized_map["sy"]])
                sr_val = safe_val(srow[normalized_map["sr"]])
                rx_val = safe_val(srow[normalized_map["rx"]])
                ry_val = safe_val(srow[normalized_map["ry"]])
                rr_val = safe_val(srow[normalized_map["rr"]])

                sx_edit, sy_edit, sr_edit, rx_edit, ry_edit, rr_edit = self.measure_sets[i]["edits"]
                sx_edit.setText(sx_val); sy_edit.setText(sy_val); sr_edit.setText(sr_val)
                rx_edit.setText(rx_val); ry_edit.setText(ry_val); rr_edit.setText(rr_val)

            self._on_measure_set_changed()
            msg_ok = f"엑셀에서 {rows_to_fill}개 세트를 적용했습니다: {file_path}"
            self._set_measure_status(msg_ok, ok=True)
            print(msg_ok)

        except Exception as ex:
            msg = f"엑셀 업로드 실패: {ex}"
            self._set_measure_status(msg, ok=False)
            print(msg)

    def _normalize_measure_columns(self, columns):
        alias_map = {
            "sx": {"sx", "start_x", "startx", "s_x", "s-x", "시작x", "시작_x"},
            "sy": {"sy", "start_y", "starty", "s_y", "s-y", "시작y", "시작_y"},
            "sr": {"sr", "start_r", "startr", "s_r", "s-r", "시작r", "시작반경", "start_radius"},
            "rx": {"rx", "ref_x", "refx", "r_x", "r-x", "기준x", "기준_x", "reference_x"},
            "ry": {"ry", "ref_y", "refy", "r_y", "r-y", "기준y", "기준_y", "reference_y"},
            "rr": {"rr", "ref_r", "refr", "r_r", "r-r", "기준r", "기준반경", "reference_radius"},
        }
        def norm(s):
            s = str(s).strip().lower()
            s = s.replace(" ", "").replace("\u00a0", "")
            s = s.replace("(", "").replace(")", "").replace("/", "").replace("\\", "")
            return s

        normalized_map = {}
        for col in columns:
            n = norm(col)
            for key, aliases in alias_map.items():
                if n in aliases:
                    normalized_map[key] = col
                    break
        return normalized_map

    # ===== 파라미터 계산 =====
    def _params(self):
        Radius_Stator = self.c / 100
        Radius_Rotor = self.d / 100
        Diameter_Aperture_Max = self.j / 100
        Diameter_Aperture_Min = self.k / 1000
        Blade_N = self.b

        Angle_Stator = 270 - 180 / Blade_N
        Angle_Stator_H = 90 - 180 / Blade_N
        Angle_Rotor = self.a / 10 / 2
        Angle_Blade_Point_1 = self.f / 10
        Angle_Blade_Point_2 = self.e / 10
        Angle_Blade_Point_3 = -60

        Size_Max = 13.3 - 0.6 - 0.2 - 0.17
        Size_Min = 5.5 + 0.6 + 0.2 + 0.17
        Radius_Blade_Edge = 0.1

        Pos_Stator_Boss = [
            Radius_Stator * np.cos(np.deg2rad(Angle_Stator)),
            Radius_Stator * np.sin(np.deg2rad(Angle_Stator)),
        ]
        
        Pos_Rotor_Boss = [
            Radius_Rotor * np.cos(np.deg2rad(Angle_Rotor)),
            Radius_Rotor * np.sin(np.deg2rad(Angle_Rotor)),
        ]

        Angle_Line = (
            np.rad2deg(
                np.acos(
                    2 * ((Diameter_Aperture_Max - Diameter_Aperture_Min) / 2) / (2 * Radius_Stator)
                    - np.cos(np.deg2rad(Angle_Stator_H - Angle_Blade_Point_2))
                )
            )
            + Angle_Stator_H
            - Angle_Blade_Point_2
        ) / 2

        Length_Extention_1 = self.g / 100
        Length_Extention_2 = (Diameter_Aperture_Max - Diameter_Aperture_Min) / 2 * np.tan(
            np.deg2rad(Angle_Stator_H - Angle_Blade_Point_2 - Angle_Line)
        )
        Angle_Blade_Rotation = 180 - 2 * Angle_Line
    
        p2 = [
            Diameter_Aperture_Min / 2 * np.cos(np.deg2rad(Angle_Blade_Point_1)),
            Diameter_Aperture_Min / 2 * np.sin(np.deg2rad(Angle_Blade_Point_1)),
        ]
        p3 = [
            Diameter_Aperture_Min / 2 * np.cos(np.deg2rad(Angle_Blade_Point_2)),
            Diameter_Aperture_Min / 2 * np.sin(np.deg2rad(Angle_Blade_Point_2)),
        ]
        p1 = [
            Diameter_Aperture_Min / 2 * np.cos(np.deg2rad(Angle_Blade_Point_1)) - Length_Extention_1 * np.sin(np.deg2rad(Angle_Blade_Point_1)),
            Diameter_Aperture_Min / 2 * np.sin(np.deg2rad(Angle_Blade_Point_1)) + Length_Extention_1 * np.cos(np.deg2rad(Angle_Blade_Point_1)),
        ]
        pe = [
            (Diameter_Aperture_Min / 2 + Radius_Blade_Edge) * np.cos(np.deg2rad(Angle_Blade_Point_1)) - Length_Extention_1 * np.sin(np.deg2rad(Angle_Blade_Point_1)),
            (Diameter_Aperture_Min / 2 + Radius_Blade_Edge) * np.sin(np.deg2rad(Angle_Blade_Point_1)) + Length_Extention_1 * np.cos(np.deg2rad(Angle_Blade_Point_1)),
        ]
        p4 = [
            Diameter_Aperture_Min / 2 * np.cos(np.deg2rad(Angle_Blade_Point_2)) + Length_Extention_2 * np.cos(np.deg2rad(270 + Angle_Blade_Point_2)),
            Diameter_Aperture_Min / 2 * np.sin(np.deg2rad(Angle_Blade_Point_2)) + Length_Extention_2 * np.sin(np.deg2rad(270 + Angle_Blade_Point_2)),
        ]
        p5 = [
            (Diameter_Aperture_Min / 2 + Radius_Blade_Edge) * np.cos(np.deg2rad(Angle_Blade_Point_1))
            - Length_Extention_1 * np.sin(np.deg2rad(Angle_Blade_Point_1))
            + Radius_Blade_Edge * np.cos(np.deg2rad(Angle_Blade_Point_1 + 30)),
            (Diameter_Aperture_Min / 2 + Radius_Blade_Edge) * np.sin(np.deg2rad(Angle_Blade_Point_1))
            + Length_Extention_1 * np.cos(np.deg2rad(Angle_Blade_Point_1))
            + Radius_Blade_Edge * np.sin(np.deg2rad(Angle_Blade_Point_1 + 30)),
        ]
        p6 = [
            p5[0] + 1.5 * np.cos(np.deg2rad(Angle_Blade_Point_1 - 60)),
            p5[1] + 1.5 * np.sin(np.deg2rad(Angle_Blade_Point_1 - 60)),
        ]

        return {
            "Radius_Stator": Radius_Stator, "Radius_Rotor": Radius_Rotor,
            "Diameter_Aperture_Max": Diameter_Aperture_Max, "Diameter_Aperture_Min": Diameter_Aperture_Min,
            "Blade_N": Blade_N,
            "Angle_Stator": Angle_Stator, "Angle_Stator_H": Angle_Stator_H, "Angle_Rotor": Angle_Rotor,
            "Angle_Blade_Point_1": Angle_Blade_Point_1, "Angle_Blade_Point_2": Angle_Blade_Point_2,
            "Angle_Blade_Point_3": Angle_Blade_Point_3,
            "Size_Max": Size_Max, "Size_Min": Size_Min,
            "Radius_Blade_Edge": Radius_Blade_Edge,
            "Pos_Stator_Boss": Pos_Stator_Boss, "Pos_Rotor_Boss": Pos_Rotor_Boss,
            "Angle_Line": Angle_Line,
            "Length_Extention_1": Length_Extention_1, "Length_Extention_2": Length_Extention_2,
            "Angle_Blade_Rotation": Angle_Blade_Rotation,
            "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "pe": pe
        }

    # ===== Plot 1: Iris =====
    def _draw_iris(self, ax):
        ax.clear()
        p = self._params()

        ax.add_patch(patches.Circle([0, 0], p["Diameter_Aperture_Min"] / 2, fc='none', ec='k', ls='--', alpha=0.5))
        ax.add_patch(patches.Circle([0, 0], p["Diameter_Aperture_Max"] / 2, fc='none', ec='k', ls='--', alpha=0.5))
        ax.add_patch(patches.Circle([0, 0], p["Size_Max"] / 2, fc='none', ec='y', ls='-', alpha=1))
        ax.add_patch(patches.Circle([0, 0], p["Size_Min"] / 2, fc='none', ec='y', ls='-', alpha=1))

        Rotation_Angle_Rotor = 10
        color = ['k', 'b', 'g', 'r']
        layer = self.i

        if self.checkbox.isChecked():
            ax.add_patch(patches.Arc([0, 0], width=2 * p["Radius_Rotor"], height=2 * p["Radius_Rotor"], angle=0,
                                     theta1=p["Angle_Stator"] + p["Angle_Rotor"] * 2 - Rotation_Angle_Rotor,
                                     theta2=p["Angle_Stator"] + p["Angle_Rotor"] * 2, fc='none', ec='r', linewidth=1))
        else:
            ax.add_patch(patches.Arc([0, 0], width=2 * p["Radius_Rotor"], height=2 * p["Radius_Rotor"], angle=0,
                                     theta1=p["Angle_Stator"] + p["Angle_Rotor"] * 2,
                                     theta2=p["Angle_Stator"] + p["Angle_Rotor"] * 2 + Rotation_Angle_Rotor, fc='none', ec='r', linewidth=1))

        for j in range(layer):
            if p["Blade_N"] / layer - int(p["Blade_N"] / layer) < 0.1:
                for i in range(int(p["Blade_N"] / layer)):
                    transform2 = Affine2D().rotate_deg_around(0, 0, (360 / p["Blade_N"]) * (layer * i + 1 + j))
                    transform3 = Affine2D().rotate_deg_around(
                        p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -p["Angle_Blade_Rotation"] / 10 * self.h / 10
                    )
                    if self.checkbox.isChecked():
                        transform4 = Affine2D().rotate_deg_around(
                            0, 0, p["Angle_Stator"] + p["Angle_Rotor"] + (360 / p["Blade_N"]) * (layer * i + 1 + j) - self.h / 10
                        ) + ax.transData
                    else:
                        transform4 = Affine2D().rotate_deg_around(
                            0, 0, p["Angle_Stator"] + p["Angle_Rotor"] + (360 / p["Blade_N"]) * (layer * i + 1 + j) + self.h / 10
                        ) + ax.transData

                    combined_transform = transform3 + transform2 + ax.transData

                    ax.add_patch(patches.Circle(p["Pos_Rotor_Boss"], 0.25, fc=color[j], ec=color[j], alpha=0.5, transform=transform4))
                    ax.add_patch(patches.Circle(p["Pos_Stator_Boss"], 0.25, fc=color[j], ec=color[j], alpha=0.5, transform=combined_transform))
                    ax.add_line(Line2D([p["p1"][0], p["p2"][0]], [p["p1"][1], p["p2"][1]], c=color[j], transform=combined_transform))
                    ax.add_patch(
                        patches.Arc([0, 0], width=p["Diameter_Aperture_Min"], height=p["Diameter_Aperture_Min"],
                                    angle=p["Angle_Blade_Point_2"], theta1=0, theta2=p["Angle_Blade_Point_1"] - p["Angle_Blade_Point_2"],
                                    fc='none', ec=color[j], linewidth=1.2, transform=combined_transform)
                    )
                    ax.add_line(Line2D([p["p3"][0], p["p4"][0]], [p["p3"][1], p["p4"][1]], c=color[j], transform=combined_transform))
                    ax.add_patch(
                        patches.Arc(rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"]),
                                    width=p["Diameter_Aperture_Max"], height=p["Diameter_Aperture_Max"],
                                    angle=p["Angle_Blade_Point_2"] + p["Angle_Blade_Point_3"],
                                    theta1=0, theta2=-p["Angle_Blade_Point_3"], fc='none', ec=color[j],
                                    linewidth=1.2, transform=combined_transform)
                    )
                    ax.add_line(Line2D([p["p5"][0], p["p6"][0]], [p["p5"][1], p["p6"][1]], c=color[j], transform=combined_transform))
                    ax.add_patch(
                        patches.Arc(p["pe"], width=2 * p["Radius_Blade_Edge"], height=2 * p["Radius_Blade_Edge"],
                                    angle=0, theta1=p["Angle_Blade_Point_1"] + 30, theta2=p["Angle_Blade_Point_1"] + 180,
                                    fc='none', ec=color[j], linewidth=1.2, transform=combined_transform)
                    )
            else:
                for i in range(p["Blade_N"]):
                    transform2 = Affine2D().rotate_deg_around(0, 0, (360 / p["Blade_N"]) * (i + 1))
                    transform3 = Affine2D().rotate_deg_around(
                        p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -p["Angle_Blade_Rotation"] / 10 * self.h / 10
                    )
                    transform4 = Affine2D().rotate_deg_around(
                        0, 0, p["Angle_Stator"] + p["Angle_Rotor"] + (360 / p["Blade_N"]) * (i + 1) + self.h / 10
                    ) + ax.transData
                    _ = transform3 + transform2 + ax.transData

        ax.set_xlim(self.xlim_current); ax.set_ylim(self.ylim_current)
        ax.set_aspect('equal')
        self.canvas.draw()

    # ===== Plot 2: Blade & Gaps =====
    def _draw_blade_and_gaps(self, ax2):
        ax2.clear()
        p = self._params()

        ax2.add_line(Line2D([p["p1"][0], p["p2"][0]], [p["p1"][1], p["p2"][1]], c='k'))
        ax2.add_patch(patches.Arc([0, 0], width=p["Diameter_Aperture_Min"], height=p["Diameter_Aperture_Min"],
                                  angle=p["Angle_Blade_Point_2"], theta1=0, theta2=p["Angle_Blade_Point_1"] - p["Angle_Blade_Point_2"],
                                  fc='none', ec='k', linewidth=1.2))
        ax2.add_line(Line2D([p["p3"][0], p["p4"][0]], [p["p3"][1], p["p4"][1]], c='k'))
        ax2.add_patch(patches.Arc(rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"]),
                                  width=p["Diameter_Aperture_Max"], height=p["Diameter_Aperture_Max"],
                                  angle=p["Angle_Blade_Point_2"] + p["Angle_Blade_Point_3"],
                                  theta1=0, theta2=-p["Angle_Blade_Point_3"], fc='none', ec='k', linewidth=1.2))
        ax2.add_line(Line2D([p["p5"][0], p["p6"][0]], [p["p5"][1], p["p6"][1]], c='k'))
        ax2.add_patch(patches.Arc(p["pe"], width=2 * p["Radius_Blade_Edge"], height=2 * p["Radius_Blade_Edge"],
                                  angle=0, theta1=p["Angle_Blade_Point_1"] + 30, theta2=p["Angle_Blade_Point_1"] + 180,
                                  fc='none', ec='k', linewidth=1.2))

        gap_blade = []
        gaps_etoe = []
        Blade_Color = ['b', 'k']
        ang = np.linspace(0, p["Angle_Blade_Rotation"], 20)
        layer = self.i

        for i in range(2):
            gaps = []
            for j in ang:
                transform5 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -j)
                transform6 = Affine2D().rotate_deg_around(0, 0, -((360 / p["Blade_N"]) + (360 / p["Blade_N"] * (layer - 1) * i)))
                transform7 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], j)

                combined_transform_2 = transform5 + transform6 + transform7 + ax2.transData
                ax2.add_patch(patches.Circle(p["pe"], p["Radius_Blade_Edge"], fc='none', ec=Blade_Color[i], linewidth=1.2, transform=combined_transform_2))
                transform_point = transform5 + transform6 + transform7
                transform_center = transform_point.transform_point(p["pe"])

                gap_line = point_to_segment_distance(p["p3"][0], p["p3"][1], p["p4"][0], p["p4"][1], transform_center[0], transform_center[1])
                arc_center = rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"])
                gap_arc = point_to_arc_distance(
                    transform_center[0], transform_center[1],
                    arc_center[0], arc_center[1],
                    p["Diameter_Aperture_Max"] / 2,
                    (p["Angle_Blade_Point_2"] - 30) % 360,
                    (p["Angle_Blade_Point_2"]) % 360,
                )
                gaps.append(gap_line)
                gaps.append(gap_arc)

                if i == 0:
                    for k in ang:
                        transform5_1 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -k)
                        transform6_1 = Affine2D().rotate_deg_around(0, 0, -((360 / p["Blade_N"]) + (360 / p["Blade_N"] * (layer - 1) * 1)))
                        transform7_1 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], k)
                        transform_point_1 = transform5_1 + transform6_1 + transform7_1
                        transform_center_1 = transform_point_1.transform_point(p["pe"])
                        gap_etoe = math.sqrt((transform_center_1[0] - transform_center[0]) ** 2 + (transform_center_1[1] - transform_center[1]) ** 2)
                        gaps_etoe.append(gap_etoe)

            min_gap = min(gaps) - p["Radius_Blade_Edge"]
            gap_blade.append(min_gap)

        gap_etoe = min(gaps_etoe) - (p["Radius_Blade_Edge"] * 2)

        if gap_blade[0] > 0:
            ax2.text(-1, 1.5, f"gap[overlap](권장 0.08이상) = {round(gap_blade[0], 4)}", ha='center', va='center')
        else:
            ax2.text(-1, 1.5, "Error", ha='center', va='center', color='r')
        if gap_blade[1] > 0:
            ax2.text(-1, 1.0, f"gap[collision](권장 0.35이상) = {round(gap_blade[1], 4)}", ha='center', va='center')
        else:
            ax2.text(-1, 1.0, "Error", ha='center', va='center', color='r')
        if gap_etoe > 0:
            ax2.text(-1, 0.5, f"gap[edge] = {round(gap_etoe, 4)}", ha='center', va='center')
        else:
            ax2.text(-1, 0.5, "Error", ha='center', va='center', color='r')

        ax2.set_xlim(self.x2lim_current); ax2.set_ylim(self.y2lim_current)
        ax2.set_aspect('equal')
        self.canvas2.draw()

    # ===== Plot 3: Aperture =====
    def _draw_aperture_shape(self, ax3):
        ax3.clear()
        p = self._params()

        p23_ang = np.linspace(p["Angle_Blade_Point_1"], p["Angle_Blade_Point_2"], 30)
        p45_ang = np.linspace(p["Angle_Blade_Point_2"], p["Angle_Blade_Point_2"] + p["Angle_Blade_Point_3"], 30)
        p23 = [(p["Diameter_Aperture_Min"] / 2 * np.cos(np.deg2rad(angle)),
                p["Diameter_Aperture_Min"] / 2 * np.sin(np.deg2rad(angle))) for angle in p23_ang]
        c_stator = rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"])
        p45 = [(c_stator[0] + p["Diameter_Aperture_Max"] / 2 * np.cos(np.deg2rad(angle)),
                c_stator[1] + p["Diameter_Aperture_Max"] / 2 * np.sin(np.deg2rad(angle)))
               for angle in p45_ang]

        points = [tuple(p["p1"])] + p23 + p45

        blade_points = []
        for x, y in points:
            xs, ys = x - p["Pos_Stator_Boss"][0], y - p["Pos_Stator_Boss"][1]
            xr = xs * np.cos(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10)) - ys * np.sin(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10))
            yr = xs * np.sin(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10)) + ys * np.cos(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10))
            blade_points.append((xr + p["Pos_Stator_Boss"][0], yr + p["Pos_Stator_Boss"][1]))
        x_blade_coords, y_blade_coords = zip(*blade_points)

        rotated_points = []
        for x, y in blade_points:
            xr = x * np.cos(np.deg2rad(360 / p["Blade_N"])) - y * np.sin(np.deg2rad(360 / p["Blade_N"]))
            yr = x * np.sin(np.deg2rad(360 / p["Blade_N"])) + y * np.cos(np.deg2rad(360 / p["Blade_N"]))
            rotated_points.append((xr, yr))
        x_rotated_coords, y_rotated_coords = zip(*rotated_points)

        intersection_point = []
        check_i = 0; check_j = 0
        for i in range(len(x_blade_coords) - 1):
            for j in range(len(x_rotated_coords) - 1):
                inter = line_intersection(
                    [x_blade_coords[i], x_blade_coords[i + 1]],
                    [y_blade_coords[i], y_blade_coords[i + 1]],
                    [x_rotated_coords[j], x_rotated_coords[j + 1]],
                    [y_rotated_coords[j], y_rotated_coords[j + 1]],
                )
                if inter:
                    intersection_point.append(inter)
                    check_i = i + 1; check_j = j + 1
        if not intersection_point:
            ax3.text(0, 0, "No intersection found", color='r')
            ax3.set_xlim(self.x3lim_current); ax3.set_ylim(self.y3lim_current)
            ax3.set_aspect('equal'); self.canvas3.draw(); return

        rotated_intersection_point = rotMat3([0, 0], [intersection_point[0][0], intersection_point[0][1]], -360 / p["Blade_N"])

        circle_x_coords = (intersection_point[0][0],) + x_blade_coords[check_i:check_j] + (rotated_intersection_point[0],)
        circle_y_coords = (intersection_point[0][1],) + y_blade_coords[check_i:check_j] + (rotated_intersection_point[1],)

        for i in range(p["Blade_N"]):
            transform8 = Affine2D().rotate_deg_around(0, 0, 360 / p["Blade_N"] * i)
            ax3.add_line(Line2D(circle_x_coords, circle_y_coords, transform=transform8 + ax3.transData))
        ax3.scatter(*intersection_point[0], color='g', marker='x')

        circle_points = list(zip(circle_x_coords, circle_y_coords))
        distances = []
        for i in range(len(circle_x_coords) - 1):
            x1, y1 = circle_x_coords[i], circle_y_coords[i]
            x2, y2 = circle_x_coords[i + 1], circle_y_coords[i + 1]
            distances.append(point_to_segment_distance(x1, y1, x2, y2, 0.0, 0.0))

        max_inscribed_radius = min(min(distances), self.m / 100 / 2)
        min_circumcircle_radius = min(circumcircle(circle_points), self.m / 100 / 2)

        ax3.text(0, 3.5, f"inner_circle = {round(2 * max_inscribed_radius, 4)}", ha='center', va='center')
        ax3.text(0, 3.0, f"roundness = {round(max_inscribed_radius / min_circumcircle_radius, 4)}", ha='center', va='center')

        self.current_inscribed_radius = max_inscribed_radius
        self.current_circum_radius = min_circumcircle_radius
        self.current_points = points

        ax3.set_xlim(self.x3lim_current); ax3.set_ylim(self.y3lim_current)
        ax3.set_aspect('equal')
        self.canvas3.draw()

    # ===== Plot 4: Roundness (곡선 데이터 저장) =====
    def _draw_roundness_curve(self, ax4):
        ax4.clear()
        p = self._params()

        graph_x_data = []
        graph_y_data = []
        result = []

        points = getattr(self, "current_points", None)
        if points is None:
            p23_ang = np.linspace(p["Angle_Blade_Point_1"], p["Angle_Blade_Point_2"], 30)
            p45_ang = np.linspace(p["Angle_Blade_Point_2"], p["Angle_Blade_Point_2"] + p["Angle_Blade_Point_3"], 30)
            p23 = [(p["Diameter_Aperture_Min"] / 2 * np.cos(np.deg2rad(angle)),
                    p["Diameter_Aperture_Min"] / 2 * np.sin(np.deg2rad(angle))) for angle in p23_ang]
            c_stator = rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"])
            p45 = [(c_stator[0] + p["Diameter_Aperture_Max"] / 2 * np.cos(np.deg2rad(angle)),
                    c_stator[1] + p["Diameter_Aperture_Max"] / 2 * np.sin(np.deg2rad(angle)))
                   for angle in p45_ang]
            points = [tuple(p["p1"])] + p23 + p45

        for k in np.linspace(0, 10, 30):
            blade_points = []
            for x, y in points:
                xs, ys = x - p["Pos_Stator_Boss"][0], y - p["Pos_Stator_Boss"][1]
                xr = xs * np.cos(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * k)) - ys * np.sin(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * k))
                yr = xs * np.sin(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * k)) + ys * np.cos(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * k))
                blade_points.append((xr + p["Pos_Stator_Boss"][0], yr + p["Pos_Stator_Boss"][1]))
            x_blade_coords, y_blade_coords = zip(*blade_points)

            rotated_points = []
            for x, y in blade_points:
                xr = x * np.cos(np.deg2rad(360 / p["Blade_N"])) - y * np.sin(np.deg2rad(360 / p["Blade_N"]))
                yr = x * np.sin(np.deg2rad(360 / p["Blade_N"])) + y * np.cos(np.deg2rad(360 / p["Blade_N"]))
                rotated_points.append((xr, yr))
            x_rotated_coords, y_rotated_coords = zip(*rotated_points)

            intersection9_point = []
            check9_i = 0; check9_j = 0
            for i in range(len(x_blade_coords) - 1):
                for j in range(len(x_rotated_coords) - 1):
                    inter = line_intersection(
                        [x_blade_coords[i], x_blade_coords[i + 1]],
                        [y_blade_coords[i], y_blade_coords[i + 1]],
                        [x_rotated_coords[j], x_rotated_coords[j + 1]],
                        [y_rotated_coords[j], y_rotated_coords[j + 1]],
                    )
                    if inter:
                        intersection9_point.append(inter)
                        check9_i = i + 1; check9_j = j + 1

            if not intersection9_point:
                continue

            rotated9_intersection_point = rotMat3([0, 0], [intersection9_point[0][0], intersection9_point[0][1]], -360 / p["Blade_N"])

            circle9_x_coords = (intersection9_point[0][0],) + x_blade_coords[check9_i:check9_j] + (rotated9_intersection_point[0],)
            circle9_y_coords = (intersection9_point[0][1],) + y_blade_coords[check9_i:check9_j] + (rotated9_intersection_point[1],)

            circle9_points = list(zip(circle9_x_coords, circle9_y_coords))
            distances9 = []
            for i in range(len(circle9_x_coords) - 1):
                x1, y1 = circle9_x_coords[i], circle9_y_coords[i]
                x2, y2 = circle9_x_coords[i + 1], circle9_y_coords[i + 1]
                distances9.append(point_to_segment_distance(x1, y1, x2, y2, 0.0, 0.0))

            max9_inscribed_radius = min(min(distances9), self.m / 100 / 2)
            min9_circumcircle_radius = min(circumcircle(circle9_points), self.m / 100 / 2)

            graph_x_data.append(round(max9_inscribed_radius * 2, 2))
            graph_y_data.append(round(max9_inscribed_radius / min9_circumcircle_radius, 4))
            result.append(np.array([round(max9_inscribed_radius * 2, 2),
                                    round(max9_inscribed_radius / min9_circumcircle_radius, 4)]))

        cur_inner = getattr(self, "current_inscribed_radius", None)
        cur_circ = getattr(self, "current_circum_radius", None)
        if cur_inner is not None and cur_circ is not None:
            ax4.scatter(cur_inner * 2, cur_inner / cur_circ, color='r', marker='x')

        ax4.plot(graph_x_data, graph_y_data)

        self.round_curve_x = graph_x_data[:]
        self.round_curve_y = graph_y_data[:]

        self.df = pd.DataFrame(result, columns=["apertrue diameter", "roundness"])

        ax4.set_xlim(self.x4lim_current); ax4.set_ylim(self.y4lim_current)
        self.canvas4.draw()

    # ===== Plot 5: measure =====
    def _draw_measure(self, ax5):
        ax5.clear()
        p = self._params()
        ax5.add_patch(patches.Circle([0, 0], p["Diameter_Aperture_Min"] / 2, fc='none', ec='k', ls='--', alpha=0.5))
        ax5.add_patch(patches.Circle([0, 0], p["Diameter_Aperture_Max"] / 2, fc='none', ec='k', ls='--', alpha=0.5))
        ax5.axhline(0, color='lightgray', lw=0.8)
        ax5.axvline(0, color='lightgray', lw=0.8)
        ax5.grid(True, linestyle='--', alpha=0.2)
        ax5.set_xlim(self.x5lim_current); ax5.set_ylim(self.y5lim_current)
        ax5.set_aspect('equal')
        self.canvas5.draw()
    
    # ===== Plot 6: pressure_angle =====
    def _draw_pressure_angle(self, ax6):
        ax6.clear()
        color = ['k', 'b', 'g', 'r']
        p = self._params()
        
        Angle_Line_2 = (np.rad2deg(np.acos(2 * ((p["Diameter_Aperture_Max"] - p["Diameter_Aperture_Min"])/2 / 2) / (2 * p["Radius_Stator"])
                                           - np.cos(np.deg2rad(90 - 180 / self.b - self.e / 10)))) + 90 - 180 / self.b - self.e / 10) / 2
        Angle_Blade_Rotation_2 = 180 - 2 * Angle_Line_2
        
        transform1 =  Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], p["Angle_Blade_Rotation"]) 
        transform2 =  Affine2D().rotate_deg_around(0, 0, 10) 
        transform3 = Affine2D().rotate_deg_around(0, 0, p["Angle_Stator"] + 2*p["Angle_Rotor"])
        
        transform1_1 =  Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], Angle_Blade_Rotation_2)
        transform2_1 = Affine2D().rotate_deg_around(0, 0, 5)
        
        if self.checkbox.isChecked():
            line_c1 = Line2D([0, p["Pos_Stator_Boss"][0]], [0, p["Pos_Stator_Boss"][1]], c=color[0], linewidth=0.2)
            line_s1 = Line2D([p["Pos_Rotor_Boss"][0], 0], [p["Pos_Rotor_Boss"][1], 0], c=color[0], linewidth=0.2, transform=transform3 + ax6.transData)
            line_c3 = Line2D([0, p["Pos_Stator_Boss"][0]], [0, p["Pos_Stator_Boss"][1]], c=color[1], linewidth=0.2, transform=transform1 + ax6.transData)
            line_s3 = Line2D([p["Pos_Rotor_Boss"][0], 0], [p["Pos_Rotor_Boss"][1], 0], c=color[1], linewidth=0.2, transform=transform3 - transform2 + transform1 + ax6.transData)
            line_c2 = Line2D([0, p["Pos_Stator_Boss"][0]], [0, p["Pos_Stator_Boss"][1]], c=color[2], linewidth=0.2, transform=transform1_1 + ax6.transData)
            line_s2 = Line2D([p["Pos_Rotor_Boss"][0], 0], [p["Pos_Rotor_Boss"][1], 0], c=color[2], linewidth=0.2, transform=transform3 - transform2_1 + transform1_1 + ax6.transData)
        else:
            line_c1 = Line2D([0, p["Pos_Stator_Boss"][0]], [0, p["Pos_Stator_Boss"][1]], c=color[0], linewidth=0.2)
            line_s1 = Line2D([p["Pos_Rotor_Boss"][0], 0], [p["Pos_Rotor_Boss"][1], 0], c=color[0], linewidth=0.2, transform=transform3 + ax6.transData)
            line_c3 = Line2D([0, p["Pos_Stator_Boss"][0]], [0, p["Pos_Stator_Boss"][1]], c=color[1], linewidth=0.2, transform=transform1 + ax6.transData)
            line_s3 = Line2D([p["Pos_Rotor_Boss"][0], 0], [p["Pos_Rotor_Boss"][1], 0], c=color[1], linewidth=0.2, transform=transform3 + transform2 + transform1 + ax6.transData)
            line_c2 = Line2D([0, p["Pos_Stator_Boss"][0]], [0, p["Pos_Stator_Boss"][1]], c=color[2], linewidth=0.2, transform=transform1_1 + ax6.transData)
            line_s2 = Line2D([p["Pos_Rotor_Boss"][0], 0], [p["Pos_Rotor_Boss"][1], 0], c=color[2], linewidth=0.2, transform=transform3 + transform2_1 + transform1_1 + ax6.transData)      
        
        ax6.add_line(line_c1), ax6.add_line(line_c2), ax6.add_line(line_c3), ax6.add_line(line_s1), ax6.add_line(line_s2), ax6.add_line(line_s3)

        def get_start_points(line, to='display'):
            """
            line: Line2D 객체
            to: 'data' | 'display' | 'axes'
                - 'data': 데이터 좌표 그대로 반환
                - 'display': 디스플레이(픽셀) 좌표로 변환
                - 'axes': 축 좌표(0~1 범위)로 변환
            """
            # 시작점 데이터 좌표
            x0 = line.get_xdata()[0]
            y0 = line.get_ydata()[0]
            start_data = np.array([[x0, y0]])

            if to == 'data':
                return (x0, y0)

            # 데이터 -> 디스플레이 좌표
            start_display = line.get_transform().transform(start_data)[0]

            if to == 'display':
                return tuple(start_display)

            if to == 'axes':
                # 디스플레이 -> 축 좌표 변환
                start_axes = ax6.transAxes.inverted().transform(start_display)
                return tuple(start_axes)

            raise ValueError("to must be one of {'data','display','axes'}")

        # 각 라인의 시작 포인트 구하기
        start_c1_data     = get_start_points(line_c1, to='data')
        start_c1_display  = get_start_points(line_c1, to='display')
        start_c1_axes     = get_start_points(line_c1, to='axes')

        start_c3_data     = get_start_points(line_c3, to='data')
        start_c3_display  = get_start_points(line_c3, to='display')
        start_c3_axes     = get_start_points(line_c3, to='axes')

        start_c2_data     = get_start_points(line_c2, to='data')
        start_c2_display  = get_start_points(line_c2, to='display')
        start_c2_axes     = get_start_points(line_c2, to='axes')
        
        start_s1_data     = get_start_points(line_s1, to='data')
        start_s1_display  = get_start_points(line_s1, to='display')
        start_s1_axes     = get_start_points(line_s1, to='axes')

        start_s3_data     = get_start_points(line_s3, to='data')
        start_s3_display  = get_start_points(line_s3, to='display')
        start_s3_axes     = get_start_points(line_s3, to='axes')

        start_s2_data     = get_start_points(line_s2, to='data')
        start_s2_display  = get_start_points(line_s2, to='display')
        start_s2_axes     = get_start_points(line_s2, to='axes')

        # print("Line A 시작점 - 데이터:", start_s1_data, "디스플레이:", start_s1_display, "축:", start_s1_axes)
        # print("Line B 시작점 - 데이터:", start_s3_data, "디스플레이:", start_s3_display, "축:", start_s3_axes)
        # print("Line C 시작점 - 데이터:", start_s2_data, "디스플레이:", start_s2_display, "축:", start_s2_axes)

                # ===== Add: draw arc passing through start_a_axes, start_b_axes, start_c_axes =====
        def axes_to_data(ax, xy_axes):
            """axes(0~1) 좌표를 data 좌표로 변환"""
            disp = ax.transAxes.transform(xy_axes)  # axes -> display
            data = ax.transData.inverted().transform(disp)  # display -> data
            return data

        # 1) 축 좌표(0~1)로 얻은 세 점을 데이터 좌표로 변환
        c1_data = np.array(axes_to_data(ax6, start_c1_axes))
        c2_data = np.array(axes_to_data(ax6, start_c2_axes))
        c3_data = np.array(axes_to_data(ax6, start_c3_axes))
        
        s1_data = np.array(axes_to_data(ax6, start_s1_axes))
        s2_data = np.array(axes_to_data(ax6, start_s2_axes))
        s3_data = np.array(axes_to_data(ax6, start_s3_axes))

        # 2) 세 점으로 원의 중심과 반지름 계산 (외접원)
        def circle_from_3pts(P, Q, R):
            """
            세 점 P, Q, R (각각 [x, y])로부터 원의 중심과 반지름을 계산.
            반환: (center_x, center_y, radius)
            """
            # 행렬식 기반 계산
            x1, y1 = P
            x2, y2 = Q
            x3, y3 = R

            # 두 벡터
            a = x2 - x1; b = y2 - y1
            c = x3 - x1; d = y3 - y1

            # 행렬식
            det = 2 * (a * d - b * c)
            if np.isclose(det, 0.0):
                # 세 점이 거의 일직선인 경우: 처리 불가
                return None

            # 제곱 길이
            e = a * (x1 + x2) + b * (y1 + y2)
            f = c * (x1 + x3) + d * (y1 + y3)

            cx = (d * e - b * f) / det
            cy = (-c * e + a * f) / det
            r = np.hypot(cx - x1, cy - y1)
            return np.array([cx, cy]), r

        result = circle_from_3pts(s1_data, s2_data, s3_data)
        if result is None:
            # 세 점이 일직선에 가까워 원을 정의할 수 없는 경우
            # 안전하게 리턴하거나, 짧은 선으로 대체
            # 여기서는 메시지만 출력
            print("세 점이 거의 일직선입니다. 원호를 그릴 수 없습니다.")
        else:
            center, radius = result
            cx, cy = center

            # 3) 세 점의 극각 계산
            def angle_of(pt):
                return np.arctan2(pt[1] - cy, pt[0] - cx)  # 라디안
            
            def angle_three_points(A, B, C, return_degrees=True, eps=1e-12):
                A = np.asarray(A, dtype=float)
                B = np.asarray(B, dtype=float)
                C = np.asarray(C, dtype=float)
                
                # 벡터 BA, BC
                u = A - B
                v = C - B

                # 노름(길이)
                nu = np.linalg.norm(u)
                nv = np.linalg.norm(v)

                if nu < eps or nv < eps:
                    raise ValueError("두 벡터 중 하나의 길이가 0입니다. 서로 다른 세 점을 입력하세요.")

                # 코사인 값 계산 (수치 안정화를 위해 clip)
                cos_theta = np.dot(u, v) / (nu * nv)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)

                theta = np.arccos(cos_theta)  # 라디안
                if return_degrees:
                    return np.degrees(theta)
                return theta

            angA = angle_of(s1_data)
            angB = angle_of(s2_data)
            angC = angle_of(s3_data)

            # 각도를 0~2π로 정규화
            def norm(a):
                a = np.mod(a, 2 * np.pi)
                return a

            angs = np.array([norm(angA), norm(angB), norm(angC)])

            # 4) 원호 방향 결정: A->B->C가 시계/반시계 중 자연스러운 방향을 선택
            # 간단히 전체 범위를 커버하는 최소 호를 그리도록 함
            ang_min = np.min(angs)
            ang_max = np.max(angs)
            # 두 호 중 더 짧은 것을 선택
            span_direct = ang_max - ang_min
            span_wrap = 2 * np.pi - span_direct

            # 기준을 A로 두고, B와 C를 포함하는 짧은 호를 선택
            # 여기서는 ang_min -> ang_max 방향으로 그리되, wrap이 더 짧으면 wrap 경로로 그림
            num = 200
            if span_direct <= span_wrap:
                theta = np.linspace(ang_min, ang_max, num)
            else:
                # wrap 경로: ang_max -> ang_min + 2π
                theta = np.linspace(ang_max, ang_min + 2 * np.pi, num)

            x_arc = cx + radius * np.cos(theta)
            y_arc = cy + radius * np.sin(theta)

            # 5) 원호 그리기
            ax6.plot(x_arc, y_arc, color='m', linewidth=1.2, label='arc through A,B,C')
            ax6.scatter([s1_data[0], s2_data[0], s3_data[0]],
                        [s1_data[1], s2_data[1], s3_data[1]],
                        s=20, color='m', zorder=5)

            ax6.add_line(Line2D([s1_data[0], cx], [s1_data[1], cy], c=color[3], linewidth=0.2))
            ax6.add_line(Line2D([s2_data[0], cx], [s2_data[1], cy], c=color[3], linewidth=0.2))
            ax6.add_line(Line2D([s3_data[0], cx], [s3_data[1], cy], c=color[3], linewidth=0.2))
            ax6.text(s1_data[0]-0.5, s1_data[1], f"{angle_three_points(c1_data, s1_data, np.array([cx,cy]), return_degrees=True):.2f}", color="#c62828", fontsize=8)
            ax6.text(s2_data[0]-0.5, s2_data[1], f"{angle_three_points(c2_data, s2_data, np.array([cx,cy]), return_degrees=True):.2f}", color="#c62828", fontsize=8)
            ax6.text(s3_data[0]-0.5, s3_data[1], f"{angle_three_points(c3_data, s3_data, np.array([cx,cy]), return_degrees=True):.2f}", color="#c62828", fontsize=8)
            
            # 6) s1, s2, s3에서 법선(normal) 벡터 표시
            def unit(v):
                n = np.linalg.norm(v)
                return v / n if n > 1e-12 else np.array([0.0, 0.0])

            center_vec = np.array([cx, cy])

            # 각 점에서의 법선 단위 벡터 (점 -> 중심 방향)
            n1 = unit(center_vec - s1_data)
            n2 = unit(center_vec - s2_data)
            n3 = unit(center_vec - s3_data)

            # 표시 스케일(상황에 맞게 조정)
            scale = max(0.5, radius * 0.25)

            # 화살표로 벡터 표시
            ax6.arrow(s1_data[0], s1_data[1], n1[0]*scale, n1[1]*scale,
                    head_width=0.08, head_length=0.12,
                    fc='tab:orange', ec='tab:orange', length_includes_head=True)
            ax6.arrow(s2_data[0], s2_data[1], n2[0]*scale, n2[1]*scale,
                    head_width=0.08, head_length=0.12,
                    fc='tab:orange', ec='tab:orange', length_includes_head=True)
            ax6.arrow(s3_data[0], s3_data[1], n3[0]*scale, n3[1]*scale,
                    head_width=0.08, head_length=0.12,
                    fc='tab:orange', ec='tab:orange', length_includes_head=True)

            # 라벨(선택)
            ax6.text(s1_data[0] + n1[0]*scale*1.05, s1_data[1] + n1[1]*scale*1.05, 'n1', color='tab:orange', fontsize=8)
            ax6.text(s2_data[0] + n2[0]*scale*1.05, s2_data[1] + n2[1]*scale*1.05, 'n2', color='tab:orange', fontsize=8)
            ax6.text(s3_data[0] + n3[0]*scale*1.05, s3_data[1] + n3[1]*scale*1.05, 'n3', color='tab:orange', fontsize=8)          
            
            # Slt 중심점 표시
            ax6.scatter([cx], [cy], s=30, color='c', zorder=5)

            # 범례 추가(선택)
            # ax6.legend(loc='best')  
        
        ax6.set_xlim(self.x6lim_current); ax6.set_ylim(self.y6lim_current)
        ax6.set_aspect('equal')
        self.canvas6.draw()
        
    # ===== Roundness@Diameter: 입력 변경 핸들러 =====
    def _on_roundness_diameter_changed(self, _txt=None):
        self._update_roundness_diameter_table()

    # ===== Roundness@Diameter 행 클릭 =====
    def _on_round_table_row_clicked(self, row: int, column: int):
        try:
            angle2_item = self.round_table.item(row, 0)
            angle1_item = self.round_table.item(row, 1)
            if angle2_item is None or angle1_item is None:
                return
            angle2 = float(angle2_item.text())
            angle1 = float(angle1_item.text())
            self.e = float(round(angle2 * 10))
            self.f = float(round(angle1 * 10))
            self.slider_5.setValue(self.e)
            self.slider_6.setValue(self.f)
            self.label_5.setText(f"Blade Angle2 : {self.e/10:.1f}")
            self.label_6.setText(f"Blade Angle1 : {self.f/10:.1f}")
            self._draw_aperture_shape(self.ax3)
            self._draw_roundness_curve(self.ax4)
            self._draw_blade_and_gaps(self.ax2)
        except Exception as ex:
            print(f"행 클릭 처리 오류: {ex}")

    def _compute_roundness_for_angles(self, angle2_deg: float, angle1_deg: float, diameter: float):
        try:
            backup_e = self.e
            backup_f = self.f
            backup_points = getattr(self, "current_points", None)

            self.e = int(round(angle2_deg * 10))
            self.f = int(round(angle1_deg * 10))

            p = self._params()

            p23_ang = np.linspace(p["Angle_Blade_Point_1"], p["Angle_Blade_Point_2"], 30)
            p45_ang = np.linspace(p["Angle_Blade_Point_2"], p["Angle_Blade_Point_2"] + p["Angle_Blade_Point_3"], 30)
            p23 = [(p["Diameter_Aperture_Min"] / 2 * np.cos(np.deg2rad(angle)),
                    p["Diameter_Aperture_Min"] / 2 * np.sin(np.deg2rad(angle))) for angle in p23_ang]
            c_stator = rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"])
            p45 = [(c_stator[0] + p["Diameter_Aperture_Max"] / 2 * np.cos(np.deg2rad(angle)),
                    c_stator[1] + p["Diameter_Aperture_Max"] / 2 * np.sin(np.deg2rad(angle)))
                   for angle in p45_ang]
            points = [tuple(p["p1"])] + p23 + p45

            blade_points = []
            for x, y in points:
                xs, ys = x - p["Pos_Stator_Boss"][0], y - p["Pos_Stator_Boss"][1]
                xr = xs * np.cos(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10)) - ys * np.sin(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10))
                yr = xs * np.sin(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10)) + ys * np.cos(np.deg2rad(-p["Angle_Blade_Rotation"] / 10 * self.h / 10))
                blade_points.append((xr + p["Pos_Stator_Boss"][0], yr + p["Pos_Stator_Boss"][1]))
            x_blade_coords, y_blade_coords = zip(*blade_points)

            rotated_points = []
            for x, y in blade_points:
                xr = x * np.cos(np.deg2rad(360 / p["Blade_N"])) - y * np.sin(np.deg2rad(360 / p["Blade_N"]))
                yr = x * np.sin(np.deg2rad(360 / p["Blade_N"])) + y * np.cos(np.deg2rad(360 / p["Blade_N"]))
                rotated_points.append((xr, yr))
            x_rotated_coords, y_rotated_coords = zip(*rotated_points)

            intersection_point = []
            check_i = 0; check_j = 0
            for i in range(len(x_blade_coords) - 1):
                for j in range(len(x_rotated_coords) - 1):
                    inter = line_intersection(
                        [x_blade_coords[i], x_blade_coords[i + 1]],
                        [y_blade_coords[i], y_blade_coords[i + 1]],
                        [x_rotated_coords[j], x_rotated_coords[j + 1]],
                        [y_rotated_coords[j], y_rotated_coords[j + 1]],
                    )
                    if inter:
                        intersection_point.append(inter)
                        check_i = i + 1; check_j = j + 1
            if not intersection_point:
                self.e = backup_e; self.f = backup_f
                if backup_points is not None:
                    self.current_points = backup_points
                return None, None

            rotated_intersection_point = rotMat3([0, 0], [intersection_point[0][0], intersection_point[0][1]], -360 / p["Blade_N"])
            circle_x_coords = (intersection_point[0][0],) + x_blade_coords[check_i:check_j] + (rotated_intersection_point[0],)
            circle_y_coords = (intersection_point[0][1],) + y_blade_coords[check_i:check_j] + (rotated_intersection_point[1],)

            circle_points = list(zip(circle_x_coords, circle_y_coords))
            distances = []
            for i in range(len(circle_x_coords) - 1):
                x1, y1 = circle_x_coords[i], circle_y_coords[i]
                x2, y2 = circle_x_coords[i + 1], circle_y_coords[i + 1]
                distances.append(point_to_segment_distance(x1, y1, x2, y2, 0.0, 0.0))

            max_inscribed_radius = min(distances) if distances else None
            if max_inscribed_radius is None or max_inscribed_radius <= 0:
                self.e = backup_e; self.f = backup_f
                if backup_points is not None:
                    self.current_points = backup_points
                return None, None

            min_circumcircle_radius = min(circumcircle(circle_points), self.m / 100 / 2)
            if min_circumcircle_radius <= 0:
                self.e = backup_e; self.f = backup_f
                if backup_points is not None:
                    self.current_points = backup_points
                return None, None

            if diameter is None or diameter <= 0:
                self.e = backup_e; self.f = backup_f
                if backup_points is not None:
                    self.current_points = backup_points
                return None, None

            roundness_val = max_inscribed_radius / min_circumcircle_radius

            self.e = backup_e; self.f = backup_f
            if backup_points is not None:
                self.current_points = backup_points

            return float(diameter), float(roundness_val)

        except Exception as ex:
            self.e = backup_e
            self.f = backup_f
            if backup_points is not None:
                self.current_points = backup_points
            print(f"[Roundness@Diameter] 계산 실패(angle2={angle2_deg}, angle1={angle1_deg}, d={diameter}): {ex}")
            return None, None

    def _compute_roundness_for_diameter(self, diameter: float):
        return None, None

    # ===== Roundness@Diameter 표 갱신 =====
    def _update_roundness_diameter_table(self):
        diam_vals = []
        for edit in self.diam_inputs:
            v = self._parse_float(edit.text())
            diam_vals.append(v if (v is not None and v > 0) else None)

        self.round_table.clear()
        self.round_table.setRowCount(0)

        def fmt_d(v):
            return "N/A" if v is None else f"{v:.4f}"

        headers = [
            "Blade Angle2",
            "Blade Angle1",
            f"R1 (D1: {fmt_d(diam_vals[0])})",
            f"R2 (D2: {fmt_d(diam_vals[1])})",
            f"R3 (D3: {fmt_d(diam_vals[2])})",
            f"R4 (D4: {fmt_d(diam_vals[3])})",
        ]
        self.round_table.setColumnCount(len(headers))
        self.round_table.setHorizontalHeaderLabels(headers)

        for angle2, angle1 in self._iter_angle_pairs_filtered():
            row = self.round_table.rowCount()
            self.round_table.insertRow(row)
            self.round_table.setItem(row, 0, QTableWidgetItem(f"{angle2}"))
            self.round_table.setItem(row, 1, QTableWidgetItem(f"{angle1}"))

            for idx in range(4):
                col = 2 + idx
                dv = diam_vals[idx]
                if dv is None:
                    item = QTableWidgetItem("N/A")
                else:
                    d_val, r_val = self._compute_roundness_for_angles(angle2, angle1, dv)
                    item = QTableWidgetItem("N/A" if (d_val is None or r_val is None) else f"{r_val:.4f}")
                    if d_val is not None:
                        item.setToolTip(f"D{idx+1}: {d_val:.4f}")
                self.round_table.setItem(row, col, item)

        self.round_table.resizeColumnsToContents()

    # ===== Master draw/update/reset =====
    def draw_graph(self):
        self._draw_iris(self.ax)
        self._draw_blade_and_gaps(self.ax2)
        self._draw_aperture_shape(self.ax3)
        self._draw_roundness_curve(self.ax4)
        self._draw_measure(self.ax5)
        self._draw_pressure_angle(self.ax6)
        self._update_roundness_diameter_table()

    def update_graph(self):
        self.a = self.slider.value(); self.label.setText(f"Slot Angle : {self.a/10:.1f}")
        self.b = self.slider_2.value(); self.label_2.setText(f"Blade N : {self.b}")
        self.c = self.slider_3.value(); self.label_3.setText(f"Radius Stator : {self.c/100:.2f}")
        self.d = self.slider_4.value(); self.label_4.setText(f"Radius Rotor : {self.d/100:.2f}")
        self.e = self.slider_5.value(); self.label_5.setText(f"Blade Angle2 : {self.e/10:.1f}")
        self.f = self.slider_6.value(); self.label_6.setText(f"Blade Angle1 : {self.f/10:.1f}")
        self.g = self.slider_7.value(); self.label_7.setText(f"extention length : {self.g/100:.2f}")
        self.h = self.slider_8.value(); self.label_8.setText(f"Rotation Rotor : {self.h/10:.1f}")
        self.i = self.slider_9.value(); self.label_9.setText(f"Layer : {self.i}")
        self.j = self.slider_10.value(); self.label_10.setText(f"Diameter_Aperture_Max : {self.j/100:.2f}")
        self.k = self.slider_11.value(); self.label_11.setText(f"Diameter_Aperture_Min : {self.k/1000:.3f}")
        self.m = self.slider_12.value(); self.label_12.setText(f"Soma : {self.m/100:.2f}")
        self.draw_graph()

    def reset_parameters(self):
        self.a = self.defaults["a"]; self.slider.setValue(self.a); self.label.setText(f"Slot Angle : {self.a/10:.1f}")
        self.b = self.defaults["b"]; self.slider_2.setValue(self.b); self.label_2.setText(f"Blade N : {self.b}")
        self.c = self.defaults["c"]; self.slider_3.setValue(self.c); self.label_3.setText(f"Radius Stator : {self.c/100:.2f}")
        self.d = self.defaults["d"]; self.slider_4.setValue(self.d); self.label_4.setText(f"Radius Rotor : {self.d/100:.2f}")
        self.e = self.defaults["e"]; self.slider_5.setValue(self.e); self.label_5.setText(f"Blade Angle2 : {self.e/10:.1f}")
        self.f = self.defaults["f"]; self.slider_6.setValue(self.f); self.label_6.setText(f"Blade Angle1 : {self.f/10:.1f}")
        self.g = self.defaults["g"]; self.slider_7.setValue(self.g); self.label_7.setText(f"extention length : {self.g/100:.2f}")
        self.h = self.defaults["h"]; self.slider_8.setValue(self.h); self.label_8.setText(f"Rotation Rotor : {self.h/10:.1f}")
        self.i = self.defaults["i"]; self.slider_9.setValue(self.i); self.label_9.setText(f"Layer : {self.i}")
        self.j = self.defaults["j"]; self.slider_10.setValue(self.j); self.label_10.setText(f"Diameter_Aperture_Max : {self.j/100:.2f}")
        self.k = self.defaults["k"]; self.slider_11.setValue(self.k); self.label_11.setText(f"Diameter_Aperture_Min : {self.k/1000:.3f}")
        self.m = self.defaults["m"]; self.slider_12.setValue(self.m); self.label_12.setText(f"Soma : {self.m/100:.2f}")

        self.checkbox.setChecked(self.defaults["reverse_slot"])

        self.xlim_current = self.xlim_init; self.ylim_current = self.ylim_init
        self.x2lim_current = self.x2lim_init; self.y2lim_current = self.y2lim_init
        self.x3lim_current = self.x3lim_init; self.y3lim_current = self.y3lim_init
        self.x4lim_current = self.x4lim_init; self.y4lim_current = self.y4lim_init
        self.x5lim_current = self.x5lim_init; self.y5lim_current = self.y5lim_init
        self.x6lim_current = self.x6lim_init; self.y6lim_current = self.y6lim_init

        init_idx = self.measure_combo.findData(self.defaults["measure_textset_count"])
        if init_idx >= 0:
            self.measure_combo.setCurrentIndex(init_idx)
        self._rebuild_measure_sets()
        self._set_measure_status("파라미터가 초기화되었습니다.", ok=True)

        for edit in self.diam_inputs:
            edit.clear()

        self.draw_graph()
    
    def _compute_gaps_for_current_params(self):
        """
        현재 self.e/self.f/self.g 상태에서 gap[overlap], gap[collision], gap[edge] 값을
        Plot 2(_draw_blade_and_gaps)의 계산과 '완전히 동일한 방식'으로 산출하여 반환.
        반환: (gap_overlap, gap_collision, gap_edge)
        """
        import numpy as np
        import math
        from matplotlib.transforms import Affine2D

        p = self._params()

        # Plot 2와 동일한 샘플 수/각도 범위 사용
        ang = np.linspace(0, p["Angle_Blade_Rotation"], 20)
        layer = self.i

        gap_blade = []   # [overlap, collision] 최소값 저장
        gaps_etoe = []   # edge-2-edge 후보들(최소값 사용)

        # i=0: overlap, i=1: collision (Plot 2와 동일)
        for i in range(2):
            gaps = []
            for j in ang:
                # Plot 2의 동일 변환(데이터 좌표 기준 Affine2D만 사용)
                transform5 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -j)
                transform6 = Affine2D().rotate_deg_around(
                    0, 0, -((360 / p["Blade_N"]) + (360 / p["Blade_N"] * (layer - 1) * i))
                )
                transform7 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], j)

                transform_point = transform5 + transform6 + transform7
                transform_center = transform_point.transform_point(p["pe"])  # 회전된 pe의 중심(데이터 좌표)

                # 선분(p3->p4)까지의 거리
                gap_line = point_to_segment_distance(
                    p["p3"][0], p["p3"][1], p["p4"][0], p["p4"][1],
                    transform_center[0], transform_center[1]
                )

                # 대구멍 호까지의 거리(Plot 2에서 쓰는 중심/반지름/각도)
                arc_center = rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"])
                gap_arc = point_to_arc_distance(
                    transform_center[0], transform_center[1],
                    arc_center[0], arc_center[1],
                    p["Diameter_Aperture_Max"] / 2,
                    (p["Angle_Blade_Point_2"] - 30) % 360,
                    (p["Angle_Blade_Point_2"]) % 360
                )

                # 후보 추가(Plot 2와 동일하게 선분과 호 거리 모두 고려)
                gaps.append(gap_line)
                gaps.append(gap_arc)

                # edge-2-edge는 i==0 루프에서만 누적(Plot 2와 동일)
                if i == 0:
                    for k in ang:
                        transform5_1 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], -k)
                        transform6_1 = Affine2D().rotate_deg_around(
                            0, 0, -((360 / p["Blade_N"]) + (360 / p["Blade_N"] * (layer - 1) * 1))
                        )
                        transform7_1 = Affine2D().rotate_deg_around(p["Pos_Stator_Boss"][0], p["Pos_Stator_Boss"][1], k)
                        transform_point_1 = transform5_1 + transform6_1 + transform7_1
                        transform_center_1 = transform_point_1.transform_point(p["pe"])
                        gap_etoe = math.sqrt(
                            (transform_center_1[0] - transform_center[0]) ** 2 +
                            (transform_center_1[1] - transform_center[1]) ** 2
                        )
                        gaps_etoe.append(gap_etoe)

            # 최소 후보에서 블레이드 에지 반경 보정(Plot 2와 동일)
            min_gap = min(gaps) - p["Radius_Blade_Edge"]
            gap_blade.append(min_gap)

        # edge-2-edge 최소값에서 에지 반경 2배 보정(Plot 2와 동일)
        gap_edge_val = min(gaps_etoe) - (p["Radius_Blade_Edge"] * 2)

        # 반환 순서: overlap, collision, edge
        return float(gap_blade[0]), float(gap_blade[1]), float(gap_edge_val)
    
    def _compute_roundness_curve_for_current_params(self):
        import math
        import numpy as np
        import pandas as pd
        try:
            p = self._params()
            p23_ang = np.linspace(p["Angle_Blade_Point_1"], p["Angle_Blade_Point_2"], 30)
            p45_ang = np.linspace(p["Angle_Blade_Point_2"], p["Angle_Blade_Point_2"] + p["Angle_Blade_Point_3"], 30)
            p23 = [(p["Diameter_Aperture_Min"] / 2 * np.cos(np.deg2rad(angle)),
                    p["Diameter_Aperture_Min"] / 2 * np.sin(np.deg2rad(angle))) for angle in p23_ang]
            c_stator = rotMat3(p["Pos_Stator_Boss"], [0, 0], p["Angle_Blade_Rotation"])
            p45 = [(c_stator[0] + p["Diameter_Aperture_Max"] / 2 * np.cos(np.deg2rad(angle)),
                    c_stator[1] + p["Diameter_Aperture_Max"] / 2 * np.sin(np.deg2rad(angle)))
                for angle in p45_ang]
            points = [tuple(p["p1"])] + p23 + p45

            xs, ys, ks = [], [], []
            for k in np.linspace(0, 10, 30):
                blade_points = []
                for x, y in points:
                    xs0, ys0 = x - p["Pos_Stator_Boss"][0], y - p["Pos_Stator_Boss"][1]
                    ang = -p["Angle_Blade_Rotation"] / 10 * k
                    xr = xs0 * np.cos(np.deg2rad(ang)) - ys0 * np.sin(np.deg2rad(ang))
                    yr = xs0 * np.sin(np.deg2rad(ang)) + ys0 * np.cos(np.deg2rad(ang))
                    blade_points.append((xr + p["Pos_Stator_Boss"][0], yr + p["Pos_Stator_Boss"][1]))
                x_blade, y_blade = zip(*blade_points)

                rot_points = []
                rot_deg = 360 / p["Blade_N"]
                for x, y in blade_points:
                    xr = x * np.cos(np.deg2rad(rot_deg)) - y * np.sin(np.deg2rad(rot_deg))
                    yr = x * np.sin(np.deg2rad(rot_deg)) + y * np.cos(np.deg2rad(rot_deg))
                    rot_points.append((xr, yr))
                x_rot, y_rot = zip(*rot_points)

                inter_pts = []
                ci = cj = 0
                for i in range(len(x_blade) - 1):
                    for j in range(len(x_rot) - 1):
                        inter = line_intersection(
                            [x_blade[i], x_blade[i + 1]],
                            [y_blade[i], y_blade[i + 1]],
                            [x_rot[j],   x_rot[j + 1]],
                            [y_rot[j],   y_rot[j + 1]],
                        )
                        if inter:
                            inter_pts.append(inter)
                            ci = i + 1; cj = j + 1

                if not inter_pts:
                    continue

                rotated_inter = rotMat3([0, 0], [inter_pts[0][0], inter_pts[0][1]], -rot_deg)
                cx = (inter_pts[0][0],) + x_blade[ci:cj] + (rotated_inter[0],)
                cy = (inter_pts[0][1],) + y_blade[ci:cj] + (rotated_inter[1],)
                circle_points = list(zip(cx, cy))

                dists = []
                for ii in range(len(cx) - 1):
                    x1, y1 = cx[ii], cy[ii]
                    x2, y2 = cx[ii + 1], cy[ii + 1]
                    dists.append(point_to_segment_distance(x1, y1, x2, y2, 0.0, 0.0))
                if not dists:
                    continue
                r_in = min(dists)
                r_in = min(r_in, self.m / 100 / 2)

                r_out = max(math.hypot(xp, yp) for xp, yp in circle_points)
                r_out = min(r_out, self.m / 100 / 2)
                if r_in <= 0 or r_out <= 0:
                    continue

                xs.append(round(r_in * 2, 2))
                ys.append(round(r_in / r_out, 4))
                ks.append(round(k, 3))

            df = pd.DataFrame({"k": ks, "apertrue diameter": xs, "roundness": ys})
            df.sort_values(["apertrue diameter", "k"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except Exception as ex:
            print(f"[curve] 계산 실패(탭 로직 동기화): {ex}")
            import pandas as pd
            return pd.DataFrame(columns=["k", "apertrue diameter", "roundness"])

    def button_clicked(self, button_name):
        if button_name == 'Button 1':
            image_path = "D:/04_Actuator/08_IRIS_ACE/00_Python/01_Blade/Blade_Setting.PNG"
            try:
                img = Image.open(image_path); img.show()
            except Exception as ex:
                print(f"이미지 열기 실패: {ex}")
                
        if button_name == 'Button 2':
            try:
                import pandas as pd
                now = datetime.datetime.now()
                default_name = f"roundness_{now.strftime('%y%m%d%H%M')}.xlsx"
                save_path, _ = QFileDialog.getSaveFileName(self, "엑셀 저장", default_name, "Excel Files (*.xlsx)")
                if not save_path:
                    print("저장이 취소되었습니다.")
                    return

                # 백업
                backup_e, backup_f, backup_g = self.e, self.f, self.g

                # roundness_all 생성(기존 로직)
                ext_start, ext_end, ext_step = self.range_extlen
                all_rows = []
                for angle2, angle1 in self._iter_angle_pairs_filtered():
                    v = ext_start
                    while v <= ext_end + 1e-9:
                        self.e = int(round(angle2 * 10))
                        self.f = int(round(angle1 * 10))
                        self.g = int(round(v * 100))

                        df_curve = self._compute_roundness_curve_for_current_params()
                        if not df_curve.empty:
                            df_curve.insert(0, "extlen", round(v, 2))
                            df_curve.insert(0, "angle1", angle1)
                            df_curve.insert(0, "angle2", angle2)
                            all_rows.append(df_curve)

                        v = round(v + ext_step, 10)

                if all_rows:
                    df_all = pd.concat(all_rows, ignore_index=True)
                    df_all.sort_values(["angle2", "angle1", "extlen", "apertrue diameter", "k"], inplace=True)
                    df_all.reset_index(drop=True, inplace=True)
                else:
                    df_all = pd.DataFrame(columns=["angle2", "angle1", "extlen", "k", "apertrue diameter", "roundness"])

                # angle_pairs 생성:
                # - extlen 스윕 중 “처음” 조건( gap_overlap>0.1 and gap_collision>0.3 )을 만족한 순간 이후 extlen 스킵
                # - angle_pairs에도 k, aperture diameter, roundness 포함
                angle_rows = []
                for angle2, angle1 in self._iter_angle_pairs_filtered():
                    satisfied_once = False  # (angle2,angle1)에서 한 번이라도 만족했는가?

                    v_float = ext_start
                    while v_float <= ext_end + 1e-9:
                        if satisfied_once:
                            break

                        self.e = int(round(angle2 * 10))
                        self.f = int(round(angle1 * 10))
                        self.g = int(round(v_float * 100))

                        gap_overlap, gap_collision, gap_edge = self._compute_gaps_for_current_params()
                        cond_ok = (gap_overlap > 0.1) and (gap_collision > 0.3)
                        if cond_ok:
                            df_curve_pair = self._compute_roundness_curve_for_current_params()
                            if df_curve_pair is not None and not df_curve_pair.empty:
                                df_curve_pair.insert(0, "extlen", round(v_float, 2))
                                df_curve_pair.insert(0, "angle1", angle1)
                                df_curve_pair.insert(0, "angle2", angle2)
                                df_curve_pair["gap_overlap"] = round(gap_overlap, 6)
                                df_curve_pair["gap_collision"] = round(gap_collision, 6)
                                df_curve_pair["gap_edge"] = round(gap_edge, 6)
                                angle_rows.append(df_curve_pair)
                            else:
                                angle_rows.append(pd.DataFrame([{
                                    "angle2": angle2, "angle1": angle1, "extlen": round(v_float, 2),
                                    "k": None, "apertrue diameter": None, "roundness": None,
                                    "gap_overlap": round(gap_overlap, 6),
                                    "gap_collision": round(gap_collision, 6),
                                    "gap_edge": round(gap_edge, 6),
                                }]))
                            satisfied_once = True

                        v_float = round(v_float + ext_step, 10)

                if angle_rows:
                    df_pairs = pd.concat(angle_rows, ignore_index=True)
                    df_pairs.sort_values(["angle2", "angle1", "extlen", "apertrue diameter", "k"],
                                        inplace=True, na_position="last")
                    df_pairs.reset_index(drop=True, inplace=True)
                else:
                    df_pairs = pd.DataFrame(columns=[
                        "angle2", "angle1", "extlen", "k", "apertrue diameter", "roundness",
                        "gap_overlap", "gap_collision", "gap_edge"
                    ])

                # roundness_filtered 생성:
                # angle2, angle1 쌍별로 roundness_all에서 "모든 extlen, 모든 k의 roundness가 0.95 이상"인 쌍만 선별
                # 1) 쌍 목록
                unique_pairs = df_all[['angle2', 'angle1']].drop_duplicates().values.tolist()
                keep_pairs = []
                for a2, a1 in unique_pairs:
                    sub = df_all[(df_all['angle2'] == a2) & (df_all['angle1'] == a1)]
                    # 모든 roundness >= 0.97 ?
                    if not sub.empty and (sub['roundness'].min() >= 0.97):
                        keep_pairs.append((a2, a1))
                # 2) angle_pairs에서 해당 (a2,a1)만 필터
                if keep_pairs and not df_pairs.empty:
                    key_df = pd.DataFrame(keep_pairs, columns=['angle2', 'angle1'])
                    df_pairs_filtered = df_pairs.merge(key_df, on=['angle2', 'angle1'], how='inner')
                    # 보기 좋게 정렬
                    df_pairs_filtered.sort_values(["angle2", "angle1", "extlen", "apertrue diameter", "k"],
                                                inplace=True, na_position="last")
                    df_pairs_filtered.reset_index(drop=True, inplace=True)
                else:
                    df_pairs_filtered = pd.DataFrame(columns=df_pairs.columns)

                # 엑셀 저장 + 차트 (roundness_all, angle_pairs, roundness_filtered)
                with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
                    # roundness_all
                    df_all.to_excel(writer, sheet_name="roundness_all", index=False)

                    # angle_pairs (원본)
                    df_pairs.to_excel(writer, sheet_name="angle_pairs", index=False)

                    # roundness_filtered (필터 적용본)
                    df_pairs_filtered.to_excel(writer, sheet_name="roundness_filtered", index=False)

                    wb = writer.book

                    # 공통 함수: 시트에 차트 추가
                    def add_scatter_chart(sheet_name, df_src, top_left_cell='H2'):
                        if df_src.empty:
                            return
                        ws = writer.sheets[sheet_name]
                        chart = wb.add_chart({'type': 'scatter', 'subtype': 'straight'})
                        chart.set_title({'name': f'{sheet_name} Curves (k sweep)'})
                        chart.set_x_axis({'name': 'inner diameter'})
                        chart.set_y_axis({'name': 'roundness', 'max': 1.0, 'min': 0.9})
                        chart.set_legend({'none': True})

                        # df_src 컬럼 인덱스 결정
                        # 기대 컬럼: angle2(0), angle1(1), extlen(2), k(3), apertrue diameter(4), roundness(5), ...
                        cols = list(df_src.columns)
                        try:
                            x_idx = cols.index('apertrue diameter')
                            y_idx = cols.index('roundness')
                            a2_idx = cols.index('angle2')
                            a1_idx = cols.index('angle1')
                            el_idx = cols.index('extlen')
                        except ValueError:
                            return  # 필수 컬럼 없으면 차트 생략

                        start_row = 1
                        combos = df_src[['angle2', 'angle1', 'extlen']].drop_duplicates().values.tolist()
                        for (a2v, a1v, elv) in combos:
                            m = (df_src["angle2"] == a2v) & (df_src["angle1"] == a1v) & (df_src["extlen"] == elv)
                            s = df_src[m].dropna(subset=['apertrue diameter', 'roundness'])
                            if s.empty:
                                continue
                            idxs = s.index.to_list()
                            first_idx, last_idx = idxs[0], idxs[-1]
                            chart.add_series({
                                'name': f"A2={int(a2v)}, A1={int(a1v)}, EL={float(elv):.2f}",
                                'categories': [sheet_name, start_row + first_idx, x_idx, start_row + last_idx, x_idx],
                                'values':     [sheet_name, start_row + first_idx, y_idx, start_row + last_idx, y_idx],
                                'marker': {'type': 'circle', 'size': 3},
                                'line':   {'width': 1.0},
                            })
                        ws.insert_chart(top_left_cell, chart, {'x_scale': 1.3, 'y_scale': 1.1})

                    # 차트 추가
                    add_scatter_chart("roundness_all", df_all, 'H2')
                    add_scatter_chart("angle_pairs", df_pairs, 'H20')
                    add_scatter_chart("roundness_filtered", df_pairs_filtered, 'H2')

                # 상태 복구
                self.e, self.f, self.g = backup_e, backup_f, backup_g
                self.slider_5.setValue(self.e); self.slider_6.setValue(self.f); self.slider_7.setValue(self.g)
                self.label_5.setText(f"Blade Angle2 : {self.e/10:.1f}")
                self.label_6.setText(f"Blade Angle1 : {self.f/10:.1f}")
                self.label_7.setText(f"extention length : {self.g/100:.2f}")

                print(f"저장 완료: {save_path} (roundness_all + angle_pairs + roundness_filtered + charts)")
            except Exception as ex:
                print(f"엑셀 저장 실패: {ex}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DesignWidget()
    window.setWindowTitle('Variable_Aperture')
    window.setGeometry(100, 100, 1600, 800)
    window.show()
    sys.exit(app.exec())
