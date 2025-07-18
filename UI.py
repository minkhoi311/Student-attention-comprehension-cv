# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'focus_analysis.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QConicalGradient, QFont
from PyQt5.QtCore import Qt


class CircularGauge(QWidget):
    def __init__(self, parent=None, title="", min_val=0, max_val=100):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.value = min_val
        self.setMinimumSize(200, 250)

    def setValue(self, value):
        self.value = max(self.min_val, min(self.max_val, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(240, 240, 240))
        painter.drawRoundedRect(self.rect(), 10, 10)

        # Calculate dimensions
        size = min(self.width(), self.height() - 50)
        x = (self.width() - size) // 2
        y = 10

        # Draw outer circle
        painter.setPen(QColor(100, 100, 100))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(x, y, size, size)

        # Draw gauge
        angle = 180 + int(180 * (self.value - self.min_val) / (self.max_val - self.min_val))
        gradient = QConicalGradient(x + size // 2, y + size // 2, 180)

        # Color based on value
        if self.value < 40:
            gradient.setColorAt(0, QColor(220, 53, 69))
            gradient.setColorAt(1, QColor(220, 53, 69))
        elif self.value < 70:
            gradient.setColorAt(0, QColor(255, 193, 7))
            gradient.setColorAt(1, QColor(255, 193, 7))
        else:
            gradient.setColorAt(0, QColor(40, 167, 69))
            gradient.setColorAt(1, QColor(40, 167, 69))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawPie(x, y, size, size, 180 * 16, (angle - 180) * 16)

        # Draw center circle
        painter.setBrush(QColor(240, 240, 240))
        painter.drawEllipse(x + size // 4, y + size // 4, size // 2, size // 2)

        # Draw value text
        painter.setPen(QColor(0, 0, 0))
        font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.value:.0f}%\n{self.title}")

        # Draw min/max values
        font = QFont("Arial", 8)
        painter.setFont(font)
        painter.drawText(x, y + size + 10, f"{self.min_val}")
        painter.drawText(x + size - 15, y + size + 10, f"{self.max_val}")


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 681)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.main_layout.setObjectName("main_layout")
        self.left_panel = QtWidgets.QVBoxLayout()
        self.left_panel.setObjectName("left_panel")
        self.camera_label = QtWidgets.QLabel(self.centralwidget)
        self.camera_label.setMinimumSize(QtCore.QSize(640, 480))
        self.camera_label.setStyleSheet("border: 2px solid #ccc; background-color: #eee;")
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setObjectName("camera_label")
        self.left_panel.addWidget(self.camera_label)
        self.stats_layout = QtWidgets.QHBoxLayout()
        self.stats_layout.setObjectName("stats_layout")
        self.focus_gauge = CircularGauge(self.centralwidget)
        self.focus_gauge.setObjectName("focus_gauge")
        self.stats_layout.addWidget(self.focus_gauge)
        self.understanding_gauge = CircularGauge(self.centralwidget)
        self.understanding_gauge.setObjectName("understanding_gauge")
        self.stats_layout.addWidget(self.understanding_gauge)
        self.emotion_group = QtWidgets.QGroupBox(self.centralwidget)
        self.emotion_group.setObjectName("emotion_group")
        self.emotion_layout = QtWidgets.QVBoxLayout(self.emotion_group)
        self.emotion_layout.setObjectName("emotion_layout")
        self.emotion_label = QtWidgets.QLabel(self.emotion_group)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.emotion_label.setFont(font)
        self.emotion_label.setAlignment(QtCore.Qt.AlignCenter)
        self.emotion_label.setObjectName("emotion_label")
        self.emotion_layout.addWidget(self.emotion_label)
        self.confidence_label = QtWidgets.QLabel(self.emotion_group)
        self.confidence_label.setAlignment(QtCore.Qt.AlignCenter)
        self.confidence_label.setObjectName("confidence_label")
        self.emotion_layout.addWidget(self.confidence_label)
        self.stats_layout.addWidget(self.emotion_group)
        self.left_panel.addLayout(self.stats_layout)
        self.main_layout.addLayout(self.left_panel)
        self.right_panel = QtWidgets.QVBoxLayout()
        self.right_panel.setObjectName("right_panel")
        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_layout.setObjectName("control_layout")
        self.start_btn = QtWidgets.QPushButton(self.centralwidget)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.setObjectName("start_btn")
        self.control_layout.addWidget(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton(self.centralwidget)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.setObjectName("stop_btn")
        self.control_layout.addWidget(self.stop_btn)
        self.report_btn = QtWidgets.QPushButton(self.centralwidget)
        self.report_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.report_btn.setObjectName("report_btn")
        self.control_layout.addWidget(self.report_btn)
        self.right_panel.addLayout(self.control_layout)
        self.tab_widget = QtWidgets.QTabWidget(self.centralwidget)
        self.tab_widget.setObjectName("tab_widget")
        self.focus_tab = QtWidgets.QWidget()
        self.focus_tab.setObjectName("focus_tab")

        # Create layout for focus tab
        self.focus_tab_layout = QtWidgets.QVBoxLayout(self.focus_tab)

        # Add time label
        self.time_label = QtWidgets.QLabel(self.focus_tab)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.time_label.setFont(font)
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.time_label.setObjectName("time_label")
        self.focus_tab_layout.addWidget(self.time_label)

        # Add warning label
        self.warning_label = QtWidgets.QLabel(self.focus_tab)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.warning_label.setFont(font)
        self.warning_label.setStyleSheet("color: #f44336;")
        self.warning_label.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_label.setObjectName("warning_label")
        self.focus_tab_layout.addWidget(self.warning_label)

        # Add status label
        self.status_label = QtWidgets.QLabel(self.focus_tab)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.status_label.setFont(font)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setObjectName("status_label")
        self.status_label.setText("Sẵn sàng bắt đầu theo dõi")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.focus_tab_layout.addWidget(self.status_label)

        # Add a spacer to push content to top
        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.focus_tab_layout.addItem(spacer)

        self.tab_widget.addTab(self.focus_tab, "")
        self.right_panel.addWidget(self.tab_widget)
        self.main_layout.addLayout(self.right_panel)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tab_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hệ thống theo dõi tập trung học tập"))
        self.focus_gauge.setProperty("title", _translate("MainWindow", "Tập trung"))
        self.understanding_gauge.setProperty("title", _translate("MainWindow", "Hiểu bài"))
        self.emotion_group.setTitle(_translate("MainWindow", "Trạng thái cảm xúc"))
        self.emotion_label.setText(_translate("MainWindow", "Đang nhận diện..."))
        self.confidence_label.setText(_translate("MainWindow", "Độ tin cậy: 0%"))
        self.start_btn.setText(_translate("MainWindow", "Bắt đầu theo dõi"))
        self.stop_btn.setText(_translate("MainWindow", "Dừng theo dõi"))
        self.report_btn.setText(_translate("MainWindow", "Xuất báo cáo"))
        self.time_label.setText(_translate("MainWindow", "Thời gian: 00:00"))
        self.warning_label.setText(_translate("MainWindow", ""))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.focus_tab),
                                   _translate("MainWindow", "Trạng thái"))