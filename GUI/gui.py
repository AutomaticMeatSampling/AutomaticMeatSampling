import sys
import time
from ImageWorker import ImageWorker
from RobotWorker import RobotWorker
from threading import Event
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QSpinBox, QComboBox
)
from CoordSelectionWidget import CoordSelectionWidget

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.stop_ongoing = False
        
        print("Creating Main Window...")

        self.setWindowTitle("Robot Controller")

        # Status Label
        self.status = QLabel("Idle")

        # -----------------------------
        # CONFIG PANEL
        # -----------------------------
        print("Creating config panel....")
        config_box = QGroupBox("Configuration Settings")
        config_layout = QVBoxLayout()

        # Number of muscle pts
        self.num_muscle_pts = QSpinBox()
        self.num_muscle_pts.setRange(0, ImageWorker.max_num_muscle)
        self.num_muscle_pts.setValue(0)
        config_layout.addWidget(QLabel("Num Muscle Sampling Points:"))
        config_layout.addWidget(self.num_muscle_pts)

        # Number of marbling pts
        self.num_marbling_pts = QSpinBox()
        self.num_marbling_pts.setRange(0, ImageWorker.max_num_marbling)
        self.num_marbling_pts.setValue(1)
        config_layout.addWidget(QLabel("Num Marbling Sampling Points:"))
        config_layout.addWidget(self.num_marbling_pts)

        # Sampling Mode (Automatic/Manual)
        self.sample_mode = QComboBox()
        self.sample_mode.addItems(["Automatic", "Manual"])
        config_layout.addWidget(QLabel("Sampling Mode:"))
        config_layout.addWidget(self.sample_mode)

        config_box.setLayout(config_layout)


        # -----------------------------
        # CONTROL BUTTONS
        # -----------------------------
        print("Creating control buttons...")
        # Buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.reset_btn = QPushButton("Reset")

        self.start_btn.clicked.connect(self.start_robot)
        self.stop_btn.clicked.connect(self.stop_robot)
        self.reset_btn.clicked.connect(self.reset_gui)

        # Button Layouts
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.reset_btn)

        # -----------------------------
        # MAIN LAYOUT
        # -----------------------------
        layout = QVBoxLayout()
        layout.addWidget(self.status)
        layout.addWidget(config_box)
        layout.addLayout(btns)
        self.setLayout(layout)

        # Setup workers with default config values
        self.image_coords_ready_event = Event()
        # self.imageWorker = ImageWorker(self.num_muscle_pts.value(), self.num_marbling_pts.value())
    

    def start_robot(self):
        if self.stop_ongoing:
            return
        
        if hasattr(self, "robotWorker") and self.robotWorker.isRunning():
            self.status.setText("Robot already running.")
            return
        
        if self.num_muscle_pts.value() + self.num_marbling_pts.value() == 0:
            self.status.setText("Error: Total Number of sampling pts is 0")
            return
        
        # Create new RobotWorker each time Start is pressed
        self.robotWorker = RobotWorker(self.sample_mode.currentText(), main_window=self)

        # Connect singals
        self.robotWorker.progress.connect(self.on_progress)

        # Start the worker
        self.robotWorker.start()
        # self.status.setText("Starting sample collection...")

    def stop_robot(self):
        self.stop_ongoing = True
        if hasattr(self, "robotWorker"):
            self.robotWorker.stop()

        if hasattr(self, "imageWorker"):
            if self.imageWorker.isRunning():
                # Don't set stop_ongoing = False
                return
                
        self.stop_ongoing = False
    def reset_gui(self):
        self.status.setText("Idle")

    def on_progress(self, msg):
        # Check msg and handle properly
        if msg == "ROBOT_START":
            self.status.setText("Starting sample collection...")
        elif msg == "ROBOT_STOP_SUCCESS":
            self.status.setText("Sucessfully completed process.")
            time.sleep(2)
            # self.reset_gui()
        elif msg == "ROBOT_STOP":
            self.status.setText("Stopped robot process.")
            time.sleep(2)
            # self.reset_gui()
        elif msg == "START_AUTOMATIC_POINT_SELECTION":
            # Create a new instance of ImageWorker
            # TODO: how does threading work here and for stopping??????????????????
            self.imageWorker = ImageWorker(self.num_muscle_pts.value(), self.num_marbling_pts.value())
            self.imageWorker.result_ready.connect(self.on_coord_result)
            self.imageWorker.start()
        elif msg == "START_MANUAL_POINT_SELECTION":
            self.coord_selector = CoordSelectionWidget()
            self.coord_selector.points_selected.connect(self.on_coord_result)
            self.coord_selector.show()
        else:
            self.status.setText(msg)
        pass

    def on_coord_result(self, results):
        # Send results to robotWorker
        if self.robotWorker.isRunning(): #Check in case "stop" has been pressed
            self.selected_points = results
            self.image_coords_ready_event.set()
        elif self.stop_ongoing:
            self.stop_ongoing = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())