from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import pyqtSignal, Qt, QPoint
import cv2
from ImageWorker import ImageWorker

class CoordSelectionWidget(QWidget):
    points_selected = pyqtSignal(dict)

    def __init__(self, max_muscle=5, max_marbling=5):
        super().__init__()
        self.setWindowTitle("Select Points")
        self.img = cv2.imread(ImageWorker.img_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        max_display_width = 800
        max_display_height = 600


        h, w, c = self.img.shape

        scale_w = max_display_width / w
        scale_h = max_display_height / h
        self.scale = min(scale_w, scale_h, 1.0)

        self.display_img = cv2.resize(self.img, (int(w*self.scale), int(h*self.scale)), interpolation=cv2.INTER_AREA)
        dh, dw, _ = self.display_img.shape

        self.qimg = QImage(self.display_img.data, dw, dh, 3*dw, QImage.Format_RGB888)

        # Max points
        self.max_muscle = max_muscle
        self.max_marbling = max_marbling

        # Track points
        self.muscle_pts = []
        self.marbling_pts = []

        # Currently selecting muscle or marbling
        self.current_type = 'marbling'

        # Widgets
        self.label = QLabel()
        self.label.setPixmap(QPixmap.fromImage(self.qimg))
        self.label.mousePressEvent = self.image_mouse_press
        self.label.setFocusPolicy(Qt.StrongFocus)

        # Buttons
        self.continue_btn = QPushButton("Continue")
        self.undo_btn = QPushButton("Undo")
        self.type_btn = QPushButton("Switch to Muscle")

        self.continue_btn.clicked.connect(self.on_continue)
        self.undo_btn.clicked.connect(self.on_undo)
        self.type_btn.clicked.connect(self.on_switch_type)

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.undo_btn)
        btn_layout.addWidget(self.type_btn)
        btn_layout.addWidget(self.continue_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.update_image()

    def update_image(self):
        # Draw points on image
        pix = QPixmap.fromImage(self.qimg.copy())
        painter = QPainter(pix)
        painter.setPen(QColor("red"))
        for pt in self.muscle_pts:
            painter.drawEllipse(QPoint(pt[0], pt[1]), 5, 5)
        painter.setPen(QColor("green"))
        for pt in self.marbling_pts:
            painter.drawEllipse(QPoint(pt[0], pt[1]), 5, 5)
        painter.end()
        self.label.setPixmap(pix)

    def image_mouse_press(self, event):
        if event.button() != Qt.LeftButton: # Only accept left-clicks
            return

        if not self.label.hasFocus(): # Must be focused
            return

        # Coordinates relative to the label (correct!)
        x, y = event.pos().x(), event.pos().y()

        # Reject clicks outside displayed image area
        if x < 0 or y < 0 or x >= self.display_img.shape[1] or y >= self.display_img.shape[0]:
            return

        # Store point
        if self.current_type == 'muscle':
            if len(self.muscle_pts) < self.max_muscle:
                self.muscle_pts.append((x, y))
                self.last_type = "muscle"
        else:
            if len(self.marbling_pts) < self.max_marbling:
                self.marbling_pts.append((x, y))
                self.last_type = "marbling"

        # Update drawing
        self.update_image()

    def on_undo(self):
        if hasattr(self, "last_type"):
            if self.last_type == 'muscle' and self.muscle_pts:
                self.muscle_pts.pop()
            elif self.last_type == 'marbling' and self.marbling_pts:
                self.marbling_pts.pop()
            self.update_image()

    def on_switch_type(self):
        if self.current_type == 'muscle':
            self.current_type = 'marbling'
            self.type_btn.setText("Switch to Muscle")
        else:
            self.current_type = 'muscle'
            self.type_btn.setText("Switch to Marbling")

    def on_continue(self):
        # Convert points back to original image coordinates
        muscle_points_full = [(int(x / self.scale), int(y / self.scale)) for x, y in self.muscle_pts]
        marbling_points_full = [(int(x / self.scale), int(y / self.scale)) for x, y in self.marbling_pts]

        # Emit the scaled-up points
        self.points_selected.emit({
            'muscle_points': muscle_points_full,
            'marbling_points': marbling_points_full
        })
        self.close()




        