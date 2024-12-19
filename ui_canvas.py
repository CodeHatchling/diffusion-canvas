from PyQt6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSizePolicy
)

from PyQt6.QtGui import QPixmap, QImage, QMouseEvent
from PyQt6.QtCore import Qt, QPoint
from typing import Callable

from ui_utils import ExceptionCatcher


class Canvas(QScrollArea):
    def __init__(self,
                 on_mouse_press: Callable[[QMouseEvent], None],
                 on_mouse_move: Callable[[QMouseEvent], None],
                 on_mouse_release: Callable[[QMouseEvent], None],
                 parent: QWidget | None = None):
        super().__init__(parent)

        self.on_mouse_press = on_mouse_press
        self.on_mouse_move = on_mouse_move
        self.on_mouse_release = on_mouse_release

        self.canvas_widget = QWidget(self)

        canvas_layout = QVBoxLayout(self.canvas_widget)
        canvas_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_widget.setLayout(canvas_layout)

        self.label = QLabel(self.canvas_widget)
        canvas_layout.addWidget(self.label)

        self.setWidget(self.canvas_widget)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(200)
        self.setMinimumHeight(200)

    def update_image(self, image: QImage):
        # Update the label pixmap
        pixmap = QPixmap.fromImage(image)

        self.label.setPixmap(pixmap)
        self.label.setFixedWidth(pixmap.width())
        self.label.setFixedHeight(pixmap.height())
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMargin(0)

        self.canvas_widget.setMinimumWidth(pixmap.width())
        self.canvas_widget.setMinimumHeight(pixmap.height())

        size_policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.canvas_widget.setSizePolicy(size_policy)

    def coord_local_to_normalized(self, local_point: QPoint):
        pixmap: QPixmap = self.label.pixmap()

        if pixmap is None:
            return 0.5, 0.5

        # Convert the point from our local space to the label's local space.
        label_local_point = self.label.mapFromGlobal(self.mapToGlobal(local_point))

        # The image is centered, so its top-left (0,0) will be
        # half the margin outward from the self's top-left (0,0).
        image_pos = (
                (self.label.width() - pixmap.width()) / 2,
                (self.label.height() - pixmap.height()) / 2
        )

        return (
            (label_local_point.x() - image_pos[0]) / pixmap.width(),
            (label_local_point.y() - image_pos[1]) / pixmap.height(),
        )

    def mousePressEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            self.on_mouse_press(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            self.on_mouse_move(event)

    def mouseReleaseEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            self.on_mouse_release(event)