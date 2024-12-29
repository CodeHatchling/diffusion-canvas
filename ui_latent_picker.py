from typing import Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QFrame, QSizePolicy
)

from ui_widgets import Slider
from diffusion_canvas_api import latent_channel_count

"""
This script is intended to provide an interface for the user to control
what latent content is provided to blending brushes.

Acts as a replacement for the previous solid color latent.

Features:
    - Color picker and color history.
    - Image upload, with tiling and resizing options
    - History (including current canvas state)
    - Offset controls for image content.
"""


class ColorPickerLatentPreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Panel)

        size_policy = self.sizePolicy()
        size_policy.setHeightForWidth(True)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Maximum)
        size_policy.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.setSizePolicy(size_policy)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return w

    def sizeHint(self) -> QSize:
        return QSize(64, 64)

    def minimumSizeHint(self) -> QSize:
        return QSize(64, 64)


class ColorPickerSwatch(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return w


class ColorPickerWidget(QWidget):
    def __init__(self,
                 generate_preview_func: Callable[[tuple[float, float, float, float]], QPixmap],
                 parent=None):
        super().__init__(parent)

        self._generate_preview_func = generate_preview_func

        """
        Layout:
        - A box showing a decoded preview of a small image filled with the current latent value.
        - A column beside the box with one slider per each channel, range -5 to 5
        - A grid of buttons with a small downscaled copy of the preview image. When clicked,
          the latent values stored within the button are assigned to the sliders, with the preview updated.
        """

        # First create the objects we need.
        overall_layout = QVBoxLayout(self)

        slider_preview_layout = QHBoxLayout(self)
        swatches_layout = QGridLayout(self)

        self._color_preview = ColorPickerLatentPreview(self)
        sliders_layout = QVBoxLayout(self)

        self._sliders: list[Slider] = []
        for i in range(latent_channel_count):
            slider = Slider(parent=self,
                            min_max=(-5.0, 5.0),
                            step_size=0.25,
                            default_value=0.0,
                            label="")
            self._sliders.append(slider)
            slider.on_ui_value_changed.append(self._on_slider_changed)

        # Then, add them to their respective containers.
        for slider in self._sliders:
            sliders_layout.addWidget(slider)

        sliders_layout.setSpacing(0)
        sliders_layout.setContentsMargins(0, 0, 0, 0)

        slider_preview_layout.addWidget(self._color_preview)
        slider_preview_layout.addLayout(sliders_layout)
        slider_preview_layout.setSpacing(0)
        slider_preview_layout.setContentsMargins(0, 0, 0, 0)

        overall_layout.addLayout(slider_preview_layout)
        overall_layout.addLayout(swatches_layout)
        overall_layout.setSpacing(0)
        overall_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(overall_layout)

    def _on_slider_changed(self):
        self.update_preview(self._get_current_latent_value())

    def update_preview(self, value: tuple[float, ...]):
        assert len(value) > 0, f"Tuple \'value\' {value} must at least contain one element."
        max_index = len(value)-1
        value = (
            value[0],
            value[min(max_index, 1)],
            value[min(max_index, 2)],
            value[min(max_index, 3)]
        )

        pixmap = self._generate_preview_func(value)
        self._color_preview.setPixmap(pixmap)

    def _get_current_latent_value(self):
        latent_value: list[float] = []
        for slider in self._sliders:
            latent_value.append(slider.value)
        return tuple(latent_value)

    def use_current_latent_value(self) -> tuple[float, ...]:
        latent_value = self._get_current_latent_value()

        # TODO: Add latent value to swatches if unique,
        #       otherwise move pre-existing swatch to front of collection.

        return latent_value

    def set_current_latent_value(self, value: tuple[float, ...]) -> None:
        for i in range(len(self._sliders)):
            self._sliders[i].value = value[i]

        self.update_preview(value)


class LatentPicker(QWidget):
    def __init__(self,
                 generate_preview_func: Callable[[tuple[float, float, float, float]], QPixmap],
                 parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.color_picker = ColorPickerWidget(generate_preview_func, parent=self)
        layout.addWidget(self.color_picker)
        self.setLayout(layout)

