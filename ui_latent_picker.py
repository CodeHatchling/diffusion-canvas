from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton
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


class ColorPickerSwatch(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)


class ColorPickerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

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
        for _ in range(latent_channel_count):
            slider = Slider(parent=self,
                            min_max=(-5.0, 5.0),
                            step_size=0.25,
                            default_value=0.0,
                            label="")
            self._sliders.append(slider)

        # Then, add them to their respective containers.
        for slider in self._sliders:
            sliders_layout.addWidget(slider)

        slider_preview_layout.addWidget(self._color_preview)
        slider_preview_layout.addLayout(sliders_layout)

        overall_layout.addLayout(slider_preview_layout)
        overall_layout.addLayout(swatches_layout)

        self.setLayout(overall_layout)

    def get_current_latent_value(self, update_swatches: bool) -> tuple[float, ...]:
        latent_value: list[float] = []
        for slider in self._sliders:
            latent_value.append(slider.value)

        # TODO: Add latent value to swatches if unique,
        #       otherwise move pre-existing swatch to front of collection.

        return tuple(latent_value)

    def set_current_latent_value(self, value: tuple[float, ...]) -> None:
        for i in range(len(self._sliders)):
            self._sliders[i].value = value[i]


class LatentPicker(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        self.color_picker = ColorPickerWidget(parent=self)
        layout.addWidget(self.color_picker)
        self.setLayout(layout)

