from typing import Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QFrame, QSizePolicy, QTabWidget
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

        self.latent_value: tuple[float, ...] = (0, 0, 0, 0)

        self.setFixedWidth(40)
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.setLayout(layout)

        self._label = QLabel()
        self._label.setScaledContents(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._label)

    def set_image(self, pixmap: QPixmap | None):
        if pixmap is None:  # It should accept none, but it doesn't?
            self._label.clear()
        else:
            self._label.setPixmap(pixmap)


class ColorPickerSwatches(QFrame):

    def __init__(self,
                 set_color_func: Callable[[tuple[float, ...]], None],
                 grid_size: tuple[int, int] = (6, 5), parent=None):
        super().__init__(parent)

        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setLineWidth(3)
        self.setContentsMargins(0, 0, 0, 0)
        self._grid_size = grid_size
        self._page_size = grid_size[0] * grid_size[1]
        self._current_page = 0
        self._set_color_func = set_color_func

        self.swatches_layout = QGridLayout(self)
        self.swatches_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        self.swatches_layout.setSpacing(0)

        self._swatch_data: dict[tuple[float, ...], QPixmap] = {}
        self._latent_values_by_index: list[tuple[float, ...]] = []

        self._swatch_buttons: list[ColorPickerSwatch] = []

        for i in range(self._page_size):
            widget = ColorPickerSwatch()
            widget.clicked.connect(lambda _, swatch_button_index=i: self.on_swatch_clicked(swatch_button_index))
            widget.setEnabled(False)
            self._swatch_buttons.append(widget)
            column = i % self._grid_size[0]
            row = i // self._grid_size[0]
            self.swatches_layout.addWidget(
                widget,
                row, column,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            )

        # Add the page change buttons.
        def change_page(increment: int):
            num_values = len(self._latent_values_by_index)
            page_count = num_values // self._page_size

            if num_values % self._page_size > 0:
                page_count += 1
            if page_count < 1:
                page_count = 1
            max_page_index = page_count-1

            new_page = self._current_page
            new_page += increment
            if new_page < 0:
                new_page = 0
            elif new_page > max_page_index:
                new_page = max_page_index

            if self._current_page != new_page:
                self._current_page = new_page
                self.update_palette()

        left_button = QPushButton(parent=self)
        left_button.setText("⬅️")
        left_button.clicked.connect(lambda: change_page(-1))
        left_button.setFixedWidth(40)
        left_button.setFixedHeight(40)

        right_button = QPushButton(parent=self)
        right_button.setText("➡️")
        right_button.clicked.connect(lambda: change_page(1))
        right_button.setFixedWidth(40)
        right_button.setFixedHeight(40)

        self.swatches_layout.addWidget(
            left_button,
            grid_size[1], 0,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        self.swatches_layout.addWidget(
            right_button,
            grid_size[1], grid_size[0]-1,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

    def _button_index_to_value_index(self, button_index: int) -> int:
        return button_index + self._current_page * self._page_size

    def _value_index_in_range(self, value_index: int) -> bool:
        return 0 <= value_index < len(self._latent_values_by_index)

    def on_color_used(self, value: tuple[float, ...], pixmap: QPixmap) -> None:
        # Move or add the most recently used swatches at the front of the list.
        if value in self._swatch_data:
            self._latent_values_by_index.remove(value)
            self._latent_values_by_index.insert(0, value)
        else:
            self._latent_values_by_index.insert(0, value)
            self._swatch_data[value] = pixmap

        self.update_palette()

    def on_swatch_clicked(self, button_index: int):
        value_index = self._button_index_to_value_index(button_index)
        if not self._value_index_in_range(value_index):
            return

        self._set_color_func(self._latent_values_by_index[value_index])

    def update_palette(self):
        # Update the swatch preview images.
        for i in range(self._page_size):
            value_index = self._button_index_to_value_index(i)
            button = self._swatch_buttons[i]
            if self._value_index_in_range(value_index):
                button.setEnabled(True)
                value = self._latent_values_by_index[value_index]
                pixmap = self._swatch_data[value]
                button.set_image(pixmap)
            else:
                button.setEnabled(False)
                button.set_image(None)


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
        self.swatch_widget = ColorPickerSwatches(
            set_color_func=self.set_current_latent_value,
            grid_size=(6, 4),
            parent=self
        )

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
        slider_preview_layout.setSpacing(10)
        slider_preview_layout.setContentsMargins(0, 0, 0, 0)

        overall_layout.addLayout(slider_preview_layout)
        overall_layout.addWidget(self.swatch_widget)
        overall_layout.setSpacing(10)
        # overall_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(overall_layout)

        # Assign the picture initially.
        self._on_slider_changed()

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
        self.swatch_widget.on_color_used(latent_value, self._color_preview.pixmap())
        return latent_value

    def set_current_latent_value(self, value: tuple[float, ...]) -> None:
        for i in range(len(self._sliders)):
            self._sliders[i].value = value[i]

        self.update_preview(value)


class LatentPicker(QTabWidget):
    def __init__(self,
                 generate_preview_func: Callable[[tuple[float, float, float, float]], QPixmap],
                 parent=None):
        super().__init__(parent)

        self.color_picker = ColorPickerWidget(generate_preview_func, parent=self)
        self.addTab(self.color_picker, "Solid Latent")
