from typing import Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QFrame, QSizePolicy, QTabWidget
)

from ui_widgets import Slider, VerticalScrollArea, ImageButton, LabelImageButton
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


class ColorPickerSwatch(ImageButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.latent_value: tuple[float, ...] = (0, 0, 0, 0)
        self.setFixedWidth(40)
        self.setFixedHeight(40)


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


class HistoryPickerWidget(VerticalScrollArea):

    class HistoryItem:
        text: str | None
        image: QPixmap | None

        def __init__(self, text: str | None, image: QPixmap | None):
            self.text = text
            self.image = image

    class HistoryInfo:
        item_count: int

        def __init__(self, item_count: int):
            self.item_count = item_count

    def __init__(self,
                 get_history_item_func: Callable[[int], HistoryItem],
                 get_history_info_func: Callable[[], HistoryInfo],
                 parent=None):
        super().__init__(parent)

        self._get_history_item_func: Callable[[int], HistoryPickerWidget.HistoryItem] = get_history_item_func
        self._get_history_info_func: Callable[[], HistoryPickerWidget.HistoryInfo] = get_history_info_func

        contents_widget = QWidget()
        self._contents_layout = QVBoxLayout(contents_widget)
        contents_widget.setLayout(self._contents_layout)
        self.setWidget(contents_widget)

        self.buttons: list[LabelImageButton] = []

        self.selected_index: int = -1

    def on_history_changed(self):
        history_info = self._get_history_info_func()

        def append_button(_index: int) -> LabelImageButton:
            _button = LabelImageButton()
            _button.clicked.connect(lambda _, _i=_index: self._on_clicked_button(_i))
            self.buttons.append(_button)
            self._contents_layout.addWidget(_button)
            return _button

        def update_button(_button: LabelImageButton, _index: int) -> None:
            _info = self._get_history_item_func(_index)
            _button.setEnabled(True)
            _button.set_image(_info.image)
            _button.set_text(_info.text)
            _button.show()

        def hide_button(_index: int) -> None:
            _button = self.buttons[_index]
            _button.hide()
            _button.set_image(None)
            _button.setEnabled(False)

        max_index = max(history_info.item_count, len(self.buttons))
        for i in range(max_index):
            history_info_exists = i < history_info.item_count
            button_slot_exists = i < len(self.buttons)

            if history_info_exists:
                if not button_slot_exists:
                    button = append_button(i)
                else:
                    button = self.buttons[i]

                update_button(button, i)
            else:
                if button_slot_exists:
                    hide_button(i)

    def _on_clicked_button(self, index: int):
        self.selected_index = index


class LatentPicker(QTabWidget):
    class SolidLatent:
        def __init__(self, latent_value: tuple[float, ...]):
            self.latent_value = latent_value

    class HistoryLatent:
        def __init__(self, history_index: int):
            self.history_index = history_index

    def __init__(self,
                 generate_preview_func: Callable[[tuple[float, float, float, float]], QPixmap],
                 get_history_item_func: Callable[[int], HistoryPickerWidget.HistoryItem],
                 get_history_info_func: Callable[[], HistoryPickerWidget.HistoryInfo],
                 parent=None):
        super().__init__(parent)

        self.color_picker = ColorPickerWidget(generate_preview_func, parent=self)
        self.addTab(self.color_picker, "Solid Latent")

        self.history_picker = HistoryPickerWidget(get_history_item_func, get_history_info_func, parent=self)
        self.addTab(self.history_picker, "Undo History")

    def get_latent_info(self) -> SolidLatent | HistoryLatent:
        selected_tab = self.currentIndex()

        if selected_tab == 0:
            return LatentPicker.SolidLatent(self.color_picker.use_current_latent_value())
        elif selected_tab == 1:
            return LatentPicker.HistoryLatent(self.history_picker.selected_index)
        else:
            # TODO add more modes.
            return LatentPicker.SolidLatent(self.color_picker.use_current_latent_value())
