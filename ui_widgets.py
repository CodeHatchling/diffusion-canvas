from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QSlider,
    QLineEdit,
    QHBoxLayout,
    QScrollArea,
    QFrame,
    QLabel,
    QFormLayout,
    QLayout,
    QPushButton
)

from PyQt6.QtCore import Qt, QSize
import math
import numpy as np
from ui_utils import ExceptionCatcher


def _call_handlers(handlers: list[callable]):
    for handler in handlers:
        try:
            handler()
        except Exception as e:
            print(e)


class Slider(QWidget):
    def __init__(self,
                 label: str,
                 default_value: int | float,
                 min_max: tuple[int, int] | tuple[float, float],
                 step_size: int | float = 1,
                 parent: QWidget | None = None,):
        super().__init__(parent)

        self.setContentsMargins(0, 0, 0, 0)

        self.on_ui_value_changed: list[callable] = []

        if not (min_max[1] > min_max[0]):
            raise ValueError(
                f"\'min_max\' parameter invalid, the minimum {min_max[0]} must be less than the maximum {min_max[1]}!")

        # This is used to prevent value handling from becoming recursive.
        self.recursion_check = False

        self._value = default_value
        self.label = label

        # Get the numbers used for conversion from the internal 'step' units to the desired units.
        # This is a work-around for the lack of floating-point value support.
        self._range = min_max
        self._span = min_max[1] - min_max[0]
        self._steps = self._span / step_size

        self._is_int = (
            isinstance(default_value, int) and
            isinstance(min_max[0], int) and
            isinstance(min_max[1], int) and
            isinstance(step_size, int)
        )

        self._steps = int(np.round(self._steps))

        # Create the slider.
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimumWidth(100)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self._steps)
        self._slider.setValue(self._value_to_step(self._value))

        tick_steps = int(np.round((10 ** np.ceil(math.log10(self._span) - 1) / self._span) * self._steps))
        self._slider.setSingleStep(tick_steps)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        # Create the value display.
        self._value_display = QLineEdit()
        self._value_display.setFixedWidth(60)

        # When one changes, update the internal value and the other widget.
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._value_display.textChanged.connect(self._on_text_changed)

        # Add these to a horizontal layout.
        layout = QHBoxLayout()
        layout.addWidget(self._slider)
        layout.addWidget(self._value_display)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add the layout to self.
        self.setLayout(layout)

        # Finally, update the other widgets to match the slider.
        self._on_slider_changed()

    def __set_value(self, value):
        self._value = value

        if self._is_int:
            self._value = int(self._value)

    def __get_value(self):
        if self._is_int:
            self._value = int(self._value)

        return self._value

    def _get_value(self):
        return self.__get_value()

    def _set_value(self, value):
        self.__set_value(value)
        self._on_value_changed()

    value = property(fget=_get_value, fset=_set_value)

    def _value_to_step(self, value: int | float) -> int:
        return int(np.clip(np.round(((value - self._range[0]) / self._span) * self._steps), a_min=0, a_max=self._steps))

    def _step_to_value(self, step: int) -> int | float:
        return np.clip((step / self._steps) * self._span + self._range[0], a_min=self._range[0], a_max=self._range[1])

    def _on_slider_changed(self):
        with ExceptionCatcher(None, "Failed to handle slider change"):
            if self.recursion_check:
                return

            self.recursion_check = True
            try:
                self._value = self._step_to_value(self._slider.value())
                if self._is_int:
                    self.__set_value(int(self._value))
                    self._value_display.setText(f"{self._value}")
                else:
                    self._value_display.setText(f"{self._value:.2f}")
                _call_handlers(self.on_ui_value_changed)
            finally:
                self.recursion_check = False

    def _on_text_changed(self):
        with ExceptionCatcher(None, "Failed to handle text change"):
            if self.recursion_check:
                return

            self.recursion_check = True
            try:
                self._value = int(self._value_display.text()) if self._is_int else float(self._value_display.text())
                self._slider.setValue(self._value_to_step(self._value))
                _call_handlers(self.on_ui_value_changed)
            except ValueError:
                return
            finally:
                self.recursion_check = False

    def _on_value_changed(self):
        with ExceptionCatcher(None, "Failed to handle text change"):
            if self.recursion_check:
                return

            self.recursion_check = True
            try:
                self._slider.setValue(self._value_to_step(self._value))

                if self._is_int:
                    self.__set_value(int(self._value))
                    self._value_display.setText(f"{self._value}")
                else:
                    self._value_display.setText(f"{self._value:.2f}")
            except ValueError:
                return
            finally:
                self.recursion_check = False


class VerticalScrollArea(QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.horizontalScrollBar().rangeChanged.connect(self._reset_min_width)

    def _reset_min_width(self):
        self.setMinimumWidth(self.minimumSizeHint().width())

    def minimumSizeHint(self) -> QSize:
        our_min_size = super().minimumSizeHint()

        our_widget = self.widget()
        if our_widget is None:
            return our_min_size

        total_frame_width = self.frameWidth()
        widget_width = our_widget.minimumSizeHint().width()
        total_width = total_frame_width + widget_width
        if our_min_size.width() < total_width:
            our_min_size.setWidth(total_width)

        return our_min_size


class HelpBox(QFrame):

    def __init__(self, contents: list[any], parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setLineWidth(3)
        self.setContentsMargins(0, 0, 0, 0)
        layout = QFormLayout(parent)
        self.setLayout(layout)
        layout.setContentsMargins(5, 5, 5, 5)

        def item_is_tuple(_item: any, element_types: tuple[any, ...]) -> bool:
            if not isinstance(_item, tuple):
                return False
            if len(_item) != len(element_types):
                return False
            for i in range(len(_item)):
                if not isinstance(_item[i], element_types[i]):
                    return False

            return True

        def add_content(_item: any):
            if isinstance(_item, QWidget):
                layout.addWidget(_item)
            elif isinstance(_item, QLayout):
                layout.addChildLayout(_item)
            elif isinstance(_item, str):
                layout.addWidget(QLabel(_item))
            elif item_is_tuple(_item, (str | QWidget, QWidget | QLayout)):
                layout.addRow(_item[0], _item[1])
            elif item_is_tuple(_item, (str | QWidget, str)):
                layout.addRow(_item[0], QLabel(_item[1]))

        for item in contents:
            add_content(item)


class ImageButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

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


class LabelImageButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.setLayout(layout)

        self._text_label = QLabel()
        self._text_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        self._image_label = QLabel()
        self._image_label.setScaledContents(True)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._text_label)
        layout.addWidget(self._image_label)

    def set_image(self, pixmap: QPixmap | None):
        if pixmap is None:  # It should accept none, but it doesn't?
            self._image_label.clear()
            self._image_label.setFixedSize(QSize(0, 0))
        else:
            self._image_label.setPixmap(pixmap)
            self._image_label.setFixedSize(pixmap.size())
        self._reset_min_size()

    def set_text(self, text: str | None):
        if text is None:
            self._text_label.clear()
        else:
            self._text_label.setText(text)
        self._reset_min_size()

    def _reset_min_size(self):
        self.setMinimumSize(self.minimumSizeHint())

    def minimumSizeHint(self) -> QSize:
        our_min_size = super().minimumSizeHint()

        min_size_text = self._text_label.minimumSizeHint()
        min_size_image_widget = self._image_label.minimumSizeHint()
        pixmap = self._image_label.pixmap()
        min_size_image = pixmap.size() if isinstance(pixmap, QPixmap) else min_size_image_widget

        widgets_total_size = QSize(
            max(min_size_image_widget.width(), min_size_image.width()) + min_size_text.width(),
            max(min_size_image_widget.height(), min_size_image.height()) + min_size_text.height(),
        )

        result_size = QSize(
            max(our_min_size.width(), widgets_total_size.width()),
            max(our_min_size.height(), widgets_total_size.height())
        )

        return result_size
