from PyQt6.QtWidgets import QWidget, QSlider, QLineEdit, QHBoxLayout

from PyQt6.QtCore import Qt
import math
import numpy as np
from ui_utils import ExceptionCatcher


class Slider(QWidget):
    def __init__(self,
                 label: str,
                 default_value: int | float,
                 min_max: tuple[int, int] | tuple[float, float],
                 step_size: int | float = 1,
                 parent: QWidget | None = None,):
        super().__init__(parent)

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

        tick_steps = int(np.round((10 ** np.round(math.log10(self._span) - 1) / self._span) * self._steps))
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

        # Add the layout to self.
        self.setLayout(layout)

        # Finally, update the other widgets to match the slider.
        self._on_slider_changed()

    def _get_value(self):
        if self._is_int:
            self._value = int(self._value)

        return self._value

    def _set_value(self, value):
        self._value = value

        if self._is_int:
            self._value = int(self._value)

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
                    self.value = int(self._value)
                    self._value_display.setText(f"{self._value}")
                else:
                    self._value_display.setText(f"{self._value:.2f}")
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
            except ValueError:
                return
            finally:
                self.recursion_check = False