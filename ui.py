# This script: ui.py - UI for DiffusionCanvas.

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.
# diffusion_canvas_api.py - Contains functions used by the UI


from PyQt6.QtWidgets import (QLabel, QMainWindow, QVBoxLayout, QWidget, QSlider, QDockWidget,
                             QFormLayout, QLineEdit, QPushButton, QScrollArea, QHBoxLayout,
                             QDialog, QDialogButtonBox, QFileDialog, QMessageBox, QGridLayout, QSpinBox, QLayout,
                             QComboBox)
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent, QKeyEvent
from PyQt6.QtCore import Qt, QTimer, QRect, QPointF
import PIL.Image
import math
import numpy as np
import torch
from PIL import Image
from sdwebui_interface import pop_intercepted, unfreeze_sd_webui
from diffusion_canvas_api import DiffusionCanvasAPI
from layer import History, Layer
from typing import Callable


global_generate_image: Callable[[int, int, int, any], QImage] | None = None


class ExceptionCatcher:
    def __init__(self, context: QWidget | None, message: str):
        self.context = context
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            import traceback as tb
            except_string = '\n'.join(tb.format_exception(exc_type, exc_value, traceback))
            QMessageBox.critical(self.context, f"Error: {exc_type}", f"{self.message}: {exc_value}\n"
                                                                     f"\n" +
                                                                     except_string)
        return True  # Suppress exceptions


class EditParamsDialog(QDialog):
    class Output:
        def __init__(self, name: str, pixmap: QImage | None):
            self.name: str = name
            self.pixmap: QPixmap | None = pixmap

    def __init__(self, current_widget: "ParamsWidget", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Params")

        self.delete = False

        self.layout = QFormLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add a name text field.
        self.name_input = QLineEdit(self)
        self.name_input.setText(current_widget.name)
        self.layout.addRow("Name", self.name_input)

        # Add a thumbnail preview.
        self.image = QLabel(self)
        self.image.setFixedWidth(200)
        self.image.setFixedHeight(200)
        self.image.setScaledContents(True)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if isinstance(current_widget.pixmap, QPixmap):
            self.image.setPixmap(current_widget.pixmap)
        else:
            self.image.setText("No Thumbnail")
        self.layout.addRow("Thumbnail", self.image)

        # Add a generate button.
        self.generate_button = QPushButton("Generate Thumbnail")
        self.generate_button.clicked.connect(lambda: self.generate_thumbnail(current_widget.params))
        self.layout.addWidget(self.generate_button)

        # Add Okay/Cancel buttons.
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def generate_thumbnail(self, params):
        with ExceptionCatcher(None, "Failed to generate thumbnail"):
            if global_generate_image is None:
                return

            if params is None:
                return

            import texture_convert as conv
            image = conv.convert(global_generate_image(512, 512, 20, params), QImage)
            image = QPixmap.fromImage(image)
            self.image.setPixmap(image)

    def get_output(self) -> Output:
        return EditParamsDialog.Output(
            name=self.name_input.text(),
            pixmap=self.image.pixmap(),
        )


class ParamsWidget(QWidget):

    params: any
    delete_handler: Callable[['ParamsWidget'], None]
    button_image: QLabel
    button_label: QLabel

    def __init__(self,
                 params: any,
                 button_name: str,
                 params_setter: Callable[[any], None],
                 delete_handler: Callable[['ParamsWidget'], None],
                 parent: QWidget | None = None,):
        super().__init__(parent)
        self.params = params
        self.delete_handler = delete_handler

        # Create a layout for the button and rename action
        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def add_button() -> QWidget:
            # Params selection button
            button = QPushButton(self)
            button.clicked.connect(lambda: params_setter(self.params))
            button.setFixedWidth(100)
            button.setFixedHeight(120)

            # Internal button layout
            button_internal_layout = QVBoxLayout(button)
            button.setLayout(button_internal_layout)
            button_internal_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.button_image = QLabel()
            self.button_image.setFixedWidth(80)
            self.button_image.setFixedHeight(80)
            self.button_image.setScaledContents(True)
            self.button_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            button_internal_layout.addWidget(self.button_image)

            self.button_label = QLabel()
            self.button_label.setText(button_name)
            self.button_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            button_internal_layout.addWidget(self.button_label)

            return button

        def add_side_buttons() -> QWidget:
            # Must be a widget.
            holder = QWidget(self)

            # Vertical layout for rename and delete buttons
            small_button_layout = QVBoxLayout()
            small_button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Edit button
            rename_button = QPushButton("📝", self)
            rename_button.setFixedWidth(30)
            rename_button.setFixedHeight(30)
            rename_button.clicked.connect(self.on_rename_params)
            small_button_layout.addWidget(rename_button)

            # Delete button
            delete_button = QPushButton("❌", self)
            delete_button.setFixedWidth(30)
            delete_button.setFixedHeight(30)
            delete_button.clicked.connect(self.on_delete)
            small_button_layout.addWidget(delete_button)

            holder.setLayout(small_button_layout)
            return holder

        main_layout.addWidget(add_button())
        main_layout.addWidget(add_side_buttons())

        # Add to layout
        self.setLayout(main_layout)

    def on_rename_params(self):
        """
        Opens a dialog to rename a params button.
        """
        with ExceptionCatcher(None, "Failed to handle rename request"):
            dialog = EditParamsDialog(current_widget=self, parent=None)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                output = dialog.get_output()
                self.button_image.setPixmap(output.pixmap)
                self.button_label.setText(output.name)

    def on_delete(self):
        with ExceptionCatcher(None, "Failed to handle delete request"):
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Delete Params")  # Set the title
            dialog.setText(f"Do you want to delete the \"{self.name}\" params?")  # Set the main message
            dialog.setInformativeText("Press \"Delete\" to delete them, press \"Cancel\" to keep them.\n"
                                      "This cannot be undone.")  # Optional detailed message
            dialog.setIcon(QMessageBox.Icon.Warning)  # Set an icon (e.g., Question, Information, Warning, Critical)

            # Add custom buttons
            accept_button = dialog.addButton("Delete", QMessageBox.ButtonRole.AcceptRole)
            dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

            # Execute the dialog
            dialog.exec()

            # Determine which button was clicked
            if dialog.clickedButton() == accept_button:
                self.delete_handler(self)

    def _get_name(self) -> str:
        return self.button_label.text()

    def _set_name(self, value: str):
        with ExceptionCatcher(None, "Failed to assign name"):
            self.button_label.setText(value)

    name = property(fget=_get_name, fset=_set_name)

    def _get_pixmap(self) -> QPixmap:
        return self.button_image.pixmap()

    def _set_pixmap(self, value: QPixmap):
        with ExceptionCatcher(None, "Failed to assign pixmap"):
            self.button_image.setPixmap(value)

    pixmap = property(fget=_get_pixmap, fset=_set_pixmap)


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


class NewCanvasDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Canvas")

        # Layout for the dialog
        self.layout = QGridLayout(self)

        # Width input
        self.layout.addWidget(QLabel("Width:"), 0, 0)
        self.width_input = QSpinBox(self)
        self.width_input.setRange(8, 8192)  # Range for width
        self.width_input.setValue(512)      # Default value
        self.layout.addWidget(self.width_input, 0, 1)

        # Height input
        self.layout.addWidget(QLabel("Height:"), 1, 0)
        self.height_input = QSpinBox(self)
        self.height_input.setRange(8, 8192)  # Range for height
        self.height_input.setValue(512)      # Default value
        self.layout.addWidget(self.height_input, 1, 1)

        # Buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons, 2, 0, 1, 2)

    def get_dimensions(self):
        with ExceptionCatcher(self, "Failed to get dimensions"):
            return self.width_input.value(), self.height_input.value()


class BaseBrushTool:
    def __init__(self,
                 icon_emoji: str,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable):

        self._tool_settings_dock = tool_settings_dock
        self._extra_on_tool_button_click = on_tool_button_click
        self.show_noisy = False

        button = QPushButton()
        button.setText(icon_emoji)
        button.setFixedWidth(40)
        button.setFixedHeight(40)
        button.setStyleSheet("font-size: 25px;")
        button.clicked.connect(self._on_tool_button_click)

        tool_dock_layout.addWidget(button)

        self._tool_settings_dock_widget = self._create_tool_settings_dock_widget()

    def _on_tool_button_click(self):
        with ExceptionCatcher(None, "Failed to handle brush button click"):
            self._tool_settings_dock.setWidget(self._tool_settings_dock_widget)
            self._extra_on_tool_button_click()

    def _create_tool_settings_dock_widget(self) -> QWidget:
        ...

    def brush_stroke_will_modify(self,
                                 layer: Layer,
                                 params,
                                 mouse_button: Qt.MouseButton,
                                 normalized_mouse_coord: (float, float)) -> bool:
        ...

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            normalized_mouse_coord: (float, float)):
        ...


class NoiseBrushTool(BaseBrushTool):

    def __init__(self,
                 api,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable):

        super().__init__("🪄", tool_dock_layout, tool_settings_dock, on_tool_button_click)
        self._api = api

    def _create_tool_settings_dock_widget(self) -> QWidget:
        sliders_widget = QWidget()
        sliders_layout = QFormLayout(sliders_widget)

        # Add a help message:
        sliders_layout.addRow("Left click", QLabel("Add noise"))
        sliders_layout.addRow("Right click", QLabel("Denoise"))

        # Add sliders:
        self.slider_noise_brush_radius = Slider(
            "Noise Radius (px)",
            64,
            (0, 512),
            0.1
        )
        self.slider_noise_brush_intensity = Slider(
            "Noise Intensity",
            3.0,
            (0.0, 5.0),
            0.01
        )
        self.slider_denoise_brush_radius = Slider(
            "Denoise Radius (px)",
            64,
            (0, 512),
            0.1
        )
        self.slider_denoise_size_x = Slider(
            "Context Width (px)",
            1024,
            (8, 4096),
            8
        )
        self.slider_denoise_size_y = Slider(
            "Context Height (px)",
            1024,
            (8, 4096),
            8
        )
        self.slider_denoise_attenuation = Slider(
            "Denoise Attenuation",
            0.25,
            (0.0, 1.0),
            0.001
        )
        self.slider_denoise_subtraction = Slider(
            "Denoise Subtraction",
            0.01,
            (0.0, 1.0),
            0.001
        )
        self.slider_denoise_bias = Slider(
            "Denoise Bias",
            0,
            (-1, 1.0),
            0.01
        )

        sliders_layout.addRow(self.slider_noise_brush_radius.label, self.slider_noise_brush_radius)
        sliders_layout.addRow(self.slider_noise_brush_intensity.label, self.slider_noise_brush_intensity)
        sliders_layout.addRow(self.slider_denoise_brush_radius.label, self.slider_denoise_brush_radius)
        sliders_layout.addRow(self.slider_denoise_size_x.label, self.slider_denoise_size_x)
        sliders_layout.addRow(self.slider_denoise_size_y.label, self.slider_denoise_size_y)
        sliders_layout.addRow(self.slider_denoise_attenuation.label, self.slider_denoise_attenuation)
        sliders_layout.addRow(self.slider_denoise_subtraction.label, self.slider_denoise_subtraction)
        sliders_layout.addRow(self.slider_denoise_bias.label, self.slider_denoise_bias)

        return sliders_widget

    def _get_noise_brush_radius(self):
        return self.slider_noise_brush_radius.value
    noise_brush_radius = property(_get_noise_brush_radius)

    def _get_noise_brush_intensity(self):
        return self.slider_noise_brush_intensity.value
    noise_brush_intensity = property(_get_noise_brush_intensity)

    def _get_denoise_brush_radius(self):
        return self.slider_denoise_brush_radius.value
    denoise_brush_radius = property(_get_denoise_brush_radius)

    def _get_denoise_context_size_x(self):
        return self.slider_denoise_size_x.value
    denoise_context_size_x = property(_get_denoise_context_size_x)

    def _get_denoise_context_size_y(self):
        return self.slider_denoise_size_y.value
    denoise_context_size_y = property(_get_denoise_context_size_y)

    def _get_denoise_attenuation(self):
        return self.slider_denoise_attenuation.value
    denoise_attenuation = property(_get_denoise_attenuation)

    def _get_denoise_subtraction(self):
        return self.slider_denoise_subtraction.value
    denoise_subtraction = property(_get_denoise_subtraction)

    def _get_denoise_bias(self):
        return self.slider_denoise_bias.value
    denoise_bias = property(_get_denoise_bias)

    def brush_stroke_will_modify(self,
                                 layer: Layer,
                                 params,
                                 mouse_button: Qt.MouseButton,
                                 normalized_mouse_coord: (float, float)) -> bool:
        return mouse_button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            normalized_mouse_coord: (float, float)):
        if mouse_button == Qt.MouseButton.LeftButton:
            self._api.draw_noise_dab(layer=layer,
                                     position_xy=normalized_mouse_coord,
                                     pixel_radius=self.noise_brush_radius,
                                     noise_intensity=self.noise_brush_intensity)
            self.show_noisy = True

        elif mouse_button == Qt.MouseButton.RightButton:
            self._api.draw_denoise_dab(layer=layer,
                                       params=params,
                                       position_xy=normalized_mouse_coord,
                                       pixel_radius=self.denoise_brush_radius,
                                       context_region_pixel_size_xy=(
                                           self.denoise_context_size_x,
                                           self.denoise_context_size_y
                                       ),
                                       attenuation_params=(self.denoise_attenuation, self.denoise_subtraction),
                                       noise_bias=2 ** self.denoise_bias,
                                       time_budget=0.1)
            self.show_noisy = False


class LatentBrushTool(BaseBrushTool):

    brush_value: tuple[float, float, float, float]

    def __init__(self,
                 api,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable):

        super().__init__("🖌️️", tool_dock_layout, tool_settings_dock, on_tool_button_click)
        self._api = api
        self.brush_value = (0.0, 0.0, 0.0, 0.0)

    def _create_tool_settings_dock_widget(self) -> QWidget:
        sliders_widget = QWidget()
        sliders_layout = QFormLayout(sliders_widget)

        # Add a help message:
        sliders_layout.addRow("Left click", QLabel("Paint Color"))
        sliders_layout.addRow("Right click", QLabel("Select Color with Eyedropper"))

        # Add a blend mode dropdown:
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(
            [mode.name for mode in DiffusionCanvasAPI.BlendMode]
        )
        sliders_layout.addRow("Blend Mode", self.blend_mode_combo)

        # Add sliders:
        self.slider_brush_radius = Slider(
            "Brush Radius (px)",
            64,
            (0, 512),
            0.1
        )

        self.slider_brush_opacity = Slider(
            "Brush Intensity",
            1.0,
            (0.0, 1.0),
            0.01
        )

        sliders_layout.addRow(self.slider_brush_radius.label, self.slider_brush_radius)
        sliders_layout.addRow(self.slider_brush_opacity.label, self.slider_brush_opacity)

        return sliders_widget

    def _get_brush_radius(self):
        return self.slider_brush_radius.value
    brush_radius = property(_get_brush_radius)

    def _get_brush_opacity(self):
        return self.slider_brush_opacity.value
    brush_opacity = property(_get_brush_opacity)

    def _get_blend_mode(self) -> DiffusionCanvasAPI.BlendMode:
        index = self.blend_mode_combo.currentIndex()
        try:
            return DiffusionCanvasAPI.BlendMode(index)
        except ValueError:
            return DiffusionCanvasAPI.BlendMode.Blend  # Default to Blend if invalid
    blend_mode = property(_get_blend_mode)

    def brush_stroke_will_modify(self,
                                 layer: Layer,
                                 params,
                                 mouse_button: Qt.MouseButton,
                                 normalized_mouse_coord: (float, float)) -> bool:
        return mouse_button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            normalized_mouse_coord: (float, float)):
        if mouse_button == Qt.MouseButton.LeftButton:
            self._api.draw_latent_dab(layer=layer,
                                      blend_mode=self.blend_mode,
                                      value=self.brush_value,
                                      position_xy=normalized_mouse_coord,
                                      pixel_radius=self.brush_radius,
                                      opacity=self.brush_opacity)

            self.show_noisy = False

        elif mouse_button == Qt.MouseButton.RightButton:
            self.brush_value = self._api.get_average_latent(layer=layer,
                                                            position_xy=normalized_mouse_coord,
                                                            pixel_radius=self.brush_radius)
            self.show_noisy = False


class DiffusionCanvasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffusion Canvas")

        self.api = DiffusionCanvasAPI()

        # Set up the generator function used by other stuff.
        global global_generate_image
        global_generate_image = self.api.generate_image

        self.setUpdatesEnabled(True)

        self.params_widgets: list[ParamsWidget] = []  # List to store params widgets

        self.label = QLabel(self)

        # Central layout setup
        self.canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_widget)
        canvas_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_widget.setLayout(canvas_layout)

        canvas_scroll_area = QScrollArea()
        canvas_scroll_area.setWidget(self.canvas_widget)
        canvas_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_scroll_area.setMinimumWidth(200)
        canvas_scroll_area.setMinimumHeight(200)

        canvas_layout.addWidget(self.label)

        self.setCentralWidget(canvas_scroll_area)

        # Add a menu widget
        menu_widget = QWidget()
        bar_layout = QHBoxLayout(menu_widget)

        new_image_button = QPushButton("📄", self)
        new_image_button.clicked.connect(self.on_clicked_new)
        new_image_button.setFixedWidth(40)
        new_image_button.setFixedHeight(40)
        new_image_button.setStyleSheet("font-size: 25px;")
        bar_layout.addWidget(new_image_button)

        load_image_button = QPushButton("📂", self)
        load_image_button.clicked.connect(self.on_clicked_load)
        load_image_button.setFixedWidth(40)
        load_image_button.setFixedHeight(40)
        load_image_button.setStyleSheet("font-size: 25px;")
        bar_layout.addWidget(load_image_button)

        save_image_button = QPushButton("💾", self)
        save_image_button.clicked.connect(self.on_clicked_save)
        save_image_button.setFixedWidth(40)
        save_image_button.setFixedHeight(40)
        save_image_button.setStyleSheet("font-size: 25px;")
        bar_layout.addWidget(save_image_button)

        unfreeze_button = QPushButton("Unfreeze sd.webui", self)
        unfreeze_button.clicked.connect(unfreeze_sd_webui)
        unfreeze_button.setFixedHeight(40)
        bar_layout.addWidget(unfreeze_button)

        self.setMenuWidget(menu_widget)

        # Create a scrollable widget for params buttons
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        self.params_widget.setLayout(self.params_layout)

        self.params_scroll_area = QScrollArea()
        self.params_scroll_area.setWidget(self.params_widget)
        self.params_scroll_area.setWidgetResizable(True)

        # Dock for params palette
        params_dock = QDockWidget("Params Palette", self)
        params_dock.setWidget(self.params_scroll_area)
        params_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, params_dock)

        # Dock widget for tool buttons
        tool_dock = QDockWidget("Tools", self)
        tool_dock.setAllowedAreas(Qt.DockWidgetArea.TopDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        tool_widget = QWidget()
        tool_layout = QHBoxLayout(tool_widget)
        tool_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        tool_dock.setWidget(tool_widget)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, tool_dock)

        # Dock widget for tool settings
        tool_settings_dock = QDockWidget("Tool Settings", self)
        tool_settings_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, tool_settings_dock)

        """
        api,
        tool_dock_layout: QLayout,
        tool_settings_dock: QDockWidget,
        on_tool_button_click: callable
        """

        self.current_tool: BaseBrushTool | None = None

        def set_current_tool(tool: BaseBrushTool | None):
            self.current_tool = tool

        self.noise_brush_tool = NoiseBrushTool(
            api=self.api,
            tool_dock_layout=tool_layout,
            tool_settings_dock=tool_settings_dock,
            on_tool_button_click=lambda: set_current_tool(self.noise_brush_tool)
        )

        self.latent_brush_tool = LatentBrushTool(
            api=self.api,
            tool_dock_layout=tool_layout,
            tool_settings_dock=tool_settings_dock,
            on_tool_button_click=lambda: set_current_tool(self.latent_brush_tool)
        )

        # Setup update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # Roughly 60 FPS

        # Add a timer to show full preview
        self.full_preview_timer = 0
        self.showing_quick_preview = False

        self.params = None

        # Track mouse dragging
        self.is_dragging = False
        self.drag_button = None
        self.show_noisy = False

        self.history = History(self.api.create_empty_layer(512//8, 512//8))
        self.create_undo = True

        self.update_canvas_view(noisy=False, full=False)

    def closeEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle close event"):
            """
            Override this method to handle tasks before the window closes.
            """
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Perform cleanup tasks here
                print("Closing the application...")

                # Clean up memory and such.
                self.release()

                event.accept()  # Accept the close event
            else:
                event.ignore()  # Ignore the close event

    def release(self):
        global global_generate_image
        global_generate_image = None

    def _get_layer(self) -> Layer:
        return self.history.layer
    layer = property(fget=_get_layer)

    def on_clicked_new(self):
        """
        Opens a dialog with width/height entry fields, and [Create New] and [Cancel] buttons.
        If [Create New] is clicked, replaces the layer with an empty latent.
        """
        with ExceptionCatcher(self, "Failed to create new image"):
            dialog = NewCanvasDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                width, height = dialog.get_dimensions()

                # Round up to the nearest whole number of latents.
                # (1x1 latent = 8x8 pixels)
                latent_size_xy: tuple[int, int] = (
                    int(np.maximum(np.ceil(width / 8), 1)),
                    int(np.maximum(np.ceil(height / 8), 1))
                )

                # Create a new latent layer with the specified dimensions
                self.history = History(self.api.create_empty_layer(latent_size_xy[0], latent_size_xy[1]))
                self.create_undo = True

                # Update the display with the new blank canvas
                self.update_canvas_view(noisy=False, full=True)
                print(f"New canvas created with dimensions: {width}x{height}")

    def on_clicked_load(self):
        """
        Opens a file open dialogue. If a file is opened, replaces the layer with the image.
        """
        with ExceptionCatcher(self, "Failed to load image"):
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

            if not file_path:
                return  # User canceled

            # Load the image and convert to a diffusion canvas layer
            image = Image.open(file_path)
            self.history = History(self.api.create_layer_from_image(image))
            self.create_undo = True

            # Redraw the canvas
            self.update_canvas_view(noisy=False, full=True)

    def on_clicked_save(self):
        """
        Opens a file save dialogue. If a destination file is chosen, saves the canvas to that file.
        """
        with ExceptionCatcher(self, "Failed to save image"):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

            if not file_path:
                return  # User canceled

            # Convert the latent space image back to a PIL image
            image = self.api.latent_to_image(self.layer.clean_latent, True, PIL.Image.Image)

            # Save the image to the chosen file path
            image.save(file_path)

            QMessageBox.information(self, "Save Successful", f"Image saved to {file_path}")

    def add_params_to_palette(self, params):
        with ExceptionCatcher(self, "Failed to add params"):
            """
            Adds a new params object to the palette and creates a corresponding button.
            """

            # Add the deletion handler.
            def delete_handler(w):
                if w in self.params_widgets:
                    self.params_widgets.remove(w)

                if self.params_layout.indexOf(w) != -1:  # Check if widget is in the layout
                    self.params_layout.removeWidget(w)
                    w.deleteLater()  # Optionally delete the widget from memory

            params_index = len(self.params_widgets) - 1
            params_widget = ParamsWidget(
                parent=self,
                params=params,
                button_name=f"Params {params_index + 1}",
                params_setter=lambda p: self.set_current_params(p),
                delete_handler=lambda p: delete_handler(p)
            )

            self.params_widgets.append(params_widget)
            self.params_layout.addWidget(params_widget)

    def set_current_params(self, params):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            """
            Sets the current params object to be used for denoising.
            """
            self.params = params
            print(f"Selected params: {self.params}")

    def update_frame(self):
        with ExceptionCatcher(self, "Error occurred in update_frame"):
            denoiser_and_params = pop_intercepted()

            if denoiser_and_params is not None:
                self.api.set_denoiser(denoiser_and_params[0])
                new_params = denoiser_and_params[1]
                self.add_params_to_palette(new_params)

            self.full_preview_timer -= 16
            if self.full_preview_timer <= 0:
                self.full_preview_timer = 0

                if self.showing_quick_preview:
                    self.update_canvas_view(full=True)

    def mousePressEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            button = event.button()
            if button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self.is_dragging = True
                self.drag_button = button
                self.apply_brush(event)

    def mouseMoveEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            if self.is_dragging:
                self.apply_brush(event)

    def mouseReleaseEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            if event.button() == self.drag_button:
                self.is_dragging = False
                self.drag_button = None

            self.create_undo = True

    @torch.no_grad()
    def apply_brush(self, event: QMouseEvent):
        if self.current_tool is None:
            return

        pixmap: QPixmap = self.label.pixmap()
        mouse_pos = event.globalPosition()
        image_rect: QRect = pixmap.rect()
        image_pos = self.label.mapToGlobal(QPointF(image_rect.x(), image_rect.y()))

        normalized_position = (
            (mouse_pos.x() - image_pos.x()) / pixmap.width(),
            (mouse_pos.y() - image_pos.y()) / pixmap.height(),
        )

        # Keep within bounds.
        if normalized_position[0] < 0 or normalized_position[0] > 1:
            return
        if normalized_position[1] < 0 or normalized_position[1] > 1:
            return

        if self.current_tool.brush_stroke_will_modify(layer=self.layer,
                                                      params=self.params,
                                                      mouse_button=self.drag_button,
                                                      normalized_mouse_coord=normalized_position):
            if self.create_undo:
                self.history.register_undo()
                self.create_undo = False

            self.current_tool.handle_brush_stroke(layer=self.layer,
                                                  params=self.params,
                                                  mouse_button=self.drag_button,
                                                  normalized_mouse_coord=normalized_position)

            self.update_canvas_view(noisy=self.current_tool.show_noisy, full=False)

    def keyPressEvent(self, event: QKeyEvent):
        with ExceptionCatcher(self, "Failed to handle key press event"):
            """
            Handles key press events to listen for Ctrl+Z and Ctrl+Shift+Z for undo and redo.
            """
            used = False
            if event.key() == Qt.Key.Key_Z:
                if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                    self.undo()  # Ctrl+Z
                    used = True
                elif event.modifiers() == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                    self.redo()  # Ctrl+Shift+Z
                    used = True

            if not used:
                super().keyPressEvent(event)

    def undo(self):
        """
        Undo the last action.
        """
        with ExceptionCatcher(None, "Failed to undo"):
            self.history.undo(1)
            self.update_canvas_view(full=False)

    def redo(self):
        """
        Redo the previously undone action.
        """
        with ExceptionCatcher(None, "Failed to redo"):
            self.history.redo(1)
            self.update_canvas_view(full=False)

    def update_canvas_view(self, noisy: bool | None = None, full: bool = True):

        if isinstance(noisy, bool):
            self.show_noisy = noisy

        latent_to_show = (
            self.layer.noisy_latent
            if self.show_noisy
            else self.layer.clean_latent
        )

        if not full:
            self.showing_quick_preview = True
            self.full_preview_timer = 1000
        else:
            self.showing_quick_preview = False

        # Convert tensor to QImage
        q_image = self.api.latent_to_image(latent_to_show, full, QImage)
        pixmap = QPixmap.fromImage(q_image)

        # Update the label pixmap
        self.label.setPixmap(pixmap)
        self.label.setFixedWidth(pixmap.width())
        self.label.setFixedHeight(pixmap.height())
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_widget.setFixedWidth(pixmap.width() + 20)
        self.canvas_widget.setFixedHeight(pixmap.height() + 20)
