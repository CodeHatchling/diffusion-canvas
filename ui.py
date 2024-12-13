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
                             QDialog, QDialogButtonBox, QFileDialog, QMessageBox, QGridLayout, QSpinBox)
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


class RenameDialog(QDialog):
    def __init__(self, parent=None, current_name=""):
        super().__init__(parent)
        self.setWindowTitle("Rename Params")

        self.layout = QVBoxLayout(self)

        self.name_input = QLineEdit(self)
        self.name_input.setText(current_name)
        self.layout.addWidget(self.name_input)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_name(self):
        return self.name_input.text()


class Slider:
    def __init__(self,
                 label: str,
                 default_value: int | float,
                 min_max: tuple[int, int] | tuple[float, float],
                 step_size: int | float = 1):

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
        self.widget = QHBoxLayout()
        self.widget.addWidget(self._slider)
        self.widget.addWidget(self._value_display)

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
        return self.width_input.value(), self.height_input.value()


class DiffusionCanvasWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffusion Canvas")
        self.api = DiffusionCanvasAPI()
        self.setUpdatesEnabled(True)
        self.params_palette = []  # List to store params objects
        self.params_buttons = []  # List to store buttons associated with params

        self.label = QLabel(self)

        # Central layout setup
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.label)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(central_widget)

        # Add an unfreeze button to return control to sd.webui
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
        self.params_scroll_area.setMinimumWidth(200)
        self.params_scroll_area.setMinimumHeight(300)

        # Dock for params palette
        params_dock = QDockWidget("Params Palette", self)
        params_dock.setWidget(self.params_scroll_area)
        params_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, params_dock)

        # Dock widget for sliders
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        sliders_widget = QWidget()
        sliders_layout = QFormLayout(sliders_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Create sliders and add them to the dock.
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

        sliders_layout.addRow(self.slider_noise_brush_radius.label, self.slider_noise_brush_radius.widget)
        sliders_layout.addRow(self.slider_noise_brush_intensity.label, self.slider_noise_brush_intensity.widget)
        sliders_layout.addRow(self.slider_denoise_size_x.label, self.slider_denoise_size_x.widget)
        sliders_layout.addRow(self.slider_denoise_size_y.label, self.slider_denoise_size_y.widget)
        sliders_layout.addRow(self.slider_denoise_attenuation.label, self.slider_denoise_attenuation.widget)
        sliders_layout.addRow(self.slider_denoise_subtraction.label, self.slider_denoise_subtraction.widget)

        dock.setWidget(sliders_widget)

        # Setup update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # Roughly 60 FPS

        self.denoiser = None
        self.params = None

        # Track mouse dragging
        self.is_dragging = False
        self.drag_button = None
        self.last_button = None

        self.history = History(self.api.create_empty_layer(512//8, 512//8))
        self.create_undo = True

        self.update_canvas_view()

    def _get_layer(self) -> Layer:
        return self.history.layer
    layer = property(fget=_get_layer)

    def _get_noise_brush_radius(self):
        return self.slider_noise_brush_radius.value
    noise_brush_radius = property(_get_noise_brush_radius)

    def _get_noise_brush_intensity(self):
        return self.slider_noise_brush_intensity.value
    noise_brush_intensity = property(_get_noise_brush_intensity)

    def _get_denoise_brush_size_x(self):
        return self.slider_denoise_size_x.value
    denoise_brush_size_x = property(_get_denoise_brush_size_x)

    def _get_denoise_brush_size_y(self):
        return self.slider_denoise_size_y.value
    denoise_brush_size_y = property(_get_denoise_brush_size_y)

    def _get_denoise_attenuation(self):
        return self.slider_denoise_attenuation.value
    denoise_attenuation = property(_get_denoise_attenuation)

    def _get_denoise_subtraction(self):
        return self.slider_denoise_subtraction.value
    denoise_subtraction = property(_get_denoise_subtraction)

    def on_clicked_new(self):
        """
        Opens a dialog with width/height entry fields, and [Create New] and [Cancel] buttons.
        If [Create New] is clicked, replaces the layer with an empty latent.
        """
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
            self.update_canvas_view()
            print(f"New canvas created with dimensions: {width}x{height}")

    def on_clicked_load(self):
        """
        Opens a file open dialogue. If a file is opened, replaces the layer with the image.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if not file_path:
            return  # User canceled

        try:
            # Load the image and convert to a diffusion canvas layer
            image = Image.open(file_path)
            self.history = History(self.api.create_layer_from_image(image))
            self.create_undo = True

            # Redraw the canvas
            self.update_canvas_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def on_clicked_save(self):
        """
        Opens a file save dialogue. If a destination file is chosen, saves the canvas to that file.
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

        if not file_path:
            return  # User canceled

        try:
            # Convert the latent space image back to a PIL image
            image = self.api.latent_to_image(self.layer.clean_latent, PIL.Image.Image)

            # Save the image to the chosen file path
            image.save(file_path)

            QMessageBox.information(self, "Save Successful", f"Image saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image: {e}")

    def add_params_to_palette(self, params):
        """
        Adds a new params object to the palette and creates a corresponding button.
        """
        self.params_palette.append(params)
        params_index = len(self.params_palette) - 1

        # Create a layout for the button and rename action
        button_layout = QHBoxLayout()

        # Params selection button
        button = QPushButton(f"Params {params_index + 1}", self)
        button.clicked.connect(lambda _, p=params: self.set_current_params(p))
        self.params_buttons.append(button)
        button_layout.addWidget(button)

        # Rename button
        rename_button = QPushButton("📝", self)
        rename_button.setFixedWidth(30)
        rename_button.clicked.connect(lambda: self.rename_params(params_index))
        button_layout.addWidget(rename_button)

        # Add to layout
        container = QWidget()
        container.setLayout(button_layout)
        self.params_layout.addWidget(container)

    def rename_params(self, index):
        """
        Opens a dialog to rename a params button.
        """
        current_name = self.params_buttons[index].text()
        dialog = RenameDialog(self, current_name=current_name)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_name = dialog.get_name()
            self.params_buttons[index].setText(new_name)

    def set_current_params(self, params):
        """
        Sets the current params object to be used for denoising.
        """
        self.params = params
        print(f"Selected params: {self.params}")

    def update_frame(self):
        denoiser_and_params = pop_intercepted()

        if denoiser_and_params is not None:
            self.api.set_denoiser(denoiser_and_params[0])
            new_params = denoiser_and_params[1]
            self.add_params_to_palette(new_params)

    def mousePressEvent(self, event):
        button = event.button()
        if button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.is_dragging = True
            self.drag_button = button
            self.last_button = button
            self.apply_brush(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.apply_brush(event)

    def mouseReleaseEvent(self, event):
        if event.button() == self.drag_button:
            self.is_dragging = False
            self.drag_button = None

        self.create_undo = True

    @torch.no_grad()
    def apply_brush(self, event: QMouseEvent):
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

        if self.drag_button == Qt.MouseButton.LeftButton:
            if self.create_undo:
                self.history.register_undo()
                self.create_undo = False
            self.api.draw_noise_dab(layer=self.layer,
                                    position_xy=normalized_position,
                                    pixel_radius=self.noise_brush_radius,
                                    noise_intensity=self.noise_brush_intensity)
            self.update_canvas_view()

        elif self.drag_button == Qt.MouseButton.RightButton:
            if self.create_undo:
                self.history.register_undo()
                self.create_undo = False
            self.api.draw_denoise_dab(layer=self.layer,
                                      params=self.params,
                                      position_xy=normalized_position,
                                      context_region_pixel_size_xy=(
                                          self.denoise_brush_size_x,
                                          self.denoise_brush_size_y
                                      ),
                                      attenuation_params=(self.denoise_attenuation, self.denoise_subtraction),
                                      time_budget=0.25)
            self.update_canvas_view()

    def keyPressEvent(self, event: QKeyEvent):
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
        self.history.undo(1)
        self.update_canvas_view()

    def redo(self):
        """
        Redo the previously undone action.
        """
        self.history.redo(1)
        self.update_canvas_view()

    def update_canvas_view(self):
        latent_to_show = (
            self.layer.noisy_latent
            if self.last_button == Qt.MouseButton.LeftButton
            else self.layer.clean_latent
        )

        # Convert tensor to QImage
        q_image = self.api.latent_to_image(latent_to_show, QImage)
        pixmap = QPixmap.fromImage(q_image)

        # Update the label pixmap
        self.label.setPixmap(pixmap)
        self.label.setFixedWidth(pixmap.width())
        self.label.setFixedHeight(pixmap.height())
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
