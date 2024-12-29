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


from PyQt6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QDockWidget,
    QPushButton,
    QScrollArea,
    QHBoxLayout,
    QDialog,
    QFileDialog,
    QMessageBox
)

from PyQt6.QtGui import QImage, QMouseEvent, QKeyEvent
from PyQt6.QtCore import Qt, QTimer

import PIL.Image
import torch
from PIL import Image

from sdwebui_interface import pop_intercepted, unfreeze_sd_webui
from diffusion_canvas_api import DiffusionCanvasAPI, latent_size_in_pixels
from layer import History, Layer

from ui_widgets import VerticalScrollArea
from ui_utils import ExceptionCatcher
from ui_params import ParamsWidget
from ui_dialogs import NewCanvasDialog
from ui_brushes import BaseBrushTool, NoiseBrushTool, LatentBrushTool, ShiftBrushTool
from ui_canvas import Canvas
from ui_latent_picker import LatentPicker

from common import *


class DiffusionCanvasWindow(QMainWindow):
    cpu_canvas_image_tensor: torch.Tensor
    cpu_canvas_q_image: QImage
    show_noisy: bool
    dirty_region_full: Bounds2D | None
    dirty_region_quick: Bounds2D | None
    history: History
    create_undo: bool

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diffusion Canvas")

        self.api = DiffusionCanvasAPI()

        self.setUpdatesEnabled(True)

        self.params_widgets: list[ParamsWidget] = []  # List to store params widgets

        self.canvas_view = Canvas(
            self.canvas_mousePressEvent,
            self.canvas_mouseMoveEvent,
            self.canvas_mouseReleaseEvent,
            self)

        # Our canvas is the central widget.
        self.setCentralWidget(self.canvas_view)

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
        self.params_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.params_widget.setLayout(self.params_layout)

        self.params_scroll_area = VerticalScrollArea()
        self.params_scroll_area.setWidget(self.params_widget)
        self.params_scroll_area.setWidgetResizable(True)

        # Dock for params palette
        params_dock = QDockWidget("Params Palette", self)
        params_dock.setWidget(self.params_scroll_area)
        params_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, params_dock)

        # Dock widget for tool buttons
        tool_dock = QDockWidget("Tools", self)
        tool_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        tool_widget = QWidget()
        tool_layout = QHBoxLayout(tool_widget)
        tool_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        tool_dock.setWidget(tool_widget)
        # Set the title bar to be vertical to save space.
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, tool_dock)
        # Dock widget for latent picker
        latent_picker_dock = QDockWidget("Latent Picker", self)
        latent_picker_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        latent_picker_widget = LatentPicker()
        latent_picker_dock.setWidget(latent_picker_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, latent_picker_dock)

        # Dock widget for tool settings
        tool_settings_dock = QDockWidget("Tool Settings", self)
        tool_settings_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, tool_settings_dock)

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
            on_tool_button_click=lambda: set_current_tool(self.latent_brush_tool),
            get_latent_value=lambda: latent_picker_widget.color_picker.get_current_latent_value(update_swatches=True),
            set_latent_value=lambda x: latent_picker_widget.color_picker.set_current_latent_value(x)
        )

        self.shift_brush_tool = ShiftBrushTool(
            api=self.api,
            tool_dock_layout=tool_layout,
            tool_settings_dock=tool_settings_dock,
            on_tool_button_click=lambda: set_current_tool(self.shift_brush_tool),
            get_latent_value=lambda: latent_picker_widget.color_picker.get_current_latent_value(update_swatches=True),
            set_latent_value=lambda x: latent_picker_widget.color_picker.set_current_latent_value(x)
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

        self.initialize_canvas(self.api.create_empty_layer(512//latent_size_in_pixels, 512//latent_size_in_pixels))

    def initialize_canvas(self, layer: Layer):
        self.show_noisy = False
        self.dirty_region_full: Bounds2D | None = None
        self.dirty_region_quick: Bounds2D | None = None

        image_size = (
            layer.clean_latent.shape[2] * latent_size_in_pixels,
            layer.clean_latent.shape[3] * latent_size_in_pixels
        )

        # Create a numpy array as the backing store
        numpy_buffer = np.zeros((image_size[0], image_size[1], 4), dtype=np.uint8)  # RGBA format

        # Create a QImage using the numpy buffer
        self.cpu_canvas_q_image = QImage(
            numpy_buffer.data,  # Pointer to the data
            image_size[1],  # width
            image_size[0],  # height
            QImage.Format.Format_RGB32  # Format
        )

        # Create a PyTorch tensor that shares the same memory
        self.cpu_canvas_image_tensor = torch.from_numpy(numpy_buffer)

        self.history = History(layer)
        self.create_undo = True

        # Update the display with the new canvas
        self.update_canvas_view(noisy=False, full=True)

    @staticmethod
    @torch.no_grad()
    def _get_cpu_image_tensor(tensor: torch.Tensor, add_alpha: bool = True):
        """
        Convert a tensor from the format used by stable diffusion VAE decoders
        to a format used by QImage.

        Args:
            tensor (torch.Tensor): Input tensor with shape (1, 3, height, width).
            add_alpha (bool): Whether to add a dummy alpha channel for QImage Format_RGB32.

        Returns:
            torch.Tensor: Tensor with shape (height, width, 4) or (height, width, 3).
        """
        # Ensure batch size is 1 and remove it
        assert tensor.shape[0] == 1, "Tensor batch size must be 1."

        tensor = tensor.squeeze(0)  # Shape: (RGB, height, width)

        # Rearrange channels to BGR if needed for QImage
        tensor = tensor[[2, 1, 0], :, :]  # Shape: (BGR, height, width)

        # Permute to (height, width, channels)
        tensor = tensor.permute(1, 2, 0)  # Shape: (height, width, BGR)

        # Map and clamp range (0, 1) to (0, 255)
        tensor = (tensor * 255).clamp(0, 255)

        # Add a dummy alpha channel if required
        if add_alpha:
            alpha_channel = torch.full(
                (tensor.shape[0], tensor.shape[1], 1),
                255,
                dtype=tensor.dtype,  # Match dtype of tensor (still likely float16 or float32)
                device=tensor.device
            )
            tensor = torch.cat((tensor, alpha_channel), dim=2)  # Shape: (height, width, BGRA)

        # Convert to uint8 on the CPU as the final step
        return tensor.to(dtype=torch.uint8, device='cpu')

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
        unfreeze_sd_webui()

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
                    int(np.maximum(np.ceil(width / latent_size_in_pixels), 1)),
                    int(np.maximum(np.ceil(height / latent_size_in_pixels), 1))
                )

                # Initialize the canvas with a new latent layer with the specified dimensions
                self.initialize_canvas(self.api.create_empty_layer(latent_size_xy[0], latent_size_xy[1]))

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
            layer = self.api.create_layer_from_image_tiled(
                image,
                max_tile_size_latents=64,
                margin_size_latents=4,
                overlap_size_latents=8
            )

            self.initialize_canvas(layer)

    def on_clicked_save(self):
        """
        Opens a file save dialogue. If a destination file is chosen, saves the canvas to that file.
        """
        with ExceptionCatcher(self, "Failed to save image"):
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")

            if not file_path:
                return  # User canceled

            # Convert the latent space image back to a PIL image
            image = self.api.latent_to_image_tiled(
                latent=self.layer.clean_latent,
                max_tile_size_latents=64,
                margin_size_latents=4,
                overlap_size_latents=8,
                full_quality=True,
                dest_type=PIL.Image.Image
            )

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
                delete_handler=lambda p: delete_handler(p),
                generate_handler=self.api.generate_image
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
                    self.update_canvas_view(full=True, region=None)

    def canvas_mousePressEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            button = event.button()
            if button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self.is_dragging = True
                self.drag_button = button
                self.apply_brush(event)

    def canvas_mouseMoveEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            if self.is_dragging:
                self.apply_brush(event)

    def canvas_mouseReleaseEvent(self, event):
        with ExceptionCatcher(self, "Failed to handle mouse event"):
            if event.button() == self.drag_button:
                self.is_dragging = False
                self.drag_button = None

            self.create_undo = True

    @torch.no_grad()
    def apply_brush(self, event: QMouseEvent):
        if self.current_tool is None:
            return

        normalized_position = self.canvas_view.coord_local_to_normalized(event.position())

        if self.current_tool.brush_stroke_will_modify(layer=self.layer,
                                                      params=self.params,
                                                      mouse_button=self.drag_button,
                                                      event=event,
                                                      normalized_mouse_coord=normalized_position):
            if self.create_undo:
                self.history.register_undo()
                self.create_undo = False

            result = self.current_tool.handle_brush_stroke(
                layer=self.layer,
                params=self.params,
                mouse_button=self.drag_button,
                event=event,
                normalized_mouse_coord=normalized_position
            )

            self.update_canvas_view(noisy=result.show_noisy, region=result.modified_bounds, full=False)

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

    def update_canvas_view(self, noisy: bool | None = None, region: Bounds2D | str | None = 'all', full: bool = True):

        from utils.time_utils import Timer

        if isinstance(noisy, bool):
            self.show_noisy = noisy

        latent_to_show = (
            self.layer.noisy_latent
            if self.show_noisy
            else self.layer.clean_latent
        )

        full_bounds = Bounds2D(
            x_bounds=(0, latent_to_show.shape[3]),
            y_bounds=(0, latent_to_show.shape[2])
        )

        if not full:
            self.showing_quick_preview = True
            self.full_preview_timer = 1000
        else:
            self.showing_quick_preview = False

        if region is not None:
            if region == 'all':
                region = full_bounds

            if isinstance(region, str):
                region = None

            if not isinstance(region, Bounds2D):
                region = None

        if isinstance(region, Bounds2D):
            self.dirty_region_quick = (
                region
                if self.dirty_region_quick is None
                else self.dirty_region_quick.get_encapsulated(region)
            )
            self.dirty_region_full = (
                region
                if self.dirty_region_full is None
                else self.dirty_region_full.get_encapsulated(region)
            )

        region_to_redraw = (
            self.dirty_region_full
            if full
            else self.dirty_region_quick
        )

        # No region to update.
        if region_to_redraw is None:
            return

        # Expand the region to account for VAE artifacts at edges
        region_to_redraw_with_padding = region_to_redraw.get_expanded(expand_amount_x=4, expand_amount_y=4)

        # Limit the regions to the actual canvas.
        region_to_redraw = region_to_redraw.get_clipped(full_bounds)
        region_to_redraw_with_padding = region_to_redraw_with_padding.get_clipped(full_bounds)

        if region_to_redraw_with_padding == full_bounds:
            with Timer("Decode"):
                decoded_tensor = self.api.latent_to_image_tiled(
                    latent_to_show,
                    max_tile_size_latents=64,
                    overlap_size_latents=8,
                    margin_size_latents=4,
                    full_quality=full,
                    dest_type=None
                )

            with Timer("Convert and Pass to CPU"):
                cpu_image_tensor = self._get_cpu_image_tensor(decoded_tensor)

            with Timer("Write to CPU buffer"):
                self.cpu_canvas_image_tensor[:, :, :] = cpu_image_tensor
        else:
            from diffusion_canvas_api import latent_size_in_pixels

            latent_view = latent_to_show[
                :, :,
                region_to_redraw_with_padding.y_bounds[0]:
                region_to_redraw_with_padding.y_bounds[1],
                region_to_redraw_with_padding.x_bounds[0]:
                region_to_redraw_with_padding.x_bounds[1],
            ]

            with Timer("Decode"):
                decoded_tensor = self.api.latent_to_image_tiled(
                    latent_view,
                    max_tile_size_latents=64,
                    overlap_size_latents=8,
                    margin_size_latents=4,
                    full_quality=full,
                    dest_type=None
                )

            # Trim the margins from the decoded view as they usually contain artifacts.
            relative_bounds = region_to_redraw_with_padding.transform_bounds(region_to_redraw)
            decoded_tensor = decoded_tensor[
                :, :,
                relative_bounds.y_bounds[0] * latent_size_in_pixels:
                relative_bounds.y_bounds[1] * latent_size_in_pixels,
                relative_bounds.x_bounds[0] * latent_size_in_pixels:
                relative_bounds.x_bounds[1] * latent_size_in_pixels,
            ]

            with Timer("Convert and Pass to CPU"):
                cpu_image_tensor = self._get_cpu_image_tensor(decoded_tensor)

            with Timer("Write to CPU buffer"):
                self.cpu_canvas_image_tensor[
                    region_to_redraw.y_bounds[0] * latent_size_in_pixels:
                    region_to_redraw.y_bounds[1] * latent_size_in_pixels,
                    region_to_redraw.x_bounds[0] * latent_size_in_pixels:
                    region_to_redraw.x_bounds[1] * latent_size_in_pixels,
                    :
                ] = cpu_image_tensor

        self.canvas_view.update_image(self.cpu_canvas_q_image)

        if full:
            self.dirty_region_full = None
            self.dirty_region_quick = None
        else:
            self.dirty_region_quick = None
