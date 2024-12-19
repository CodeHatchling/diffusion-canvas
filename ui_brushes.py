from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QDockWidget,
    QPushButton,
    QLayout, QFormLayout, QComboBox
)

from PyQt6.QtCore import Qt

from diffusion_canvas_api import DiffusionCanvasAPI
from layer import Layer

from ui_utils import ExceptionCatcher
from ui_widgets import Slider


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
                                 event: QMouseEvent,
                                 normalized_mouse_coord: (float, float)) -> bool:
        ...

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            event: QMouseEvent,
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
        sliders_layout.addRow("Ctrl+Left click", QLabel("Remove noise"))
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
            (8, 2048),
            8
        )
        self.slider_denoise_size_y = Slider(
            "Context Height (px)",
            1024,
            (8, 2048),
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
                                 event: QMouseEvent,
                                 normalized_mouse_coord: (float, float)) -> bool:
        return mouse_button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            event: QMouseEvent,
                            normalized_mouse_coord: (float, float)):
        ctrl_modifier = event.modifiers() & Qt.KeyboardModifier.ControlModifier  # Check if Ctrl is held

        if mouse_button == Qt.MouseButton.LeftButton:
            if ctrl_modifier:
                self._api.draw_remove_noise_dab(layer=layer,
                                                position_xy=normalized_mouse_coord,
                                                pixel_radius=self.noise_brush_radius,
                                                noise_intensity=self.noise_brush_intensity)
            else:
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
                                 event: QMouseEvent,
                                 normalized_mouse_coord: (float, float)) -> bool:
        return mouse_button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            event: QMouseEvent,
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