from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QDockWidget,
    QPushButton,
    QLayout,
    QFormLayout,
    QComboBox,
    QCheckBox
)

from PyQt6.QtCore import Qt

from diffusion_canvas_api import DiffusionCanvasAPI
from layer import Layer

from ui_utils import ExceptionCatcher
from ui_widgets import Slider

from common import *


def _add_slider(layout, slider):
    layout.addRow(slider.label, slider)


class BrushStrokeInfo:
    modified_bounds: Bounds2D | None
    show_noisy: bool

    def __init__(self,
                 modified_bounds: Bounds2D | None,
                 show_noisy: bool):
        self.modified_bounds = modified_bounds
        self.show_noisy = show_noisy


class BaseBrushTool:
    def __init__(self,
                 icon_emoji: str,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable):

        self._tool_settings_dock = tool_settings_dock
        self._extra_on_tool_button_click = on_tool_button_click

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
                            normalized_mouse_coord: (float, float)) -> BrushStrokeInfo:
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

        _add_slider(sliders_layout, self.slider_noise_brush_radius)
        _add_slider(sliders_layout, self.slider_noise_brush_intensity)
        _add_slider(sliders_layout, self.slider_denoise_brush_radius)
        _add_slider(sliders_layout, self.slider_denoise_size_x)
        _add_slider(sliders_layout, self.slider_denoise_size_y)
        _add_slider(sliders_layout, self.slider_denoise_attenuation)
        _add_slider(sliders_layout, self.slider_denoise_subtraction)
        _add_slider(sliders_layout, self.slider_denoise_bias)

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
                            normalized_mouse_coord: (float, float)) -> BrushStrokeInfo:
        ctrl_modifier = event.modifiers() & Qt.KeyboardModifier.ControlModifier  # Check if Ctrl is held

        if mouse_button == Qt.MouseButton.LeftButton:
            if ctrl_modifier:
                bounds = self._api.draw_remove_noise_dab(
                    layer=layer,
                    position_xy=normalized_mouse_coord,
                    pixel_radius=self.noise_brush_radius,
                    noise_intensity=self.noise_brush_intensity
                )
            else:
                bounds = self._api.draw_noise_dab(
                    layer=layer,
                    position_xy=normalized_mouse_coord,
                    pixel_radius=self.noise_brush_radius,
                    noise_intensity=self.noise_brush_intensity
                )
            return BrushStrokeInfo(
                show_noisy=True,
                modified_bounds=bounds
            )

        elif mouse_button == Qt.MouseButton.RightButton:
            bounds = self._api.draw_denoise_dab(
                layer=layer,
                params=params,
                position_xy=normalized_mouse_coord,
                pixel_radius=self.denoise_brush_radius,
                context_region_pixel_size_xy=(
                   self.denoise_context_size_x,
                   self.denoise_context_size_y
                ),
                attenuation_params=(self.denoise_attenuation, self.denoise_subtraction),
                noise_bias=2 ** self.denoise_bias,
                time_budget=0.1
            )
            return BrushStrokeInfo(
                show_noisy=False,
                modified_bounds=bounds
            )


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

        _add_slider(sliders_layout, self.slider_brush_radius)
        _add_slider(sliders_layout, self.slider_brush_opacity)

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
                            normalized_mouse_coord: (float, float)) -> BrushStrokeInfo:
        if mouse_button == Qt.MouseButton.LeftButton:
            bounds = self._api.draw_latent_dab(
                layer=layer,
                blend_mode=self.blend_mode,
                value=self.brush_value,
                position_xy=normalized_mouse_coord,
                pixel_radius=self.brush_radius,
                opacity=self.brush_opacity
            )
            return BrushStrokeInfo(
                show_noisy=False,
                modified_bounds=bounds
            )

        elif mouse_button == Qt.MouseButton.RightButton:
            self.brush_value = self._api.get_average_latent(
                layer=layer,
                position_xy=normalized_mouse_coord,
                pixel_radius=self.brush_radius
            )
            return BrushStrokeInfo(
                show_noisy=False,
                modified_bounds=None
            )


class ShiftBrushTool(BaseBrushTool):
    brush_value: tuple[float, float, float, float]
    _color_mode: bool
    _color_shift_mode_widgets: list[tuple[str, QWidget]]
    _shift_mode_widgets: list[tuple[str, QWidget]]

    def __init__(self,
                 api,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable):

        super().__init__("✨️️", tool_dock_layout, tool_settings_dock, on_tool_button_click)
        self._api = api
        self.brush_value = (0.0, 0.0, 0.0, 0.0)
        self._color_mode = True

    @staticmethod
    def _add_widget(
            widget_list: list[tuple[str, QWidget]],
            label: str,
            widget: QWidget) -> None:
        widget_list.append((label, widget))

    @staticmethod
    def _add_slider(
            widget_list: list[tuple[str, QWidget]],
            slider: Slider) -> None:
        widget_list.append((slider.label, slider))

    def _create_tool_settings_dock_widget(self) -> QWidget:
        self._color_mode = True
        self._color_shift_mode_widgets: list[tuple[str, QWidget]] = []
        self._shift_mode_widgets: list[tuple[str, QWidget]] = []

        sliders_widget = QWidget()
        self._sliders_layout = QFormLayout(sliders_widget)

        ## Create widgets for settings common to both modes
        self.paint_mode_toggle = QCheckBox()
        self.paint_mode_toggle.setCheckState(Qt.CheckState.Checked)
        self.paint_mode_toggle.stateChanged.connect(self._on_toggle_paint_mode)

        self.slider_brush_radius = Slider(
            "Brush Radius (px)",
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
        self.slider_denoise_steps = Slider(
            "Denoise Steps",
            5,
            (1, 10),
            1
        )
        self.slider_denoise_bias = Slider(
            "Denoise Bias",
            0,
            (-1, 1.0),
            0.01
        )

        ## Widgets for paint shift mode
        # Add a blend mode dropdown:
        self.blend_mode_combo = QComboBox()
        self.blend_mode_combo.addItems(
            [mode.name for mode in DiffusionCanvasAPI.BlendMode]
        )
        self.slider_brush_opacity = Slider(
            "Brush Opacity",
            1.0,
            (0.0, 1.0),
            0.01
        )
        self.slider_noise_scale = Slider(
            "Noise Intensity Scale",
            1.0,
            (0.0, 2.0),
            0.01
        )
        self.slider_noise_radius_scale = Slider(
            "Noise Radius Scale",
            1.5,
            (0, 3),
            0.1
        )

        ## Widgets for shift only mode
        self.slider_noise_brush_intensity = Slider(
            "Noise Intensity",
            3.0,
            (0.0, 5.0),
            0.01
        )

        ## Create the color shift mode list.
        # Add toggle between modes.
        self._add_widget(self._color_shift_mode_widgets, "Paint Mode", self.paint_mode_toggle)
        # Add a help message:
        self._add_widget(self._color_shift_mode_widgets, "Left click", QLabel("Paint and Shift"))
        self._add_widget(self._color_shift_mode_widgets, "Right click", QLabel("Select Color with Eyedropper"))
        # Add sliders:
        self._add_widget(self._color_shift_mode_widgets, "Blend Mode", self.blend_mode_combo)
        self._add_slider(self._color_shift_mode_widgets, self.slider_brush_radius)
        self._add_slider(self._color_shift_mode_widgets, self.slider_brush_opacity)
        self._add_slider(self._color_shift_mode_widgets, self.slider_noise_scale)
        self._add_slider(self._color_shift_mode_widgets, self.slider_noise_radius_scale)
        self._add_slider(self._color_shift_mode_widgets, self.slider_denoise_size_x)
        self._add_slider(self._color_shift_mode_widgets, self.slider_denoise_size_y)
        self._add_slider(self._color_shift_mode_widgets, self.slider_denoise_steps)
        self._add_slider(self._color_shift_mode_widgets, self.slider_denoise_bias)

        ## Create the shift mode list.
        # Add toggle between modes.
        self._add_widget(self._shift_mode_widgets, "Paint Mode", self.paint_mode_toggle)
        # Add a help message:
        self._add_widget(self._shift_mode_widgets, "Left click", QLabel("Shift"))
        # Add sliders:
        self._add_slider(self._shift_mode_widgets, self.slider_brush_radius)
        self._add_slider(self._shift_mode_widgets, self.slider_noise_brush_intensity)
        self._add_slider(self._shift_mode_widgets, self.slider_denoise_size_x)
        self._add_slider(self._shift_mode_widgets, self.slider_denoise_size_y)
        self._add_slider(self._shift_mode_widgets, self.slider_denoise_steps)
        self._add_slider(self._shift_mode_widgets, self.slider_denoise_bias)

        self._populate_dock_layout()

        return sliders_widget

    def _populate_dock_layout(self):
        # Clear the layout object, keeping the widgets, if any items are in it.
        while self._sliders_layout.count() > 0:
            item = self._sliders_layout.takeAt(0)  # Remove the item from the layout

            if item.widget():
                item.widget().setParent(None)  # Detach the widget from the layout

        widgets_list: list[tuple[str, QWidget]] = (
            self._color_shift_mode_widgets
            if self._color_mode
            else self._shift_mode_widgets
        )

        for label, widget in widgets_list:
            self._sliders_layout.addRow(label, widget)

    def _on_toggle_paint_mode(self):
        self._color_mode = self.paint_mode_toggle.isChecked()
        self._populate_dock_layout()

    def _get_blend_mode(self) -> DiffusionCanvasAPI.BlendMode:
        index = self.blend_mode_combo.currentIndex()
        try:
            return DiffusionCanvasAPI.BlendMode(index)
        except ValueError:
            return DiffusionCanvasAPI.BlendMode.Blend  # Default to Blend if invalid
    blend_mode = property(_get_blend_mode)

    def _get_brush_radius(self):
        return self.slider_brush_radius.value
    brush_radius = property(_get_brush_radius)

    def _get_brush_opacity(self):
        return self.slider_brush_opacity.value
    brush_opacity = property(_get_brush_opacity)

    def _get_noise_radius_scale(self):
        return self.slider_noise_radius_scale.value
    noise_radius_scale = property(_get_noise_radius_scale)

    def _get_noise_intensity(self):
        return self.slider_noise_brush_intensity.value
    noise_intensity = property(_get_noise_intensity)

    def _get_noise_scale(self):
        return self.slider_noise_scale.value
    noise_scale = property(_get_noise_scale)

    def _get_denoise_size_x(self):
        return self.slider_denoise_size_x.value
    denoise_size_x = property(_get_denoise_size_x)

    def _get_denoise_size_y(self):
        return self.slider_denoise_size_y.value
    denoise_size_y = property(_get_denoise_size_y)

    def _get_denoise_steps(self) -> int:
        return int(self.slider_denoise_steps.value)
    denoise_steps = property(_get_denoise_steps)

    def _get_denoise_bias(self):
        return self.slider_denoise_bias.value
    denoise_bias = property(_get_denoise_bias)

    def brush_stroke_will_modify(self,
                                 layer: Layer,
                                 params,
                                 mouse_button: Qt.MouseButton,
                                 event: QMouseEvent,
                                 normalized_mouse_coord: (float, float)) -> bool:
        if self._color_mode:
            return mouse_button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
        else:
            return mouse_button == Qt.MouseButton.LeftButton

    def handle_brush_stroke(self,
                            layer: Layer,
                            params,
                            mouse_button: Qt.MouseButton,
                            event: QMouseEvent,
                            normalized_mouse_coord: (float, float)) -> BrushStrokeInfo:
        if self._color_mode:
            if mouse_button == Qt.MouseButton.LeftButton:
                bounds = self._api.draw_color_shift_dab(
                    params=params,
                    layer=layer,
                    blend_mode=self.blend_mode,
                    value=self.brush_value,
                    position_xy=normalized_mouse_coord,
                    pixel_radius=self.brush_radius,
                    opacity=self.brush_opacity,
                    noise_pixel_radius=self.brush_radius * self.noise_radius_scale,
                    noise_scale=self.noise_scale,
                    noise_bias=2 ** self.denoise_bias,
                    context_region_pixel_size_xy=(
                        self.denoise_size_x,
                        self.denoise_size_y
                    ),
                    denoise_steps=self.denoise_steps
                )
                return BrushStrokeInfo(
                    show_noisy=False,
                    modified_bounds=bounds
                )

            elif mouse_button == Qt.MouseButton.RightButton:
                self.brush_value = self._api.get_average_latent(
                    layer=layer,
                    position_xy=normalized_mouse_coord,
                    pixel_radius=self.brush_radius
                )
                return BrushStrokeInfo(
                    show_noisy=False,
                    modified_bounds=None
                )
        else:
            if mouse_button == Qt.MouseButton.LeftButton:
                bounds = self._api.draw_shift_dab(
                    params=params,
                    layer=layer,
                    position_xy=normalized_mouse_coord,
                    pixel_radius=self.brush_radius,
                    noise_intensity=self.noise_intensity,
                    noise_bias=2 ** self.denoise_bias,
                    context_region_pixel_size_xy=(
                        self.denoise_size_x,
                        self.denoise_size_y
                    ),
                    denoise_steps=self.denoise_steps
                )
                return BrushStrokeInfo(
                    show_noisy=False,
                    modified_bounds=bounds
                )
