from typing import Callable

from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
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
import torch

from ui_utils import ExceptionCatcher
from ui_widgets import Slider, HelpBox

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
        help_box = HelpBox([
            ("Left click", "Add noise"),
            ("Ctrl+Left click", "Remove noise"),
            ("Right click", "Denoise"),
        ])
        sliders_layout.addRow(help_box)

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
    def __init__(self,
                 api,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable,
                 get_source_latent: Callable[[tuple[int, int]], torch.Tensor],
                 set_latent_value: Callable[[tuple[float, float, float, float]], None]):

        super().__init__("🖌️️", tool_dock_layout, tool_settings_dock, on_tool_button_click)
        self._api = api
        self._get_source_latent = get_source_latent
        self._set_latent_value = set_latent_value

    def _create_tool_settings_dock_widget(self) -> QWidget:
        sliders_widget = QWidget()
        sliders_layout = QFormLayout(sliders_widget)

        # Add a help message:
        help_box = HelpBox([
            ("Left click", "Paint color"),
            ("Right click", "Select color with eyedropper"),
        ])
        sliders_layout.addRow(help_box)

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
                source_tensor=self._get_source_latent((layer.clean_latent.shape[3], layer.clean_latent.shape[2])),
                position_xy=normalized_mouse_coord,
                pixel_radius=self.brush_radius,
                opacity=self.brush_opacity
            )
            return BrushStrokeInfo(
                show_noisy=False,
                modified_bounds=bounds
            )

        elif mouse_button == Qt.MouseButton.RightButton:
            self._set_latent_value(self._api.get_average_latent(
                layer=layer,
                position_xy=normalized_mouse_coord,
                pixel_radius=self.brush_radius
            ))
            return BrushStrokeInfo(
                show_noisy=False,
                modified_bounds=None
            )


class ShiftBrushTool(BaseBrushTool):
    _color_mode: bool
    _blend_transitions: bool
    _color_shift_mode_widgets: list[tuple[str | None, QWidget]]
    _shift_mode_widgets: list[tuple[str | None, QWidget]]

    def __init__(self,
                 api,
                 tool_dock_layout: QLayout,
                 tool_settings_dock: QDockWidget,
                 on_tool_button_click: callable,
                 get_source_latent: Callable[[tuple[int, int]], torch.Tensor],
                 set_latent_value: Callable[[tuple[float, float, float, float]], None]):

        super().__init__("✨️️", tool_dock_layout, tool_settings_dock, on_tool_button_click)
        self._api = api
        self._color_mode = True

        self._get_source_latent = get_source_latent
        self._set_latent_value = set_latent_value

    @staticmethod
    def _add_widget(
            widget_list: list[tuple[str | None, QWidget]],
            label: str | None,
            widget: QWidget) -> None:
        widget_list.append((label, widget))

    @staticmethod
    def _add_slider(
            widget_list: list[tuple[str | None, QWidget]],
            slider: Slider) -> None:
        widget_list.append((slider.label, slider))

    def _create_tool_settings_dock_widget(self) -> QWidget:
        self._color_mode = True
        self._blend_transitions = False
        self._color_shift_mode_widgets: list[tuple[str | None, QWidget]] = []
        self._shift_mode_widgets: list[tuple[str | None, QWidget]] = []

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

        self.blend_transitions_toggle = QCheckBox()
        self.blend_transitions_toggle.setCheckState(Qt.CheckState.Unchecked)
        self.blend_transitions_toggle.stateChanged.connect(self._on_toggle_transitions_mode)

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
        help_box = HelpBox([
            ("Left click", "Paint and shift"),
            ("Right click", "Select color with eyedropper"),
        ])
        self._add_widget(self._color_shift_mode_widgets, None, help_box)
        # Add sliders:
        self._add_widget(self._color_shift_mode_widgets, "Blend Mode", self.blend_mode_combo)
        self._add_widget(self._color_shift_mode_widgets, "Transitions Only", self.blend_transitions_toggle)
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
        help_box = HelpBox([
            ("Left click", "Shift")
        ])
        self._add_widget(self._shift_mode_widgets, None, help_box)
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

        widgets_list: list[tuple[str | None, QWidget]] = (
            self._color_shift_mode_widgets
            if self._color_mode
            else self._shift_mode_widgets
        )

        for label, widget in widgets_list:
            if label is None:
                self._sliders_layout.addRow(widget)
            else:
                self._sliders_layout.addRow(label, widget)

    def _on_toggle_paint_mode(self):
        self._color_mode = self.paint_mode_toggle.isChecked()
        self._populate_dock_layout()

    def _on_toggle_transitions_mode(self):
        self._blend_transitions = self.blend_transitions_toggle.isChecked()

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
                    blend_transitions=self._blend_transitions,
                    source_tensor=self._get_source_latent((layer.clean_latent.shape[3], layer.clean_latent.shape[2])),
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
                self._set_latent_value(self._api.get_average_latent(
                    layer=layer,
                    position_xy=normalized_mouse_coord,
                    pixel_radius=self.brush_radius
                ))
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
