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

import torch
from diffusion_canvas_api import DiffusionCanvasAPI
from layer import Layer

from ui_utils import ExceptionCatcher
from ui_widgets import Slider

from common import *

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


class LatentDabContentProvider:

    def __init__(self):
        pass

    def get_latent(self,
                   affected_bounds_latents: Bounds2D,
                   dtype: torch.dtype,
                   device: torch.device):
        ...


class SolidDabContentProvider(LatentDabContentProvider):
    latent_value = tuple[float, float, float, float]

    def __init__(self, latent_value: tuple[float, float, float, float]):
        super().__init__()
        self.latent_value = latent_value

    def get_latent(self,
                   affected_bounds_latents: Bounds2D,
                   dtype: torch.dtype,
                   device: torch.device):
        # Create a tensor with the same shape as dimensions as the affected boundary,
        # where each channel is set to value[channel]
        value_tensor = torch.tensor(
            self.latent_value,
            dtype=dtype,
            device=device
        ).view(1, -1, affected_bounds_latents.span[1], affected_bounds_latents.span[0])
        return value_tensor


class ImageContentProvider(LatentDabContentProvider):
    content: torch.Tensor

    def __init__(self, content: torch.Tensor):
        super().__init__()
        # TODO: Check that the tensor follows the (batch=1, channels=4, height, width) format.
        self.content = content

    def get_latent(self,
                   affected_bounds_latents: Bounds2D,
                   dtype: torch.dtype,
                   device: torch.device):
        # Return a view of the current latent.
        # TODO: Handle tiling, stretching, offsets, etc.
        return self.content[
            :, :,
            affected_bounds_latents.y_bounds[0]:affected_bounds_latents.y_bounds[1],
            affected_bounds_latents.x_bounds[0]:affected_bounds_latents.x_bounds[1]
        ]