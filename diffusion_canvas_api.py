# diffusion_canvas_api.py - Contains functions used by the UI

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.
# diffusion_canvas_api.py - Contains functions used by the UI

import math
import PIL.Image
import numpy as np
import torch
import texture_convert as conv
from brushes import Brushes
from layer import Layer
from sdwebui_interface import encode_image, decode_image, denoise
import modules.shared as shared
from time_utils import TimeBudget


def _center_crop_for_sd(image: PIL.Image.Image, rounding: int):
    new_width = int(np.floor(image.width / rounding)) * rounding
    new_height = int(np.floor(image.height / rounding)) * rounding

    diff = (image.width - new_width, image.height - new_height)
    corner = (diff[0] // 2, diff[1] // 2)

    image = image.crop((corner[0], corner[1], corner[0] + new_width, corner[1] + new_height))
    return image


def _get_cropped_1d(center: int, cropped_size: int, original_size: int) -> tuple[int, int]:
    """
    Args:
        center (int): Desired center for the crop
        cropped_size (int): Desired cropping size along this dimension
        original_size (int): Original size

    Returns: Starting and ending indices (tuple(int, int))
    """
    if cropped_size >= original_size:
        return 0, original_size

    start = center - cropped_size // 2
    if start < 0:
        start = 0

    end = start + cropped_size
    if end > original_size:
        end = original_size
        start = end - cropped_size

    return start, end


def _position_to_latent_coords(position_xy: tuple[float, float],
                               tensor: torch.Tensor) -> (float, float, float):
    tensor_width = tensor.shape[3]
    tensor_height = tensor.shape[2]
    latent_x = position_xy[0] * tensor_width
    latent_y = position_xy[1] * tensor_height
    latent_y_flipped = (1 - position_xy[1]) * tensor_height

    return latent_x, latent_y, latent_y_flipped


class DiffusionCanvasAPI:
    def __init__(self):
        self._brushes = Brushes()
        self._denoiser = None

    @staticmethod
    def create_layer_from_image(image: PIL.Image.Image) -> Layer:
        image = _center_crop_for_sd(image, 8).convert(mode="RGB")
        image_tensor = conv.convert(image, torch.Tensor).to(shared.device)
        encoded = encode_image(image_tensor)
        noise_amp_shape = list(encoded.shape)
        noise_amp_shape[1] = 1
        noise_amp_shape = tuple(noise_amp_shape)
        noise_amplitude = torch.zeros(noise_amp_shape, dtype=encoded.dtype, device=encoded.device)
        return Layer(
            encoded,
            encoded.clone(),
            noise_amplitude
        )

    def create_empty_layer(self, latent_width: int, latent_height: int):
        clean = torch.zeros(size=(1, 4, latent_height, latent_width), dtype=torch.float32, device=shared.device)
        noisy = torch.zeros(size=(1, 4, latent_height, latent_width), dtype=torch.float32, device=shared.device)
        amp = torch.zeros(size=(1, 1, latent_height, latent_width), dtype=torch.float32, device=shared.device)
        return Layer(
            clean,
            noisy,
            amp
        )

    @staticmethod
    def latent_to_image(latent: torch.Tensor, dest_type):
        decoded = decode_image(latent)
        converted = conv.convert(decoded, dest_type)
        return converted

    def set_denoiser(self, denoiser):
        self._denoiser = denoiser

    @torch.no_grad()
    def draw_noise_dab(self,
                       layer: Layer, 
                       position_xy: tuple[float, float],
                       pixel_radius: float,
                       noise_intensity: float):

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )

        amplitude = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            pixel_radius / 8,
            (1, 1, 1, 1),
            opacity=noise_intensity,
            mode="add"
        ).to(layer.noise_amplitude.device)

        layer.add_noise(amplitude)

    @torch.no_grad()
    def draw_denoise_dab(self,
                         params,
                         layer: Layer,
                         position_xy: tuple[float, float],
                         context_region_pixel_size_xy: tuple[int, int],
                         attenuation_params: tuple[float, float],
                         time_budget: float = 0.25):

        if self._denoiser is None:
            print("No denoiser! Press [Generate] to send the denoiser to Diffusion Canvas.")
            return

        if params is None:
            print("No params! Press [Generate] to send denoising parameters to Diffusion Canvas.")
            return

        latent_size_xy = (
            np.maximum(int(math.ceil(context_region_pixel_size_xy[0] / 8)), 8),
            np.maximum(int(math.ceil(context_region_pixel_size_xy[1] / 8)), 8)
        )

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )

        y_bounds = _get_cropped_1d(int(latent_y), latent_size_xy[1], layer.clean_latent.shape[2])
        x_bounds = _get_cropped_1d(int(latent_x), latent_size_xy[0], layer.clean_latent.shape[3])

        '''mask = self.brushes.draw_dab(
            torch.zeros_like(self.layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            self.noise_brush_radius,
            (1, 1, 1, 1),
            opacity=1,
            mode="blend"
        ).to(shared.device)'''

        for _ in TimeBudget(time_budget):
            layer.step(lambda x, y: denoise(self._denoiser, x, y, params),
                       lambda x: np.maximum(x * (1.0 - attenuation_params[0])
                                            - attenuation_params[1], 0),
                       brush_mask=None,  # mask,
                       y_bounds=y_bounds,
                       x_bounds=x_bounds)