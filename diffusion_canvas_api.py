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
import torch
import utils.texture_convert as conv
from brushes import Brushes
from layer import Layer
from sdwebui_interface import encode_image, decode_image, denoise
import modules.shared as shared
from utils.time_utils import TimeBudget
from enum import Enum
import tilemap as tm
from common import *


latent_size_in_pixels: int = 8
latent_channel_count: int = 4
color_channel_count: int = 3


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


def _get_brush_bounds(latent_xy: tuple[float, float],
                      latent_radius: float,
                      tensor_size_xy: tuple[float, float]) -> Bounds2D:
    x_min = int(np.clip(
        np.floor(latent_xy[0] - latent_radius),
        a_min=0,
        a_max=tensor_size_xy[0]
    ))
    x_max = int(np.clip(
        np.ceil(latent_xy[0] + latent_radius),
        a_min=0,
        a_max=tensor_size_xy[0]
    ))

    y_min = int(np.clip(
        np.floor(latent_xy[1] - latent_radius),
        a_min=0,
        a_max=tensor_size_xy[1]
    ))
    y_max = int(np.clip(
        np.ceil(latent_xy[1] + latent_radius),
        a_min=0,
        a_max=tensor_size_xy[1]
    ))

    return Bounds2D(
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max)
    )


def cubic_interpolation(
        t: float,
        start_value: float,
        end_value: float,
        start_steepness: float,
        end_steepness: float
) -> float:
    """
    Computes f(t) for a cubic polynomial f that satisfies:
      f(0)   = start_value
      f(1)   = end_value
      f'(0)  = start_steepness
      f'(1)  = end_steepness

    The function is sampled at t in the range [0, 1].
    """
    # Boundary conditions
    d = start_value
    c = start_steepness

    # Temporary helpers
    A = (end_value - start_value) - c
    B = end_steepness - c

    # Solve for a and b
    a = B - 2 * A
    b = 3 * A - B

    # Evaluate the cubic polynomial at t
    return a * t ** 3 + b * t ** 2 + c * t + d


class DiffusionCanvasAPI:
    class BlendMode(Enum):
        Blend = 0
        Add = 1
        Merge = 2

    def __init__(self):
        self._brushes = Brushes()
        self._denoiser = None

    @staticmethod
    @torch.no_grad()
    def create_layer_from_image(image: PIL.Image.Image) -> Layer:
        image = _center_crop_for_sd(image, latent_size_in_pixels).convert(mode="RGB")
        image_tensor = conv.convert(image, torch.Tensor).to(shared.device)
        encoded = DiffusionCanvasAPI.image_to_latent(image_tensor)
        noise_amp_shape = list(encoded.shape)
        noise_amp_shape[1] = 1
        noise_amp_shape = tuple(noise_amp_shape)
        noise_amplitude = torch.zeros(noise_amp_shape, dtype=encoded.dtype, device=encoded.device)
        return Layer(
            encoded,
            encoded.clone(),
            noise_amplitude
        )

    @staticmethod
    @torch.no_grad()
    def create_layer_from_image_tiled(image: PIL.Image.Image,
                                      max_tile_size_latents: int,
                                      margin_size_latents: int,
                                      overlap_size_latents: int) -> Layer:
        image = _center_crop_for_sd(image, latent_size_in_pixels).convert(mode="RGB")
        image_tensor = conv.convert(image, torch.Tensor).to(shared.device)
        encoded = DiffusionCanvasAPI.image_to_latent_tiled(
            image_tensor,
            max_tile_size_latents,
            margin_size_latents,
            overlap_size_latents
        )
        noise_amp_shape = list(encoded.shape)
        noise_amp_shape[1] = 1
        noise_amp_shape = tuple(noise_amp_shape)
        noise_amplitude = torch.zeros(noise_amp_shape, dtype=encoded.dtype, device=encoded.device)
        return Layer(
            encoded,
            encoded.clone(),
            noise_amplitude
        )

    @torch.no_grad()
    def create_empty_layer(self, latent_width: int, latent_height: int):
        clean = torch.zeros(size=(1, latent_channel_count, latent_height, latent_width), dtype=torch.float32, device=shared.device)
        noisy = torch.zeros(size=(1, latent_channel_count, latent_height, latent_width), dtype=torch.float32, device=shared.device)
        amp = torch.zeros(size=(1, 1, latent_height, latent_width), dtype=torch.float32, device=shared.device)
        return Layer(
            clean,
            noisy,
            amp
        )

    @staticmethod
    @torch.no_grad()
    def image_to_latent(image_tensor: torch.Tensor):
        return encode_image(image_tensor)

    @staticmethod
    @torch.no_grad()
    def image_to_latent_tiled(image_tensor: torch.Tensor,
                              max_tile_size_latents: int,
                              margin_size_latents: int,
                              overlap_size_latents: int):
        latent_width = image_tensor.shape[3] // latent_size_in_pixels
        latent_height = image_tensor.shape[2] // latent_size_in_pixels

        tiles = tm.get_or_create_tilemap(
            size_latents=(latent_width, latent_height),
            max_tile_size_latents=max_tile_size_latents,
            margin_size_latents=margin_size_latents,
            overlap_size_latents=overlap_size_latents,
            pixels_per_latent=latent_size_in_pixels,
            dtype=image_tensor.dtype,
            device=image_tensor.device
        )

        latent_tensor_shape = (1, latent_channel_count, latent_height, latent_width)
        latent_tensor = torch.zeros(size=latent_tensor_shape, dtype=image_tensor.dtype, device=image_tensor.device)

        for tile_index in tiles.enumerate_tiles():
            view = tiles.get_encode_view(image_tensor, tile_index)
            view = DiffusionCanvasAPI.image_to_latent(view)
            tiles.write_encoded_tile(tile_index, view, latent_tensor)

        return latent_tensor

    @staticmethod
    @torch.no_grad()
    def latent_to_image(latent: torch.Tensor, full_quality: bool, dest_type):
        decoded = decode_image(latent, full_quality)

        if dest_type is None:
            return decoded

        converted = conv.convert(decoded, dest_type)
        return converted

    @staticmethod
    @torch.no_grad()
    def latent_to_image_tiled(latent: torch.Tensor,
                              max_tile_size_latents: int,
                              margin_size_latents: int,
                              overlap_size_latents: int,
                              full_quality: bool,
                              dest_type):

        tiles = tm.get_or_create_tilemap(
            size_latents=(latent.shape[3], latent.shape[2]),
            max_tile_size_latents=max_tile_size_latents,
            margin_size_latents=margin_size_latents,
            overlap_size_latents=overlap_size_latents,
            pixels_per_latent=latent_size_in_pixels,
            dtype=latent.dtype,
            device=latent.device
        )

        image_tensor_shape = (1, color_channel_count, latent.shape[2]*latent_size_in_pixels, latent.shape[3]*latent_size_in_pixels)
        image_tensor = torch.zeros(size=image_tensor_shape, dtype=latent.dtype, device=latent.device)

        for tile_index in tiles.enumerate_tiles():
            view = tiles.get_decode_view(latent, tile_index)
            view = DiffusionCanvasAPI.latent_to_image(view, full_quality, dest_type=None)
            tiles.write_decoded_tile(tile_index, view, image_tensor)

        if dest_type is None:
            return image_tensor

        converted = conv.convert(image_tensor, dest_type)
        return converted

    def set_denoiser(self, denoiser):
        self._denoiser = denoiser

    @torch.no_grad()
    def generate_image(self, width: int, height: int, steps: int, params: any) -> torch.Tensor:
        width = int(np.maximum(1, np.ceil(width / latent_size_in_pixels)))
        height = int(np.maximum(1, np.ceil(height / latent_size_in_pixels)))

        sigma = 20
        latent = torch.randn(size=(1, latent_channel_count, height, width), dtype=torch.float32, device=shared.device) * sigma

        for i in range(steps):
            sigma_to_remove = (i+1) / steps
            denoised = denoise(denoiser=self._denoiser, latent=latent, sigma=sigma, params=params)

            if i+1 == steps:
                latent = denoised
                sigma = 0
            else:
                latent = denoised * sigma_to_remove + latent * (1-sigma_to_remove)
                sigma = sigma * (1-sigma_to_remove)

        return decode_image(latent, full_quality=True)

    @staticmethod
    @torch.no_grad()
    def _get_blended(layer: Layer,
                     value: tuple[float, float, float, float],
                     alpha: torch.Tensor,
                     blend_mode: 'DiffusionCanvasAPI.BlendMode'):
        """
        Args:
            layer: Layer to which the blending procedure is applied.
            value: The current latent color to blend.
            alpha: Opacity mask.
            blend_mode: The current blend mode.

        Returns:
            A modified copy of the latent with the current blending operation applied.
        """

        if blend_mode == DiffusionCanvasAPI.BlendMode.Add:
            solid = layer.create_solid_latent(value)
            return solid * alpha + layer.clean_latent

        elif blend_mode == DiffusionCanvasAPI.BlendMode.Merge:
            average = layer.get_average_latent(alpha)
            difference = tuple(v - a for v, a in zip(value, average))
            solid = layer.create_solid_latent(difference)
            return solid * alpha + layer.clean_latent

        else:
            solid = layer.create_solid_latent(value)
            return solid * alpha + layer.clean_latent * (1-alpha)

    @torch.no_grad()
    def draw_latent_dab(self,
                        layer: Layer,
                        blend_mode: 'DiffusionCanvasAPI.BlendMode',
                        value: tuple[float, float, float, float],
                        position_xy: tuple[float, float],
                        pixel_radius: float,
                        opacity: float) -> Bounds2D:

        if blend_mode not in DiffusionCanvasAPI.BlendMode:
            blend_mode = DiffusionCanvasAPI.BlendMode.Blend

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        latent_radius = pixel_radius / latent_size_in_pixels

        alpha = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            latent_radius,
            (1, 1, 1, 1),
            opacity=opacity,
            mode="blend"
        ).to(layer.clean_latent.device)

        layer.replace_clean_latent(self._get_blended(layer, value, alpha, blend_mode))

        return _get_brush_bounds(
            (latent_x, latent_y),
            latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )

    @torch.no_grad()
    def get_average_latent(self,
                           layer: Layer,
                           position_xy: tuple[float, float],
                           pixel_radius: float):

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        latent_radius = pixel_radius / latent_size_in_pixels

        weight = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            latent_radius,
            (1, 1, 1, 1),
            opacity=1,
            mode="blend"
        ).to(layer.clean_latent.device)

        return layer.get_average_latent(weight)

    @torch.no_grad()
    def draw_noise_dab(self,
                       layer: Layer, 
                       position_xy: tuple[float, float],
                       pixel_radius: float,
                       noise_intensity: float) -> Bounds2D:

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        latent_radius = pixel_radius / latent_size_in_pixels

        amplitude = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            latent_radius,
            (1, 1, 1, 1),
            opacity=noise_intensity,
            mode="add"
        ).to(layer.noise_amplitude.device)

        layer.add_noise(amplitude)

        return _get_brush_bounds(
            (latent_x, latent_y),
            latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )

    @torch.no_grad()
    def draw_remove_noise_dab(self,
                              layer: Layer,
                              position_xy: tuple[float, float],
                              pixel_radius: float,
                              noise_intensity: float) -> Bounds2D:

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        latent_radius = pixel_radius / latent_size_in_pixels

        amplitude = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            latent_radius,
            (1, 1, 1, 1),
            opacity=noise_intensity,
            mode="add"
        ).to(layer.noise_amplitude.device)

        layer.remove_noise(amplitude)

        return _get_brush_bounds(
            (latent_x, latent_y),
            latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )

    @torch.no_grad()
    def draw_denoise_dab(self,
                         params,
                         layer: Layer,
                         position_xy: tuple[float, float],
                         pixel_radius: float,
                         context_region_pixel_size_xy: tuple[int, int],
                         attenuation_params: tuple[float, float],
                         noise_bias: float,
                         time_budget: float = 0.25) -> Bounds2D | None:

        if self._denoiser is None:
            print("No denoiser! Press [Generate] to send the denoiser to Diffusion Canvas.")
            return None

        if params is None:
            print("No params! Press [Generate] to send denoising parameters to Diffusion Canvas.")
            return None

        latent_size_xy = (
            np.maximum(int(math.ceil(context_region_pixel_size_xy[0] / latent_size_in_pixels)), latent_size_in_pixels),
            np.maximum(int(math.ceil(context_region_pixel_size_xy[1] / latent_size_in_pixels)), latent_size_in_pixels)
        )

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        latent_radius = pixel_radius / latent_size_in_pixels

        y_bounds = _get_cropped_1d(int(latent_y), latent_size_xy[1], layer.clean_latent.shape[2])
        x_bounds = _get_cropped_1d(int(latent_x), latent_size_xy[0], layer.clean_latent.shape[3])

        mask = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            latent_radius,
            (1, 1, 1, 1),  # The Y, Z, and W components are ignored.
            opacity=1,
            mode="blend"
        ).to(shared.device)

        for _ in TimeBudget(time_budget):
            layer.step(lambda x, y: denoise(self._denoiser, x, y, params),
                       lambda x: np.maximum(x * (1.0 - attenuation_params[0])
                                            - attenuation_params[1], 0),
                       brush_mask=mask,
                       noise_bias=noise_bias,
                       y_bounds=y_bounds,
                       x_bounds=x_bounds)

        brush_bounds = _get_brush_bounds(
            (latent_x, latent_y),
            latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )

        context_bounds = Bounds2D(x_bounds=x_bounds, y_bounds=y_bounds)

        return brush_bounds.get_clipped(context_bounds)

    @torch.no_grad()
    def draw_shift_dab(self,
                       params,
                       layer: Layer,
                       position_xy: tuple[float, float],
                       pixel_radius: float,
                       noise_intensity: float,
                       noise_bias: float,
                       context_region_pixel_size_xy: tuple[int, int],
                       denoise_steps: int) -> Bounds2D | None:

        if self._denoiser is None:
            print("No denoiser! Press [Generate] to send the denoiser to Diffusion Canvas.")
            return None

        if params is None:
            print("No params! Press [Generate] to send denoising parameters to Diffusion Canvas.")
            return None

        latent_size_xy = (
            np.maximum(int(math.ceil(context_region_pixel_size_xy[0] / latent_size_in_pixels)), latent_size_in_pixels),
            np.maximum(int(math.ceil(context_region_pixel_size_xy[1] / latent_size_in_pixels)), latent_size_in_pixels)
        )

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        noise_latent_radius = pixel_radius / latent_size_in_pixels

        # Define the bounds.
        y_bounds = _get_cropped_1d(int(latent_y), latent_size_xy[1], layer.clean_latent.shape[2])
        x_bounds = _get_cropped_1d(int(latent_x), latent_size_xy[0], layer.clean_latent.shape[3])
        noise_bounds = _get_brush_bounds(
            (latent_x, latent_y),
            noise_latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )
        context_bounds = Bounds2D(x_bounds=x_bounds, y_bounds=y_bounds)
        noise_bounds = noise_bounds.get_clipped(context_bounds)

        noise_mask = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            noise_latent_radius,
            (1, 1, 1, 1),  # The Y, Z, and W components are ignored.
            opacity=1,
            mode="blend"
        ).to(shared.device)

        #    2.a. Return early if the total noise is zero.
        if noise_intensity <= 0:
            return None

        layer.add_noise(noise_mask * noise_intensity)

        # 3. Denoise the region.
        noise_start = 1
        for i in range(denoise_steps):
            t_end = (i+1) / denoise_steps
            noise_end = cubic_interpolation(t_end,
                                            start_value=1,
                                            end_value=0,
                                            start_steepness=-1,
                                            end_steepness=0)

            attenuation = noise_end / noise_start if noise_start != 0 else 0
            if math.isnan(attenuation) or math.isinf(attenuation):
                attenuation = 0

            layer.step(lambda x, y: denoise(self._denoiser, x, y, params),
                       lambda x: x * attenuation,
                       brush_mask=None,
                       noise_bias=noise_bias,
                       y_bounds=y_bounds,
                       x_bounds=x_bounds)

            noise_start = noise_end

        return noise_bounds

    @torch.no_grad()
    def draw_color_shift_dab(self,
                             params,
                             layer: Layer,
                             blend_mode: 'DiffusionCanvasAPI.BlendMode',
                             value: tuple[float, float, float, float],
                             position_xy: tuple[float, float],
                             pixel_radius: float,
                             opacity: float,
                             noise_pixel_radius: float,
                             noise_scale: float,
                             noise_bias: float,
                             context_region_pixel_size_xy: tuple[int, int],
                             denoise_steps: int) -> Bounds2D | None:

        if self._denoiser is None:
            print("No denoiser! Press [Generate] to send the denoiser to Diffusion Canvas.")
            return None

        if params is None:
            print("No params! Press [Generate] to send denoising parameters to Diffusion Canvas.")
            return None

        latent_size_xy = (
            np.maximum(int(math.ceil(context_region_pixel_size_xy[0] / latent_size_in_pixels)), latent_size_in_pixels),
            np.maximum(int(math.ceil(context_region_pixel_size_xy[1] / latent_size_in_pixels)), latent_size_in_pixels)
        )

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )
        draw_latent_radius = pixel_radius / latent_size_in_pixels
        noise_latent_radius = noise_pixel_radius / latent_size_in_pixels

        # Define the bounds.
        y_bounds = _get_cropped_1d(int(latent_y), latent_size_xy[1], layer.clean_latent.shape[2])
        x_bounds = _get_cropped_1d(int(latent_x), latent_size_xy[0], layer.clean_latent.shape[3])
        draw_bounds = _get_brush_bounds(
            (latent_x, latent_y),
            draw_latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )
        noise_bounds = _get_brush_bounds(
            (latent_x, latent_y),
            noise_latent_radius,
            (layer.clean_latent.shape[3], layer.clean_latent.shape[2])
        )
        context_bounds = Bounds2D(x_bounds=x_bounds, y_bounds=y_bounds)
        noise_bounds = noise_bounds.get_clipped(context_bounds)
        total_bounds = draw_bounds.get_encapsulated(noise_bounds)

        paint_mask = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            draw_latent_radius,
            (1, 1, 1, 1),  # The Y, Z, and W components are ignored.
            opacity=1,
            mode="blend"
        ).to(shared.device)

        noise_mask = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            noise_latent_radius,
            (1, 1, 1, 1),  # The Y, Z, and W components are ignored.
            opacity=1,
            mode="blend"
        ).to(shared.device)

        # 1. Apply the blend procedure to the affected area,
        #    and get the amplitude of difference introduced by the change.
        blended = self._get_blended(layer, value, paint_mask * opacity, blend_mode)

        #    1.a. Calculate the amplitude of the change from the old to the new.
        difference = layer.clean_latent - blended

        #    1.b. Compute the average norm of each latent
        difference = torch.norm(
            input=difference,
            p=2,
            dim=1)
        difference = difference.mean().squeeze().item()

        #    1.c. Scale the difference by the mask.
        average_mask_value = paint_mask.mean().squeeze().item()
        if average_mask_value > 0:
            difference /= average_mask_value

        #    1.d. Replace the latent with the blended.
        layer.replace_clean_latent(blended)

        # 2. Add noise overtop of the latent proportional to the difference introduced.
        noise_amplitude = difference * noise_scale

        #    2.a. Return early if the total noise is zero.
        if noise_amplitude <= 0:
            return draw_bounds

        layer.add_noise(noise_mask * noise_amplitude)

        # 3. Denoise the region.
        noise_start = 1
        for i in range(denoise_steps):
            t_end = (i+1) / denoise_steps
            noise_end = cubic_interpolation(t_end,
                                            start_value=1,
                                            end_value=0,
                                            start_steepness=-1,
                                            end_steepness=0)

            attenuation = noise_end / noise_start if noise_start != 0 else 0
            if math.isnan(attenuation) or math.isinf(attenuation):
                attenuation = 0

            layer.step(lambda x, y: denoise(self._denoiser, x, y, params),
                       lambda x: x * attenuation,
                       brush_mask=None,
                       noise_bias=noise_bias,
                       y_bounds=y_bounds,
                       x_bounds=x_bounds)

            noise_start = noise_end

        return total_bounds
