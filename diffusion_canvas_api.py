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
import utils.texture_convert as conv
from brushes import Brushes
from layer import Layer
from sdwebui_interface import encode_image, decode_image, denoise
import modules.shared as shared
from utils.time_utils import TimeBudget
from enum import Enum


latent_size_in_pixels: int = 8


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


cache = {}


def compute_tile_count(total_size, max_tile_size, margin_size, overlap_size) -> int:
    extra = margin_size * 2 + overlap_size

    def solve():
        if extra <= 0:
            return total_size / max_tile_size

        term1 = extra + max_tile_size

        term2 = (
                + extra * extra
                + 2 * extra * max_tile_size
                + max_tile_size * max_tile_size
                - 4 * extra * total_size
        )

        if term2 < 0:
            return total_size / max_tile_size

        term2 = term2 ** 0.5
        term3 = 2 * extra

        solution = (term1 - term2) / term3

        if solution < 0:
            return total_size / max_tile_size

        return solution

    return int(np.ceil(solve()))


def get_corrected_bounds_1d(bounds: tuple[int, int]) -> tuple[int, int]:
    if bounds[0] >= bounds[1]:
        average = (bounds[0] + bounds[1]) / 2
        return (
            int(np.floor(average)),
            int(np.ceil(average))
        )
    else:
        return bounds


def get_expanded_bounds_1d(bounds: tuple[int, int], expand_amount: tuple[int, int] | int) -> tuple[int, int]:
    if expand_amount == 0:
        return bounds

    if isinstance(expand_amount, int):
        return get_corrected_bounds_1d((
            bounds[0] - expand_amount,
            bounds[1] + expand_amount
        ))
    else:
        return get_corrected_bounds_1d((
            bounds[0] - expand_amount[0],
            bounds[1] + expand_amount[1]
        ))


def get_clipped_coord_1d(coord: int, clip: tuple[int, int]) -> int:
    if coord < clip[0]:
        coord = clip[0]
    if coord > clip[1]:
        coord = clip[1]
    return coord


def get_clipped_bounds_1d(bounds: tuple[int, int], clip: tuple[int, int]) -> tuple[int, int]:
    min_clipping_range = (clip[0], clip[1] - 1)
    max_clipping_range = (clip[0] + 1, clip[1])

    return (
        get_clipped_coord_1d(bounds[0], min_clipping_range),
        get_clipped_coord_1d(bounds[1], max_clipping_range)
    )


def get_offset_bounds_1d(bounds: tuple[int, int], offset: int) -> tuple[int, int]:
    return (
        bounds[0] + offset,
        bounds[1] + offset
    )


class Bounds:
    x_bounds: tuple[int, int]
    y_bounds: tuple[int, int]

    def __init__(self, x_bounds: tuple[int, int], y_bounds: tuple[int, int]):
        self.x_bounds = get_corrected_bounds_1d(x_bounds)
        self.y_bounds = get_corrected_bounds_1d(y_bounds)

    def get_expanded(self, expand_amount_x: tuple[int, int] | int, expand_amount_y: tuple[int, int] | int):
        return Bounds(
            get_expanded_bounds_1d(self.x_bounds, expand_amount_x),
            get_expanded_bounds_1d(self.y_bounds, expand_amount_y)
        )

    def get_clipped(self, clip: 'Bounds') -> 'Bounds':
        return Bounds(
            get_clipped_bounds_1d(self.x_bounds, clip.x_bounds),
            get_clipped_bounds_1d(self.y_bounds, clip.y_bounds)
        )

    def transform_bounds(self, other: 'Bounds'):
        return Bounds(
            get_offset_bounds_1d(other.x_bounds, -self.x_bounds[0]),
            get_offset_bounds_1d(other.y_bounds, -self.y_bounds[0]),
        )


class TileMap:
    latent_tile_bounds_x: list[tuple[int, int]]
    latent_tile_bounds_y: list[tuple[int, int]]
    latent_tile_bounds_with_margins_x: list[tuple[int, int]]
    latent_tile_bounds_with_margins_y: list[tuple[int, int]]
    tile_count: tuple[int, int]

    def __init__(self,
                 latent_size: tuple[int, int],
                 max_tile_size_latents: int,
                 margin_size_latents: int,
                 overlap_size_latents: int):

        self.tile_count = (
            compute_tile_count(
                latent_size[0],
                max_tile_size_latents,
                margin_size_latents,
                overlap_size_latents
            ),
            compute_tile_count(
                latent_size[1],
                max_tile_size_latents,
                margin_size_latents,
                overlap_size_latents
            ),
        )

        def compute_boundaries(size, divisions) -> list[int]:
            d: list[int] = []
            gap_size = size / divisions
            for _i in range(divisions):
                d.append(int(np.round(_i * gap_size)))
            d.append(size)
            return d

        boundaries_x: list[int] = compute_boundaries(latent_size[0], self.tile_count[0])
        boundaries_y: list[int] = compute_boundaries(latent_size[1], self.tile_count[1])

        half_overlap = overlap_size_latents / 2
        min_expand = int(np.floor(half_overlap))
        max_expand = int(np.ceil(half_overlap))

        self.latent_tile_bounds_x = []
        self.latent_tile_bounds_with_margins_x = []
        for i in range(self.tile_count[0]):
            self.latent_tile_bounds_x.append((
                boundaries_x[i] - (min_expand if (i > 0) else 0),
                boundaries_x[i + 1] + (max_expand if ((i + 1) < self.tile_count[0]) else 0)
            ))
            self.latent_tile_bounds_with_margins_x.append((
                boundaries_x[i] - ((min_expand + margin_size_latents) if (i > 0) else 0),
                boundaries_x[i + 1] + ((max_expand + margin_size_latents) if ((i + 1) < self.tile_count[0]) else 0)
            ))

        self.latent_tile_bounds_y = []
        self.latent_tile_bounds_with_margins_y = []
        for i in range(self.tile_count[1]):
            self.latent_tile_bounds_y.append((
                boundaries_y[i] - (min_expand if (i > 0) else 0),
                boundaries_y[i + 1] + (max_expand if ((i + 1) < self.tile_count[1]) else 0)
            ))
            self.latent_tile_bounds_with_margins_y.append((
                boundaries_y[i] - ((min_expand + margin_size_latents) if (i > 0) else 0),
                boundaries_y[i + 1] + ((max_expand + margin_size_latents) if ((i + 1) < self.tile_count[1]) else 0)
            ))

    def get_bounds(self, tile_coord: tuple[int, int], include_margins: bool):
        bounds_array_x = (
            self.latent_tile_bounds_with_margins_x
            if include_margins
            else self.latent_tile_bounds_x
        )
        bounds_array_y = (
            self.latent_tile_bounds_with_margins_y
            if include_margins
            else self.latent_tile_bounds_y
        )

        bounds_x = bounds_array_x[tile_coord[0]]
        bounds_y = bounds_array_y[tile_coord[1]]

        return Bounds(bounds_x, bounds_y)

    def calculate_weights(self, coord: float, vertical: bool) -> list[float]:
        bounds_list = self.latent_tile_bounds_y if vertical else self.latent_tile_bounds_x

        def enforce_01(value: float, default: float):
            if math.isnan(value):
                return default
            elif value > 1:
                return 1
            elif value < 0:
                return 0
            else:
                return value

        def calculate_raw_weight(tile_index):
            bounds = bounds_list[tile_index]

            start = bounds[0]
            end = bounds[1]
            middle = (start + end) * 0.5

            if coord < start:
                return 0
            if coord > end:
                return 0

            if coord < middle:
                value = (coord - start) / (middle - start)
            else:
                value = 1 - ((coord - middle) / (end - middle))

            return enforce_01(value, 0.5)

        weight_total: float = 0
        weights_list: list[float] = []

        for i in range(len(bounds_list)):
            weight = calculate_raw_weight(i)
            weight_total += weight
            weights_list.append(weight)

        default_value = 1 / len(bounds_list)
        for i in range(len(bounds_list)):
            weights_list[i] = (
                default_value
                if weight_total == 0
                else enforce_01(weights_list[i] / weight_total, default_value)
            )

        return weights_list


def create_mask_tensors(tilemap: TileMap, latent_size: int, pixel_size: int, vertical: bool, dtype, device) \
        -> list[torch.Tensor]:
    tile_count = tilemap.tile_count[1] if vertical else tilemap.tile_count[0]

    arrays: list[np.array] = []

    for _ in range(tile_count):
        arrays.append(np.empty(pixel_size, dtype=np.float32))

    pixel_to_latent = latent_size / pixel_size

    for pixel_index in range(pixel_size):
        # We add a slight offset because we want to sample in the middle of the latent.
        weights = tilemap.calculate_weights((pixel_index + 0.5) * pixel_to_latent, vertical)
        for tile_index in range(tile_count):
            arrays[tile_index][pixel_index] = weights[tile_index]

    tensors: list[torch.Tensor] = []

    for i in range(tile_count):
        tensor = torch.from_numpy(arrays[i])

        if vertical:
            tensor = tensor.view(1, 1, -1, 1)
        else:
            tensor = tensor.view(1, 1, 1, -1)

        tensor = tensor.to(dtype=dtype, device=device)

        tensors.append(tensor)

    return tensors


def get_or_create(_latent_size: tuple[int, int],
                  dtype, device,
                  _max_tile_size_latents: int,
                  _margin_size_latents: int,
                  _overlap_size_latents: int):
    key = (_latent_size, dtype, device, _max_tile_size_latents, _margin_size_latents, _overlap_size_latents)

    if key in cache:
        return cache[key]

    _tilemap = TileMap(
        latent_size=_latent_size,
        max_tile_size_latents=_max_tile_size_latents,
        margin_size_latents=_margin_size_latents,
        overlap_size_latents=_overlap_size_latents
    )

    # Broadcastable masks.
    _h_pixel_masks = create_mask_tensors(
        tilemap=_tilemap,
        latent_size=_latent_size[0],
        pixel_size=_latent_size[0] * latent_size_in_pixels,
        vertical=False,
        dtype=dtype,
        device=device,
    )

    _v_pixel_masks = create_mask_tensors(
        tilemap=_tilemap,
        latent_size=_latent_size[1],
        pixel_size=_latent_size[1] * latent_size_in_pixels,
        vertical=True,
        dtype=dtype,
        device=device,
    )

    value = (
        _tilemap,
        _h_pixel_masks,
        _v_pixel_masks
    )

    cache[key] = value
    return value


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

    @torch.no_grad()
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

        latent_size = (latent.shape[3], latent.shape[2])
        image_tensor_shape = (1, 3, latent_size[1] * latent_size_in_pixels, latent_size[0] * latent_size_in_pixels)
        image_tensor = torch.zeros(size=image_tensor_shape, dtype=latent.dtype, device=latent.device)

        tilemap, h_pixel_masks, v_pixel_masks = get_or_create(
            _latent_size=latent_size,
            _max_tile_size_latents=max_tile_size_latents,
            _margin_size_latents=margin_size_latents,
            _overlap_size_latents=overlap_size_latents,
            dtype=latent.dtype,
            device=latent.device
        )

        for x in range(tilemap.tile_count[0]):
            for y in range(tilemap.tile_count[1]):
                bounds_with_margins = tilemap.get_bounds((x, y), include_margins=True)

                latent_view = latent[
                    :, :,
                    bounds_with_margins.y_bounds[0]:bounds_with_margins.y_bounds[1],
                    bounds_with_margins.x_bounds[0]:bounds_with_margins.x_bounds[1]
                ]

                decoded_view = DiffusionCanvasAPI.latent_to_image(
                    latent_view,
                    full_quality,
                    dest_type=None
                )

                bounds_without_margins = tilemap.get_bounds((x, y), include_margins=False)
                relative_bounds = bounds_with_margins.transform_bounds(bounds_without_margins)
                trimmed_decoded_view = decoded_view[
                    :, :,

                    relative_bounds.y_bounds[0] * latent_size_in_pixels:
                    relative_bounds.y_bounds[1] * latent_size_in_pixels,

                    relative_bounds.x_bounds[0] * latent_size_in_pixels:
                    relative_bounds.x_bounds[1] * latent_size_in_pixels
                ]

                trimmed_decoded_view *= h_pixel_masks[x][
                    :, :,

                    :,

                    bounds_without_margins.x_bounds[0] * latent_size_in_pixels:
                    bounds_without_margins.x_bounds[1] * latent_size_in_pixels
                ]

                trimmed_decoded_view *= v_pixel_masks[y][
                    :, :,

                    bounds_without_margins.y_bounds[0] * latent_size_in_pixels:
                    bounds_without_margins.y_bounds[1] * latent_size_in_pixels,

                    :
                ]

                image_tensor[
                    :, :,

                    bounds_without_margins.y_bounds[0] * latent_size_in_pixels:
                    bounds_without_margins.y_bounds[1] * latent_size_in_pixels,

                    bounds_without_margins.x_bounds[0] * latent_size_in_pixels:
                    bounds_without_margins.x_bounds[1] * latent_size_in_pixels
                ] += trimmed_decoded_view

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
        latent = torch.randn(size=(1, 4, height, width), dtype=torch.float32, device=shared.device) * sigma

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

    @torch.no_grad()
    def draw_latent_dab(self,
                        layer: Layer,
                        blend_mode: 'DiffusionCanvasAPI.BlendMode',
                        value: tuple[float, float, float, float],
                        position_xy: tuple[float, float],
                        pixel_radius: float,
                        opacity: float):

        if blend_mode not in DiffusionCanvasAPI.BlendMode:
            blend_mode = DiffusionCanvasAPI.BlendMode.Blend

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )

        alpha = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            pixel_radius / latent_size_in_pixels,
            (1, 1, 1, 1),
            opacity=opacity,
            mode="blend"
        ).to(layer.clean_latent.device)

        if blend_mode == DiffusionCanvasAPI.BlendMode.Add:
            solid = layer.create_solid_latent(value)
            new_clean_latent = solid * alpha + layer.clean_latent
            layer.replace_clean_latent(new_clean_latent)

        elif blend_mode == DiffusionCanvasAPI.BlendMode.Merge:
            average = layer.get_average_latent(alpha)
            difference = tuple(v - a for v, a in zip(value, average))
            solid = layer.create_solid_latent(difference)
            new_clean_latent = solid * alpha + layer.clean_latent
            layer.replace_clean_latent(new_clean_latent)
            pass

        else:
            solid = layer.create_solid_latent(value)
            new_clean_latent = solid * alpha + layer.clean_latent * (1-alpha)
            layer.replace_clean_latent(new_clean_latent)

    @torch.no_grad()
    def get_average_latent(self,
                           layer: Layer,
                           position_xy: tuple[float, float],
                           pixel_radius: float):

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )

        weight = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            pixel_radius / latent_size_in_pixels,
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
                       noise_intensity: float):

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )

        amplitude = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            pixel_radius / latent_size_in_pixels,
            (1, 1, 1, 1),
            opacity=noise_intensity,
            mode="add"
        ).to(layer.noise_amplitude.device)

        layer.add_noise(amplitude)

    @torch.no_grad()
    def draw_remove_noise_dab(self,
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
            pixel_radius / latent_size_in_pixels,
            (1, 1, 1, 1),
            opacity=noise_intensity,
            mode="add"
        ).to(layer.noise_amplitude.device)

        layer.remove_noise(amplitude)

    @torch.no_grad()
    def draw_denoise_dab(self,
                         params,
                         layer: Layer,
                         position_xy: tuple[float, float],
                         pixel_radius: float,
                         context_region_pixel_size_xy: tuple[int, int],
                         attenuation_params: tuple[float, float],
                         noise_bias: float,
                         time_budget: float = 0.25):

        if self._denoiser is None:
            print("No denoiser! Press [Generate] to send the denoiser to Diffusion Canvas.")
            return

        if params is None:
            print("No params! Press [Generate] to send denoising parameters to Diffusion Canvas.")
            return

        latent_size_xy = (
            np.maximum(int(math.ceil(context_region_pixel_size_xy[0] / latent_size_in_pixels)), latent_size_in_pixels),
            np.maximum(int(math.ceil(context_region_pixel_size_xy[1] / latent_size_in_pixels)), latent_size_in_pixels)
        )

        latent_x, latent_y, latent_y_flipped = _position_to_latent_coords(
            position_xy,
            layer.noise_amplitude
        )

        y_bounds = _get_cropped_1d(int(latent_y), latent_size_xy[1], layer.clean_latent.shape[2])
        x_bounds = _get_cropped_1d(int(latent_x), latent_size_xy[0], layer.clean_latent.shape[3])

        mask = self._brushes.draw_dab(
            torch.zeros_like(layer.noise_amplitude),
            (latent_x, latent_y_flipped),
            pixel_radius / latent_size_in_pixels,
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
