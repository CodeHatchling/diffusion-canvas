import math
import torch
from typing import Iterator
from common import *


def compute_tile_count(total_size, max_tile_size, margin_size, overlap_size) -> int:
    """
    Args:
        total_size: The total size of the space to divide into tiles.
        max_tile_size: The maximum size of tiles allowed (with margins and overlap).
        margin_size: The size of the tiles' margins.
        overlap_size: The size of overlap between tiles.

    Returns:
        The number of tiles required to tile a region with "total_size", given the
        margins, overlap, and maximum total size for tiles.
    """

    # TODO: Constrain the margin and overlap arguments within a range whereby a solution exists.

    if total_size <= 0:
        raise ValueError(f"Argument \"total_size\" ({total_size}) must be greater than 0.")
    if margin_size < 0:
        margin_size = 0
    if overlap_size < 0:
        overlap_size = 0
    if max_tile_size < 1:
        max_tile_size = 1

    def round_up_to_int(value: float) -> int:
        return int(np.ceil(value))

    # This is the amount of "extra" latents between tiles (overlaps and margins).
    extra = margin_size * 2 + overlap_size

    # If there are no "extra" latents per tile, we can solve this simply:
    # max_tile_size = (total_size / tile_count)
    if extra <= 0:
        return round_up_to_int(total_size / max_tile_size)

    # Solution to the following equation when solving for tile_count:
    # max_tile_size = (total_size / tile_count) + (margin_size_latents * 2 + overlap_size_latents) * (tile_count - 1)
    term1 = extra + max_tile_size

    term2 = (
            + extra * extra
            + 2 * extra * max_tile_size
            + max_tile_size * max_tile_size
            - 4 * extra * total_size
    )

    if term2 < 0:
        return round_up_to_int(total_size / max_tile_size)

    term2 = term2 ** 0.5
    term3 = 2 * extra

    solution = (term1 - term2) / term3

    if solution < 0:
        return round_up_to_int(total_size / max_tile_size)

    return round_up_to_int(solution)


class TileGeometry:
    size_latents: tuple[int, int]
    latent_tile_bounds_x: list[tuple[int, int]]
    latent_tile_bounds_y: list[tuple[int, int]]
    latent_tile_bounds_with_margins_x: list[tuple[int, int]]
    latent_tile_bounds_with_margins_y: list[tuple[int, int]]
    tile_count: tuple[int, int]

    def __init__(self,
                 size_latents: tuple[int, int],
                 max_tile_size_latents: int,
                 margin_size_latents: int,
                 overlap_size_latents: int):
        self.size_latents = size_latents

        self.tile_count = (
            compute_tile_count(
                size_latents[0],
                max_tile_size_latents,
                margin_size_latents,
                overlap_size_latents
            ),
            compute_tile_count(
                size_latents[1],
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

        boundaries_x: list[int] = compute_boundaries(size_latents[0], self.tile_count[0])
        boundaries_y: list[int] = compute_boundaries(size_latents[1], self.tile_count[1])

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

    def get_bounds(self, tile_coord: tuple[int, int], include_margins: bool) -> Bounds2D:
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

        return Bounds2D(bounds_x, bounds_y)

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

    def enumerate_tiles(self) -> Iterator[tuple[int, int]]:
        for x in range(self.tile_count[0]):
            for y in range(self.tile_count[1]):
                yield x, y  # Yield each (x, y) combination


class Tilemap:
    _geometry: TileGeometry
    _pixel_size: tuple[int, int]
    _pixels_per_latent: int
    _encode_masks: tuple[list[torch.Tensor], list[torch.Tensor]] | None
    _decode_masks: tuple[list[torch.Tensor], list[torch.Tensor]] | None
    _device: torch.device
    _dtype: torch.dtype

    def __init__(self,
                 size_latents: tuple[int, int],
                 max_tile_size_latents: int,
                 margin_size_latents: int,
                 overlap_size_latents: int,
                 pixels_per_latent: int,
                 device: torch.device,
                 dtype: torch.dtype):

        self._geometry = TileGeometry(
            size_latents,
            max_tile_size_latents,
            margin_size_latents,
            overlap_size_latents
        )

        self._pixels_per_latent = pixels_per_latent
        self._pixel_size = (
                size_latents[0] * pixels_per_latent,
                size_latents[1] * pixels_per_latent
        )
        self._device = device
        self._dtype = dtype

        # These are generated when requested.
        self._encode_masks = None
        self._decode_masks = None

    def _get_encode_masks(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._encode_masks is not None:
            return self._encode_masks

        masks_horizontal = self._create_masks_1d(self._geometry.size_latents[0], vertical=False)
        masks_vertical = self._create_masks_1d(self._geometry.size_latents[1], vertical=True)

        self._encode_masks = (masks_horizontal, masks_vertical)
        return self._encode_masks

    def _get_decode_masks(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._decode_masks is not None:
            return self._decode_masks

        masks_horizontal = self._create_masks_1d(self._pixel_size[0], vertical=False)
        masks_vertical = self._create_masks_1d(self._pixel_size[1], vertical=True)

        self._decode_masks = (masks_horizontal, masks_vertical)
        return self._decode_masks

    def _create_masks_1d(self, dest_size: int, vertical: bool) -> list[torch.Tensor]:
        axis: int = 1 if vertical else 0
        tile_count = self._geometry.tile_count[axis]
        size_latents = self._geometry.size_latents[axis]
        coordinate_scale = size_latents / dest_size

        tensors: list[torch.Tensor] = []
        tensor_shape: tuple[int] = (dest_size,)

        for _ in range(tile_count):
            tensors.append(torch.empty(size=tensor_shape, dtype=self._dtype, device='cpu'))

        # Populate the tensors with weight values.
        for dest_index in range(dest_size):
            # We add a half-element offset because we want to sample in the middle of the element,
            # and the geometry describes boundaries at the lower edges.
            weights = self._geometry.calculate_weights((dest_index + 0.5) * coordinate_scale, vertical)
            for tile_index in range(tile_count):
                tensors[tile_index][dest_index] = weights[tile_index]

        # Expand the masks to match the shape of tensors used in sd.webui.
        for i in range(tile_count):
            if vertical:
                tensors[i] = tensors[i].view(1, 1, -1, 1).to(self._device)
            else:
                tensors[i] = tensors[i].view(1, 1, 1, -1).to(self._device)

        return tensors

    def enumerate_tiles(self) -> Iterator[tuple[int, int]]:
        return self._geometry.enumerate_tiles()

    def get_encode_view(self, image_tensor: torch.Tensor, tile_index: tuple[int, int]):

        # Check the input dimensionality.
        expected_shape_length = 4
        input_shape = image_tensor.shape
        assert len(input_shape) == expected_shape_length, \
            (f"Expected {expected_shape_length}-dimensional latent tensor, "
             f"was given tensor with dimensionality of {len(input_shape)}.")

        # Check height and width.
        expected_shape = (
            input_shape[0],       # Batch Count
            input_shape[1],       # Channel Count
            self._pixel_size[1],  # Height
            self._pixel_size[0]   # Width
        )
        assert image_tensor.shape == expected_shape, \
            (f"Argument tensor \"image_tensor\" has unexpected shape of {image_tensor.shape}"
             f"expected shape of {expected_shape}.")

        bounds_with_margins = self._geometry.get_bounds(tile_index, include_margins=True)

        image_view = image_tensor[
            :, :,
            bounds_with_margins.y_bounds[0] * self._pixels_per_latent:
            bounds_with_margins.y_bounds[1] * self._pixels_per_latent,
            bounds_with_margins.x_bounds[0] * self._pixels_per_latent:
            bounds_with_margins.x_bounds[1] * self._pixels_per_latent
        ]

        return image_view

    def write_encoded_tile(self,
                           tile_index: tuple[int, int],
                           encoded_latent_tensor: torch.Tensor,
                           dest_latent_tensor: torch.Tensor):

        assert (0 <= tile_index[0] < self._geometry.tile_count[0] and
                0 <= tile_index[1] < self._geometry.tile_count[1]), \
               (f"Argument \"tile_index\" value {tile_index} is out-of-range, "
                f"valid range is (0, 0) to {self._geometry.tile_count}")

        # Check the inputs' dimension count.
        expected_shape_length = 4
        assert len(encoded_latent_tensor.shape) == expected_shape_length, \
            (f"Expected {expected_shape_length}-dimensional latent tensor, "
             f"argument \"encoded_latent_tensor\" has dimensionality of {len(encoded_latent_tensor.shape)}.")
        assert len(dest_latent_tensor.shape) == expected_shape_length, \
            (f"Expected {expected_shape_length}-dimensional latent tensor, "
             f"argument \"dest_latent_tensor\" has dimensionality of {len(dest_latent_tensor.shape)}.")

        bounds_with_margins = self._geometry.get_bounds(tile_index, include_margins=True)

        # Check encoded_latent_tensor's shape.
        expected_encoded_shape = (
            encoded_latent_tensor.shape[0],  # Batch Count
            encoded_latent_tensor.shape[1],  # Channel Count
            bounds_with_margins.span[1],     # Height
            bounds_with_margins.span[0]      # Width
        )
        assert encoded_latent_tensor.shape == expected_encoded_shape, \
            (f"Argument tensor \"encoded_latent_tensor\" has unexpected shape of {encoded_latent_tensor.shape}"
             f"expected shape of {expected_encoded_shape}.")

        # Check destination image's shape.
        expected_dest_shape = (
            dest_latent_tensor.shape[0],  # Batch Count
            dest_latent_tensor.shape[1],  # Channel Count
            self._geometry.size_latents[1],  # Height
            self._geometry.size_latents[0]  # Width
        )
        assert dest_latent_tensor.shape == expected_dest_shape, \
            (f"Argument tensor \"dest_latent_tensor\" has unexpected shape of {dest_latent_tensor.shape}"
             f"expected shape of {expected_dest_shape}.")

        bounds_without_margins = self._geometry.get_bounds(tile_index, include_margins=False)
        relative_bounds = bounds_with_margins.transform_bounds(bounds_without_margins)

        trimmed_encoded_view = encoded_latent_tensor[
            :, :,
            relative_bounds.y_bounds[0]:relative_bounds.y_bounds[1],
            relative_bounds.x_bounds[0]:relative_bounds.x_bounds[1]
        ]

        masks = self._get_encode_masks()
        h_pixel_mask = masks[0][tile_index[0]]
        v_pixel_mask = masks[1][tile_index[1]]

        # Multiply in the masks.
        trimmed_encoded_view *= h_pixel_mask[
            :, :,
            :,
            bounds_without_margins.x_bounds[0]:bounds_without_margins.x_bounds[1]
        ]

        trimmed_encoded_view *= v_pixel_mask[
            :, :,
            bounds_without_margins.y_bounds[0]:bounds_without_margins.y_bounds[1],
            :
        ]

        dest_latent_tensor[
            :, :,
            bounds_without_margins.y_bounds[0]:bounds_without_margins.y_bounds[1],
            bounds_without_margins.x_bounds[0]:bounds_without_margins.x_bounds[1]
        ] += trimmed_encoded_view

    def get_decode_view(self, latent_tensor: torch.Tensor, tile_index: tuple[int, int]):

        # Check the input dimensionality.
        expected_shape_length = 4
        input_shape = latent_tensor.shape
        assert len(input_shape) == expected_shape_length, \
            (f"Expected {expected_shape_length}-dimensional latent tensor, "
             f"was given tensor with dimensionality of {len(input_shape)}.")

        # Check height and width.
        expected_shape = (
            input_shape[0],          # Batch Count
            input_shape[1],          # Channel Count
            self._geometry.size_latents[1],  # Height
            self._geometry.size_latents[0]   # Width
        )
        assert latent_tensor.shape == expected_shape, \
            (f"Argument tensor \"latent_tensor\" has unexpected shape of {latent_tensor.shape}"
             f"expected shape of {expected_shape}.")

        bounds_with_margins = self._geometry.get_bounds(tile_index, include_margins=True)

        latent_view = latent_tensor[
            :, :,
            bounds_with_margins.y_bounds[0]:bounds_with_margins.y_bounds[1],
            bounds_with_margins.x_bounds[0]:bounds_with_margins.x_bounds[1]
        ]

        return latent_view

    def write_decoded_tile(self,
                           tile_index: tuple[int, int],
                           decoded_pixel_tensor: torch.Tensor,
                           dest_pixel_tensor: torch.Tensor):

        assert (0 <= tile_index[0] < self._geometry.tile_count[0] and
                0 <= tile_index[1] < self._geometry.tile_count[1]), \
               (f"Argument \"tile_index\" value {tile_index} is out-of-range, "
                f"valid range is (0, 0) to {self._geometry.tile_count}")

        # Check the inputs' dimension count.
        expected_shape_length = 4
        assert len(decoded_pixel_tensor.shape) == expected_shape_length, \
            (f"Expected {expected_shape_length}-dimensional latent tensor, "
             f"argument \"decoded_pixel_tensor\" has dimensionality of {len(decoded_pixel_tensor.shape)}.")
        assert len(dest_pixel_tensor.shape) == expected_shape_length, \
            (f"Expected {expected_shape_length}-dimensional latent tensor, "
             f"argument \"decoded_pixel_tensor\" has dimensionality of {len(dest_pixel_tensor.shape)}.")

        bounds_with_margins = self._geometry.get_bounds(tile_index, include_margins=True)

        # Check decoded_pixel_tensor's shape.
        expected_decoded_shape = (
            decoded_pixel_tensor.shape[0],  # Batch Count
            decoded_pixel_tensor.shape[1],  # Channel Count
            bounds_with_margins.span[1] * self._pixels_per_latent,  # Height
            bounds_with_margins.span[0] * self._pixels_per_latent  # Width
        )
        assert decoded_pixel_tensor.shape == expected_decoded_shape, \
            (f"Argument tensor \"decoded_pixel_tensor\" has unexpected shape of {decoded_pixel_tensor.shape}"
             f"expected shape of {expected_decoded_shape}.")

        # Check destination image's shape.
        expected_dest_shape = (
            dest_pixel_tensor.shape[0],  # Batch Count
            dest_pixel_tensor.shape[1],  # Channel Count
            self._pixel_size[1],  # Height
            self._pixel_size[0]  # Width
        )
        assert dest_pixel_tensor.shape == expected_dest_shape, \
            (f"Argument tensor \"dest_pixel_tensor\" has unexpected shape of {dest_pixel_tensor.shape}"
             f"expected shape of {expected_dest_shape}.")

        bounds_without_margins = self._geometry.get_bounds(tile_index, include_margins=False)
        relative_bounds = bounds_with_margins.transform_bounds(bounds_without_margins)

        trimmed_decoded_view = decoded_pixel_tensor[
            :, :,
            relative_bounds.y_bounds[0] * self._pixels_per_latent:
            relative_bounds.y_bounds[1] * self._pixels_per_latent,
            relative_bounds.x_bounds[0] * self._pixels_per_latent:
            relative_bounds.x_bounds[1] * self._pixels_per_latent
        ]

        masks = self._get_decode_masks()
        h_pixel_mask = masks[0][tile_index[0]]
        v_pixel_mask = masks[1][tile_index[1]]

        # Multiply in the masks.
        trimmed_decoded_view *= h_pixel_mask[
            :, :,
            :,
            bounds_without_margins.x_bounds[0] * self._pixels_per_latent:
            bounds_without_margins.x_bounds[1] * self._pixels_per_latent
        ]

        trimmed_decoded_view *= v_pixel_mask[
            :, :,
            bounds_without_margins.y_bounds[0] * self._pixels_per_latent:
            bounds_without_margins.y_bounds[1] * self._pixels_per_latent,
            :
        ]

        dest_pixel_tensor[
            :, :,
            bounds_without_margins.y_bounds[0] * self._pixels_per_latent:
            bounds_without_margins.y_bounds[1] * self._pixels_per_latent,
            bounds_without_margins.x_bounds[0] * self._pixels_per_latent:
            bounds_without_margins.x_bounds[1] * self._pixels_per_latent
        ] += trimmed_decoded_view


cache = {}


def get_or_create_tilemap(
        size_latents: tuple[int, int],
        max_tile_size_latents: int,
        margin_size_latents: int,
        overlap_size_latents: int,
        pixels_per_latent: int,
        device: torch.device,
        dtype: torch.dtype):

    key = (
        size_latents,
        max_tile_size_latents,
        margin_size_latents,
        overlap_size_latents,
        pixels_per_latent,
        device,
        dtype
    )

    if key in cache:
        return cache[key]

    value = Tilemap(*key)

    cache[key] = value
    return value
