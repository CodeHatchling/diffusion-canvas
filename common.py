import numpy as np


def get_corrected_bounds_1d(bounds: tuple[int, int]) -> tuple[int, int]:
    """
    Corrects the bounds provided by ensuring the span is at least 1.

    Args:
        bounds (tuple[int, int]): The bounds to possibly correct.

    Returns:
        (tuple[int, int]): If the input already has a span of 1 or more, the input is returned.
        Otherwise, the best-fitting bounds with a span of at least 1 is returned.
    """
    if bounds[0] >= bounds[1]:
        average = (bounds[0] + bounds[1]) / 2
        minimum = int(np.floor(average))
        maximum = int(np.ceil(average))
        if minimum == maximum:
            maximum += 1

        return (
            minimum,
            maximum
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


def get_encapsulated_bounds_1d(bounds: tuple[int, int], other: tuple[int, int]) -> tuple[int, int]:
    return (
        np.minimum(bounds[0], other[0]),
        np.maximum(bounds[1], other[1])
    )


def get_offset_bounds_1d(bounds: tuple[int, int], offset: int) -> tuple[int, int]:
    return (
        bounds[0] + offset,
        bounds[1] + offset
    )


class Bounds2D:
    x_bounds: tuple[int, int]
    y_bounds: tuple[int, int]

    def __init__(self, x_bounds: tuple[int, int], y_bounds: tuple[int, int]):
        self.x_bounds = get_corrected_bounds_1d(x_bounds)
        self.y_bounds = get_corrected_bounds_1d(y_bounds)

    def get_expanded(self, expand_amount_x: tuple[int, int] | int, expand_amount_y: tuple[int, int] | int):
        return Bounds2D(
            get_expanded_bounds_1d(self.x_bounds, expand_amount_x),
            get_expanded_bounds_1d(self.y_bounds, expand_amount_y)
        )

    def get_clipped(self, clip: 'Bounds2D') -> 'Bounds2D':
        return Bounds2D(
            get_clipped_bounds_1d(self.x_bounds, clip.x_bounds),
            get_clipped_bounds_1d(self.y_bounds, clip.y_bounds)
        )

    def get_encapsulated(self, other: 'Bounds2D') -> 'Bounds2D':
        return Bounds2D(
            get_encapsulated_bounds_1d(self.x_bounds, other.x_bounds),
            get_encapsulated_bounds_1d(self.y_bounds, other.y_bounds)
        )

    def transform_bounds(self, other: 'Bounds2D'):
        return Bounds2D(
            get_offset_bounds_1d(other.x_bounds, -self.x_bounds[0]),
            get_offset_bounds_1d(other.y_bounds, -self.y_bounds[0]),
        )

    def __eq__(self, other):
        if not hasattr(other, "x_bounds"):
            return False
        if not hasattr(other, "y_bounds"):
            return False

        return self.x_bounds == other.x_bounds and self.y_bounds == other.y_bounds

    def _get_span(self):
        return self.x_bounds[1] - self.x_bounds[0], self.y_bounds[1] - self.y_bounds[0],
    span = property(fget=_get_span)
