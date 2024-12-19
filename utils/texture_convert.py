# This script: texture_convert.py - Automatic conversion of various representations of texture data.

# Supports conversion of the following data types:
# - torch stable diffusion image tensors [batch, channels, height, width]
# - PIL Images
# - ModernGL textures (must provide factory due to needing a context object)
# - wrapped numpy ndarray with shape descriptor

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.
# diffusion_canvas_api.py - Contains functions used by the UI



from typing import NamedTuple
import warnings
import numpy as np
import PIL
from PIL import Image
import torch
from PyQt6.QtGui import QImage
import moderngl


supported_types = PIL.Image.Image | torch.Tensor | QImage


class DTypeDesc:
    # The data represents the values exactly as defined by their underlying type.
    raw = "raw"

    # The data represents values within unit distance of 0; only applicable to non-float types.
    # Zero (0) is fixed, meaning it is consistent between the interpreted value
    # and the value represented by underlying type.
    normalized = "normalized"


class LabeledArray(NamedTuple):
    data: np.ndarray
    shape_desc: tuple[str, ...]
    dtype_desc: str

    def get_dimension(self, label: str) -> int:
        """
        Returns: The size of the dimension labelled <label>, or 1, if not provided.
        """
        return self.data.shape[self.shape_desc.index(label)] if label in self.shape_desc else 1

    def get_dimensions(self, labels: tuple[str, ...]) -> tuple[int, ...]:
        """
        Returns: The sizes of the dimensions labelled. Absent dimensions are given a size of 1.
        """
        return tuple(self.get_dimension(x) for x in labels)

    def get_converted_shape(self, target_desc: tuple[str, ...]) -> 'LabeledArray':
        current_desc = self.shape_desc

        if current_desc == target_desc:
            return self

        # Step 1: Unsqueeze dimensions that are present in the target shape, but absent in the current shape.
        reshaped_array = self.data
        reshaped_desc = list(current_desc)

        for dim in target_desc:
            if dim not in current_desc:
                reshaped_array = np.expand_dims(reshaped_array, axis=len(reshaped_desc))
                reshaped_desc.append(dim)

        # Step 2: For dimensions absent in the target shape but present in the current shape:
        for dim in current_desc:
            if dim not in target_desc:
                dim_index = reshaped_desc.index(dim)
                # If the dimension's size is not 1, warn and truncate it.
                if reshaped_array.shape[dim_index] != 1:
                    warnings.warn(f"Warning: Truncating non-singleton dimension '{dim}' to size 1.")
                    reshaped_array = np.take(reshaped_array, indices=0, axis=dim_index)
                # Squeeze the dimension.
                reshaped_array = np.squeeze(reshaped_array, axis=dim_index)
                reshaped_desc.pop(dim_index)

        # Step 3: Reorder the dimensions to match the target shape.
        axis_map = [reshaped_desc.index(dim) for dim in target_desc]
        reshaped_array = np.transpose(reshaped_array, axes=axis_map)

        return LabeledArray(reshaped_array, shape_desc=target_desc, dtype_desc=self.dtype_desc)

    def get_converted_type(self, dest_dtype, dest_dtype_desc: str):

        data = self.data
        source_dtype = self.data.dtype
        source_dtype_desc = self.dtype_desc
        dest_dtype = np.dtype(dest_dtype)

        # If the data type is already in floating-point format, ignore the type_desc parameter.
        if source_dtype.kind == 'f':
            source_scaling_factor = 1
        elif source_dtype_desc == DTypeDesc.raw:
            source_scaling_factor = 1
        elif source_dtype_desc == DTypeDesc.normalized:
            source_scaling_factor = np.iinfo(source_dtype).max
        else:
            raise ValueError(f"Unexpected dtype_desc in source: {source_dtype_desc}")

        # If the data type is already in floating-point format, ignore the type_desc parameter.
        if dest_dtype.kind == 'f':
            dest_scaling_factor = 1
        elif dest_dtype_desc == DTypeDesc.raw:
            dest_scaling_factor = 1
        elif dest_dtype_desc == DTypeDesc.normalized:
            clamp = True
            dest_scaling_factor = np.iinfo(dest_dtype).max
        else:
            raise ValueError(f"Unexpected dest_dtype_desc: {dest_dtype_desc}")

        # Scale if needed
        if source_scaling_factor != dest_scaling_factor:
            # TODO: Is this optimal for conversions, or will it introduce unnecessary precision loss?
            data = (data * (dest_scaling_factor / source_scaling_factor))

        # Clip
        if dest_dtype.kind in 'iu':
            data = data.clip(np.iinfo(dest_dtype).min, np.iinfo(dest_dtype).max)

        data = data.astype(dest_dtype)

        return LabeledArray(data,
                            shape_desc=self.shape_desc,
                            dtype_desc=dest_dtype_desc)

    def get_converted(self, target_shape, target_dtype, target_dtype_desc):
        return self.get_converted_shape(target_shape).get_converted_type(target_dtype, target_dtype_desc)


def moderngl_dtype_to_labeled_dtype_desc(dtype: str) -> tuple:
    """
    Convert a moderngl dtype alias string to a NumPy dtype.

    Parameters:
    dtype (str): moderngl dtype alias string.

    Returns:
    np.dtype: The corresponding moderngl NumPy dtype.
    """
    alias_map = {
        'f1': (np.uint8, DTypeDesc.normalized),
        'f2': (np.float16, DTypeDesc.raw),
        'f4': (np.float32, DTypeDesc.raw),
        'f8': (np.float64, DTypeDesc.raw),
        'i1': (np.int8, DTypeDesc.raw),
        'i2': (np.int16, DTypeDesc.raw),
        'i4': (np.int32, DTypeDesc.raw),
        'i8': (np.int64, DTypeDesc.raw),
        'u1': (np.uint8, DTypeDesc.raw),
        'u2': (np.uint16, DTypeDesc.raw),
        'u4': (np.uint32, DTypeDesc.raw),
        'u8': (np.uint64, DTypeDesc.raw),
    }
    if dtype in alias_map:
        return alias_map[dtype]
    else:
        raise ValueError(f"Unsupported moderngl dtype alias string: '{dtype}'")


def mgltex_to_array(mgltex: moderngl.Texture) -> 'LabeledArray':
    """
    Read texture data from the GPU and convert it to a NumPy array.

    Parameters:
    - tex_gpu (moderngl.Texture): The GPU texture to read from.
    - flipud (bool): Whether to flip the array vertically.

    Returns:
    - output_data (LabeledArray): The texture data as a wrapped nparray array with labels.
    """
    raw_data = mgltex.read()
    dtype, dtype_desc = moderngl_dtype_to_labeled_dtype_desc(mgltex.dtype)
    width, height = mgltex.size
    output_data = np.frombuffer(raw_data, dtype=dtype)

    # Reshape the data to (height, width, components)
    output_data = output_data.reshape((height, width, mgltex.components))

    # Flip the array vertically to match image coordinate systems
    output_data = np.flipud(output_data)

    return LabeledArray(output_data.copy(), ("height", "width", "channels"), dtype_desc)


def array_to_mgltex(array: LabeledArray, mgltex_factory: callable) -> moderngl.Texture:
    """
    Converts a LabeledArray texture to a mgl texture on the GPU.

    Parameters:
    - array (LabeledArray): The texture data as a NumPy array.
    - mgltex_factory (callable): A callable method for instantiating textures.
                                 Args: (width: int, height: int), channels: int, data: list[byte], dtype
                                 Returns: (moderngl.Texture)
    - flipud (bool): Whether to flip the image vertically.
    - u8_as_f1 (bool): Whether to interpret np.uint8 as single-byte values ranging from 0 to 1.

    Returns:
    - tex_gpu (moderngl.Texture): The created GPU texture.
    """

    array = array.get_converted_shape(("height", "width", "channels"))
    height, width, channels = array.get_dimensions(("height", "width", "channels"))

    # Convert the array to a mgl-compatible format.
    src_dtype, src_desc = array.data.dtype, array.dtype_desc
    converted = array

    if (src_dtype, src_desc) == (np.uint8, DTypeDesc.normalized):
        dest_format = 'f1'
    elif src_desc == DTypeDesc.raw or src_dtype.kind == 'f':
        alias_map = {
            np.float16: 'f2',
            np.float32: 'f4',
            np.float64: 'f8',
            np.int8: 'i1',
            np.int16: 'i2',
            np.int32: 'i4',
            np.int64: 'i8',
            np.uint8: 'u1',
            np.uint16: 'u2',
            np.uint32: 'u4',
            np.uint64: 'u8',
        }

        if src_dtype.type in alias_map:
            # If dtype is a type, try this first.
            dest_format = alias_map[src_dtype.type]
        else:
            raise ValueError(f"Unsupported format: {src_dtype}")
    else:
        size = src_dtype.itemsize
        if size < 2:
            converted = converted.get_converted_type(np.float16, DTypeDesc.raw)
            dest_format = 'f2'
        elif size < 4:
            converted = converted.get_converted_type(np.float32, DTypeDesc.raw)
            dest_format = 'f4'
        elif size < 8:
            converted = converted.get_converted_type(np.float64, DTypeDesc.raw)
            dest_format = 'f8'
        else:
            raise ValueError(f"Unsupported format: {src_dtype}")

    # Note that the array is flipped vertically to match image coordinate systems
    mgltex = mgltex_factory((width, height), channels, data=np.flipud(converted.data).tobytes(), dtype=dest_format)

    return mgltex


def pil_to_array(pil: PIL.Image.Image) -> 'LabeledArray':
    unit_data_types = ('L', 'RGB', 'RGBA', 'RGBX', 'LA', 'CMYK', 'YCbCr')
    single_channel_data_types = ('L', 'F', '1', 'P', 'I')
    data = np.array(pil)

    if pil.mode in single_channel_data_types:
        shape_desc = ("height", "width")
    else:
        shape_desc = ("height", "width", "channels")

    if pil.mode in unit_data_types and data.dtype.kind != 'f':
        dtype_desc = DTypeDesc.normalized
    else:
        dtype_desc = DTypeDesc.raw

    array = LabeledArray(data, shape_desc, dtype_desc)
    return array


def array_to_pil(array: LabeledArray) -> PIL.Image:
    """
    Convert a NumPy array texture to a PIL Image.

    Parameters:
    - array (np.ndarray): The texture data as a NumPy array.
    - remap (bool): Whether to remap the data to uint8.

    Returns:
    - result_image (PIL.Image): The resulting PIL Image.
    """
    array = array.get_converted(("height", "width", "channels"), np.uint8, DTypeDesc.normalized)
    components = array.get_dimension("channels")
    data = array.data

    # Determine the image mode based on the number of components
    if components == 4:
        mode = 'RGBA'
    elif components == 3:
        mode = 'RGB'
    elif components == 2:
        mode = 'LA'  # Luminance with alpha
    elif components == 1:
        mode = 'L'   # Luminance (grayscale)
        data = data.squeeze()
    else:
        raise NotImplementedError(f"Conversion from numpy array texture with {components} components to PIL is not implemented.")

    pil = Image.fromarray(data, mode)
    return pil


def array_to_sdimage(array: LabeledArray) -> torch.Tensor:
    array = array.get_converted(("batch", "channels", "height", "width"), np.float32, DTypeDesc.raw)
    return torch.tensor(array.data)


def sdimage_to_array(sdtensor: torch.Tensor) -> 'LabeledArray':
    tensor_cpu = sdtensor.cpu()
    if hasattr(tensor_cpu, "detach"):
        tensor_cpu = tensor_cpu.detach()

    return LabeledArray(tensor_cpu.numpy(), ("batch", "channels", "height", "width"), DTypeDesc.raw)


def array_to_qimage(array: LabeledArray) -> 'QImage':
    array = array.get_converted_shape(("height", "width", "channels"))
    width, height, channels = array.get_dimensions(("width", "height", "channels"))

    # Values are (QImage format, append alpha: bool)
    table = {
        (3, np.uint8): (QImage.Format.Format_RGB888, False),
        (4, np.uint8): (QImage.Format.Format_RGBA8888, False),
        (1, np.uint8): (QImage.Format.Format_Grayscale8, False),
        (1, np.uint16): (QImage.Format.Format_Grayscale16, False),
        (3, np.uint16): (QImage.Format.Format_RGBX64, True),
        (3, np.float16): (QImage.Format.Format_RGBX16FPx4, True),
        (3, np.float32): (QImage.Format.Format_RGBX32FPx4, True),
        (4, np.uint16): (QImage.Format.Format_RGBA64, False),
        (4, np.float16): (QImage.Format.Format_RGBA16FPx4, False),
        (4, np.float32): (QImage.Format.Format_RGBA32FPx4, False)
    }

    key = (channels, array.data.dtype.type)
    if key in table:
        conversion = table[key]
    else:
        raise TypeError(f"Unsupported source format: ({channels} channels, {array.data.dtype})")

    data = array.data

    # Requires a redundant zero-filled alpha channel?
    if conversion[1]:
        channels = 4
        zero_shape = list(data.shape)  # Convert the tuple to a list
        zero_shape[2] = 1  # Modify the desired dimension
        zero_shape = tuple(zero_shape)  # Convert it back to a tuple
        zero_channel = np.zeros(zero_shape, dtype=data.dtype)
        data = np.concatenate((data, zero_channel), axis=-1)

    # Convert numpy array to QImage
    q_image = QImage(data.tobytes(), width, height, channels * width * data.dtype.itemsize, conversion[0])
    return q_image


def convert_to_labelled_array(source: LabeledArray | PIL.Image.Image | torch.Tensor | QImage | moderngl.Texture) \
        -> 'LabeledArray':

    if isinstance(source, LabeledArray):
        return source

    # convert to labelled array
    elif isinstance(source, PIL.Image.Image):
        source = pil_to_array(source)
    elif isinstance(source, torch.Tensor):
        source = sdimage_to_array(source)
    elif isinstance(source, moderngl.Texture):
        source = mgltex_to_array(source)
    # TODO: Implement QImage to LabelledArray
    # elif isinstance(source, QImage):
    #     source = qimage_to_array(source)
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    return source


def convert(source: LabeledArray | PIL.Image.Image | torch.Tensor | QImage | moderngl.Texture, dest_type) \
        -> LabeledArray | PIL.Image.Image | torch.Tensor | QImage | moderngl.Texture:

    source = convert_to_labelled_array(source)

    # convert to destination type
    if dest_type == PIL.Image.Image:
        return array_to_pil(source)
    elif dest_type == torch.Tensor:
        return array_to_sdimage(source)
    elif dest_type == QImage:
        return array_to_qimage(source)
    else:
        raise ValueError(f"Unsupported destination type: {dest_type}")


def convert_to_moderngl_texture(source: LabeledArray | PIL.Image.Image | torch.Tensor | QImage | moderngl.Texture,
                                moderngl_factory: callable) -> moderngl.Texture:

    source = convert_to_labelled_array(source)

    return array_to_mgltex(source, moderngl_factory)
