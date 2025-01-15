import builtins

import torch
from typing_extensions import BinaryIO, Literal


def _get_dtype_size(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def _get_dtype_name(dtype):
    return dtype.name


dtypes = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128
]

dtype_sizes = {dtype: _get_dtype_size(dtype) for dtype in dtypes}

dtype_to_name = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.complex64: "complex64",
    torch.complex128: "complex128"
}

name_to_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "complex128": torch.complex128
}


def _read_int(file: BinaryIO, byteorder: Literal['big', 'little'] = 'little', signed=True) -> int:
    # Read 4 bytes (32-bit int)
    return int.from_bytes(file.read(4), byteorder=byteorder, signed=signed)


def _write_int(file: BinaryIO, value: int, byteorder: Literal['big', 'little'] = 'little', signed=True) -> None:
    data = value.to_bytes(length=4, byteorder=byteorder, signed=signed)
    file.write(data)


def _read_dtype(file: BinaryIO) -> torch.dtype:
    # Read the dtype string (e.g., "float32")
    dtype_str_length = _read_int(file)  # Read length of dtype string
    dtype_str = file.read(dtype_str_length).decode('utf-8')

    if dtype_str not in name_to_dtype:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return name_to_dtype[dtype_str]


def _write_dtype(file: BinaryIO, dtype: torch.dtype) -> None:
    if dtype not in dtype_to_name:
        raise ValueError(f"Unsupported dtype: {dtype}")

    dtype_str: str = dtype_to_name[dtype]

    encoded = dtype_str.encode('utf-8')
    encoded_length = len(encoded)
    _write_int(file, encoded_length)
    file.write(encoded)


def _get_byte_count(shape, dtype):
    element_count = torch.prod(torch.tensor(shape)).item()  # Product of all dimensions
    return element_count * dtype_sizes[dtype]  # Size of dtype


def _read_bytes(file, byte_count) -> bytes:
    # Read the required number of bytes
    data = file.read(byte_count)
    if len(data) != byte_count:
        raise IOError(f"Expected {byte_count} bytes, got {len(data)} bytes")
    return data


def _create_tensor(shape: tuple[int, ...], dtype: torch.dtype, data: bytes) -> torch.Tensor:
    # Convert raw data to a torch tensor
    flat_tensor = torch.frombuffer(data, dtype=dtype)
    return flat_tensor.reshape(shape)


def read_tensor(file_path: str) -> torch.Tensor:
    try:
        with builtins.open(file_path, "rb") as file:
            version = _read_int(file)
            if version == 0:
                # Format:
                # - Dimension count
                # - Dimension size for each dimension
                # - dtype
                # - tensor data

                dim_count = _read_int(file)
                shape = tuple(_read_int(file) for _ in range(dim_count))

                dtype: torch.dtype = _read_dtype(file)
                byte_count = _get_byte_count(shape, dtype)
                data = _read_bytes(file, byte_count)

                return _create_tensor(
                    shape=shape,
                    dtype=dtype,
                    data=data
                )
            else:
                raise Exception(f"The provided latent file did not have a recognized version number.")
    except Exception as e:
        raise IOError(f"Failed to write tensor to {file_path}: {e}")


def write_tensor(file_path: str, tensor: torch.Tensor) -> None:
    try:
        with builtins.open(file_path, "wb") as file:
            _write_int(file, 0)  # Version
            _write_int(file, len(tensor.shape))
            for dim in tensor.shape:
                _write_int(file, dim)

            dtype = tensor.dtype
            _write_dtype(file, dtype)

            file.write(tensor.contiguous().cpu().numpy().tobytes())
    except Exception as e:
        raise IOError(f"Failed to write tensor to {file_path}: {e}")



