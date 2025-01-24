# This script: sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we
#                                     might do to get things we need from sd.webui.

# Overview of all scripts in project:
# scripts/diffusion_canvas.py - Script that interfaces with sd.webui and is the entry point for launch.
# brushes.py - Tools for image data manipulation.
# sdwebui_interface.py - Acts as a layer of abstraction, hiding away all the potentially hacky things we might do to get
#                        things we need from sd.webui.
# shader_runner.py - Used to execute shader-based math on tensors.
# texture_convert.py - Automatic conversion of various representations of texture data.
# ui.py - UI for DiffusionCanvas.
# diffusion_canvas_api.py - Contains functions used by the UI

import modules.processing as processing
import modules.shared as shared
import modules.scripts as scripts
from modules.sd_samplers_common import (
    images_tensor_to_samples,
    samples_to_images_tensor,
    InterruptedException
)
from modules import devices
from modules.script_callbacks import BeforeDenoiserForwardParams

import torch

from utils.time_utils import Timer


class DenoiserParams:
    sigma = None
    uncond = None
    cond = None
    cond_scale = None
    s_min_uncond = None
    image_cond = None


# Used for inter-thread communication
interrupt: bool = False
freeze: bool = False
our_call: bool = False
intercepted_params: DenoiserParams = DenoiserParams()
intercepted_denoiser = None
has_intercepted_values: bool = False


@torch.no_grad()
def decode_image(latent, full_quality: bool = True):
    if full_quality:
        image = processing.decode_latent_batch(shared.sd_model, latent)
        image = torch.stack(image).float()
    else:
        image = samples_to_images_tensor(latent, approximation=3)

    return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)


@torch.no_grad()
def encode_image(image):
    image = torch.clamp(image, min=0.0, max=1.0)
    image = image.to(shared.device, dtype=devices.dtype_vae)
    return images_tensor_to_samples(image, approximation=0)  # Full


@torch.no_grad()
def denoise(denoiser: any, latent: torch.Tensor, sigma: float, cfg_scale: float, params: any) -> torch.Tensor:
    with Timer("denoise"):
        if denoiser is None:
            return latent
        if params is None:
            return latent

        global our_call
        params.sigma = (params.sigma * 0) + sigma
        our_call = True

        with Timer("denoiser.forward"), torch.no_grad():
            dtype = latent.dtype
            latent = denoiser.forward(latent, params.sigma, params.uncond, params.cond, cfg_scale, params.s_min_uncond,
                                    params.image_cond).to(dtype)

        our_call = False
        return latent


def begin_interrupt():
    global interrupt

    if interrupt:
        return

    global intercepted_params, intercepted_denoiser, has_intercepted_values
    intercepted_params = DenoiserParams()
    intercepted_denoiser = None
    has_intercepted_values = False

    interrupt = True


def end_interrupt():
    global interrupt
    interrupt = False


def intercept_and_interrupt(params: BeforeDenoiserForwardParams):
    global interrupt, our_call
    if not interrupt:
        return

    # Do not intercept our own call.
    if our_call:
        return

    global intercepted_params, intercepted_denoiser, has_intercepted_values

    # why_tho.jpg its initialized above lol
    if intercepted_params is None:
        intercepted_params = DenoiserParams()

    # Grab the denoiser
    intercepted_denoiser = params.denoiser

    # Grab the arguments passed to the "forward" call.
    intercepted_params.sigma = params.sigma
    intercepted_params.uncond = params.uncond
    intercepted_params.cond = params.cond
    intercepted_params.cond_scale = params.cond_scale
    intercepted_params.s_min_uncond = params.s_min_uncond
    intercepted_params.image_cond = params.image_cond

    has_intercepted_values = True

    global freeze
    freeze = True
    import time
    while freeze:
        time.sleep(0.1)

    # Cancel execution
    raise InterruptedException


def unfreeze_sd_webui():
    global freeze
    freeze = False


def pop_intercepted() -> tuple[any, DenoiserParams] | None:
    global intercepted_params, intercepted_denoiser, has_intercepted_values

    if not has_intercepted_values:
        return None

    has_intercepted_values = False
    out = (intercepted_denoiser, intercepted_params)
    intercepted_denoiser = None
    intercepted_params = DenoiserParams()
    return out


scripts.script_callbacks.on_before_denoiser_forward(intercept_and_interrupt)
