import math

import numpy as np
import torch
from typing import Callable
from time_utils import Timer


@torch.no_grad()
def safe_sqrt(x):
    return torch.clamp(x, min=0) ** 0.5


@torch.no_grad()
def apply_uniform_maximized_noise(noisy_latent: torch.Tensor,
                                  noise_amplitude: torch.Tensor) -> (torch.Tensor, float):
    max_amplitude = torch.max(noise_amplitude)
    added_amplitude = safe_sqrt(max_amplitude**2 - noise_amplitude**2)
    return noisy_latent + torch.randn_like(noisy_latent) * added_amplitude, max_amplitude.squeeze().item()


class Layer:
    '''def __init__(self, latent):
        self.clean_latent = latent
        self.noisy_latent = latent.clone()
        noise_amp_shape = list(latent.shape)
        noise_amp_shape[1] = 1
        noise_amp_shape = tuple(noise_amp_shape)
        self.noise_amplitude = torch.zeros(noise_amp_shape, dtype=latent.dtype, device=latent.device)'''

    def __init__(self, clean_latent, noisy_latent, noise_amplitude):
        assert clean_latent.shape == noisy_latent.shape, \
            "clean_latent and noisy_latent shape mismatch" \
            f"{clean_latent.shape} != {noisy_latent.shape}."

        assert clean_latent.shape[1] == 4, \
            f"clean_latent invalid shape in dim [1], expected 4, got {clean_latent.shape[1]}."

        assert noise_amplitude.shape[1] == 1, \
            f"noise_amplitude invalid shape in dim [1], expected 1, got {noise_amplitude.shape[1]}."

        assert clean_latent.shape[2] == noise_amplitude.shape[2], \
            f"clean_latent and noise_amplitude mismatched shape dim[2]," \
            f"{clean_latent.shape[2]} != {noise_amplitude.shape[2]}."

        assert clean_latent.shape[3] == noise_amplitude.shape[3], \
            "clean_latent and noise_amplitude mismatched shape dim[3]," \
            f"{clean_latent.shape[3]} != {noise_amplitude.shape[3]}."

        assert clean_latent is not noisy_latent, \
            "clean_latent and noisy_latent must be separate tensors."

        assert clean_latent.data_ptr() != noisy_latent.data_ptr(), \
            "clean_latent and noisy_latent must not share memory."

        assert clean_latent.data_ptr() != noise_amplitude.data_ptr(), \
            "clean_latent and noise_amplitude must not share memory."

        assert noisy_latent.data_ptr() != noise_amplitude.data_ptr(), \
            "noisy_latent and noise_amplitude must not share memory."

        self.clean_latent = clean_latent
        self.noisy_latent = noisy_latent
        self.noise_amplitude = noise_amplitude

    @torch.no_grad()
    def add_noise(self, desired_amplitude_increase: torch.Tensor):
        """
        Args:
            desired_amplitude_increase: How much to increase the noise amplitude by per latent.
        """
        amplitude_to_add = safe_sqrt(
            desired_amplitude_increase ** 2 + 2 * self.noise_amplitude * desired_amplitude_increase)
        self.noisy_latent += torch.randn_like(self.noisy_latent) * amplitude_to_add
        self.noise_amplitude = safe_sqrt(self.noise_amplitude**2 + amplitude_to_add**2)

    @torch.no_grad()
    def step(self,
             denoise: Callable[[torch.Tensor, float], torch.Tensor],
             attenuation_func: Callable[[float], float],
             brush_mask: torch.Tensor | None = None,
             noise_bias: float = 1,
             y_bounds: tuple[int, int] = (0, -1), x_bounds: tuple[int, int] = (0, -1)):

        # Convert these tensors into "windowed" versions according to the bounds.
        if isinstance(brush_mask, torch.Tensor):
            brush_mask = brush_mask[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
        self_clean_latent = self.clean_latent[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
        self_noisy_latent = self.noisy_latent[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
        self_noise_amplitude = self.noise_amplitude[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]

        # Add noise to the latent to make it uniform.
        # Though, when masking is used, some regions are noisier than the "maximum" as they are given less weight.
        # The noise level which exceeds the maximum is output in "extra_amplitude", or is None when masks are not used.
        uniform_noisy_latent, max_amplitude = apply_uniform_maximized_noise(
            self_noisy_latent,
            self_noise_amplitude)

        # Get the maxed maximum amplitude, which will bias the target noise level towards what the brush is centered
        # over.
        max_masked_amplitude = torch.max(
            self_noise_amplitude * brush_mask
            if isinstance(brush_mask, torch.Tensor)
            else self_noise_amplitude
        ).squeeze().item()

        # Denoise!
        denoised_latent = denoise(uniform_noisy_latent, max_amplitude * noise_bias)

        # Use the attenuation function to calculate our next maximum noise level.
        attenuated_max_amplitude: float = attenuation_func(max_masked_amplitude)

        # How much should we blend the noisy and denoised latent together, given the previous
        # and attenuated noise levels?
        ratio: float = attenuated_max_amplitude / max_amplitude
        if math.isnan(ratio) or math.isinf(ratio):
            ratio = 0

        # Reduce the noise level less for unmasked regions.
        # The noise level and ratio may become per-latent here.
        if isinstance(brush_mask, torch.Tensor):
            attenuated_max_amplitude = attenuated_max_amplitude * brush_mask + max_amplitude * (1-brush_mask)
            ratio = ratio * brush_mask + (1-brush_mask)

        # Re-add noise back in.
        renoised_latent = uniform_noisy_latent * ratio + denoised_latent * (1-ratio)

        # Create a mask for pixels where noise_amplitude exceeds attenuated_max_amplitude
        mask = self_noise_amplitude > attenuated_max_amplitude  # Shape: (batch_count, 1, height, width)

        # Broadcast the mask to match the shape of the latent tensors
        expanded_mask = mask.expand_as(self_noisy_latent)  # Shape: (batch_count, channels, height, width)

        # Apply the updates conditionally using the mask
        self_noisy_latent[expanded_mask] = renoised_latent[expanded_mask].to(self_clean_latent.dtype)
        self_clean_latent[expanded_mask] = denoised_latent[expanded_mask].to(self_clean_latent.dtype)

        # Update noise_amplitude using the original mask (no need to expand)
        self_noise_amplitude[mask] = (
            attenuated_max_amplitude[mask]
            if isinstance(attenuated_max_amplitude, torch.Tensor)
            else attenuated_max_amplitude
        )

        # Copy the values back into the whole tensor.
        # This step may not be needed, but is done to be safe.
        self.clean_latent[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]] = self_clean_latent
        self.noisy_latent[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]] = self_noisy_latent
        self.noise_amplitude[:, :, y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]] = self_noise_amplitude

    @torch.no_grad()
    def clone(self):
        return Layer(self.clean_latent.clone(), self.noisy_latent.clone(), self.noise_amplitude.clone())


class History:
    def __init__(self, init_layer: Layer):
        self._undo_index: int = 0
        self._undo_stack: list[Layer] = [init_layer]

    def _get_layer(self) -> Layer:
        return self._undo_stack[self._undo_index]

    layer = property(fget=_get_layer)

    def register_undo(self):
        current_layer = self._undo_stack[self._undo_index]

        # Remove the current layer from the stack, and all "redo" steps ahead of it.
        self._undo_stack = self._undo_stack[0:self._undo_index]

        # Add a clone of the current state we can switch back to, AND the active state.
        self._undo_stack += [current_layer.clone(), current_layer]

        self._undo_index = len(self._undo_stack) - 1

    def _clamp_undo_index(self, index: int):
        return np.maximum(
            0,
            np.minimum(
                len(self._undo_stack) - 1,
                index
            )
        )

    def undo(self, steps: int = 1):
        self._undo_index = self._clamp_undo_index(self._undo_index - steps)

    def redo(self, steps: int = 1):
        self._undo_index = self._clamp_undo_index(self._undo_index + steps)


