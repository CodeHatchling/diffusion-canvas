import math

import numpy as np
import torch
from typing import Callable


@torch.no_grad()
def safe_sqrt(x):
    return torch.clamp(x, min=0) ** 0.5


@torch.no_grad()
def apply_uniform_maximized_noise(noisy_latent: torch.Tensor,
                                  noise_amplitude: torch.Tensor) -> (torch.Tensor, float):
    max_amplitude = torch.max(noise_amplitude)
    added_amplitude = safe_sqrt(max_amplitude**2 - noise_amplitude**2)
    return noisy_latent + torch.randn_like(noisy_latent) * added_amplitude, max_amplitude.squeeze().item()


@torch.no_grad()
def apply_uniform_noise(clean_latent: torch.Tensor,
                        noisy_latent: torch.Tensor,
                        noise_amplitude: torch.Tensor,
                        target_noise_amplitude: float) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """

    Args:
        clean_latent (torch.Tensor): The latent without noise.
        noisy_latent (torch.Tensor): The latent with noise.
        noise_amplitude (torch.Tensor): The amplitude of the noise signal added to clean latent to get noisy latent.
        target_noise_amplitude (float): The desired uniform noise level.

    Returns:
        [0] (torch.Tensor): The latent with uniform noise at the specified amplitude.
        [1] (torch.Tensor): The noise subtracted from noisy_latent to produce [0].
        [2] (torch.Tensor): The amplitude of [1].

    """
    noise_only = noisy_latent - clean_latent

    if target_noise_amplitude <= 0:
        return clean_latent, noise_only, target_noise_amplitude
    else:
        # Remove noise that is above the target amount.
        attenuation = 1 / torch.clamp(noise_amplitude / target_noise_amplitude, min=1)
        attenuated_noise = noise_only * attenuation
        attenuated_amplitude = noise_amplitude * attenuation

        # These values contain the contribution of noise that is above the target amplitude.
        # Since we are not creating new noise -- there is correlation --
        # we do not need to use sqrt or squares (arithmetic on variance).
        subtracted_noise = noise_only - attenuated_noise
        subtracted_noise_amplitude = noise_amplitude - attenuated_amplitude

        added_noise_amplitude = safe_sqrt(target_noise_amplitude ** 2 - attenuated_amplitude ** 2)
        added_noise = torch.randn_like(noisy_latent) * added_noise_amplitude

        return (
            clean_latent + attenuated_noise + added_noise,
            subtracted_noise,
            subtracted_noise_amplitude
        )


class Layer:
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
    def create_solid_latent(self, value: (float, float, float, float)):
        # Create a tensor with the same shape as clean_latent, where each channel is set to value[channel]
        value_tensor = torch.tensor(
            value,
            dtype=self.clean_latent.dtype,
            device=self.clean_latent.device
        ).view(1, -1, 1, 1)
        return value_tensor.expand_as(self.clean_latent)  # Broadcast to match clean_latent shape

    @torch.no_grad()
    def replace_clean_latent(self, new_clean_latent: torch.Tensor):
        """
        Replaces the contents of clean_latent while maintaining the difference to its noisy counterpart.
        Useful for editing.
        """
        # Get the difference between noisy and clean latent
        noise_only = self.noisy_latent - self.clean_latent

        # Replace
        self.clean_latent = new_clean_latent

        # Recalculate the noisy latent.
        self.noisy_latent = self.clean_latent + noise_only

    @torch.no_grad()
    def get_average_latent(self, mask: torch.Tensor | None):
        if isinstance(mask, torch.Tensor):
            mask_mean = mask.mean().squeeze().item()

            if mask_mean > 0:
                masked = self.clean_latent * mask
                average = masked.mean(dim=(0, 2, 3)) / mask_mean
            else:
                average = self.clean_latent.mean(dim=(0, 2, 3))
        else:
            average = self.clean_latent.mean(dim=(0, 2, 3))

        # Convert to (float, float, float, float) tuple
        result = tuple(average.tolist())

        return result

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

        if isinstance(brush_mask, torch.Tensor):
            # Get the maximum noise value after masking.
            masked_maximum_noise_amplitude = torch.max(self_noise_amplitude * brush_mask).squeeze().item()

            # Attenuate existing noise, or add new noise, in order to create a uniformly noisy latent.
            (uniform_noisy_latent, subtracted_noise, subtracted_noise_amplitude) = apply_uniform_noise(
                clean_latent=self_clean_latent,
                noisy_latent=self_noisy_latent,
                noise_amplitude=self_noise_amplitude,
                target_noise_amplitude=masked_maximum_noise_amplitude
            )

            # Denoise!
            denoised_latent = denoise(uniform_noisy_latent, masked_maximum_noise_amplitude * noise_bias)

            # Get the fraction of noise to add back in according to our attenuation function.
            renoise_fraction: float = attenuation_func(masked_maximum_noise_amplitude) / masked_maximum_noise_amplitude
            if math.isnan(renoise_fraction) or math.isinf(renoise_fraction):
                renoise_fraction = 0

            # Change the fraction so that noise is completely restored for unmasked regions.
            renoise_fraction: torch.Tensor = renoise_fraction * brush_mask + (1-brush_mask)

            # Restore the given fraction of noise removed by the denoiser step.
            renoised_latent: torch.Tensor = (
                    uniform_noisy_latent * renoise_fraction +
                    denoised_latent * (1-renoise_fraction)
            )
            renoised_amplitude: torch.Tensor = masked_maximum_noise_amplitude * renoise_fraction

            # Restore the noise removed to enforce uniform noise levels.
            renoised_latent += subtracted_noise
            renoised_amplitude += subtracted_noise_amplitude

            # Create a mask for pixels where noise_amplitude exceeds attenuated_max_amplitude
            # Shape: (batch_count, 1, height, width)
            mask: torch.Tensor = (self_noise_amplitude > renoised_amplitude) & (brush_mask > 0)
            expanded_mask = mask.expand_as(self_noisy_latent)  # Shape: (batch_count, channels, height, width)

            # Apply the updates conditionally using the mask
            self_clean_latent[expanded_mask] = denoised_latent[expanded_mask].to(self_clean_latent.dtype)
            self_noisy_latent[expanded_mask] = renoised_latent[expanded_mask].to(self_noisy_latent.dtype)
            self_noise_amplitude[mask] = renoised_amplitude[mask].to(self_noise_amplitude.dtype)
        else:
            # Add noise to the latent to make it uniform.
            uniform_noisy_latent, max_amplitude = apply_uniform_maximized_noise(
                self_noisy_latent,
                self_noise_amplitude)

            # Denoise!
            denoised_latent = denoise(uniform_noisy_latent, max_amplitude * noise_bias)

            # Use the attenuation function to calculate our next maximum noise level.
            attenuated_max_amplitude: float = attenuation_func(max_amplitude)

            # How much should we blend the noisy and denoised latent together, given the previous
            # and attenuated noise levels?
            renoise_fraction: float = attenuated_max_amplitude / max_amplitude
            if math.isnan(renoise_fraction) or math.isinf(renoise_fraction):
                renoise_fraction = 0

            # Re-add noise back in.
            renoised_latent = uniform_noisy_latent * renoise_fraction + denoised_latent * (1 - renoise_fraction)

            # Create a mask for pixels where noise_amplitude exceeds attenuated_max_amplitude
            # Shape: (batch_count, 1, height, width)
            mask = self_noise_amplitude > attenuated_max_amplitude

            # Broadcast the mask to match the shape of the latent tensors
            # Shape: (batch_count, channels, height, width)
            expanded_mask = mask.expand_as(self_noisy_latent)

            # Apply the updates conditionally using the mask
            self_noisy_latent[expanded_mask] = renoised_latent[expanded_mask].to(self_clean_latent.dtype)
            self_clean_latent[expanded_mask] = denoised_latent[expanded_mask].to(self_clean_latent.dtype)

            # Update noise_amplitude using the original mask (no need to expand)
            self_noise_amplitude[mask] = attenuated_max_amplitude

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
