"""Loss / noise scheduling helpers extracted from ``library.train_util``.

This module hosts the per-step loss building blocks shared by every
trainer:

- :func:`get_timesteps` — sample diffusion timesteps for a batch.
- :func:`get_noise_noisy_latents_and_timesteps` — sample noise + timestep
  and produce noisy latents (with noise-offset / multires-noise / IP-noise
  variants applied, and optional immiscible KNN noise pairing).
- :func:`apply_dream` — DREAM rectification of the noisy input / noise target
  using one extra no-grad model prediction.
- :func:`get_huber_threshold_if_needed` — per-timestep Huber/Smooth-L1
  threshold schedule (exponential / SNR / constant).
- :func:`conditional_loss` — dispatch over ``l2 / l1 / huber / smooth_l1``.

These used to live in ``library.train_util`` and are still re-exported
from there for backward compatibility. New code should import from this
module.
"""

import math
from typing import Callable, Optional, Tuple

import torch

from library import custom_train_functions


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    # Sample noise that we'll add to the latents
    immiscible_k = getattr(args, "immiscible_noise_k", None)
    if immiscible_k is not None and immiscible_k > 1:
        # immiscible pairing: pick the nearest of k noise candidates for each latent
        noise = custom_train_functions.immiscible_knn_noise(latents, immiscible_k)
    else:
        noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
    timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # This moves the alphas_cumprod back to the CPU after it is moved in noise_scheduler.add_noise
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps


@torch.no_grad()
def apply_dream(
    args,
    noise_scheduler,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    keep_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DREAM: Diffusion Rectification and Estimation-Adaptive Models (Zhou et al., CVPR 2024, arXiv 2312.00210).

    Runs one extra no-grad forward pass and moves the training input/target toward the model's own
    prediction, reducing the train/sampling mismatch (exposure bias):

        lambda_t = sqrt(1 - alpha_bar_t) ** p
        delta    = lambda_t * sg(noise - eps_hat)
        x_t'     = x_t + sqrt(1 - alpha_bar_t) * delta
        noise'   = noise + delta          (the target is built from noise' as usual)

    eps_hat is the model output for eps-prediction; for v-prediction it is recovered as
    eps_hat = sqrt(alpha_bar) * v_hat + sqrt(1 - alpha_bar) * x_t.

    Args:
        predict_fn: maps noisy latents (same shape as ``noisy_latents``) to the raw model output.
        keep_mask: optional bool tensor (B,) of samples to leave unchanged (e.g. differential output preservation).

    Returns:
        (noise', noisy_latents'), with the dtypes of the inputs.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()
    alpha_bar = alphas_cumprod[timesteps.cpu()].to(device=noisy_latents.device, dtype=torch.float32)
    alpha_bar = alpha_bar.view(-1, *([1] * (noisy_latents.ndim - 1)))
    sqrt_alpha_bar = alpha_bar.sqrt()
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    p = getattr(args, "dream_detail_preservation", 1.0)
    dream_lambda = sqrt_one_minus_alpha_bar**p

    model_output = predict_fn(noisy_latents).float()
    if args.v_parameterization:
        eps_hat = sqrt_alpha_bar * model_output + sqrt_one_minus_alpha_bar * noisy_latents.float()
    else:
        eps_hat = model_output

    delta = dream_lambda * (noise.float() - eps_hat)
    if keep_mask is not None:
        delta = delta * (~keep_mask).to(delta.dtype).view(-1, *([1] * (delta.ndim - 1)))

    new_noisy_latents = (noisy_latents.float() + sqrt_one_minus_alpha_bar * delta).to(noisy_latents.dtype)
    new_noise = (noise.float() + delta).to(noise.dtype)
    return new_noise, new_noisy_latents


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps.cpu())
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full((b_size,), args.huber_c * args.huber_scale, device=timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str, huber_c: Optional[torch.Tensor] = None
):
    """
    NOTE: if you're using the scheduled version, huber_c has to depend on the timesteps already
    """
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        # Reshape huber_c to broadcast with model_pred (supports 4D and 5D tensors)
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        # Reshape huber_c to broadcast with model_pred (supports 4D and 5D tensors)
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss
