from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler

from library import custom_train_functions
from library.loss import apply_dream, get_noise_noisy_latents_and_timesteps


def make_scheduler(prediction_type="epsilon"):
    return DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type=prediction_type,
    )


def make_args(**kwargs):
    defaults = dict(
        noise_offset=None,
        noise_offset_random_strength=False,
        adaptive_noise_scale=None,
        multires_noise_iterations=None,
        multires_noise_discount=0.3,
        min_timestep=None,
        max_timestep=None,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        immiscible_noise_k=None,
        v_parameterization=False,
        dream_detail_preservation=1.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def alpha_bar_for(scheduler, timesteps, ndim=4):
    ab = scheduler.alphas_cumprod.cpu()[timesteps.cpu()].float()
    return ab.view(-1, *([1] * (ndim - 1)))


# ---------------------------------------------------------------------------
# immiscible KNN noise
# ---------------------------------------------------------------------------


def test_immiscible_shape_dtype():
    latents = torch.randn(3, 4, 16, 16)
    noise = custom_train_functions.immiscible_knn_noise(latents, 4)
    assert noise.shape == latents.shape
    assert noise.dtype == latents.dtype


def test_immiscible_returns_nearest_candidate():
    latents = torch.randn(2, 4, 8, 8)
    k = 5
    torch.manual_seed(123)
    noise = custom_train_functions.immiscible_knn_noise(latents, k)
    torch.manual_seed(123)
    candidates = torch.randn((2, k, 4, 8, 8))
    dist = (candidates - latents.unsqueeze(1)).flatten(2).pow(2).sum(-1)
    expected = candidates[torch.arange(2), dist.argmin(1)]
    assert torch.equal(noise, expected)


def test_immiscible_reduces_distance():
    torch.manual_seed(0)
    latents = torch.randn(64, 4, 8, 8)
    plain = torch.randn_like(latents)
    paired = custom_train_functions.immiscible_knn_noise(latents, 8)
    d_plain = (plain - latents).flatten(1).pow(2).sum(-1).mean()
    d_paired = (paired - latents).flatten(1).pow(2).sum(-1).mean()
    assert d_paired < d_plain


@pytest.mark.parametrize("k", [None, 1])
def test_immiscible_disabled_matches_baseline(k):
    scheduler = make_scheduler()
    latents = torch.randn(2, 4, 8, 8)
    torch.manual_seed(7)
    noise_a, noisy_a, t_a = get_noise_noisy_latents_and_timesteps(make_args(), scheduler, latents)
    torch.manual_seed(7)
    noise_b, noisy_b, t_b = get_noise_noisy_latents_and_timesteps(make_args(immiscible_noise_k=k), scheduler, latents)
    assert torch.equal(noise_a, noise_b) and torch.equal(noisy_a, noisy_b) and torch.equal(t_a, t_b)


def test_immiscible_batch_size_one_with_ip_noise():
    scheduler = make_scheduler()
    latents = torch.randn(1, 4, 8, 8)
    args = make_args(immiscible_noise_k=4, ip_noise_gamma=0.1)
    noise, noisy, t = get_noise_noisy_latents_and_timesteps(args, scheduler, latents)
    ab = alpha_bar_for(scheduler, t)
    # returned target noise excludes the input perturbation
    recovered = (noisy - ab.sqrt() * latents) / (1 - ab).sqrt()
    assert not torch.allclose(recovered, noise, atol=1e-4)


# ---------------------------------------------------------------------------
# DREAM
# ---------------------------------------------------------------------------


def _setup(b=3):
    scheduler = make_scheduler()
    latents = torch.randn(b, 4, 8, 8)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([10, 500, 990][:b])
    noisy = scheduler.add_noise(latents, noise, timesteps)
    return scheduler, latents, noise, timesteps, noisy


def test_dream_perfect_prediction_is_noop():
    scheduler, latents, noise, t, noisy = _setup()
    new_noise, new_noisy = apply_dream(make_args(), scheduler, noise, noisy, t, lambda x: noise)
    assert torch.allclose(new_noise, noise) and torch.allclose(new_noisy, noisy)


def test_dream_perfect_v_prediction_is_noop():
    scheduler, latents, noise, t, noisy = _setup()
    v = scheduler.get_velocity(latents, noise, t)
    new_noise, new_noisy = apply_dream(make_args(v_parameterization=True), scheduler, noise, noisy, t, lambda x: v)
    assert torch.allclose(new_noise, noise, atol=1e-5) and torch.allclose(new_noisy, noisy, atol=1e-5)


def test_dream_matches_reference_eps():
    scheduler, latents, noise, t, noisy = _setup()
    pred = torch.randn_like(noise)
    new_noise, new_noisy = apply_dream(make_args(), scheduler, noise, noisy, t, lambda x: pred)

    # reference: diffusers' compute_dream_and_update_latents math
    ab = alpha_bar_for(scheduler, t)
    s = (1 - ab).sqrt()
    delta = (noise - pred) * s**1.0
    assert torch.allclose(new_noisy, noisy + s * delta, atol=1e-5)
    assert torch.allclose(new_noise, noise + delta, atol=1e-5)
    # rectified input is a valid forward-diffused sample of the rectified noise
    assert torch.allclose(new_noisy, scheduler.add_noise(latents, new_noise, t), atol=1e-4)


def test_dream_v_pred_recovers_eps_hat():
    scheduler, latents, noise, t, noisy = _setup()
    eps_fake = torch.randn_like(noise)
    # a v prediction consistent with x_t: x0_hat = (x_t - s * eps_fake) / a, v_hat = a * eps_fake - s * x0_hat
    ab = alpha_bar_for(scheduler, t)
    a, s = ab.sqrt(), (1 - ab).sqrt()
    x0_hat = (noisy - s * eps_fake) / a
    v_fake = a * eps_fake - s * x0_hat
    args = make_args(v_parameterization=True)
    new_noise, _ = apply_dream(args, scheduler, noise, noisy, t, lambda x: v_fake)
    eps_noise, _ = apply_dream(make_args(), scheduler, noise, noisy, t, lambda x: eps_fake)
    assert torch.allclose(new_noise, eps_noise, atol=1e-4)


def test_dream_lambda_schedule():
    scheduler, latents, noise, t, noisy = _setup()
    pred = torch.zeros_like(noise)
    # p = 0 -> lambda = 1 -> new_noise = 2 * noise when prediction is 0
    new_noise, _ = apply_dream(make_args(dream_detail_preservation=0.0), scheduler, noise, noisy, t, lambda x: pred)
    assert torch.allclose(new_noise, 2 * noise, atol=1e-5)
    # p = 1 -> change is smallest at low t
    new_noise, _ = apply_dream(make_args(), scheduler, noise, noisy, t, lambda x: pred)
    rel = ((new_noise - noise).flatten(1).norm(dim=1) / noise.flatten(1).norm(dim=1)).tolist()
    assert rel[0] < rel[1] < rel[2]
    assert rel[0] < 0.2


def test_dream_keep_mask():
    scheduler, latents, noise, t, noisy = _setup()
    pred = torch.randn_like(noise)
    keep = torch.tensor([True, False, True])
    new_noise, new_noisy = apply_dream(make_args(), scheduler, noise, noisy, t, lambda x: pred, keep)
    assert torch.equal(new_noise[0], noise[0]) and torch.equal(new_noisy[2], noisy[2])
    assert not torch.allclose(new_noise[1], noise[1])


def test_dream_preserves_dtype_and_cpu_alphas():
    scheduler, latents, noise, t, noisy = _setup()
    new_noise, new_noisy = apply_dream(
        make_args(), scheduler, noise.half(), noisy.half(), t, lambda x: torch.zeros_like(x)
    )
    assert new_noise.dtype == torch.float16 and new_noisy.dtype == torch.float16
    assert scheduler.alphas_cumprod.device.type == "cpu"
