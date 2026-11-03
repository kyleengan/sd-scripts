# Session context: immiscible noise + DREAM for SDXL training

This is a handoff written at the end of a Claude Code session (2026-11-03) so the work can continue from another location: this sd-scripts copy is being moved into a subdirectory of a Kohya GUI, and the next step is to **add the new flags to that GUI**.

Paths and line numbers below are relative to the sd-scripts root and were accurate when this was written.

---

## 1. What was decided, and why

| Topic | Decision | Reasoning |
|---|---|---|
| Input perturbation | **Left as-is.** It already exists as `--ip_noise_gamma` / `--ip_noise_gamma_random_strength` (`library/loss.py`, in `get_noise_noisy_latents_and_timesteps`). | Two variants were considered and rejected as not worth adding: a per-timestep γ schedule and per-sample random strength. `add_noise` already scales the perturbation by √(1−ᾱ_t), so it falls off on its own. Per-sample strength is identical to the current behaviour at batch size 1. |
| Immiscible noise | **KNN variant only** (Li et al., "Improved Immiscible Diffusion"). For each latent, draw k noise candidates and keep the nearest by L2 distance. | Needs no scipy and works at batch size 1, which is common for SDXL. The original batch-wide linear-assignment variant was rejected because it does nothing at batch size 1 and adds a dependency. |
| Epsilon scaling (Ning et al., ICLR 2024, arXiv 2308.15321) | **Not implemented.** | It is a sampling-time fix for exposure bias and can't be moved into the training loss in a meaningful way. The other ideas were preview images with epsilon scaling and a λ_t calibration script; both were offered and not chosen. |
| DREAM (Zhou et al., CVPR 2024, arXiv 2312.00210) | **Implemented as opt-in, for both eps- and v-prediction.** Applies to `sdxl_train_network.py` / `train_network.py` and `sdxl_train.py`. **ControlNet scripts are excluded.** | DREAM is the training-side counterpart to epsilon scaling. Its gains are only demonstrated for super-resolution trained from scratch, so for SDXL LoRA it is **experimental** and needs an A/B test. |

### DREAM research notes
- **Algorithm:**
  - `λ_t = √(1−ᾱ_t)^p` (p=1 is best in the paper).
  - `Δ = λ_t·sg(ε − ε̂)`
  - `x_t' = x_t + √(1−ᾱ_t)·Δ`
  - target noise `ε' = ε + Δ`
  - The paper uses an L1 loss; here the existing loss type is kept.
- **Same as diffusers:** this matches `compute_dream_and_update_latents` (`--dream_training`, `--dream_detail_preservation` in `train_text_to_image.py`). Diffusers only supports eps-prediction.
- **v-prediction support:** recover `ε̂ = √ᾱ·v̂ + √(1−ᾱ)·x_t`. Since `x_t'` is just a forward-diffused x₀ with noise ε', the v target is `get_velocity(x0, ε', t)`.
- **Paper overhead:**
  - Training time: 1.08–1.38×.
  - Memory: 1.06–1.13×.
- **Estimated SDXL overhead:** about +25–35% per step. LoRA doesn't reduce this much, because gradients still have to pass back through the frozen UNet.
- **Paper results (super-resolution only):**
  - About 2× faster convergence.
  - Example (DIV2K, SR3): DREAM at 470k iterations reached PSNR 28.07 / FID 14.72, versus the baseline at 980k reaching 27.02 / 16.72.
  - The paper has some internal inconsistencies in its tables.
  - No text-to-image, fine-tuning or LoRA results exist.
- **Caveat:** the effective residual is roughly `(1+λ)(ε−ε̂)`, so part of DREAM acts like up to 4× loss upweighting at high noise levels. Some of the speedup may be reproducible with loss weighting or a higher LR, and it may make overfitting happen sooner.
- **Break-even:** DREAM only shortens training if it reaches the same quality in ≳1.3× fewer steps.

---

## 2. Code changes (all done, uncommitted)

| File | Change |
|---|---|
| `library/custom_train_functions.py:488` | New `immiscible_knn_noise(latents, k)` (no-grad). |
| `library/loss.py:42` | `get_noise_noisy_latents_and_timesteps` uses KNN noise when `immiscible_noise_k > 1`. This happens **before** noise offset and multires noise. The returned `noise` (the loss target) never includes the ip perturbation. |
| `library/loss.py:83` | New `apply_dream(args, noise_scheduler, noise, noisy_latents, timesteps, predict_fn, keep_mask=None) -> (noise', noisy_latents')`. The math runs in float32 and the input dtypes are preserved. `alphas_cumprod` stays on the CPU. |
| `library/args.py:485-503` | New CLI flags (see §3). |
| `library/args.py:891` | `verify_training_args(args, supports_dream=False)` raises if `--dream_training` is used by a script that doesn't support it. It also validates `immiscible_noise_k >= 1` and `dream_detail_preservation >= 0`. |
| `train_network.py:~265-310` | DREAM in `NetworkTrainer.get_noise_pred_and_target`: training only (skipped when `is_train=False`, so validation loss stays comparable); a `build_unet_input` helper handles the inpainting 9-channel input; differential output preservation samples are excluded via `keep_mask`. The gradient-checkpointing `requires_grad_` block now runs **after** DREAM. |
| `train_network.py:594` | Metadata: `ss_immiscible_noise_k`, `ss_dream_training`, `ss_dream_detail_preservation`. |
| `train_network.py:930` | Passes `supports_dream=True`, and raises if a subclass overrides `get_noise_pred_and_target` (Flux, SD3, Lumina, Anima, HunyuanImage, Moss). |
| `sdxl_train.py:107, ~710-745` | Passes `supports_dream=True`. DREAM runs after noisy latents are created; `build_unet_input` handles the inpainting concat. |
| `tests/library/test_loss.py` | New: 13 tests, all passing. |

---

## 3. Flags to expose in the GUI

All three are defined in `add_training_arguments` (`library/args.py`), so they exist on every training script. However, `--dream_training` **errors** on unsupported scripts.

| Flag | Type / default | Valid values | GUI notes |
|---|---|---|---|
| `--immiscible_noise_k` | int, `None` | ≥1 (1 or unset = off). Suggested starting range 4–8. | Number field; empty means off. Works with **all** scripts that use the shared noise function, including SD1.x and the SDXL ControlNet / LLLite scripts. Not used by Flux, SD3 or Lumina, which build their own noise. |
| `--dream_training` | store_true | — | Checkbox. Show it, or enable it, **only** for `sdxl_train_network.py`, `train_network.py` and `sdxl_train.py`. Label it experimental and note the ~+25–35% step time. |
| `--dream_detail_preservation` | float, `1.0` | ≥0 | Only meaningful when DREAM is on, so disable it otherwise. p=0 means λ=1 everywhere; larger p weakens the effect at low-noise timesteps. |

Related existing flags that may already be in the GUI: `--ip_noise_gamma` (float, ~0.1 recommended) and `--ip_noise_gamma_random_strength` (bool). Both work with the new features.

If the GUI writes a TOML config instead of CLI args, the keys are the same names without `--`: `immiscible_noise_k`, `dream_training`, `dream_detail_preservation`.

---

## 4. Verification status

- **Done:**
  - `pytest tests/library/test_loss.py` → 13 passed.
  - `py_compile` passes on all edited files.
  - The argparse parse of the new flags was checked.
  - These ran in a throwaway venv (Python 3.13 + torch 2.13 CPU + diffusers 0.40.0), because the user's global Python has no pytest or diffusers.
- **Not done** (no training environment or SDXL model was available):
  1. Short `sdxl_train_network.py` run with `--immiscible_noise_k 8 --ip_noise_gamma 0.1`.
  2. `--dream_training` with and without `--v_parameterization`: check the loss is finite, step time is about +25–35%, and the `ss_*` metadata is present.
  3. A few steps of `sdxl_train.py --dream_training`.
  4. A run with no new flags, to confirm behaviour is unchanged.
  5. DREAM A/B test: two LoRA runs with the same seed and data, compared by wall-clock time to an acceptable result.

---

## 4b. GUI integration (LoRA_Easy_Training_Scripts), done later in the same session

The GUI is at `C:\LoraTraining\LoRA_Easy_Training_Scripts`. Its bundled `backend/sd_scripts` is an older kohya build: the `sd3` branch at `5a18a03`, running in a venv with diffusers 0.25, transformers 4.44, accelerate 0.33 and torch 2.6. The new copy was therefore **not** dropped in. The changes were **ported** into the old layout instead:

- `backend/sd_scripts/library/custom_train_functions.py`: adds `immiscible_knn_noise`.
- `backend/sd_scripts/library/train_util.py`: KNN noise in `get_noise_noisy_latents_and_timesteps`; new `apply_dream`; the three CLI flags; `verify_training_args(args, supports_dream=False)`.
- `backend/sd_scripts/train_network.py`: DREAM call (the old code has no inpainting path); metadata; subclass guard.
- `backend/sd_scripts/sdxl_train.py`: DREAM call and `supports_dream=True`.

GUI changes:

- **`ui_files/NoiseOffsetUI.ui`:** adds two rows.
  - Immiscible Noise + k: SpinBox, range 2–64, default 8.
  - DREAM (experimental) + p: DoubleSpinBox, range 0.05–5, default 1.0.
  - The minimum is 0.05 rather than 0 because `backend/utils/validation.py:87` drops falsy values.
- **`ui_files/NoiseOffsetUI.py`:** regenerated from the `.ui` with `venv\Scripts\pyside6-uic.exe`.
- **`main_ui_files/NoiseOffsetUI.py`:**
  - Panel renamed to "Noise Args". The section key stays `noise_args`, so saved TOMLs still load.
  - New handlers. `set_flux` / `set_ti` / `update_support` disable DREAM for Flux and textual inversion, and immiscible for Flux.
- **`main_ui_files/ArgsListUI.py`:**
  - Keeps `self.noise_widget`.
  - Wires the Flux `Toggled` signal plus the sdxl/v2 signals to `set_flux`.
  - `set_ti_training` / `set_lora_training` now call `set_ti`.

Verified:

- The 13 unit tests pass against the bundled backend in its own venv. Pytest was loaded from a scratch `--target` directory, so the venv was not modified.
- 17 headless (offscreen) GUI checks pass.
- End-to-end: GUI args → `validate_args` → `process_args` TOML → `sdxl_train_network` parser gives `(6, True, 0.5)`.

**Updater caution:** `update.bat` runs `git submodule update --init --recursive`. If upstream changes these files, that command will refuse or conflict with the local edits in `backend/sd_scripts`. Keep a patch, or commit to a local branch inside the submodule.

## 5. Loose ends
- Git refused this repo with "dubious ownership" (the drive doesn't record ownership). The fix is `git config --global --add safe.directory <path>`, which was left to the user. **Nothing has been committed.**
- `library/custom_train_functions.py:174` has a pre-existing `SyntaxWarning: invalid escape sequence '\('`. It is unrelated to these changes.
- Since sd-scripts is moving into the GUI's tree, check how the GUI launches it (via `accelerate launch` or a configured script path) so the GUI calls the modified copy and not an older bundled one.
