# Changelog

Everything worth noting lands here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org). While we're pre-1.0, treat the minor version as the place breaking changes can show up.

## [0.2.0] — 2026-06-16

The array backend moves from NumPy to JAX (ADR-004). v0.1 was a proof of concept; this is the groundwork for the autodiff and batching v0.2 is aiming at.

### Changed

- The core runs on `jax.numpy`. `Belief` and `LinearGaussianModel` now hold `jax.Array`s, and the Kalman filter and LQR are pure `jnp`. If you were reaching past the public API and expecting `numpy.ndarray` off `belief.mean`, you'll get a `jax.Array` now — both still hand off to NumPy, so most code won't notice.
- Importing `cpomdp` switches JAX into float64 mode (`jax_enable_x64`) process-wide. The library is validated to 1e-9 against the RxInfer oracle and JAX defaults to float32, so this keeps the numbers right — but it does change float behaviour for any other JAX code in the same process.

### Added

- `Belief` and `LinearGaussianModel` are registered JAX pytrees, so they flow through `jit`, `vmap`, and `grad` as data.
- The Kalman step is split into pure, `jit`-compiled kernels — one filter step now `vmap`s over a batch of beliefs.

### Dependencies

- Added `jax` and `jaxtyping`. NumPy stays: JAX pulls it in anyway, and the RxInfer backend still hands real NumPy arrays across the Julia bridge.

## [0.1.1] — 2026-06-15

A metadata-only re-release, functionally identical to 0.1.0. The 0.1.0 release has
been removed from PyPI, so use 0.1.1.

### Changed

- Trimmed the author entry in the package metadata.
- README: the DECISIONS.md link is now an absolute URL, so it resolves on PyPI.

## [0.1.0] — 2026-06-15

The first cut. Linear-Gaussian active inference, end to end: perceive with a Kalman filter, act with LQR, all behind a pymdp-style `Agent`.

### Added

- `Agent`, the stateful façade you actually drive. `infer_states` to perceive, `sample_action` to act, the same loop pymdp users know. It remembers the last action it took, so you don't have to thread that back in by hand. Build it without a goal and it's a pure tracker that perceives but won't act.
- `LinearGaussianModel`, the generative model. Matrices are named for their role (`dynamics`, `control`, `sensor_model`, `dynamics_noise`, `sensor_noise`), with the control-theory letters (`A`/`B`/`C`/`Q`/`R`) kept as aliases for when you're reading the maths.
- `Belief`, an immutable Gaussian belief: a mean and a covariance, validated on the way in.
- `KalmanBackend`, exact Kalman filtering. Has an optional steady-state mode that solves the gain once up front and reuses it.
- `LQRController`, steady-state LQR action selection. For a linear-Gaussian sensor this *is* the expected-free-energy-optimal action rather than a stand-in for it (the why is in DECISIONS.md, ADR-003).
- `RxInferBackend`, an optional [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) (Julia) backend. It re-derives the same filtering results through completely separate machinery and exists as a correctness oracle for the native path. Lives behind the `rxinfer` extra so the core install stays Julia-free.
- `InferenceBackend`, the protocol the backends satisfy, so you can drop in your own engine.

This is pre-alpha. The API works and is tested against the RxInfer oracle, but it can still move before 1.0.
