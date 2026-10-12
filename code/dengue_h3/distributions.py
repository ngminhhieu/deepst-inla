from __future__ import annotations

import numpy as np


SUPPORTED_LIKELIHOODS = ("lognormal", "negative_binomial")


def predictive_sample(
    parameter_1: np.ndarray,
    parameter_2: np.ndarray,
    *,
    likelihood: str,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw predictive case counts from network-predicted distribution parameters.

    The neural network always predicts distribution parameters first. This function
    is the single sampling implementation used at prediction time:

    * lognormal: ``parameter_1`` is Normal location on log1p(cases) and
      ``parameter_2`` is the Normal scale.
    * negative_binomial: ``parameter_1`` is the count mean ``mu`` and
      ``parameter_2`` is the NB dispersion/size ``r``.

    The returned array appends a final ``draw`` dimension.
    """
    if draws <= 0:
        raise ValueError("draws must be positive")
    likelihood = str(likelihood).lower()
    if likelihood == "lognormal":
        latent = rng.normal(
            loc=parameter_1[..., None],
            scale=parameter_2[..., None],
            size=(*parameter_1.shape, draws),
        )
        return np.clip(np.expm1(latent), 0.0, None).astype(np.float32)
    if likelihood == "negative_binomial":
        mu = np.clip(parameter_1, 1e-8, None)
        dispersion = np.clip(parameter_2, 1e-8, None)
        gamma_rate = rng.gamma(
            shape=dispersion[..., None],
            scale=(mu / dispersion)[..., None],
            size=(*mu.shape, draws),
        )
        return rng.poisson(gamma_rate).astype(np.float32)
    raise ValueError(
        f"Unsupported likelihood {likelihood!r}; choose one of {SUPPORTED_LIKELIHOODS}"
    )


def parameter_names(likelihood: str) -> tuple[str, str]:
    likelihood = str(likelihood).lower()
    if likelihood == "lognormal":
        return "log_location", "log_scale"
    if likelihood == "negative_binomial":
        return "mu", "dispersion"
    raise ValueError(
        f"Unsupported likelihood {likelihood!r}; choose one of {SUPPORTED_LIKELIHOODS}"
    )
