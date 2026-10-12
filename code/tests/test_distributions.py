import numpy as np

from dengue_h3.distributions import parameter_names, predictive_sample


def test_negative_binomial_sampling_has_expected_shape_and_nonnegative_counts():
    rng = np.random.default_rng(7)
    mean = np.full((2, 3, 3), 4.0)
    dispersion = np.full_like(mean, 2.0)
    samples = predictive_sample(
        mean, dispersion, likelihood="negative_binomial", draws=11, rng=rng
    )
    assert samples.shape == (2, 3, 3, 11)
    assert np.all(samples >= 0)
    assert np.all(np.equal(samples, np.floor(samples)))
    assert parameter_names("negative_binomial") == ("mu", "dispersion")


def test_lognormal_sampling_has_expected_shape_and_nonnegative_values():
    rng = np.random.default_rng(9)
    location = np.zeros((1, 2, 3))
    scale = np.full_like(location, 0.2)
    samples = predictive_sample(
        location, scale, likelihood="lognormal", draws=13, rng=rng
    )
    assert samples.shape == (1, 2, 3, 13)
    assert np.all(samples >= 0)
    assert parameter_names("lognormal") == ("log_location", "log_scale")
