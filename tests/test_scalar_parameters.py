"""Regression tests: the models that smooth with ``gaussian_filter1d`` accept
the shape-(1,) parameter arrays astropy passes to ``evaluate``.

SciPy 1.18 (with NumPy 2.5) raises ``TypeError: only 0-dimensional arrays can
be converted to Python scalars`` when ``sigma`` is such an array; older versions
emit a ``DeprecationWarning``, which these tests turn into an error so that they
fail without the fix on either version.

Run with::

    python -m pytest tests/test_scalar_parameters.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.modeling import fitting
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import galspec                                                      # noqa: E402
from galspec.constants import ls_km                                 # noqa: E402
from galspec.line_profile import Line_Exponential                  # noqa: E402
from galspec.utils import _scalar                                   # noqa: E402

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")

WAVE = np.linspace(4400.0, 5500.0, 600)


def _expected_iron(model, sigma: float, z: float = 0.0, dv: float = 0.0) -> np.ndarray:
    """The template broadened to ``sigma`` [km/s], with scalar arithmetic only."""
    sig_pix = np.sqrt(sigma**2 - model._stddev_intr**2) / model._vchan
    flux = gaussian_filter1d(model._flux_temp, sig_pix)
    return np.interp(WAVE, model._wave_temp * (1 + z + dv / ls_km), flux)


def test_scalar_accepts_floats_and_size_one_arrays():
    assert _scalar(2.5) == 2.5
    assert _scalar(np.array([2.5])) == 2.5
    assert isinstance(_scalar(np.array([[2.5]])), float)
    with pytest.raises(ValueError):
        _scalar(np.array([1.0, 2.0]))


@pytest.mark.parametrize("template_name", ["park2022", "boroson1992"])
def test_iron_template_new_call(template_name: str):
    model = galspec.IronTemplate_new(amplitude=1.0, sigma=1500.0, dv=200.0,
                                     template_name=template_name)
    np.testing.assert_allclose(model(WAVE),
                               _expected_iron(model, 1500.0, dv=200.0), rtol=1e-12)


def test_iron_template_new_evaluate_with_array_parameters():
    model = galspec.IronTemplate_new(amplitude=1.0, sigma=1500.0)
    one = np.array([1.0])
    got = model.evaluate(WAVE, one, np.array([1500.0]), np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(got, _expected_iron(model, 1500.0), rtol=1e-12)


def test_iron_template_call():
    model = galspec.IronTemplate(amplitude=2.0, stddev=1500.0, z=0.01)
    np.testing.assert_allclose(model(WAVE),
                               2.0 * _expected_iron(model, 1500.0, z=0.01), rtol=1e-12)


def test_iron_template_tied_without_kernel_call():
    model = galspec.IronTemplate_tied(amplitude=1.0, stddev=1500.0)
    np.testing.assert_allclose(model(WAVE), _expected_iron(model, 1500.0), rtol=1e-12)


def test_iron_template_tied_with_kernel_call():
    kernel = galspec.Line_MultiGauss(n_components=1, amp_c=1.0, dv_c=0.0, sigma_c=1500.0,
                                     wavec=4862.68)
    model = galspec.IronTemplate_tied(amplitude=1.0, kernel=kernel)
    np.testing.assert_allclose(model(WAVE), _expected_iron(model, 1500.0), rtol=1e-12)


def test_iron_template_new_can_be_fitted():
    truth = galspec.IronTemplate_new(amplitude=1.0, sigma=1800.0, dv=150.0)
    flux = truth(WAVE)
    start = galspec.IronTemplate_new(amplitude=0.8, sigma=1500.0, dv=0.0)
    fitted = fitting.LevMarLSQFitter()(start, WAVE, flux, maxiter=500)
    assert fitted.sigma.value == pytest.approx(1800.0, rel=1e-3)
    assert fitted.dv.value == pytest.approx(150.0, abs=1.0)


def test_line_exponential_call():
    wavec = 4862.68
    wave = np.linspace(4700.0, 5000.0, 801)
    model = Line_Exponential(amplitude=1.0, dv=0.0, sigma=300.0, w=500.0, wavec=wavec)
    velocity = (wave - wavec) / wavec * ls_km
    expected = gaussian_filter1d(np.exp(-np.abs(velocity / 500.0)),
                                 300.0 / (velocity[1] - velocity[0]))
    np.testing.assert_allclose(model(wave), expected, rtol=1e-12)


def test_log_rebin_with_a_given_velscale():
    from galspec.stellar_continuum import log_rebin

    wave = np.linspace(4000.0, 6000.0, 1001)
    flux_rebin, ln_wave = log_rebin(wave, np.ones_like(wave), velscale=100.0)
    np.testing.assert_allclose(np.diff(ln_wave), 100.0 / ls_km, rtol=1e-10)
    np.testing.assert_allclose(flux_rebin[1:-1], 1.0, rtol=1e-10)
