"""Tests for :class:`galspec.Line_Lorentzian`.

Run with::

    python -m pytest tests/test_line_lorentzian.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.modeling import fitting

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import galspec                                                      # noqa: E402
from galspec.constants import ls_km                                 # noqa: E402

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")

WAVEC = 4862.68  # Hbeta, vacuum [A]


def _wave_from_velocity(velocity: np.ndarray) -> np.ndarray:
    """Wavelength [A] at velocity [km/s] from WAVEC, as the model defines it."""
    return WAVEC * (1 + velocity / ls_km)


def test_lorentzian_is_exported():
    assert "Line_Lorentzian" in galspec.line_profile.__all__
    assert galspec.Line_Lorentzian is galspec.line_profile.Line_Lorentzian


def test_peak_at_dv_and_fwhm_recovered():
    amplitude, dv, fwhm = 3.0, 350.0, 4000.0
    model = galspec.Line_Lorentzian(amplitude=amplitude, dv=dv, fwhm=fwhm, wavec=WAVEC)
    velocity = np.linspace(-20000.0, 20000.0, 400001)  # 0.1 km/s steps
    flux = model(_wave_from_velocity(velocity))

    assert velocity[np.argmax(flux)] == pytest.approx(dv, abs=0.1)
    assert flux.max() == pytest.approx(amplitude, rel=1e-9)
    above = velocity[flux >= amplitude / 2]
    assert above[-1] - above[0] == pytest.approx(fwhm, abs=0.2)


def test_integral_matches_analytic_flux():
    amplitude, fwhm = 2.0, 3000.0
    half_window = 1.0e6  # km/s, wide enough that the tails are small but finite
    model = galspec.Line_Lorentzian(amplitude=amplitude, dv=0.0, fwhm=fwhm, wavec=WAVEC)
    wave = _wave_from_velocity(np.linspace(-half_window, half_window, 2000001))
    integral = np.trapezoid(model(wave), wave)

    infinite = np.pi / 2 * amplitude * fwhm * WAVEC / ls_km
    # the window misses the tails beyond +-half_window, a fraction
    # 1 - (2 / pi) * arctan(2 * half_window / fwhm), about fwhm / (pi * half_window)
    covered = 2 / np.pi * np.arctan(2 * half_window / fwhm)
    assert integral == pytest.approx(infinite * covered, rel=1e-6)
    assert integral / infinite == pytest.approx(1 - fwhm / (np.pi * half_window), rel=1e-6)


def test_evaluate_with_size_one_parameter_arrays():
    wave = np.linspace(4700.0, 5000.0, 301)
    one = np.array
    got = galspec.Line_Lorentzian.evaluate(wave, one([2.0]), one([100.0]), one([3000.0]),
                                           one([WAVEC]))
    velocity = (wave - WAVEC) / WAVEC * ls_km
    np.testing.assert_allclose(got, 2.0 / (1 + (2 * (velocity - 100.0) / 3000.0)**2),
                               rtol=1e-12)
    model = galspec.Line_Lorentzian(amplitude=2.0, dv=100.0, fwhm=3000.0, wavec=WAVEC)
    np.testing.assert_allclose(model(wave), got, rtol=1e-12)


def test_trf_fit_recovers_dv_and_fwhm():
    rng = np.random.default_rng(20261029)
    truth = galspec.Line_Lorentzian(amplitude=5.0, dv=-250.0, fwhm=3500.0, wavec=WAVEC)
    wave = np.linspace(4700.0, 5030.0, 600)
    error = np.full_like(wave, 0.1)
    flux = truth(wave) + rng.normal(0.0, error)

    start = galspec.Line_Lorentzian(amplitude=4.0, dv=0.0, fwhm=2500.0, wavec=WAVEC)
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    fitted = fitter(start, wave, flux, weights=1 / error, maxiter=1000)

    names = list(fitted.param_names)
    free = [name for name in names if not fitted.fixed[name]]
    std = dict(zip(free, np.sqrt(np.diag(fitter.fit_info["param_cov"]))))
    assert abs(fitted.dv.value - truth.dv.value) < 4 * std["dv"]
    assert abs(fitted.fwhm.value - truth.fwhm.value) < 4 * std["fwhm"]
    assert std["fwhm"] < 0.05 * truth.fwhm.value
