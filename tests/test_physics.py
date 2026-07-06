"""Unit tests for src.physics (pure numpy, no ROOT / uproot needed).

Run with: pytest tests/test_physics.py
"""

import numpy as np
import pytest

from src.physics import (
    delta_phi,
    transverse_mass,
    delta_r,
    neutrino_pz,
    reconstruct_w_mass,
)


# --------------------------------------------------------------------------- #
# transverse_mass
# --------------------------------------------------------------------------- #
def test_transverse_mass_back_to_back():
    """lep_pt = met = 40, back-to-back (dphi = pi) -> mT = 80 exactly.

    mT = sqrt(2*40*40*(1 - cos(pi))) = sqrt(2*1600*2) = sqrt(6400) = 80.
    """
    mt = transverse_mass(lep_pt=40.0, lep_phi=0.0, met_et=40.0, met_phi=np.pi)
    assert mt == pytest.approx(80.0, abs=1e-6)


def test_transverse_mass_collinear_is_zero():
    """Collinear lepton and MET (dphi = 0) -> mT = 0."""
    mt = transverse_mass(lep_pt=50.0, lep_phi=1.3, met_et=25.0, met_phi=1.3)
    assert mt == pytest.approx(0.0, abs=1e-9)


def test_transverse_mass_vectorized():
    """Works elementwise over arrays."""
    lep_pt = np.array([40.0, 50.0])
    met_et = np.array([40.0, 25.0])
    lep_phi = np.array([0.0, 1.3])
    met_phi = np.array([np.pi, 1.3])
    mt = transverse_mass(lep_pt, lep_phi, met_et, met_phi)
    assert mt.shape == (2,)
    np.testing.assert_allclose(mt, [80.0, 0.0], atol=1e-6)


# --------------------------------------------------------------------------- #
# delta_phi
# --------------------------------------------------------------------------- #
def test_delta_phi_wraps_two_pi():
    """A difference of 2*pi wraps to ~0."""
    assert delta_phi(2.0 * np.pi, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_delta_phi_wraps_above_pi():
    """pi + 0.1 wraps to -(pi - 0.1) (i.e. into (-pi, pi])."""
    assert delta_phi(np.pi + 0.1, 0.0) == pytest.approx(-(np.pi - 0.1), abs=1e-9)


def test_delta_phi_upper_edge_included():
    """Exactly +pi stays +pi (interval is (-pi, pi], closed at the top)."""
    assert delta_phi(np.pi, 0.0) == pytest.approx(np.pi, abs=1e-9)


def test_delta_phi_vectorized_in_range():
    """Vectorized, and every result lies in (-pi, pi]."""
    rng = np.random.default_rng(0)
    phi1 = rng.uniform(-10, 10, size=500)
    phi2 = rng.uniform(-10, 10, size=500)
    d = delta_phi(phi1, phi2)
    assert d.shape == (500,)
    assert np.all(d > -np.pi - 1e-9)
    assert np.all(d <= np.pi + 1e-9)


# --------------------------------------------------------------------------- #
# delta_r
# --------------------------------------------------------------------------- #
def test_delta_r_basic():
    """dR = sqrt(deta^2 + dphi^2) with wrapped dphi."""
    # deta = 0.3, dphi wraps 2*pi to ~0 -> dR ~= 0.3
    assert delta_r(1.0, 2.0 * np.pi, 0.7, 0.0) == pytest.approx(0.3, abs=1e-9)
    # deta = 3-0 = 3, dphi = 4 wraps to 4 - 2pi ~= -2.2831853 -> dR
    expected = np.sqrt(3.0**2 + (4.0 - 2.0 * np.pi) ** 2)
    assert delta_r(3.0, 4.0, 0.0, 0.0) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# neutrino_pz  (the physics reconstruction, Task B)
# --------------------------------------------------------------------------- #
def _massless_energy(px, py, pz):
    return np.sqrt(px * px + py * py + pz * pz)


def test_neutrino_pz_round_trip():
    """Build an exact W -> l nu event and recover the true neutrino pz.

    Strategy: pick a massless lepton 4-vector and a TRUE neutrino
    (px, py, pz). Compute the ACTUAL invariant mass of (lep + nu) and feed
    that value in as ``mW`` -- then by construction the constraint is exactly
    satisfied and one of the two solver roots must equal the true pz.
    """
    # Lepton (massless).
    lep_px, lep_py, lep_pz = 30.0, 12.0, 20.0
    lep_E = _massless_energy(lep_px, lep_py, lep_pz)

    # True neutrino (massless). MET = its transverse momentum.
    met_x, met_y = -18.0, 25.0
    true_pz = 15.0
    nu_E = _massless_energy(met_x, met_y, true_pz)

    # Exact invariant mass of (lep + nu) -> use as the W mass so the
    # constraint is satisfied with no rounding slack.
    E = lep_E + nu_E
    px = lep_px + met_x
    py = lep_py + met_y
    pz = lep_pz + true_pz
    m_true = np.sqrt(E * E - (px * px + py * py + pz * pz))

    pz1, pz2 = neutrino_pz(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, mW=m_true)

    # Two real python floats.
    assert isinstance(pz1, float) and isinstance(pz2, float)
    assert np.isfinite(pz1) and np.isfinite(pz2)

    # One of the two roots recovers the true pz.
    assert min(abs(pz1 - true_pz), abs(pz2 - true_pz)) < 1e-3

    # And the recovered pz reconstructs back to the input W mass.
    m_reco = reconstruct_w_mass(
        lep_px, lep_py, lep_pz, lep_E, met_x, met_y,
        true_pz if abs(pz1 - true_pz) < abs(pz2 - true_pz) else pz1,
    )
    assert float(m_reco) == pytest.approx(float(m_true), abs=1e-3)


def test_neutrino_pz_negative_discriminant_returns_equal_reals():
    """Smeared MET can kill the discriminant; expect two equal real floats.

    Force it by using a tiny mW so ``mu`` is small and the term under the
    sqrt goes negative. Result must be real, finite, and pz1 == pz2 (never
    nan or complex).
    """
    lep_px, lep_py, lep_pz = 40.0, 0.0, 10.0
    lep_E = _massless_energy(lep_px, lep_py, lep_pz)
    met_x, met_y = 0.0, 60.0  # large transverse MET, no way to hit a small mW

    pz1, pz2 = neutrino_pz(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, mW=1.0)

    assert isinstance(pz1, float) and isinstance(pz2, float)
    assert not isinstance(pz1, complex) and not isinstance(pz2, complex)
    assert np.isfinite(pz1) and np.isfinite(pz2)
    assert pz1 == pz2  # degenerate real part returned for both roots


def test_neutrino_pz_always_two_floats():
    """Whatever the inputs, the return is always a 2-tuple of python floats."""
    out = neutrino_pz(25.0, 5.0, 8.0, _massless_energy(25.0, 5.0, 8.0), 10.0, -4.0)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(v, float) for v in out)


# --------------------------------------------------------------------------- #
# reconstruct_w_mass
# --------------------------------------------------------------------------- #
def test_reconstruct_w_mass_matches_constraint():
    """With a solver root plugged back in, the reco mass equals the W mass.

    Uses the PDG mW; picks an event where the discriminant is positive so a
    real solution exists, then checks reconstruct_w_mass returns ~mW.
    """
    mW = 80.379
    lep_px, lep_py, lep_pz = 35.0, 10.0, 22.0
    lep_E = _massless_energy(lep_px, lep_py, lep_pz)
    met_x, met_y = -20.0, 15.0

    pz1, pz2 = neutrino_pz(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, mW=mW)
    # Discriminant should be positive here -> genuine two-solution case.
    assert pz1 != pz2

    for pz in (pz1, pz2):
        m = reconstruct_w_mass(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, pz)
        assert float(m) == pytest.approx(mW, abs=1e-3)


def test_reconstruct_w_mass_vectorized():
    """Vectorized over event arrays."""
    lep_px = np.array([35.0, 30.0])
    lep_py = np.array([10.0, 12.0])
    lep_pz = np.array([22.0, 20.0])
    lep_E = _massless_energy(lep_px, lep_py, lep_pz)
    met_x = np.array([-20.0, -18.0])
    met_y = np.array([15.0, 25.0])
    nu_pz = np.array([5.0, 15.0])

    m = reconstruct_w_mass(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, nu_pz)
    assert m.shape == (2,)
    assert np.all(np.isfinite(m))
