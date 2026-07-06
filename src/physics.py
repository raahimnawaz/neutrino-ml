"""Pure-numpy collider kinematics for the W -> l nu analysis.

Everything here works in GeV (the read layer converts ATLAS Open Data's
native MeV to GeV before anything downstream runs). No ROOT / uproot import:
this module is deliberately dependency-light so it can be unit tested on its
own and reused from any environment.

Conventions
-----------
* Azimuthal angle ``phi`` is in radians; deltas are wrapped to ``(-pi, pi]``.
* The neutrino is invisible -- its transverse momentum IS the missing
  transverse energy ``(met_x, met_y)``. Its longitudinal momentum ``pz`` is
  unmeasured and is what :func:`neutrino_pz` reconstructs from the W-mass
  constraint.
* Leptons and the neutrino are treated as massless (a fine approximation at
  the W scale), so ``E^2 = px^2 + py^2 + pz^2``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "delta_phi",
    "transverse_mass",
    "delta_r",
    "neutrino_pz",
    "reconstruct_w_mass",
]


def delta_phi(phi1, phi2):
    """Signed azimuthal difference ``phi1 - phi2`` wrapped into ``(-pi, pi]``.

    Vectorized over numpy arrays. ``np.mod`` maps the raw difference into
    ``[0, 2*pi)``; subtracting ``pi`` and re-wrapping recentres it on zero so
    the result lands in the half-open interval ``(-pi, pi]``.
    """
    d = np.asarray(phi1, dtype=float) - np.asarray(phi2, dtype=float)
    # Wrap to (-pi, pi]: shift into (0, 2pi], then subtract pi.
    return np.pi - np.mod(np.pi - d, 2.0 * np.pi)


def transverse_mass(lep_pt, lep_phi, met_et, met_phi):
    """Transverse mass of the lepton + missing-energy (W candidate) system.

    ``mT = sqrt( 2 * lep_pt * met_et * (1 - cos(dphi)) )`` where ``dphi`` is
    the azimuthal opening between the lepton and the MET. For a real W the mT
    distribution has a Jacobian edge at ``m_W``. Vectorized; returns GeV.
    """
    lep_pt = np.asarray(lep_pt, dtype=float)
    met_et = np.asarray(met_et, dtype=float)
    dphi = delta_phi(lep_phi, met_phi)
    # 2*lep_pt*met_et*(1-cos) is >= 0 analytically; clip guards tiny negative
    # round-off before the sqrt.
    mt2 = 2.0 * lep_pt * met_et * (1.0 - np.cos(dphi))
    return np.sqrt(np.clip(mt2, 0.0, None))


def delta_r(eta1, phi1, eta2, phi2):
    """Angular separation ``dR = sqrt(deta^2 + dphi^2)`` in eta-phi space.

    ``dphi`` is wrapped into ``(-pi, pi]`` via :func:`delta_phi`. Vectorized.
    """
    deta = np.asarray(eta1, dtype=float) - np.asarray(eta2, dtype=float)
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta * deta + dphi * dphi)


def neutrino_pz(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, mW=80.379):
    r"""Reconstruct the neutrino longitudinal momentum from the W-mass constraint.

    A W -> l nu decay leaves the neutrino invisible: we measure its transverse
    momentum as the MET ``(met_x, met_y)`` but not its ``pz``. Imposing that
    the lepton + neutrino invariant mass equals ``m_W`` gives a quadratic in
    ``pz`` with (generically) two real roots.

    Derivation
    ----------
    Treat both the lepton and the neutrino as massless. The invariant-mass
    constraint is

        m_W^2 = (p_lep + p_nu)^2
              = 2 * (E_lep * E_nu - px_lep*px_nu - py_lep*py_nu - pz_lep*pz_nu)

    with the neutrino transverse momentum fixed to the MET
    (``px_nu = met_x``, ``py_nu = met_y``) and ``E_nu = sqrt(met_x^2 + met_y^2 + pz_nu^2)``.

    Collect the measured transverse pieces into

        mu = m_W^2 / 2 + (px_lep*met_x + py_lep*met_y)
           = m_W^2 / 2 + pt_lep . pt_nu          (transverse dot product)

    so the constraint becomes  ``mu = E_lep*E_nu - pz_lep*pz_nu``, i.e.
    ``E_lep*E_nu = mu + pz_lep*pz_nu``. Squaring and substituting
    ``E_nu^2 = ptnu2 + pz_nu^2`` (with ``ptnu2 = met_x^2 + met_y^2``) yields

        E_lep^2 * (ptnu2 + pz_nu^2) = mu^2 + 2*mu*pz_lep*pz_nu + pz_lep^2*pz_nu^2

    which rearranges to  ``A*pz_nu^2 + B*pz_nu + C = 0`` with

        A = E_lep^2 - pz_lep^2 = px_lep^2 + py_lep^2 = pt_lep^2   (massless lepton)
        B = -2 * mu * pz_lep
        C = E_lep^2 * ptnu2 - mu^2

    Hence the two roots

        pz_nu = ( mu*pz_lep  +/-  sqrt( mu^2*pz_lep^2 - pt_lep^2 * (E_lep^2*ptnu2 - mu^2) ) )
                / pt_lep^2

    which is the standard collider result (equivalently
    ``a = mu*pz_lep/pt_lep^2`` and the term under the sqrt is
    ``mu^2*pz_lep^2/pt_lep^4 - (E_lep^2*ptnu2 - mu^2)/pt_lep^2``).

    Complex / negative-discriminant case
    ------------------------------------
    When the MET is smeared by resolution the discriminant can go negative and
    there is no exact real solution. Following common practice we drop the
    imaginary part and return the real part ``mu*pz_lep/pt_lep^2`` for BOTH
    roots (so ``pz1 == pz2``). This function NEVER returns nan or complex.

    Parameters
    ----------
    lep_px, lep_py, lep_pz, lep_E : float
        Lepton four-momentum components in GeV.
    met_x, met_y : float
        Missing transverse energy components (neutrino px, py) in GeV.
    mW : float, optional
        W boson mass in GeV (default 80.379, the PDG value).

    Returns
    -------
    (pz1, pz2) : tuple[float, float]
        The two real solutions as python floats. Equal when the discriminant
        is negative.
    """
    lep_px = float(lep_px)
    lep_py = float(lep_py)
    lep_pz = float(lep_pz)
    lep_E = float(lep_E)
    met_x = float(met_x)
    met_y = float(met_y)

    pt_lep2 = lep_px * lep_px + lep_py * lep_py          # A = pt_lep^2
    ptnu2 = met_x * met_x + met_y * met_y                # neutrino pT^2

    # Degenerate lepton with no transverse momentum: the quadratic collapses.
    # This does not occur for physical selected leptons (pt > 30 GeV) but we
    # guard it so we never divide by zero.
    if pt_lep2 == 0.0:
        return (0.0, 0.0)

    mu = 0.5 * mW * mW + (lep_px * met_x + lep_py * met_y)

    # Real part of both roots: -B / (2A) = mu*pz_lep / pt_lep^2.
    base = mu * lep_pz / pt_lep2

    # Discriminant / (2A)^2, i.e. the term under the sqrt in the boxed formula:
    #   mu^2*pz_lep^2/pt_lep^4 - (E_lep^2*ptnu2 - mu^2)/pt_lep^2
    disc = (mu * mu * lep_pz * lep_pz) / (pt_lep2 * pt_lep2) - (
        lep_E * lep_E * ptnu2 - mu * mu
    ) / pt_lep2

    if disc < 0.0:
        # No real solution (smeared MET): return the real part twice.
        return (base, base)

    root = np.sqrt(disc)
    return (float(base + root), float(base - root))


def reconstruct_w_mass(lep_px, lep_py, lep_pz, lep_E, met_x, met_y, nu_pz):
    """Invariant mass of the lepton + reconstructed-neutrino system (GeV).

    The neutrino is built as ``(px, py, pz) = (met_x, met_y, nu_pz)`` with
    ``E = sqrt(px^2 + py^2 + pz^2)`` (massless). With a correct ``nu_pz`` from
    :func:`neutrino_pz` this returns ``m_W`` by construction; used as a sanity
    check that the reconstruction peaks where it should.

    Vectorized over numpy inputs.
    """
    lep_px = np.asarray(lep_px, dtype=float)
    lep_py = np.asarray(lep_py, dtype=float)
    lep_pz = np.asarray(lep_pz, dtype=float)
    lep_E = np.asarray(lep_E, dtype=float)
    met_x = np.asarray(met_x, dtype=float)
    met_y = np.asarray(met_y, dtype=float)
    nu_pz = np.asarray(nu_pz, dtype=float)

    nu_E = np.sqrt(met_x * met_x + met_y * met_y + nu_pz * nu_pz)

    E = lep_E + nu_E
    px = lep_px + met_x
    py = lep_py + met_y
    pz = lep_pz + nu_pz

    m2 = E * E - (px * px + py * py + pz * pz)
    return np.sqrt(np.clip(m2, 0.0, None))
