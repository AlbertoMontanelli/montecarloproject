"""
Optimization of target thickness for pi- p interactions.

Goal: maximize significance and resolution of pi0 and eta peaks in the
two-photon invariant mass spectrum.
For each target thickness L:

- Simulate N_pi- interactions in the target of thickness L.
- For each interaction, sample the channel according to cross sections.
- For "other" channels, generate a toy background of two uncorrelated
  photons (log-uniform energy, uniform forward direction).
- For pi0 and eta channels, generate forced decays to two photons.
- Simulate detection, reconstruction, and selection of the two photons.
- Fill weighted histograms for pi0 and eta signals, and unweighted
  histogram for background.
- Fit the pi0 and eta peaks in the forced signal histograms to extract
  the mass resolution sigma_m.
- Integrate counts in +/- n*sigma_m windows around the nominal meson
  masses in the weighted histograms to get expected S and B and compute
  the significance S/sqrt(B).

Finally, plot the significance and mass resolution vs target thickness
L.
"""

import json

import numpy as np
import ROOT
from loguru import logger

from detector import detect_two_photons
from kinematics import generate_event_from_t
from scattering import (
    DATA_DIR,
    CrossSections,
    sample_depth,
)


def unit_vec_from_eta_phi(eta, phi):
    """
    Convert (eta, phi) to a 3D unit direction vector.

    Parameters
    ----------
    eta : float
        Pseudorapidity.
    phi : float
        Azimuthal angle.

    Returns
    -------
    nx, ny, nz : float
        Unit vector components.
    """
    theta = 2.0 * np.atan(np.exp(-eta))
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return nx, ny, nz


def opening_angle_from_meas(eta1, phi1, eta2, phi2):
    """
    Angle between two reconstructed directions from (eta,phi).

    Parameters
    ----------
    eta1, phi1 : float
        Direction from photon 1.
    eta2, phi2 : float
        Direction from photon 2.

    Returns
    -------
    alpha : float
        Opening angle in [0, pi].
    """
    n1x, n1y, n1z = unit_vec_from_eta_phi(eta1, phi1)
    n2x, n2y, n2z = unit_vec_from_eta_phi(eta2, phi2)
    dot = n1x * n2x + n1y * n2y + n1z * n2z
    dot = max(-1.0, min(1.0, dot))
    return np.acos(dot)


def mgg_from_meas(E1, E2, eta1, phi1, eta2, phi2):
    """
    Reconstruct invariant mass from measured energies and directions.

    Parameters
    ----------
    E1, E2 : float
        Measured photon energies in GeV.
    eta1, phi1 : float
        Direction from photon 1.
    eta2, phi2 : float
        Direction from photon 2.

    Returns
    -------
    m : float
        Reconstructed invariant mass in GeV.
    """
    alpha = opening_angle_from_meas(eta1, phi1, eta2, phi2)
    m2 = 2.0 * E1 * E2 * (1.0 - np.cos(alpha))
    return np.sqrt(max(m2, 0.0))


def toy_background_two_photons(rng):
    """
    Very simple background: two uncorrelated photons.

    - energies: log-uniform between 0.1 and 50 GeV
    - directions: uniform in phi, and forward in eta (eta in [2, 6])

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    g1_lab, g2_lab : ROOT.TLorentzVector
    """

    def one_gamma():
        # log-uniform energy
        Emin, Emax = 0.1, 50.0
        u = rng.random()
        E = Emin * (Emax / Emin) ** u

        # choose eta forward
        eta = rng.uniform(2.0, 6.0)
        phi = rng.uniform(-np.pi, np.pi)

        nx, ny, nz = unit_vec_from_eta_phi(eta, phi)
        px, py, pz = E * nx, E * ny, E * nz
        return ROOT.TLorentzVector(px, py, pz, E)

    return one_gamma(), one_gamma()


def expected_interactions(n_pions, L_cm, rng):
    """
    Compute the expected number of true interactions in the target.

    Parameters
    ----------
    n_pions : int
        Number of incident pi-.
    L_cm : float
        Target thickness (cm).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    N_pi0_phys : int
        Expected number of pi0 interactions.
    N_eta_phys : int
        Expected number of eta interactions.
    N_bkg_phys : int
        Expected number of background interactions.
    """
    # Fractions from cross sections
    p_pi0 = CrossSections.SIGMA_PI0_CM2 / CrossSections.SIGMA_TOT_CM2
    p_eta = CrossSections.SIGMA_ETA_CM2 / CrossSections.SIGMA_TOT_CM2
    p_other = 1.0 - p_pi0 - p_eta

    z_vtx = sample_depth(rng, n_pions)
    inside = z_vtx < L_cm
    idx_int = np.where(inside)[0]
    n_int = len(idx_int)

    # expected physical yields
    N_pi0_phys = n_int * p_pi0
    N_eta_phys = n_int * p_eta
    N_bkg_phys = n_int * p_other

    return N_pi0_phys, N_eta_phys, N_bkg_phys


def oversampled_background(N_bkg, L_cm, rng, N_pions):
    """
    Generate an oversampled background histogram.

    Parameters
    ----------
    N_bkg : int
        Number of background events to simulate.
    L_cm : float
        Target thickness (cm).
    rng : np.random.Generator
        Random number generator.
    N_pions : int
        Number of incident pi-.

    Returns
    -------
    h_bkg : ROOT.TH1
        Background oversampled histogram.
    meta : dict
        Metadata for the background simulation.
    """
    h_bkg = ROOT.TH1F(
        f"h_bkg_L{L_cm:.2f}",
        "bkg; m_{#gamma#gamma} [GeV]; Events",
        400,
        0.0,
        1.0,
    )
    # Estimate weights from expected background interactions
    _, _, N_bkg_phys = expected_interactions(N_pions, L_cm, rng)
    w_bkg = N_bkg_phys / N_bkg

    # Fill histogram with oversampled background events
    N_reco = 0
    for _ in range(N_bkg):
        # Sample z position inside the target
        zv = float(sample_depth(rng, 1)[0])
        while zv > L_cm:
            zv = float(sample_depth(rng, 1)[0])

        g1, g2 = toy_background_two_photons(rng)
        ok, meas = detect_two_photons(rng, g1, g2, zv, L_cm)
        if ok:
            m = mgg_from_meas(
                meas["E1_meas"],
                meas["E2_meas"],
                meas["eta1_meas"],
                meas["phi1_meas"],
                meas["eta2_meas"],
                meas["phi2_meas"],
            )
            h_bkg.Fill(m)
            N_reco += 1

    meta = {
        "N_expected": N_bkg_phys,
        "w_phys": w_bkg,
        "N_gen": int(N_bkg),
        "N_reco": int(N_reco),
        "L_cm": float(L_cm),
    }
    return h_bkg, meta


def oversampled_signal(N_sig, channel, L_cm, rng, N_pions):
    """
    Generate an oversampled signal histogram.

    Parameters
    ----------
    N_sig : int
        Number of forced signal events to generate.
    channel : str
        "pi0" or "eta".
    L_cm : float
        Target thickness (cm).
    rng : np.random.Generator
        Random number generator.
    N_pions : int
        Number of incident pi-.

    Returns
    -------
    h_meson : ROOT.TH1
        Signal oversampled histogram.
    meta : dict
        Metadata for the background simulation.
    """
    N_pi0_phys, N_eta_phys, _ = expected_interactions(N_pions, L_cm, rng)
    if channel == "pi0":
        h_meson = ROOT.TH1F(
            f"h_pi0_L{L_cm:.2f}",
            "pi0; m_{#gamma#gamma} [GeV]; Events",
            120,
            0.0,
            0.30,
        )
        w_meson = N_pi0_phys / N_sig
    elif channel == "eta":
        h_meson = ROOT.TH1F(
            f"h_eta_L{L_cm:.2f}",
            "eta; m_{#gamma#gamma} [GeV]; Events",
            160,
            0.35,
            0.75,
        )
        w_meson = N_eta_phys / N_sig

    N_reco = 0
    for _ in range(N_sig):
        # Sample z position inside the target
        zv = float(sample_depth(rng, 1)[0])
        while zv > L_cm:
            zv = float(sample_depth(rng, 1)[0])

        ev = generate_event_from_t(rng, 1, channel=channel)[0]
        g1, g2 = ev["g1_lab"], ev["g2_lab"]

        ok, meas = detect_two_photons(rng, g1, g2, zv, L_cm)
        if ok:
            m = mgg_from_meas(
                meas["E1_meas"],
                meas["E2_meas"],
                meas["eta1_meas"],
                meas["phi1_meas"],
                meas["eta2_meas"],
                meas["phi2_meas"],
            )
            h_meson.Fill(m)
            N_reco += 1

    meta = {
        "N_expected": N_pi0_phys if channel == "pi0" else N_eta_phys,
        "w_phys": w_meson,
        "N_gen": int(N_sig),
        "N_reco": int(N_reco),
        "L_cm": float(L_cm),
    }

    return h_meson, meta


def save_histograms(
    L_values,
    rng,
    N_pions=100_000,
    N_gen_bkg=10_000,
    N_gen_pi0=10_000,
    N_gen_eta=10_000,
):
    """
    Compute and save signal and background histograms for each L.

    Parameters
    ----------
    L_values : array-like
        Array of target thickness values to scan.
    rng : np.random.Generator
        Random number generator.
    N_pions : int
        Number of pi- to simulate per L.
    N_gen_pi0 : int
        Number of pi0 oversampled events to generate per interaction.
    N_gen_eta : int
        Number of eta oversampled events to generate per interaction.
    """
    if isinstance(L_values, (int, float)):
        L_values = [L_values]

    results_pi0 = []
    results_eta = []
    results_bkg = []

    dir = str(DATA_DIR / "histogram.root")
    f = ROOT.TFile(dir, "RECREATE")

    for L in L_values:
        h_eta, meta_eta = oversampled_signal(
            N_pions=N_pions, N_sig=N_gen_eta, channel="eta", L_cm=L, rng=rng
        )
        results_eta.append(meta_eta)
        h_eta.Write()
        logger.info(f"Write eta hist on root file for L={L:.2f}")

        h_pi0, meta_pi0 = oversampled_signal(
            N_pions=N_pions, N_sig=N_gen_pi0, channel="pi0", L_cm=L, rng=rng
        )
        results_pi0.append(meta_pi0)
        h_pi0.Write()
        logger.info(f"Write pi0 hist on root file for L={L:.2f}")

        h_bkg, meta_bkg = oversampled_background(
            N_pions=N_pions, N_bkg=N_gen_bkg, L_cm=L, rng=rng
        )
        results_bkg.append(meta_bkg)
        h_bkg.Write()
        logger.info(f"Write bkg hist on root file for L={L:.2f}")

        del h_eta, h_pi0, h_bkg

    f.Close()

    logger.info("Writing on ROOT file completed. Saving metadata")
    with open(DATA_DIR / "metadata_pi0.json", "w") as f:
        json.dump(results_pi0, f, indent=2)
    with open(DATA_DIR / "metadata_eta.json", "w") as f:
        json.dump(results_eta, f, indent=2)
    with open(DATA_DIR / "metadata_bkg.json", "w") as f:
        json.dump(results_bkg, f, indent=2)
    logger.info("Metadata saved on json files")


if __name__ == "__main__":
    L_values = np.linspace(0.5, 20.5, 21)  # cm
    seed = 42
    rng = np.random.default_rng(seed)
    save_histograms(L_values, rng)
