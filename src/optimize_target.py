"""
Optimization of target thickness for pi- interactions.

Optimize target thickness L by simulating:
- pi- interaction depth and channel choice
- signal kinematics pi0/eta -> gamma gamma
- toy detector response
- reconstructed m_gg spectrum and simple figures of merit vs L
"""

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from loguru import logger

from detector import detect_two_photons
from kinematics import M_ETA, M_PI0, generate_event_from_t
from scattering import (
    sample_channels,
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


def scan_L(
    rng,
    L_list_cm,
    n_pions,
    z_cal_cm=1000.0,
    R_calo_cm=100.0,
    R_tgt_cm=100.0,
    deltaR_min=0.02,
):
    """
    Execute the main scan over target thickness L.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    L_list_cm : array-like
        List of target thicknesses to scan (cm).
    n_pions : int
        Number of pi- to simulate per L.
    z_cal_cm : float
        Distance from the target to the calorimeter (cm).
    R_calo_cm : float
        Calorimeter radius (cm)
    R_tgt_cm : float
        Radius of the target (cm).
    deltaR_min : float
        Minimum delta R for photon detection.

    Returns
    -------
    results : dict
        Per-L metrics and spectra summaries.
    """
    out = {}

    for L_cm in L_list_cm:
        # book arrays for mgg (signal and background)
        mgg_pi0 = []
        mgg_eta = []
        mgg_bkg = []

        n_int = 0
        n_pi0_true = 0
        n_eta_true = 0
        n_other_true = 0

        n_pi0_reco = 0
        n_eta_reco = 0
        n_bkg_reco = 0

        # sample depths once
        z_vtx = sample_depth(rng, n_pions)
        inside = z_vtx < L_cm

        # choose channels for those interacting
        idx_int = np.where(inside)[0]
        n_int = len(idx_int)
        if n_int == 0:
            out[L_cm] = {}
            continue

        channels = sample_channels(rng, n_int)
        # print(channels)

        # loop only over interacting events
        for idx, ch in zip(idx_int, channels):
            zv = float(z_vtx[idx])
            # print(ch)

            if ch == "pi0":
                n_pi0_true += 1
                ev = generate_event_from_t(rng, 1, channel=ch)[0]
                g1, g2 = ev["g1_lab"], ev["g2_lab"]

                ok, meas = detect_two_photons(
                    rng,
                    g1,
                    g2,
                    zv,
                    L_cm,
                    z_cal_cm,
                    E_thr_GeV=0.1,
                    deltaR_min=deltaR_min,
                    sigma_xy_cm=0.5,
                    res_a=0.12,
                    res_b=0.02,
                    R_tgt_cm=R_tgt_cm,
                    R_calo_cm=R_calo_cm,
                )
                if ok:
                    n_pi0_reco += 1
                    m = mgg_from_meas(
                        meas["E1_meas"],
                        meas["E2_meas"],
                        meas["eta1_meas"],
                        meas["phi1_meas"],
                        meas["eta2_meas"],
                        meas["phi2_meas"],
                    )
                    mgg_pi0.append(m)

            elif ch == "eta":
                n_eta_true += 1
                ev = generate_event_from_t(rng, 1, channel=ch)[0]
                g1, g2 = ev["g1_lab"], ev["g2_lab"]

                ok, meas = detect_two_photons(
                    rng,
                    g1,
                    g2,
                    zv,
                    L_cm,
                    z_cal_cm,
                    E_thr_GeV=0.1,
                    deltaR_min=deltaR_min,
                    sigma_xy_cm=0.5,
                    res_a=0.12,
                    res_b=0.02,
                    R_tgt_cm=R_tgt_cm,
                    R_calo_cm=R_calo_cm,
                )
                if ok:
                    n_eta_reco += 1
                    m = mgg_from_meas(
                        meas["E1_meas"],
                        meas["E2_meas"],
                        meas["eta1_meas"],
                        meas["phi1_meas"],
                        meas["eta2_meas"],
                        meas["phi2_meas"],
                    )
                    mgg_eta.append(m)

            else:
                # background: two uncorrelated photons produced at
                # same zv (toy)
                n_other_true += 1
                g1, g2 = toy_background_two_photons(rng)

                ok, meas = detect_two_photons(
                    rng,
                    g1,
                    g2,
                    zv,
                    L_cm,
                    z_cal_cm,
                    E_thr_GeV=0.1,
                    deltaR_min=deltaR_min,
                    sigma_xy_cm=0.5,
                    res_a=0.12,
                    res_b=0.02,
                    R_tgt_cm=R_tgt_cm,
                    R_calo_cm=R_calo_cm,
                )
                if ok:
                    n_bkg_reco += 1
                    m = mgg_from_meas(
                        meas["E1_meas"],
                        meas["E2_meas"],
                        meas["eta1_meas"],
                        meas["phi1_meas"],
                        meas["eta2_meas"],
                        meas["phi2_meas"],
                    )
                    mgg_bkg.append(m)

        # --------- simple metrics ---------
        # windows around true masses

        def count_in_window(arr, m0, w):
            arr = np.asarray(arr, dtype=float)
            return int(np.sum((arr > (m0 - w)) & (arr < (m0 + w))))

        # choose fixed windows (toy). You can make them L-dependent or
        # use fit later.
        S_pi0 = count_in_window(mgg_pi0, M_PI0, 0.02)
        S_eta = count_in_window(mgg_eta, M_ETA, 0.03)

        # background in same windows
        B_pi0 = count_in_window(mgg_bkg, M_PI0, 0.02)
        B_eta = count_in_window(mgg_bkg, M_ETA, 0.03)

        def signif(S, B):
            if S + B <= 0:
                return 0.0
            return S / np.sqrt(S + B)

        Z_pi0 = signif(S_pi0, B_pi0)
        Z_eta = signif(S_eta, B_eta)

        # mass resolution estimate: RMS in window (quick & dirty)
        def rms_in_window(arr, m0, w):
            arr = np.asarray(arr, dtype=float)
            sel = arr[(arr > (m0 - w)) & (arr < (m0 + w))]
            if len(sel) < 10:
                return float("nan")
            return float(np.std(sel))

        sigm_pi0 = rms_in_window(mgg_pi0, M_PI0, 0.03)
        sigm_eta = rms_in_window(mgg_eta, M_ETA, 0.05)

        out[L_cm] = {
            "n_int": n_int,
            "n_pi0_true": n_pi0_true,
            "n_eta_true": n_eta_true,
            "n_other_true": n_other_true,
            "n_pi0_reco": n_pi0_reco,
            "n_eta_reco": n_eta_reco,
            "n_bkg_reco": n_bkg_reco,
            "S_pi0": S_pi0,
            "B_pi0": B_pi0,
            "Z_pi0": Z_pi0,
            "sigma_m_pi0": sigm_pi0,
            "S_eta": S_eta,
            "B_eta": B_eta,
            "Z_eta": Z_eta,
            "sigma_m_eta": sigm_eta,
            "mgg_pi0": np.asarray(mgg_pi0, dtype=float),
            "mgg_eta": np.asarray(mgg_eta, dtype=float),
            "mgg_bkg": np.asarray(mgg_bkg, dtype=float),
        }

        logger.info(
            f"L={L_cm:6.2f} cm | int={n_int:7d} | pi0 reco={n_pi0_reco:5d} "
            f"| Z_pi0={Z_pi0:6.3f} | eta reco={n_eta_reco:5d} | "
            f"Z_eta={Z_eta:6.3f}"
        )
        fractions = [
            n_pi0_true / n_int,
            n_eta_true / n_int,
            n_other_true / n_int,
        ]
        logger.info(f"fractions: {fractions}")

    return out


def plot_scan(results):
    """
    Plot Z(L) and sigma_m(L) for pi0 and eta.

    Parameters
    ----------
    results : dict
        Output of scan_L function.
    """
    L = np.array(sorted(results.keys()), dtype=float)

    Z_pi0 = np.array([results[x]["Z_pi0"] for x in L], dtype=float)
    Z_eta = np.array([results[x]["Z_eta"] for x in L], dtype=float)
    s_pi0 = np.array([results[x]["sigma_m_pi0"] for x in L], dtype=float)
    s_eta = np.array([results[x]["sigma_m_eta"] for x in L], dtype=float)

    plt.figure()
    plt.plot(L, Z_pi0, marker="o", label="pi0")
    plt.plot(L, Z_eta, marker="o", label="eta")
    plt.xlabel("Target thickness L (cm)")
    plt.ylabel("Significance Z = S/sqrt(S+B)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(L, s_pi0, marker="o", label="pi0")
    plt.plot(L, s_eta, marker="o", label="eta")
    plt.xlabel("Target thickness L (cm)")
    plt.ylabel("Mass resolution (RMS in window) [GeV]")
    plt.grid(True)
    plt.legend()

    plt.show()


def main():
    """Execute the main function to run the L scan and plot results."""
    rng = np.random.default_rng(12345)

    # scan thicknesses (cm)
    L_list = np.linspace(0.5, 20.0, 12)

    # detector and target geometry
    z_cal_cm = 1000.0
    R_calo_cm = 100.0
    R_tgt_cm = 10.0

    # cluster separation
    deltaR_min = 0.02

    results = scan_L(
        rng,
        L_list,
        n_pions=10_000_000,
        z_cal_cm=z_cal_cm,
        R_calo_cm=R_calo_cm,
        R_tgt_cm=R_tgt_cm,
        deltaR_min=deltaR_min,
    )

    plot_scan(results)


if __name__ == "__main__":
    main()
