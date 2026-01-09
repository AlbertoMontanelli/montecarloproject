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

import ctypes

import numpy as np
import ROOT
from loguru import logger

from detector import detect_two_photons
from kinematics import M_ETA, M_PI0, generate_event_from_t
from scattering import CrossSections, sample_channels, sample_depth


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


def fit_meson_peak(h, m_meson, fit_range=None, sigma_guess=0.010):
    """
    Fit the meson peak in a m(gg) histogram with a Gaussian.

    This is meant to be used on signal-only (forced) spectra to extract
    the intrinsic mass resolution vs target thickness L.

    Parameters
    ----------
    h : ROOT.TH1
        Histogram of m(gg), possibly weighted. Sumw2() should be enabled.
    m0 : float
        Nominal pi0 mass in GeV (default: PDG value ~0.13498).
    fit_range : tuple or None
        (xmin, xmax) fit range in GeV. If None, uses a sensible default
        around the pi0 mass.
    sigma_guess : float
        Initial guess for Gaussian sigma in GeV.

    Returns
    -------
    out : dict
        Fit results. Keys:
        - ok : bool
        - mu, mu_err : float
        - sigma, sigma_err : float
        - chi2, ndf : float, int
        - fit_range : (float, float)
        - f : ROOT.TF1 (the fitted function) or None
    """
    out = {
        "ok": False,
        "mu": float("nan"),
        "mu_err": float("nan"),
        "sigma": float("nan"),
        "sigma_err": float("nan"),
        "chi2": float("nan"),
        "ndf": -1,
        "fit_range": None,
        "f": None,
    }

    if (h is None) or (h.GetEntries() <= 0):
        return out
    c = ROOT.TCanvas("meson peak fit", "meson peak fit")
    xax = h.GetXaxis()
    xmin_h = xax.GetXmin()
    xmax_h = xax.GetXmax()

    if fit_range is None and m_meson == M_PI0:
        # Default window wide enough to catch the smeared peak
        xmin = max(xmin_h, m_meson - 0.03)
        xmax = min(xmax_h, m_meson + 0.03)
    else:
        xmin = max(xmin_h, m_meson - 0.03)
        xmax = min(xmax_h, m_meson + 0.03)

    if xmax <= xmin:
        return out

    # Initial guesses from histogram
    bmax = h.GetMaximumBin()
    amp_guess = max(1e-12, h.GetBinContent(bmax))
    mu_guess = xax.GetBinCenter(bmax)

    f = ROOT.TF1(f"f_gaus_{h.GetName()}", "gaus", xmin, xmax)
    f.SetParameters(amp_guess, mu_guess, sigma_guess)
    fit_res = h.Fit(f, "ILRSQ")

    # ROOT returns a TFitResultPtr; check status if available
    status = int(fit_res.Status()) if hasattr(fit_res, "Status") else 0
    if status != 0:
        # Fit failed or did not converge
        out["fit_range"] = (xmin, xmax)
        out["f"] = f
        return out

    mu = f.GetParameter(1)
    sigma = abs(f.GetParameter(2))

    out.update(
        {
            "ok": True,
            "mu": float(mu),
            "mu_err": float(f.GetParError(1)),
            "sigma": float(sigma),
            "sigma_err": float(f.GetParError(2)),
            "chi2": float(f.GetChisquare()),
            "ndf": int(f.GetNDF()),
            "fit_range": (float(xmin), float(xmax)),
            "f": f,
        }
    )
    c.Draw()
    # input()
    return out


def compute_window_counts(h, m0, sigma, nsig=2.0):
    """
    Integrate histogram counts in a +/- nsig*sigma window around m0.

    Uses IntegralAndError so it works correctly with weighted
    histograms (Sumw2 enabled).

    Parameters
    ----------
    h : ROOT.TH1
        Histogram (possibly weighted).
    m0 : float
        Window center in GeV.
    sigma : float
        Window half-width is nsig*sigma (GeV).
    nsig : float
        Number of sigmas for the half-window.

    Returns
    -------
    out : dict
        Keys:
        - n : float
            Integral in the window.
        - err : float
            Statistical uncertainty from Sumw2.
        - low, high : float
            Window bounds.
        - b1, b2 : int
            Bin indices used (inclusive).
    """
    xax = h.GetXaxis()
    low = m0 - nsig * sigma
    high = m0 + nsig * sigma

    # Clip to histogram range
    low = max(low, xax.GetXmin())
    high = min(high, xax.GetXmax())

    b1 = xax.FindBin(low)
    b2 = xax.FindBin(high)

    err = ctypes.c_double()
    # IntegralAndError consider weighted histograms correctly
    n = h.IntegralAndError(b1, b2, err)

    return {
        "n": float(n),
        "err": float(err.value),
        "low": float(low),
        "high": float(high),
        "b1": int(b1),
        "b2": int(b2),
    }


def scan_L(
    rng,
    L_cm,
    n_pions,
    n_gen_pi0=1_000,
    n_gen_eta=1_000,
    z_cal_cm=1000.0,
    R_calo_cm=100.0,
    R_tgt_cm=10.0,
    deltaR_min=0.02,
):
    """
    Execute the main scan over target thickness L.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    L_cm : float
        Target thickness to scan (cm).
    n_pions : int
        Number of pi- to simulate per L.
    n_gen_pi0 : int
        Number of pi0 oversampled events to generate per interaction.
    n_gen_eta : int
        Number of eta oversampled events to generate per interaction.
    z_cal_cm : float
        Distance from the target to the calorimeter (cm).
    R_calo_cm : float
        Calorimeter radius (cm).
    R_tgt_cm : float
        Radius of the target (cm).
    deltaR_min : float
        Minimum delta R for photon detection.

    Returns
    -------
    results : dict
        Per-L metrics and spectra summaries.
    """
    # Fractions from cross sections
    p_pi0 = CrossSections.SIGMA_PI0_CM2 / CrossSections.SIGMA_TOT_CM2
    p_eta = CrossSections.SIGMA_ETA_CM2 / CrossSections.SIGMA_TOT_CM2

    # Histograms (weighted)
    h_pi0 = ROOT.TH1F(
        f"h_pi0_L{L_cm:.2f}",
        "pi0; m_{#gamma#gamma} [GeV]; Events",
        120,
        0.0,
        0.30,
    )
    h_pi0_shaped = h_pi0.Clone(f"h_pi0_shaped_L{L_cm:.2f}")
    h_eta = ROOT.TH1F(
        f"h_eta_L{L_cm:.2f}",
        "eta; m_{#gamma#gamma} [GeV]; Events",
        160,
        0.35,
        0.75,
    )
    h_eta_shaped = h_eta.Clone(f"h_eta_shaped_L{L_cm:.2f}")
    h_bkg = ROOT.TH1F(
        f"h_bkg_L{L_cm:.2f}",
        "bkg; m_{#gamma#gamma} [GeV]; Events",
        200,
        0.0,
        1.0,
    )

    # IMPORTANT for weighted errors
    h_pi0.Sumw2()
    h_eta.Sumw2()
    h_bkg.Sumw2()

    # --- realistic interactions + channel sampling
    # (for background normalisation) ---
    z_vtx = sample_depth(rng, n_pions)
    inside = z_vtx < L_cm
    idx_int = np.where(inside)[0]
    n_int = len(idx_int)

    # sample channels only to populate the background from "other"
    channels = sample_channels(rng, n_int)

    n_other_true = 0
    n_bkg_reco = 0

    for idx, ch in zip(idx_int, channels):
        if ch != "other":
            continue

        n_other_true += 1
        zv = float(z_vtx[idx])
        g1, g2 = toy_background_two_photons(rng)

        ok, meas = detect_two_photons(
            rng,
            g1,
            g2,
            zv,
            L_cm,
            z_cal_cm=z_cal_cm,
            R_calo_cm=R_calo_cm,
            E_thr_GeV=0.1,
            deltaR_min=deltaR_min,
            sigma_xy_cm=0.5,
            res_a=0.12,
            res_b=0.02,
            R_tgt_cm=R_tgt_cm,
            en_material_smearing=True,
            xy_material_smearing=True,
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
            h_bkg.Fill(m, 1.0)

    # --- forced signal samples with weights ---
    # expected physical yields
    N_pi0_phys = n_int * p_pi0
    N_eta_phys = n_int * p_eta

    w_pi0 = (N_pi0_phys / n_gen_pi0) if n_gen_pi0 > 0 else 0.0
    w_eta = (N_eta_phys / n_gen_eta) if n_gen_eta > 0 else 0.0

    n_pi0_reco = 0
    n_eta_reco = 0

    # For forced signals, sample z_vtx from the same conditional
    # distribution inside [0,L]
    def sample_z_inside(L_cm=L_cm):
        while True:
            z = float(sample_depth(rng, 1)[0])
            if z < L_cm:
                return z

    # pi0 forced
    for _ in range(n_gen_pi0):
        zv = sample_z_inside()
        ev = generate_event_from_t(rng, 1, channel="pi0")[0]
        g1, g2 = ev["g1_lab"], ev["g2_lab"]

        ok, meas = detect_two_photons(
            rng,
            g1,
            g2,
            zv,
            L_cm,
            z_cal_cm=z_cal_cm,
            R_calo_cm=R_calo_cm,
            E_thr_GeV=0.1,
            deltaR_min=deltaR_min,
            sigma_xy_cm=0.5,
            res_a=0.12,
            res_b=0.02,
            R_tgt_cm=R_tgt_cm,
            en_material_smearing=True,
            xy_material_smearing=True,
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
            h_pi0.Fill(m, w_pi0)
            h_pi0_shaped.Fill(m)

    # eta forced
    for _ in range(n_gen_eta):
        zv = sample_z_inside()
        ev = generate_event_from_t(rng, 1, channel="eta")[0]
        g1, g2 = ev["g1_lab"], ev["g2_lab"]

        ok, meas = detect_two_photons(
            rng,
            g1,
            g2,
            zv,
            L_cm,
            z_cal_cm=z_cal_cm,
            R_calo_cm=R_calo_cm,
            E_thr_GeV=0.1,
            deltaR_min=deltaR_min,
            sigma_xy_cm=0.5,
            res_a=0.12,
            res_b=0.02,
            R_tgt_cm=R_tgt_cm,
            en_material_smearing=True,
            xy_material_smearing=True,
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
            h_eta.Fill(m, w_eta)
            h_eta_shaped.Fill(m)

    out = {
        "n_int": n_int,
        "p_pi0": p_pi0,
        "p_eta": p_eta,
        "w_pi0": w_pi0,
        "w_eta": w_eta,
        "n_bkg_reco": n_bkg_reco,
        "n_pi0_reco_forced": n_pi0_reco,
        "n_eta_reco_forced": n_eta_reco,
        "h_pi0": h_pi0,
        "h_eta": h_eta,
        "h_bkg": h_bkg,
        "h_pi0_shaped": h_pi0_shaped,
        "h_eta_shaped": h_eta_shaped,
    }

    logger.info(
        f"L={L_cm:.4f} cm | int={n_int:7d} | "
        f"w_pi0={w_pi0:.3e} w_eta={w_eta:.3e} | "
        f"pi0 reco(forced)={n_pi0_reco:6d} | "
        f"eta reco(forced)={n_eta_reco:6d} | "
        f"bkg reco={n_bkg_reco:6d}"
    )

    return out


def compute_significance(S, B, kind="asymptotic"):
    """
    Compute significance from expected signal/background yields.

    Parameters
    ----------
    S, B : float
        Expected yields in the chosen mass window.
    kind : str
        "simple": S/sqrt(S+B)
        "asymptotic": Asimov approximation (recommended)

    Returns
    -------
    Z : float
    """
    if kind == "simple":
        return S / ((S + B) ** 0.5)

    # Asimov
    return (2.0 * ((S + B) * np.log(1.0 + S / B) - S)) ** 0.5


def main():
    """Execute the main function to run the L scan and plot results."""
    rng = np.random.default_rng(42)

    # scan thicknesses (cm)
    L_values = np.linspace(0.5, 20.0, 12)
    results = []

    for L in L_values:
        out = scan_L(
            rng,
            L,
            n_pions=100_000,
            n_gen_pi0=10_000,
            n_gen_eta=10_000,
            z_cal_cm=1000.0,
            R_calo_cm=100.0,
            R_tgt_cm=10.0,
            deltaR_min=0.02,
        )
        h_pi0 = out["h_pi0"]  # weighted
        h_eta = out["h_eta"]  # weighted
        h_pi0_shaped = out["h_pi0_shaped"]  # shape-only, weights=1
        h_eta_shaped = out["h_eta_shaped"]  # shape-only, weights=1
        h_bkg = out["h_bkg"]  # shape-only, weights=1
        # 1) Fit resolution on signal-only
        fit_pi0 = fit_meson_peak(
            h=h_pi0_shaped, m_meson=M_PI0, sigma_guess=0.023
        )
        fit_eta = fit_meson_peak(h=h_eta_shaped, m_meson=M_ETA)

        if not fit_pi0["ok"]:
            results.append(
                {
                    "L": L,
                    "ok": False,
                    "Z": 0.0,
                    "sigma": float("nan"),
                    "S": 0.0,
                    "B": 0.0,
                }
            )
            continue

        mu = fit_pi0["mu"]
        sigma = fit_pi0["sigma"]

        # 2) Count events in adaptive window +/- nsig*sigma
        nsig = 2.0
        win_sig = compute_window_counts(h_pi0, mu, sigma, nsig=nsig)
        win_bkg = compute_window_counts(h_bkg, mu, sigma, nsig=nsig)

        # 3) Convert to physical expected yields
        # Because h_* were filled with weight 1, the integral is
        # "number of selected generated events"
        S = win_sig["n"]
        B = win_bkg["n"]

        Z = compute_significance(S, B, kind="asymptotic")

        results.append(
            {
                "L": L,
                "ok": True,
                "mu": mu,
                "sigma": sigma,
                "S": S,
                "B": B,
                "Z": Z,
                "win_low": win_sig["low"],
                "win_high": win_sig["high"],
            }
        )

        logger.info(
            f"L={L:.4f} cm | mu={mu:.4f}| sigma={sigma:.4f} GeV |"
            f" S={S:.3e} | B={B:.3e} | Z={Z:.3e}"
        )

    # Qui poi fai plot di Z(L) e sigma(L), o salvi su file
    # ...

    return results


if __name__ == "__main__":
    main()
