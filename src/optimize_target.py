"""
Use metrics to scan target length.

Plot the results.
"""

import ctypes
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from loguru import logger

from kinematics import M_ETA, M_PI0
from scattering import PLOT_DIR

# Update matplotlib.pyplot parameters.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


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


def fit_meson_peak(h, m_meson, L, k_range=2, n_iter=2):
    """
    Fit the meson peak and plot the results.

    Fit the m(gg) histogram h with a Gaussian in the range
    mu ± k_range*sigma, iterating n_iter times to refine mu and sigma.
    This is meant to be used on signal-only (forced) spectra to extract
    the intrinsic mass resolution vs target thickness L.

    Parameters
    ----------
    h : ROOT.TH1
        Histogram of m(gg), possibly weighted. Sumw2() should be enabled.
    m_meson : float
        Meson mass to fit (M_PI0 or M_ETA).
    L : float
        Target thickness (cm).
    k_range : float
        Fit range is mu ± k_range*sigma.
    n_iter : int
        Number of fit iterations to refine mu and sigma.

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
    xax = h.GetXaxis()
    xmin = xax.GetXmin()
    xmax = xax.GetXmax()

    # Initial guesses from histogram
    bmax = h.GetMaximumBin()
    amp_guess = max(1e-12, h.GetBinContent(bmax))
    if m_meson == M_PI0:
        mu_guess = M_PI0
        sigma_guess = 0.03
        meson_name = r"\pi^{0}"
    else:
        mu_guess = M_ETA
        sigma_guess = 0.05
        meson_name = r"\eta"

    # Iterative fit to refine mu and sigma
    for _ in range(n_iter):
        xmin_fit = max(xmin, mu_guess - k_range * sigma_guess)
        xmax_fit = min(xmax, mu_guess + k_range * sigma_guess)

        f = ROOT.TF1(f"f_gaus_{h.GetName()}", "gaus", xmin_fit, xmax_fit)

        amp_guess = max(1e-12, h.GetBinContent(bmax))
        f.SetParameters(amp_guess, mu_guess, sigma_guess)

        h.Fit(f, "ILRSQ")
        mu_guess = float(f.GetParameter(1))
        sigma_guess = abs(float(f.GetParameter(2)))

    # Plot with final fit results
    c = ROOT.TCanvas(
        f"Fit_{meson_name}_{L:.2f}",
        f"Fit {meson_name} peak at L={L:.2f} cm",
        800,
        600,
    )
    c.SetBatch(True)

    c.SetLeftMargin(0.08)
    c.SetRightMargin(0.04)
    c.SetBottomMargin(0.1)
    c.SetTopMargin(0.08)

    h.SetStats(0)
    h.SetTitle(f"Fit {meson_name} peak at L={L:.2f} cm")

    ay = h.GetYaxis()
    ay.SetTitle("Events")
    ay.SetLabelFont(42)
    ay.SetTitleFont(42)
    ay.SetLabelSize(0.040)
    ay.SetTitleSize(0.042)
    ay.SetTitleOffset(0.9)

    ax = h.GetXaxis()
    ax.SetTitle("")
    ax.SetLabelOffset(999)
    ax.SetTickLength(0)

    h.Draw()
    f.Draw("same")

    num_bins = h.GetNbinsX()
    num_entries = h.GetEntries()
    ndof = f.GetNDF()
    chi_square = f.GetChisquare()

    mu = mu_guess * 1e3
    mu_err = float(f.GetParError(1)) * 1e3
    sigma = sigma_guess * 1e3
    sigma_err = float(f.GetParError(2)) * 1e3

    legend = ROOT.TLegend(0.60, 0.50, 0.94, 0.90)
    legend.SetBorderSize(1)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.038)
    legend.SetMargin(0.15)
    legend.SetEntrySeparation(0.35)

    legend.AddEntry(h, f"Entries: {num_entries:.0f}", "l")
    legend.AddEntry(f, "Fit: A e^{-(x-#mu)^{2}/(2#sigma^{2})}", "l")
    legend.AddEntry("", f"Bins: {num_bins:.0f}", "")
    legend.AddEntry("", f"L = {L:.2f} cm", "")
    legend.AddEntry("", f"#mu = {mu:.2f} #pm {mu_err:.2f} MeV", "")
    legend.AddEntry("", f"#sigma = {sigma:.2f} #pm {sigma_err:.2f} MeV", "")
    legend.AddEntry("", f"#chi^{{2}}/ndof = {chi_square:.0f}/{ndof:.0f}", "")

    legend.Draw()

    ROOT.gPad.Update()
    xax = h.GetXaxis()
    xmin = xax.GetXmin()
    xmax = xax.GetXmax()
    axis_mev = ROOT.TGaxis(xmin, 0, xmax, 0, xmin * 1e3, xmax * 1e3, 510, "S")
    axis_mev.SetTitle("m_{#gamma#gamma} [MeV]")
    axis_mev.SetLabelFont(42)
    axis_mev.SetTitleFont(42)
    axis_mev.SetLabelSize(0.040)
    axis_mev.SetLabelOffset(0.015)
    axis_mev.SetTitleSize(0.042)
    axis_mev.SetTitleOffset(1.1)
    axis_mev.Draw()

    c.Update()
    meson_tag = "pi0" if m_meson == M_PI0 else "eta"
    dir_name = PLOT_DIR / "fits"
    os.mkdir(dir_name) if not dir_name.exists() else None
    fname = str(dir_name / f"fit_{meson_tag}_L{L:.2f}.pdf")
    c.SaveAs(fname)

    # Return fit results
    out = {
        "mu": mu_guess,
        "mu_err": float(f.GetParError(1)),
        "sigma": sigma_guess,
        "sigma_err": float(f.GetParError(2)),
        "chi2": float(f.GetChisquare()),
        "ndf": int(f.GetNDF()),
        "fit_range": (float(xmin_fit), float(xmax_fit)),
    }
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
        Significance.
    """
    if kind == "simple":
        return S / ((S + B) ** 0.5)

    # Asimov
    if B <= 0.0:
        Z = np.sqrt(2.0 * S)
    else:
        Z = (2.0 * ((S + B) * np.log(1.0 + S / B) - S)) ** 0.5

    return Z


def plot_oversampled_background(h_bkg, L_cm):
    """
    Plot the background histogram.

    Parameters
    ----------
    h_bkg : ROOT.TH1
        Background histogram to plot.
    L_cm : array-like of float
        Target thicknesses (cm).
    """
    c = ROOT.TCanvas(
        f"bkg_{L_cm:.2f}",
        f"Background at L={L_cm:.2f} cm",
        800,
        600,
    )
    c.SetBatch(True)

    c.SetLeftMargin(0.08)
    c.SetRightMargin(0.04)
    c.SetBottomMargin(0.1)
    c.SetTopMargin(0.08)

    h_bkg.SetStats(0)
    h_bkg.SetTitle(f"Background at L={L_cm:.2f} cm")

    ay = h_bkg.GetYaxis()
    ay.SetTitle("Events")
    ay.SetLabelFont(42)
    ay.SetTitleFont(42)
    ay.SetLabelSize(0.040)
    ay.SetTitleSize(0.042)
    ay.SetTitleOffset(0.9)

    ax = h_bkg.GetXaxis()
    ax.SetTitle("")
    ax.SetLabelOffset(999)
    ax.SetTickLength(0)

    h_bkg.Draw()

    num_bins = h_bkg.GetNbinsX()
    num_entries = h_bkg.GetEntries()

    legend = ROOT.TLegend(0.7, 0.70, 0.94, 0.90)
    legend.SetBorderSize(1)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.038)
    legend.SetMargin(0.2)
    legend.SetEntrySeparation(0.35)

    legend.AddEntry(h_bkg, f"Entries: {num_entries:.0f}", "l")
    legend.AddEntry("", f"Bins: {num_bins:.0f}", "")
    legend.AddEntry("", f"L = {L_cm:.2f} cm", "")

    legend.Draw()

    ROOT.gPad.Update()
    xax = h_bkg.GetXaxis()
    xmin = xax.GetXmin()
    xmax = xax.GetXmax()
    axis_mev = ROOT.TGaxis(xmin, 0, xmax, 0, xmin * 1e3, xmax * 1e3, 510, "S")
    axis_mev.SetTitle("m_{#gamma#gamma} [MeV]")
    axis_mev.SetLabelFont(42)
    axis_mev.SetTitleFont(42)
    axis_mev.SetLabelSize(0.040)
    axis_mev.SetLabelOffset(0.015)
    axis_mev.SetTitleSize(0.042)
    axis_mev.SetTitleOffset(1.1)
    axis_mev.Draw()

    c.Update()
    dir_name = PLOT_DIR / "bkg"
    os.mkdir(dir_name) if not dir_name.exists() else None
    fname = str(dir_name / f"bkg_L{L_cm:.2f}.pdf")
    c.SaveAs(fname)
    logger.info(f"Plotted oversampled background for L={L_cm:.2f} cm")


def plot_scan_results(results, meson="pi0"):
    """
    Plot sigma(L) and Z(L) in two stacked panels.

    Parameters
    ----------
    results : list[dict]
        Output list from main() scan.
    meson : str
        "pi0" or "eta".

    """
    meson_name = r"$\pi^{0}$" if meson == "pi0" else r"$\eta$"
    L = np.array([r["L"] for r in results], dtype=float)
    sigma = np.array([r["sigma"] for r in results], dtype=float)
    Z = np.array([r["Z"] for r in results], dtype=float)
    sigma_err = np.array([r["sigma_err"] for r in results], dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), constrained_layout=True
    )

    # top: sigma(L)
    ax1.set_title(f"Scan target thickness - {meson_name}")
    ax1.set_ylabel(r"$\sigma_{m_{\gamma\gamma}}$ [MeV]")
    ax1.errorbar(L, sigma * 1e3, yerr=sigma_err * 1e3, fmt="o", capsize=2)
    ax1.grid(True)

    # bottom: Z(L)
    ax2.set_xlabel("Target thickness L [cm]")
    ax2.set_ylabel("Significance Z")
    ax2.plot(L, Z, "o-")
    ax2.grid(True)

    fig.savefig(PLOT_DIR / f"scan_results_{meson}.pdf", dpi=1200)


"""
    for i, L in enumerate(L_values):
        out = scan_L(
            rng,
            L,
            n_pions=n_pions,
            n_gen_pi0=n_gen_pi0,
            n_gen_eta=n_gen_eta,
        )
        h_bkg = out["h_bkg"]
        for meson in meson_list:
            m_meson = M_ETA if meson == "eta" else M_PI0
            h_meson = out[f"h_{meson}"]  # weighted
            h_meson_shaped = out[f"h_{meson}_shaped"]  # shape-only, weights=1

            # 1) Fit resolution on signal-only
            fit_meson = fit_meson_peak(h=h_meson_shaped, m_meson=m_meson, L=L)
            mu = fit_meson["mu"]
            sigma = fit_meson["sigma"]

            # 2) Count events in adaptive window +/- nsig*sigma
            win_sig = compute_window_counts(h_meson, mu, sigma)
            win_bkg = compute_window_counts(h_bkg, mu, sigma)

            # 3) Convert to physical expected yields
            S = win_sig["n"]
            B = win_bkg["n"]
            Z = compute_significance(S, B, kind="asymptotic")

            logger.info(
                f"{meson} | L = {L:.4f} cm | mu = {mu:.4f} GeV |"
                f" sigma = {sigma:.4f} GeV | S = {S:.3e} | B = {B:.3e}"
                f"| Z = {Z:.3e}"
            )

            results[i].append(
                {
                    "L": L,
                    "ok": True,
                    "mu": mu,
                    "mu_err": fit_meson["mu_err"],
                    "sigma": sigma,
                    "sigma_err": fit_meson["sigma_err"],
                    "S": S,
                    "B": B,
                    "Z": Z,
                    "win_low": win_sig["low"],
                    "win_high": win_sig["high"],
                }
            )
    for i, meson in enumerate(meson_list):
        results_meson = [r[i] for r in results]
        plot_scan_results(results_meson, meson=meson)

    # Return results for possible further analysis or saving
    return results
    """
