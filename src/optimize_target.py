"""
Use metrics to scan target length.

Plot the results.
"""

import argparse
import ctypes
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from loguru import logger

from kinematics import M_ETA, M_PI0
from scattering import DATA_DIR, PLOT_DIR

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


def fit_meson_peak(h, m_meson, L, k_range=2, n_iter=2, suffix=None):
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
    suffix : str or None
        Optional suffix for output files.

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
    fname = (
        str(dir_name / f"fit_{meson_tag}_L{L:.2f}.pdf")
        if suffix is None
        else str(dir_name / f"fit_{meson_tag}_L{L:.2f}_{suffix}.pdf")
    )
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


def load_histograms(suffix=None):
    """
    Load oversampled histograms and metadata from file.

    Parameters
    ----------
    suffix : str or None
        Optional suffix for input files.

    Returns
    -------
    histogram_list : list of dict
        Each dict has keys:
        - h_bkg : ROOT.TH1
        - h_pi0 : ROOT.TH1
        - h_eta : ROOT.TH1
    meta_bkg, meta_pi0, meta_eta : dict
        Metadata for background, pi0, and eta.
    """
    meta = {}
    for name in ["bkg", "pi0", "eta"]:
        dir_meta = (
            str(DATA_DIR / f"metadata_{name}_{suffix}.json")
            if suffix
            else str(DATA_DIR / f"metadata_{name}.json")
        )
        meta[name] = json.load(open(dir_meta))

    L_values = [item["L_cm"] for item in meta["bkg"]]
    dir = (
        str(DATA_DIR / f"histograms_{suffix}.root")
        if suffix
        else str(DATA_DIR / "histograms.root")
    )
    f = ROOT.TFile.Open(dir, "READ")
    histogram_list = []
    for L in L_values:
        h_bkg = f.Get(f"h_bkg_L{L:.2f}")
        h_pi0 = f.Get(f"h_pi0_L{L:.2f}")
        h_eta = f.Get(f"h_eta_L{L:.2f}")
        out = {"h_bkg": h_bkg, "h_pi0": h_pi0, "h_eta": h_eta}
        histogram_list.append(out)
        for name, h in [("bkg", h_bkg), ("pi0", h_pi0), ("eta", h_eta)]:
            if not h:
                raise RuntimeError(f"Missing histogram {name} for L={L:.2f}")
            h.SetDirectory(0)
    f.Close()

    return histogram_list, meta["bkg"], meta["pi0"], meta["eta"]


def plot_oversampled_background(suffix=None):
    """
    Plot oversampled background histograms for each target thickness.

    Parameters
    ----------
    suffix : str or None
        Optional suffix for output files.
    """
    hist_list, meta_bkg, _, _ = load_histograms(suffix)
    L_values = [item["L_cm"] for item in meta_bkg]
    for i, L_cm in enumerate(L_values):
        h_bkg = hist_list[i]["h_bkg"]
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
        axis_mev = ROOT.TGaxis(
            xmin, 0, xmax, 0, xmin * 1e3, xmax * 1e3, 510, "S"
        )
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
        fname = (
            str(dir_name / f"bkg_L{L_cm:.2f}.pdf")
            if suffix is None
            else str(dir_name / f"bkg_L{L_cm:.2f}_{suffix}.pdf")
        )
        c.SaveAs(fname)
        logger.info(f"Plotted oversampled background for L={L_cm:.2f} cm")


def plot_normalized_histograms(suffix=None, logy=False):
    """
    Overlay normalized m(gg) histograms for pi0, eta, and background.

    For each target thickness L, overlay the normalized m(gg)
    histograms for pi0, eta, and background on the same plot.
    Normalization is done to unit area (Integral over all bins).

    Parameters
    ----------
    suffix : str or None
        Optional suffix for input/output files.
    logy : bool
        If True, use log scale on y axis (useful to compare tails).
    """
    histogram_list, meta_bkg, _, _ = load_histograms(suffix=suffix)
    L_values = [item["L_cm"] for item in meta_bkg]

    c_pi0 = ROOT.TColor.GetColor("#0072B2")
    c_eta = ROOT.TColor.GetColor("#D55E00")
    c_bkg = ROOT.TColor.GetColor("#009E73")

    out_dir = PLOT_DIR / "normalized_spectra"
    os.mkdir(out_dir) if not out_dir.exists() else None

    for i, L in enumerate(L_values):
        h_bkg = histogram_list[i]["h_bkg"]
        h_pi0 = histogram_list[i]["h_pi0"]
        h_eta = histogram_list[i]["h_eta"]

        # Normalize to unit area (shape comparison)
        for h in (h_bkg, h_pi0, h_eta):
            integ = h.Integral(1, h.GetNbinsX())
            if integ > 0:
                h.Scale(1.0 / integ)
            h.SetDirectory(0)
            h.SetStats(0)
            h.SetLineWidth(3)

        h_pi0.SetLineColor(c_pi0)
        h_eta.SetLineColor(c_eta)
        h_bkg.SetLineColor(c_bkg)

        # Canvas
        c = ROOT.TCanvas(
            f"c_norm_L{L:.2f}",
            f"Normalized m(gg) at L={L:.2f} cm",
            900,
            650,
        )
        c.SetBatch(True)
        c.SetLeftMargin(0.10)
        c.SetRightMargin(0.04)
        c.SetBottomMargin(0.12)
        c.SetTopMargin(0.08)
        if logy:
            c.SetLogy(True)

        # Title + axes styling
        h_bkg.SetTitle(f"Normalized m_{{#gamma#gamma}} at L={L:.2f} cm")
        ay = h_bkg.GetYaxis()
        ay.SetTitle("Arbitrary units (area = 1)")
        ay.SetLabelFont(42)
        ay.SetTitleFont(42)
        ay.SetLabelSize(0.040)
        ay.SetTitleSize(0.045)
        ay.SetTitleOffset(1.05)

        ax = h_bkg.GetXaxis()
        ax.SetTitle("")
        ax.SetLabelOffset(999)
        ax.SetTickLength(0)

        # Choose a sensible y-maximum
        ymax = max(h_bkg.GetMaximum(), h_pi0.GetMaximum(), h_eta.GetMaximum())
        h_bkg.SetMaximum(1.25 * ymax)

        h_bkg.Draw("HIST")
        h_pi0.Draw("HIST SAME")
        h_eta.Draw("HIST SAME")

        leg = ROOT.TLegend(0.62, 0.68, 0.94, 0.90)
        leg.SetBorderSize(1)
        leg.SetFillStyle(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.038)
        leg.SetMargin(0.18)
        leg.AddEntry(h_pi0, "#pi^{0} (normalized)", "l")
        leg.AddEntry(h_eta, "#eta (normalized)", "l")
        leg.AddEntry(h_bkg, "bkg (normalized)", "l")
        leg.Draw()

        # Add MeV axis
        ROOT.gPad.Update()
        xax = h_bkg.GetXaxis()
        xmin = xax.GetXmin()
        xmax = xax.GetXmax()
        axis_mev = ROOT.TGaxis(
            xmin, 0, xmax, 0, xmin * 1e3, xmax * 1e3, 510, "S"
        )
        axis_mev.SetTitle("m_{#gamma#gamma} [MeV]")
        axis_mev.SetLabelFont(42)
        axis_mev.SetTitleFont(42)
        axis_mev.SetLabelSize(0.040)
        axis_mev.SetLabelOffset(0.015)
        axis_mev.SetTitleSize(0.045)
        axis_mev.SetTitleOffset(1.15)
        axis_mev.Draw()

        c.Update()

        fname = (
            str(out_dir / f"mgg_norm_L{L:.2f}.pdf")
            if suffix is None
            else str(out_dir / f"mgg_norm_L{L:.2f}_{suffix}.pdf")
        )
        c.SaveAs(fname)

        logger.info(f"Plotted normalized overlays for L={L:.2f} cm")


def scan_target_length(suffix=None):
    """
    Scan target thickness and compute efficiencies and significances.

    Parameters
    ----------
    suffix : str or None
        Optional suffix for input files.

    Returns
    -------
    results : dict
        Keys:
        - L_values : array of float
        - eff_pi0, eff_eta, eff_bkg : array of float
            Reconstruction efficiencies.
        - S_pi0, B_pi0, Z_pi0 : array of float
            Metrics for pi0.
        - S_eta, B_eta, Z_eta : array of float
            Metrics for eta.
        - sigma_pi0, sigma_pi0_err : array of float
            Mass resolution and uncertainty for pi0 (MeV).
        - sigma_eta, sigma_eta_err : array of float
            Mass resolution and uncertainty for eta (MeV).
    """
    histogram_list, meta_bkg, meta_pi0, meta_eta = load_histograms(
        suffix=suffix
    )
    L_values = [item["L_cm"] for item in meta_bkg]

    eff_pi0 = []
    eff_eta = []
    eff_bkg = []
    S_pi0 = []
    B_pi0 = []
    Z_pi0 = []
    S_eta = []
    B_eta = []
    Z_eta = []
    sigma_pi0 = []
    sigma_eta = []
    sigma_pi0_err = []
    sigma_eta_err = []
    for i, L in enumerate(L_values):
        eff_bkg.append(meta_bkg[i]["N_reco"] / meta_bkg[i]["N_gen"])
        for meson in ["pi0", "eta"]:
            m_meson = M_ETA if meson == "eta" else M_PI0
            h_bkg = histogram_list[i]["h_bkg"]
            h_meson = histogram_list[i][f"h_{meson}"]

            # Fit resolution on signal-only
            fit_meson = fit_meson_peak(
                h=h_meson, m_meson=m_meson, L=L, suffix=suffix
            )
            mu = fit_meson["mu"]
            sigma = fit_meson["sigma"]
            sigma_err = fit_meson["sigma_err"]

            # Create weighted histogram for signal and background
            w = (
                meta_pi0[i]["w_phys"]
                if meson == "pi0"
                else meta_eta[i]["w_phys"]
            )
            h_meson_weighted = h_meson.Clone()
            h_meson_weighted.Scale(w)
            h_bkg_weighted = h_bkg.Clone()
            h_bkg_weighted.Scale(meta_bkg[i]["w_phys"])

            # Count events in adaptive window +/- nsig*sigma
            win_sig = compute_window_counts(h_meson_weighted, mu, sigma)
            win_bkg = compute_window_counts(h_bkg_weighted, mu, sigma)

            # Compute physical expected yields
            S = win_sig["n"]
            B = win_bkg["n"]
            Z = compute_significance(S, B, kind="asymptotic")

            if meson == "pi0":
                eff_pi0.append(meta_pi0[i]["N_reco"] / meta_pi0[i]["N_gen"])
                S_pi0.append(S)
                B_pi0.append(B)
                Z_pi0.append(Z)
                sigma_pi0.append(sigma)
                sigma_pi0_err.append(sigma_err)
            else:
                eff_eta.append(meta_eta[i]["N_reco"] / meta_eta[i]["N_gen"])
                S_eta.append(S)
                B_eta.append(B)
                Z_eta.append(Z)
                sigma_eta.append(sigma)
                sigma_eta_err.append(sigma_err)

    results = {
        "L_values": L_values,
        "N_gen_bkg": meta_bkg[0]["N_gen"],
        "N_gen_pi0": meta_pi0[0]["N_gen"],
        "N_gen_eta": meta_eta[0]["N_gen"],
        "eff_pi0": eff_pi0,
        "eff_eta": eff_eta,
        "eff_bkg": eff_bkg,
        "S_pi0": S_pi0,
        "B_pi0": B_pi0,
        "Z_pi0": Z_pi0,
        "S_eta": S_eta,
        "B_eta": B_eta,
        "Z_eta": Z_eta,
        "sigma_pi0": sigma_pi0,
        "sigma_pi0_err": sigma_pi0_err,
        "sigma_eta": sigma_eta,
        "sigma_eta_err": sigma_eta_err,
    }
    dir = (
        DATA_DIR / f"metrics_{suffix}.json"
        if suffix
        else DATA_DIR / "metrics.json"
    )
    with open(dir, "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_significance(meson="pi0", suffix=None):
    """
    Plot significance and purity vs target thickness after scan.

    Parameters
    ----------
    results : dict
        Output dictionary from scan_target_length().
    meson : str
        "pi0" or "eta".
    suffix : str or None
        Optional suffix for output files.

    """
    dir = (
        DATA_DIR / f"metrics_{suffix}.json"
        if suffix
        else DATA_DIR / "metrics.json"
    )
    results = json.load(open(dir))
    meson_name = r"$\pi^{0}$" if meson == "pi0" else r"$\eta$"

    L = np.array(results["L_values"], dtype=float)
    Z = np.array(results[f"Z_{meson}"], dtype=float)
    S = np.array(results[f"S_{meson}"], dtype=float)
    B = np.array(results[f"B_{meson}"], dtype=float)
    purity = S / (S + B)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), constrained_layout=True
    )
    ax1.set_title(f"Significance and purity for {meson_name}")
    # Top: Z(L)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel(r"Significance $Z(L)$")
    ax1.plot(
        L,
        Z,
        "o-",
        label=r"$Z(L)=$"
        r"$\sqrt{2\left[(S+B)\log\left(1+ \frac{S}{B}\right)-S\right]}$",
    )
    ax1.grid(True)
    ax1.legend()

    # Bottom: S/B (L)
    ax2.set_xlabel("Target thickness L [cm]")
    ax2.set_ylabel(r"Purity $P(L)$")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.plot(L, purity, "o-", color="orange", label=r"$P(L)=\frac{S}{S+B}$")
    ax2.grid(True)
    ax2.legend()

    dir = PLOT_DIR / (
        f"significance_{meson}_{suffix}.pdf"
        if suffix
        else f"significance_{meson}.pdf"
    )
    fig.savefig(dir, dpi=1200)


def plot_sigma(meson="pi0", suffix=None):
    """
    Plot mass resolution and efficiency vs target thickness after scan.

    Parameters
    ----------
    results : dict
        Output dictionary from scan_target_length().
    meson : str
        "pi0" or "eta".
    suffix : str or None
        Optional suffix for output files.

    """
    dir = (
        DATA_DIR / f"metrics_{suffix}.json"
        if suffix
        else DATA_DIR / "metrics.json"
    )
    results = json.load(open(dir))
    meson_name = r"\pi^{0}" if meson == "pi0" else r"\eta"

    L = np.array(results["L_values"], dtype=float)
    sigma = np.array(results[f"sigma_{meson}"], dtype=float)
    sigma_err = np.array(results[f"sigma_{meson}_err"], dtype=float)
    eff_meson = np.array(results[f"eff_{meson}"], dtype=float)
    eff_bkg = np.array(results["eff_bkg"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), constrained_layout=True
    )
    ax1.set_title(
        f"${meson_name}$ mass resolution;"
        r" $\epsilon$ for signal and background"
    )
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel(r"$\sigma_{m_{\gamma\gamma}}$ [MeV]")
    ax1.errorbar(
        L,
        sigma * 1e3,
        yerr=sigma_err * 1e3,
        fmt="o",
        label=r"$\sigma_{m_{\gamma\gamma}}$",
        markersize=4,
        capsize=3,
        elinewidth=1,
    )
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel("Target thickness L [cm]")
    N_gen_meson = (
        results["N_gen_pi0"] if meson == "pi0" else results["N_gen_eta"]
    )
    N_gen_bkg = results["N_gen_bkg"]
    err_bin_meson = np.sqrt((eff_meson * (1.0 - eff_meson)) / N_gen_meson)
    err_bin_bkg = np.sqrt((eff_bkg * (1.0 - eff_bkg)) / N_gen_bkg)
    ax2.set_ylabel(r"Efficiency $\epsilon(L)$")
    ax2.errorbar(
        L,
        eff_meson,
        yerr=err_bin_meson,
        fmt="o",
        color="#D55E00",
        label=rf"$\epsilon_{{{meson_name}}}(L)$ - binomial error",
        markersize=4,
        capsize=3,
        elinewidth=1,
    )
    ax2.errorbar(
        L,
        eff_bkg,
        yerr=err_bin_bkg,
        fmt="o",
        color="green",
        label=r"$\epsilon_{bkg}(L)$ - binomial error",
        markersize=4,
        capsize=3,
        elinewidth=1,
    )
    ax2.grid(True)
    ax2.legend()

    dir = PLOT_DIR / (
        f"sigma_{meson}_{suffix}.pdf" if suffix else f"sigma_{meson}.pdf"
    )
    fig.savefig(dir, dpi=1200)


def main():
    """
    Execute the main function to parse arguments and run scan/plot.

    CLI Parameters
    --------------
    --scan : bool
        Perform the target length scan and save results in a file.
    --plot : bool
        Plot the significance and mass resolution results after
        scanning, loading from file.
    --plot_bkg : bool
        Plot only the background histograms for the each target length.
    --plot_normalized : bool
        Plot the normalized signal and background histograms for each
        target length.
    --suffix : str or None
        Optional suffix for input/output files.
    --meson : str
        Meson to plot results for (pi0 and/or eta).
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scan",
        default="False",
        choices=["True", "False"],
        help="Perform the target length scan and save results in a file.",
    )
    parser.add_argument(
        "--plot",
        default="False",
        choices=["True", "False"],
        help="Plot the significance and mass resolution results after "
        "scanning various target lengths loading from file.",
    )

    parser.add_argument(
        "--plot_bkg",
        default="False",
        choices=["True", "False"],
        help="Plot the background histograms for each target length",
    )

    parser.add_argument(
        "--plot_normalized",
        default="False",
        choices=["True", "False"],
        help=(
            "Plot the normalized signal and background histograms for"
            " each target length"
        ),
    )

    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional suffix for input/output files.",
    )
    parser.add_argument(
        "--meson",
        default=["pi0", "eta"],
        nargs="+",
        choices=["pi0", "eta"],
        help="Meson to plot results for (pi0 and/or eta).",
    )
    args = parser.parse_args()
    if args.scan == "True":
        scan_target_length(suffix=args.suffix)
    if args.plot == "True":
        mesons = [args.meson] if isinstance(args.meson, str) else args.meson
        for meson in mesons:
            plot_significance(meson=meson, suffix=args.suffix)
            plot_sigma(meson=meson, suffix=args.suffix)
    if args.plot_bkg == "True":
        plot_oversampled_background(suffix=args.suffix)
    if args.plot_normalized == "True":
        plot_normalized_histograms(suffix=args.suffix, logy=False)


if __name__ == "__main__":
    main()
