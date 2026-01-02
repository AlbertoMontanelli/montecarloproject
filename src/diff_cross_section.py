"""
Sample Mandelstam t from binned dsigma/dt tables stored in ROOT files.

This module provides functionality to:
- Load dsigma/dt data from ROOT TGraphAsymmErrors.
- Build a piecewise-constant PDF and CDF for tau = -t.
- Sample t values according to the PDF.
- Convert sampled t to cos(theta*) in the CM frame for pi^- p -> X n
"""

import ctypes
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import ROOT

from scattering import _dir_path_finder

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


@dataclass
class DtPdf:
    """
    Piecewise-constant PDF built from a binned dsigma/dt table.

    Attributes
    ----------
    tau_lo, tau_hi:
        Bin edges for tau=-t in GeV^2.
    cdf:
        Cumulative distribution normalized to 1, same length as bins.
    """

    tau_lo: list[float]
    tau_hi: list[float]
    cdf: list[float]


def load_graph(root_path):
    """
    Load Table dsigma/dt stored as TGraphAsymmErrors Graph1D_y1.

    Parameters
    ----------
    root_path:
        Path to the ROOT file.

    Returns
    -------
    ROOT.TGraphAsymmErrors
        The graph object.
    """
    f = ROOT.TFile.Open(root_path, "READ")
    d3 = f.Get("Table 3")
    g = d3.Get("Graph1D_y1")
    return g


def build_pdf_from_graph(g):
    """
    Build a CDF from a TGraphAsymmErrors containing binned dsigma/dt.

    For each point i:

    - tau bin = [x - ex_low, x + ex_high]
    - weight w_i = (dsigma/dt)_i * (tau_hi - tau_lo)

    Parameters
    ----------
    g : ROOT.TGraphAsymmErrors
        Graph containing binned dsigma/dt.

    Returns
    -------
    DtPdf
        Precomputed bin edges + CDF.
    """
    n = g.GetN()
    tau_lo: list[float] = []
    tau_hi: list[float] = []
    w: list[float] = []

    for i in range(n):
        x = ctypes.c_double()
        y = ctypes.c_double()
        g.GetPoint(i, x, y)
        xval = x.value
        yval = y.value

        lo = xval - float(g.GetErrorXlow(i))
        hi = xval + float(g.GetErrorXhigh(i))
        width = hi - lo

        # Central-value generation: clip negative y just in case
        weight = max(0.0, yval) * width

        tau_lo.append(lo)
        tau_hi.append(hi)
        w.append(weight)

    # Build CDF
    cdf: list[float] = []
    acc = 0.0
    for wi in w:
        acc += wi / sum(w)
        cdf.append(acc)
    cdf[-1] = 1.0  # numerical robustness

    return DtPdf(tau_lo=tau_lo, tau_hi=tau_hi, cdf=cdf)


def sample_t(pdf, rng, size):
    """
    Sample Mandelstam t (negative) from the dsigma/dt CDF table.

    Parameters
    ----------
    pdf : DtPdf
        Precomputed PDF and CDF.
    rng : numpy.random.Generator
        Random number generator (same instance used everywhere).
    size : int
        Number of samples

    Returns
    -------
    t: float
        Sampled Mandelstam t value.
    """
    cdf = np.asarray(pdf.cdf, dtype=float)

    u = rng.random(size)  # shape (size,)
    # Find the minimum bin index where CDF >= u
    idx = np.searchsorted(cdf, u, side="left")

    tau_lo = np.asarray(pdf.tau_lo, dtype=float)
    tau_hi = np.asarray(pdf.tau_hi, dtype=float)

    tau = rng.uniform(tau_lo[idx], tau_hi[idx])  # vectorized
    t = -tau

    if size == 1:
        return float(t[0])
    return t


def plot_binned_pdf_cdf(pdf, rng=None, n_samples=0, plot_name=None):
    """
    Plot the binned PDF and CDF from the provided DtPdf object.

    Plot:

    - Top: piecewise-constant PDF p(tau) + (optional) histogram of
      sampled tau=-t
    - Bottom: CDF F(tau)

    Parameters
    ----------
    pdf : DtPdf
        Precomputed bin edges and CDF.
    rng : numpy.random.Generator | None
        If provided and n_samples>0, samples are drawn and overlaid as
        a histogram.
    n_samples : int
        Number of tau=-t samples for the diagnostic histogram.
    plot_name : str
        Name of the output plot file (without extension).
    """
    # Bin probabilities from CDF
    cdf = np.asarray(pdf.cdf, dtype=float)
    cdf_prev = np.concatenate(([0.0], cdf[:-1]))
    P = cdf - cdf_prev

    # Piecewise-constant PDF density in each bin
    tau_lo = np.asarray(pdf.tau_lo, dtype=float)
    tau_hi = np.asarray(pdf.tau_hi, dtype=float)
    widths = tau_hi - tau_lo
    p = P / widths  # 1/GeV^2

    # Build edges for step plotting
    edges = np.concatenate((tau_lo[:1], tau_hi))  # length nbins+1

    # --- figure with 2 rows ---
    fig, (ax_pdf, ax_cdf) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if plot_name is None:
        title = "Binned PDF and cumulative distribution for t-sampling"
    if plot_name == "pi0":
        title = (
            r"Binned pdf and CDF for t-sampling from "
            r"$\frac{d\sigma_{\pi^0}}{dt}$"
        )
    if plot_name == "eta":
        title = (
            r"Binned pdf and CDF for t-sampling from "
            r"$\frac{d\sigma_{\eta}}{dt}$"
        )

    fig.suptitle(title)
    # --- Top: PDF as step ---
    ax_pdf.step(
        edges, np.concatenate((p, [p[-1]])), where="post", label="Binned PDF"
    )  # duplicate last value for step plot

    # Optional: overlay sampled tau histogram
    if rng is not None and n_samples and n_samples > 0:
        t_samples = sample_t(pdf, rng, size=n_samples)
        tau_samples = -np.asarray(t_samples, dtype=float)

        # Use the *same binning* as the table for a clean comparison
        ax_pdf.hist(
            tau_samples,
            bins=edges,
            density=True,
            histtype="step",
            label=f"Sampled $-t$, N={n_samples:}",
        )

    ax_pdf.set_ylabel(r"$p(\tau)\ \mathrm{[1/GeV^2]}$")
    ax_pdf.set_ylim(bottom=0.0)
    ax_pdf.legend(loc="best")
    ax_pdf.grid(True, alpha=0.3)

    # --- Bottom: CDF as step vs upper bin edges ---
    ax_cdf.step(tau_hi, cdf, where="post")
    ax_cdf.set_xlabel(r"$\tau = -t\ \mathrm{[GeV^2]}$")
    ax_cdf.set_ylabel(r"$P(\mathrm{bin}\leq i)$")
    ax_cdf.set_ylim(0.0, 1.05)
    ax_cdf.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_dir = _dir_path_finder(data=False)
    if plot_name is None:
        fig_name = "dt_pdf_cdf.pdf"
    else:
        fig_name = f"{plot_name}_dt_pdf_cdf.pdf"
    plt.savefig(plot_dir / fig_name, dpi=1200)
    plt.close()


def _kallen(x, y, z):
    """Kallen function: lambda(x,y,z)."""
    return x * x + y * y + z * z - 2 * x * y - 2 * x * z - 2 * y * z


def cos_theta_from_t(s, m_a, m_b, m_c, m_d, filename, rng, n_samples=1000):
    """
    Convert Mandelstam t -> cos(theta*) in the CM frame for a+b -> c+d.

    Convention: t = (p_a - p_c)^2, metric (+,-,-,-).
    theta* is the angle between incoming a and outgoing c in the CM.

    Parameters
    ----------
    s : float
        Mandelstam s in (GeV)^2.
    m_a, m_b, m_c, m_d : float
        Particle masses in GeV.
    filename : str
        Path to the ROOT file containing the dsigma/dt table.
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of samples to generate.
    Returns
    -------
    cos_theta : float or np.ndarray
        cos(theta*) corresponding to t.
    """
    g = load_graph(filename)
    pdf = build_pdf_from_graph(g)
    t_samples = sample_t(pdf, rng, size=n_samples)

    sqrt_s = np.sqrt(s)

    p_a = np.sqrt(np.maximum(_kallen(s, m_a * m_a, m_b * m_b), 0.0)) / (
        2.0 * sqrt_s
    )
    p_c = np.sqrt(np.maximum(_kallen(s, m_c * m_c, m_d * m_d), 0.0)) / (
        2.0 * sqrt_s
    )

    E_a = (s + m_a * m_a - m_b * m_b) / (2.0 * sqrt_s)
    E_c = (s + m_c * m_c - m_d * m_d) / (2.0 * sqrt_s)

    denom = 2.0 * p_a * p_c
    if np.any(denom == 0):
        raise ValueError(
            "Denominator 2*p_a*p_c is zero (check kinematics / thresholds)."
        )

    cos_th = (t_samples - m_a * m_a - m_c * m_c + 2.0 * E_a * E_c) / denom
    cos_th = np.clip(cos_th, -1.0, 1.0)

    return cos_th


if __name__ == "__main__":
    data_dir = _dir_path_finder(data=True)
    filenames = [
        str(data_dir / "dsigma_dt_pi0.root"),
        str(data_dir / "dsigma_dt_eta.root"),
    ]
    plot_names = ["pi0", "eta"]

    seed = 42
    rng = np.random.default_rng(seed)
    n_samples = 100_000

    for f, plot_name in zip(filenames, plot_names):
        g = load_graph(f)
        pdf = build_pdf_from_graph(g)
        plot_binned_pdf_cdf(
            pdf, rng=rng, n_samples=n_samples, plot_name=plot_name
        )
