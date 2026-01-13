"""
Sample Mandelstam t from binned dsigma/dt tables stored in ROOT files.

Provide functionality to:

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

    For each point i in the graph:

    - Get x_i, y_i, and bin width from asymmetric errors.
    - Compute weight w_i = max(0, y_i) * width_i for central value
      generation.
    - Build CDF from normalized weights.

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
    tot = sum(w)
    for wi in w:
        acc += wi / tot
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


def plot_binned_pdf_cdf(pdf, rng, n_samples, plot_name="pi0"):
    """
    Plot the binned PDF and CDF from the provided DtPdf object.

    Plot:

    - Top: piecewise-constant PDF p(tau) + histogram of sampled tau=-t
    - Bottom: CDF F(tau)

    Parameters
    ----------
    pdf : DtPdf
        Precomputed bin edges and CDF.
    rng : numpy.random.Generator
        Random number generator (same instance used everywhere).
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

    # Overlay sampled tau histogram
    t_samples = sample_t(pdf, rng, size=n_samples)
    tau_samples = -np.asarray(t_samples, dtype=float)

    # Use the same binning as the table for a clean comparison
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
    fig_name = f"{plot_name}_dt_pdf_cdf.pdf"
    plt.savefig(PLOT_DIR / fig_name, dpi=1200)
    plt.close()


def _kallen(x, y, z):
    """Kallen function: lambda(x,y,z)."""
    return x * x + y * y + z * z - 2 * x * y - 2 * x * z - 2 * y * z


def cos_theta_from_t(s, m_a, m_b, m_c, m_d, rng, pdf, n_samples=1000):
    """
    Convert Mandelstam t -> cos(theta*) in the CM frame for a+b -> c+d.

    Convention:

    - t = (p_a - p_c)^2, metric (+,-,-,-).
    - theta* is the angle between incoming a and outgoing c in the CM.

    Parameters
    ----------
    s : float
        Mandelstam s in (GeV)^2.
    m_a, m_b, m_c, m_d : float
        Particle masses in GeV.
    rng : np.random.Generator
        Random number generator.
    pdf : DtPdf
        Precomputed PDF and CDF for t sampling.
    n_samples : int
        Number of samples to generate.
    Returns
    -------
    cos_theta : float or np.ndarray
        cos(theta*) corresponding to t.
    """
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
    eps = 1e-12
    cos_th = np.clip(cos_th, -1.0 + eps, 1.0 - eps)

    return cos_th


if __name__ == "__main__":
    filenames = [
        str(DATA_DIR / "dsigma_dt_pi0.root"),
        str(DATA_DIR / "dsigma_dt_eta.root"),
    ]
    plot_names = ["pi0", "eta"]

    rng = np.random.default_rng(42)

    for f, plot_name in zip(filenames, plot_names):
        g = load_graph(f)
        pdf = build_pdf_from_graph(g)
        plot_binned_pdf_cdf(
            pdf, rng=rng, n_samples=100_000, plot_name=plot_name
        )
