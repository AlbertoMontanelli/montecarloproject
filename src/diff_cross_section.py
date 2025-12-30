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

    # Keep the file alive by attaching it to the graph
    g._root_file = f
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

    sumw = sum(w)
    # Build normalized CDF
    cdf: list[float] = []
    acc = 0.0
    for wi in w:
        acc += wi / sumw
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
    u = rng.random(size)
    # binary search in python list
    lo_i, hi_i = 0, len(pdf.cdf) - 1
    while lo_i < hi_i:
        mid = (lo_i + hi_i) // 2
        if u <= pdf.cdf[mid]:
            hi_i = mid
        else:
            lo_i = mid + 1

    i = lo_i
    t = -float(rng.uniform(pdf.tau_lo[i], pdf.tau_hi[i]))
    return t


def plot_binned_pdf_cdf(pdf):
    """
    Plot the piecewise-constant PDF and the CDF built from DtPdf.

    Parameters
    ----------
    pdf : DtPdf
        Precomputed bin edges and CDF.
    """
    tau_lo = np.array(pdf.tau_lo, dtype=float)
    tau_hi = np.array(pdf.tau_hi, dtype=float)
    cdf = np.array(pdf.cdf, dtype=float)

    widths = tau_hi - tau_lo

    # bin probabilities P_i from the cumulative
    cdf_prev = np.concatenate(([0.0], cdf[:-1]))
    P = cdf - cdf_prev

    # piecewise-constant PDF value per bin (density)
    p = P / widths

    # For step plots we want edges and values
    edges = np.concatenate((tau_lo[:1], tau_hi))
    # Repeat p for step plot: len(values)=len(edges)-1
    values = p

    # --- PDF plot ---
    plt.figure()
    plt.step(edges, np.concatenate((values, [values[-1]])), where="post")
    plt.xlabel(r"$\tau = -t\ \mathrm{[GeV^2]}$")
    plt.ylabel(r"$p(\tau)\ \mathrm{[1/GeV^2]}$")
    plt.title("Binned PDF for tau=-t")
    plt.ylim(bottom=0)

    # --- CDF plot ---
    plt.figure()
    # CDF is constant within bin then jumps at tau_hi;
    # show as step vs upper edges.
    upper_edges = tau_hi
    plt.step(upper_edges, cdf, where="post")
    plt.xlabel(r"$\tau = -t\ \mathrm{[GeV^2]}$")
    plt.ylabel(r"$F(\tau)$")
    plt.title("Binned CDF for tau=-t")
    plt.ylim(0, 1.05)

    plt.show()


if __name__ == "__main__":
    data_dir = _dir_path_finder(data=True)
    pi0_file = data_dir / "dsgima_dt_pi0_100GeV.root"
    print(pi0_file)
    eta_file = data_dir / "dsgima_dt_eta_100GeV.root"

    g_pi0 = load_graph(pi0_file)
    pdf_pi0 = build_pdf_from_graph(g_pi0)
    plot_binned_pdf_cdf(pdf_pi0)

    g_eta = load_graph(eta_file)
    pdf_eta = build_pdf_from_graph(g_eta)
    plot_binned_pdf_cdf(pdf_eta)

    seed = 42
    rng = np.random.default_rng(seed)
    n_samples = 100_000
