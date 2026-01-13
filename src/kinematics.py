"""
Event kinematics for pi^- p -> X n with X-> gamma gamma.

Calculation steps:

- Build initial-state 4-vectors in the LAB frame.
- Sample cos(theta*) in the CM frame using t-distribution.
- Build outgoing meson 4-vector in CM frame.
- Boost meson 4-vector to LAB frame.
- Isotropic two-body decay of meson to two photons in LAB frame.
- Compute kinematic variables and plot distributions.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import ROOT
from matplotlib.lines import Line2D

from diff_cross_section import (
    _kallen,
    build_pdf_from_graph,
    cos_theta_from_t,
    load_graph,
)
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

# --- Particle masses (GeV) ---
pdg = ROOT.TDatabasePDG.Instance()
M_PI_MINUS = pdg.GetParticle("pi-").Mass()
M_P = pdg.GetParticle("proton").Mass()
M_N = pdg.GetParticle("neutron").Mass()
M_PI0 = pdg.GetParticle("pi0").Mass()
M_ETA = pdg.GetParticle("eta").Mass()
E_BEAM = 100.0  # GeV

# Precompute PDFs from dsigma/dt graphs
PDF_PI0 = build_pdf_from_graph(
    load_graph(str(DATA_DIR / "dsigma_dt_pi0.root"))
)
PDF_ETA = build_pdf_from_graph(
    load_graph(str(DATA_DIR / "dsigma_dt_eta.root"))
)


def build_initial_state():
    """
    Build initial-state 4-vectors in the LAB frame for beam + target.

    Returns
    -------
    p_beam : ROOT.TLorentzVector
        Beam 4-vector in LAB (along +z).
    p_target : ROOT.TLorentzVector
        Target 4-vector in LAB (at rest).
    p_tot : ROOT.TLorentzVector
        Total initial 4-vector in LAB.
    s : float
        Mandelstam s in (GeV)^2.
    beta_cm : ROOT.TVector3
        Boost vector that takes CM -> LAB (i.e. +beta_cm).
    """
    pz = math.sqrt(E_BEAM * E_BEAM - M_PI_MINUS * M_PI_MINUS)
    p_beam = ROOT.TLorentzVector(0.0, 0.0, pz, E_BEAM)
    p_target = ROOT.TLorentzVector(0.0, 0.0, 0.0, M_P)

    p_tot = p_beam + p_target
    # Invariant mass s squared
    s = p_tot.M2()
    # Boost vector that takes a 4-vector from CM to LAB.
    beta_cm = p_tot.BoostVector()

    return p_beam, p_target, p_tot, s, beta_cm


def two_body_momentum_cm(s, m_c, m_d):
    """
    Compute the CM momentum magnitude p* for a -> c+d at fixed s.

    Parameters
    ----------
    s : float
        Mandelstam s in (GeV)^2.
    m_c, m_d : float
        Final-state masses in GeV.

    Returns
    -------
    p_star : float
        CM momentum magnitude in GeV.
    """
    lam = _kallen(s, m_c * m_c, m_d * m_d)
    if lam < 0:
        # Below threshold (numerical or invalid input)
        return 0.0
    return math.sqrt(lam) / (2.0 * math.sqrt(s))


def meson_fourvec_cm(s, m_meson, cos_th, rng):
    """
    Build the outgoing meson 4-vector in the CM frame using sampled t.

    Parameters
    ----------
    s : float
        Mandelstam s in (GeV)^2.
    m_meson : float
        Outgoing meson mass (pi0 or eta) in GeV.
    cos_th : float
        cos(theta*) used for the direction.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    p_meson_cm : ROOT.TLorentzVector
        Meson 4-vector in CM.
    cos_th : float
        cos(theta*) used for the direction.
    phi : float
        Azimuth in radians.
    """
    sin_th = np.sqrt(1.0 - cos_th * cos_th)
    phi = float(rng.uniform(0.0, 2.0 * math.pi))

    p_star = two_body_momentum_cm(s, m_meson, M_N)
    e_star = np.sqrt(p_star * p_star + m_meson * m_meson)

    px = p_star * sin_th * math.cos(phi)
    py = p_star * sin_th * math.sin(phi)
    pz = p_star * cos_th
    p_meson_cm = ROOT.TLorentzVector(px, py, pz, e_star)
    return p_meson_cm, cos_th, phi


def boost_cm_to_lab(p4_cm, beta_cm):
    """
    Boost a 4-vector from CM to LAB.

    Parameters
    ----------
    p4_cm : ROOT.TLorentzVector
        4-vector in the CM frame.
    beta_cm : ROOT.TVector3
        Boost vector that takes CM -> LAB.

    Returns
    -------
    p4_lab : ROOT.TLorentzVector
        Boosted 4-vector in the LAB frame.
    """
    p4_lab = ROOT.TLorentzVector(p4_cm)
    p4_lab.Boost(beta_cm)
    return p4_lab


def decay_to_two_photons_isotropic(p_mother_lab, rng):
    """
    Isotropic two-body decay mother -> gamma gamma.

    Parameters
    ----------
    p_mother_lab : ROOT.TLorentzVector
        Mother 4-vector in the LAB frame.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    g1_lab, g2_lab : ROOT.TLorentzVector
        Photon 4-vectors in the LAB frame.
    """
    m = p_mother_lab.M()
    if m <= 0:
        raise ValueError("Mother mass must be positive.")

    # Photons in mother rest frame: E = m/2, |p| = E, back-to-back.
    e = 0.5 * m
    cos_th = float(rng.uniform(-1.0, 1.0))
    sin_th = math.sqrt(max(1.0 - cos_th * cos_th, 0.0))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))

    px = e * sin_th * math.cos(phi)
    py = e * sin_th * math.sin(phi)
    pz = e * cos_th

    g1_rest = ROOT.TLorentzVector(px, py, pz, e)
    g2_rest = ROOT.TLorentzVector(-px, -py, -pz, e)

    # Boost rest -> LAB with mother's beta
    beta = p_mother_lab.BoostVector()
    g1_lab = ROOT.TLorentzVector(g1_rest)
    g2_lab = ROOT.TLorentzVector(g2_rest)
    g1_lab.Boost(beta)
    g2_lab.Boost(beta)
    return g1_lab, g2_lab


def generate_event_from_t(rng, n_samples, channel="pi0"):
    """
    Generate a single event given a sampled Mandelstam t.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of samples to generate.
    channel : str
        Either "pi0" or "eta".

    Returns
    -------
    event_list : list[dict]
        List of dictionaries with ROOT TLorentzVectors and useful
        scalars:
        - cos_theta_star
        - p_meson_cm, p_meson_lab
        - g1_lab, g2_lab
        - m_gg (reconstructed invariant mass from LAB photons)
        - opening_angle (radians) between LAB photons
    """
    if channel not in {"pi0", "eta"}:
        raise ValueError('channel must be "pi0" or "eta".')

    m_meson = M_PI0 if channel == "pi0" else M_ETA
    pdf = PDF_PI0 if channel == "pi0" else PDF_ETA
    _, _, _, s, beta_cm = build_initial_state()

    cos_th_list = cos_theta_from_t(
        s, M_PI_MINUS, M_P, m_meson, M_N, rng, pdf, n_samples
    )
    # Ensure output is a list even for single sample
    if n_samples == 1:
        cos_th_list = [cos_th_list]

    event_list = []
    for cos_th in cos_th_list:
        p_meson_cm, _, _ = meson_fourvec_cm(
            s=s, m_meson=m_meson, cos_th=cos_th, rng=rng
        )

        p_meson_lab = boost_cm_to_lab(p_meson_cm, beta_cm)
        g1_lab, g2_lab = decay_to_two_photons_isotropic(p_meson_lab, rng)

        p_gg = g1_lab + g2_lab
        m_gg = p_gg.M()

        # Opening angle in LAB between photon 3-momenta
        v1 = g1_lab.Vect()
        v2 = g2_lab.Vect()
        opening_angle = v1.Angle(v2)

        # Compute deltaR
        eta1, eta2 = g1_lab.Eta(), g2_lab.Eta()
        phi1, phi2 = g1_lab.Phi(), g2_lab.Phi()
        dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
        deta = eta1 - eta2
        dR = np.sqrt(deta**2 + dphi**2)

        dict = {
            "channel": channel,
            "cos_theta_star": float(cos_th),
            "p_meson_cm": p_meson_cm,
            "p_meson_lab": p_meson_lab,
            "g1_lab": g1_lab,
            "g2_lab": g2_lab,
            "m_gg": float(m_gg),
            "opening_angle": float(opening_angle),
            "dR": float(dR),
        }
        event_list.append(dict)

    return event_list


def plot_kinematics(n_samples, rng):
    """
    Plot kinematic distributions for generated events.

    Plots:

    - Meson pseudorapidity in LAB.
    - Photon pseudorapidity in LAB from meson decay.
    - DeltaR between the two photons from meson decay.
    - Photon energy in LAB from meson decay.
    - Meson energy in LAB.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    rng : np.random.Generator
        Random number generator.
    """
    etas_g1, etas_g2 = {}, {}
    etas_meson = {}
    dRs = {}
    E_g1, E_g2 = {}, {}
    E_meson = {}

    for channel in ["pi0", "eta"]:
        etas_g1[channel] = []
        etas_g2[channel] = []
        etas_meson[channel] = []
        E_meson[channel] = []
        dRs[channel] = []
        E_g1[channel] = []
        E_g2[channel] = []

        ev_list = generate_event_from_t(
            rng, n_samples=n_samples, channel=channel
        )

        for ev in ev_list:
            p_meson_lab = ev["p_meson_lab"]
            g1_lab = ev["g1_lab"]
            g2_lab = ev["g2_lab"]
            eta1, eta2 = g1_lab.Eta(), g2_lab.Eta()

            etas_g1[channel].append(eta1)
            etas_g2[channel].append(eta2)
            etas_meson[channel].append(p_meson_lab.Eta())
            E_meson[channel].append(p_meson_lab.E())
            dRs[channel].append(ev["dR"])
            E_g1[channel].append(g1_lab.E())
            E_g2[channel].append(g2_lab.E())

    # Plot pseudorapidity of mesons
    plt.figure(figsize=(12, 5), dpi=1200)
    plt.hist(
        etas_meson["pi0"], bins=80, alpha=0.6, label=r"$\pi^0$", range=(4, 10)
    )
    plt.hist(
        etas_meson["eta"], bins=80, alpha=0.6, label=r"$\eta$", range=(4, 10)
    )
    plt.xlabel(r"$\eta_X^{LAB}$")
    plt.ylabel("counts")
    plt.grid(True, alpha=0.3)
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$"),
        )
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    plt.title("Meson pseudorapidity in LAB")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "meson_eta_LAB.pdf", dpi=1200)
    plt.close()

    # Plot pseudorapidity of the two photons from meson decay
    fig, (ax_pi0, ax_eta) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.suptitle(r"$\gamma$ pseudorapidity in LAB from meson decay")
    ax_pi0.hist(
        etas_g1["pi0"], bins=80, alpha=0.6, label=r"$\gamma_1$", range=(0, 10)
    )
    ax_pi0.hist(
        etas_g2["pi0"], bins=80, alpha=0.6, label=r"$\gamma_2$", range=(0, 10)
    )
    ax_pi0.set_xlabel(r"$\eta_\gamma^{LAB}$")
    ax_pi0.set_ylabel("counts")
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$, $\pi^0$ decay"),
        )
    ]
    handles, labels = ax_pi0.get_legend_handles_labels()
    ax_pi0.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    ax_pi0.grid(True, alpha=0.3)

    ax_eta.hist(
        etas_g1["eta"], bins=80, alpha=0.6, label=r"$\gamma_1$", range=(0, 10)
    )
    ax_eta.hist(
        etas_g2["eta"], bins=80, alpha=0.6, label=r"$\gamma_2$", range=(0, 10)
    )
    ax_eta.set_xlabel(r"$\eta_\gamma^{LAB}$")
    ax_eta.set_ylabel("counts")
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$, $\eta$ decay"),
        )
    ]
    handles, labels = ax_eta.get_legend_handles_labels()
    ax_eta.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    ax_eta.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gamma_eta_from_meson.pdf", dpi=1200)
    plt.close()

    # Plot deltaR between the two photons from meson decay
    plt.figure(figsize=(12, 5), dpi=1200)
    plt.hist(dRs["pi0"], bins=80, alpha=0.6, label=r"$\pi^0$", range=(0, 6))
    plt.hist(dRs["eta"], bins=80, alpha=0.6, label=r"$\eta$", range=(0, 6))
    plt.xlabel(
        r"$\Delta R(\gamma,\gamma)=\sqrt{(\Delta \eta)^2 + (\Delta \phi)^2}$"
    )
    plt.ylabel("counts")
    plt.grid(True, alpha=0.3)
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$"),
        )
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    plt.title(r"$\gamma$ separation from meson decay")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gamma_separation.pdf", dpi=1200)
    plt.close()

    # Plot energy in LAB of the two photons from meson decay
    fig, (ax_pi0, ax_eta) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 7),
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.suptitle(r"$\gamma$ energy in LAB from meson decay")
    ax_pi0.hist(
        E_g1["pi0"],
        bins=80,
        alpha=0.6,
        label=r"$\gamma_1$",
    )
    ax_pi0.hist(
        E_g2["pi0"],
        bins=80,
        alpha=0.6,
        label=r"$\gamma_2$",
    )
    ax_pi0.set_xlabel(r"$E_\gamma^{LAB}$ [GeV]")
    ax_pi0.set_ylabel("counts")
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$, $\pi^0$ decay"),
        )
    ]
    handles, labels = ax_pi0.get_legend_handles_labels()
    ax_pi0.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
        fontsize=14,
        loc="lower left",
    )
    ax_pi0.grid(True, alpha=0.3)

    ax_eta.hist(
        E_g1["eta"],
        bins=80,
        alpha=0.6,
        label=r"$\gamma_1$",
    )
    ax_eta.hist(
        E_g2["eta"],
        bins=80,
        alpha=0.6,
        label=r"$\gamma_2$",
    )
    ax_eta.set_xlabel(r"$E_\gamma^{LAB}$ [GeV]")
    ax_eta.set_ylabel("counts")
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$, $\eta$ decay"),
        )
    ]
    handles, labels = ax_eta.get_legend_handles_labels()
    ax_eta.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
        fontsize=14,
        loc="lower left",
    )
    ax_eta.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gamma_energy_from_meson.pdf", dpi=1200)
    plt.close()

    # Plot meson energy in LAB
    plt.figure(figsize=(12, 5), dpi=1200)
    plt.hist(
        E_meson["pi0"],
        bins=80,
        alpha=0.6,
        label=r"$\pi^0$",
    )
    plt.hist(
        E_meson["eta"],
        bins=80,
        alpha=0.6,
        label=r"$\eta$",
    )
    plt.xlabel(r"$E_X^{LAB}$ [GeV]")
    plt.ylabel("counts")
    plt.grid(True, alpha=0.3)
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$N = {n_samples}$"),
        )
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    plt.title("Meson energy in LAB")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "meson_E_LAB.pdf", dpi=1200)
    plt.close()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    plot_kinematics(n_samples=100_000, rng=rng)
