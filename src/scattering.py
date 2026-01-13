"""
Simulation of pi- p interactions in a polyethylene target.

Simulate the first interaction depth and channel choice of incoming pi-
particles in a polyethylene target of varying thickness L.
Produces plots comparing Monte Carlo results to analytical expectations
for:

- Accepted interaction counts vs target length L.
- Rare channel counts meson n -> 2 gamma with Garwood confidence
  intervals.
- Histogram of first interaction depths x for a fixed target length L,
  compared to the expected truncated exponential distribution.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.constants import N_A as AVOGADRO
from scipy.stats import chi2

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

# Directories for data and plots
base_dir = Path(__file__).resolve().parent
DATA_DIR = base_dir / "data"
PLOT_DIR = base_dir / "plots"
DATA_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


class CrossSections:
    """Static container for cross sections and derived quantities."""

    # --- Cross sections (cm^2) ---
    SIGMA_TOT_CM2 = 25.0e-27
    SIGMA_PI0_CM2 = 3.2e-30
    SIGMA_ETA_CM2 = 0.33e-30

    # --- Material properties ---
    DENSITY_G_CM3 = 0.93
    MOLAR_MASS = 16.0
    NR_PROTONS = 2.0

    @staticmethod
    def lambda_cm():
        """
        Compute the mean free path lambda in the target material.

        Returns
        -------
        float
            Mean free path in cm.
        """
        return CrossSections.MOLAR_MASS / (
            CrossSections.NR_PROTONS
            * AVOGADRO
            * CrossSections.DENSITY_G_CM3
            * CrossSections.SIGMA_TOT_CM2
        )

    @staticmethod
    def channel_probabilities():
        """
        Interaction channel probabilities from cross-section ratios.

        Returns
        -------
        dict
            Dictionary with keys: 'pi0n', 'etan_2g', 'other'.
        """
        p_pi0 = CrossSections.SIGMA_PI0_CM2 / CrossSections.SIGMA_TOT_CM2
        p_eta = CrossSections.SIGMA_ETA_CM2 / CrossSections.SIGMA_TOT_CM2
        p_other = 1.0 - (p_pi0 + p_eta)
        return {
            "pi0": p_pi0,
            "eta": p_eta,
            "other": p_other,
        }


def sample_depth(rng, size):
    """
    Sample first interaction depth using inverse CDF.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator (seeded once in main).
    size : int
        Number of samples.

    Returns
    -------
    x_cm : numpy.ndarray
        Sampled depths in cm.
    """
    u = rng.random(size)
    u = np.clip(u, 1e-15, 1.0)  # avoid log(0)
    x_cm = -CrossSections.lambda_cm() * np.log(u)
    return x_cm


def expected_prob(L_cm):
    """
    Compute the expected probability to have the first interaction.

    Parameters
    ----------
    L_cm : float
        Target length in cm.

    Returns
    -------
    p_acc : float
        Probability of acceptance.
    """
    return 1.0 - np.exp(-L_cm / CrossSections.lambda_cm())


def sample_channels(rng, n):
    """
    Sample interaction channels for accepted events.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator (same instance used everywhere).
    n : int
        Number of accepted events.

    Returns
    -------
    channels : numpy.ndarray
        Array of channel labels of length n.
    """
    labels = list(CrossSections.channel_probabilities().keys())
    p = np.array(
        [CrossSections.channel_probabilities()[k] for k in labels],
        dtype=float,
    )
    p = p / p.sum()
    channels = rng.choice(labels, size=n, p=p)
    return channels


def run_length_scan(rng, lengths_cm, n_events):
    """
    Run the simulation for a scan of target lengths.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    lengths_cm : numpy.ndarray
        Array of target lengths (cm).
    n_events : int
        Number of trials per target length.

    Returns
    -------
    results : dict
        Dictionary containing arrays of results vs length.
    """
    nL = len(lengths_cm)

    accepted_mc = np.zeros(nL, dtype=int)
    rejected_mc = np.zeros(nL, dtype=int)

    pi0_mc = np.zeros(nL, dtype=int)
    eta_mc = np.zeros(nL, dtype=int)
    other_mc = np.zeros(nL, dtype=int)

    for i, L_cm in enumerate(lengths_cm):
        x = sample_depth(rng, size=n_events)
        mask_acc = x < L_cm

        n_acc = int(mask_acc.sum())
        accepted_mc[i] = n_acc
        rejected_mc[i] = n_events - n_acc

        channels = sample_channels(rng, n=n_acc)
        pi0_mc[i] = int(np.sum(channels == "pi0"))
        eta_mc[i] = int(np.sum(channels == "eta"))
        other_mc[i] = int(np.sum(channels == "other"))

    results = {
        "lambda_cm": CrossSections.lambda_cm(),
        "L_cm": lengths_cm,
        "accepted_mc": accepted_mc,
        "rejected_mc": rejected_mc,
        "pi0_mc": pi0_mc,
        "eta_mc": eta_mc,
        "other_mc": other_mc,
    }
    return results


def plot_accepted_mc_vs_expected(results, n_events):
    """
    Plot accepted counts: MC vs expected with binomial errors.

    Parameters
    ----------
    results : dict
        Output of run_length_scan.
    n_events : int
        Number of trials per target length.
    """
    lam = CrossSections.lambda_cm()
    L = results["L_cm"]
    acc_mc = results["accepted_mc"]

    p_mc = acc_mc / n_events
    err_bin = np.sqrt(n_events * p_mc * (1.0 - p_mc))

    filename = PLOT_DIR / "accepted_counts_mc_vs_expected.pdf"
    plt.figure(figsize=(12, 5), dpi=1200)
    plt.errorbar(
        L,
        acc_mc,
        yerr=err_bin,
        fmt="o",
        label="MC interaction counts - binomial errors",
        markersize=3,
        elinewidth=1,
        capsize=4,
        capthick=1,
    )
    x = np.linspace(0, L.max(), 1000)
    plt.plot(
        x,
        n_events * expected_prob(x),
        label=(r"Interaction counts expected: $1-e^{-\frac{L}{\lambda}}$"),
    )

    ax = plt.gca()
    minor = np.array([0.1])
    major = np.arange(0.5, 6, 0.5)
    xticks = np.concatenate([minor, major]) * lam
    ax.set_xticks(xticks)
    labels = [rf"${m:g}\lambda$" for m in np.concatenate([minor, major])]
    ax.set_xticklabels(labels)
    ax.set_xlim(0, 5.2 * lam)
    ax.set_xlabel(r"Target length $L$ in units of $\lambda$")

    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$\lambda= {lam:.0f}\,\mathrm{{cm}}$, $N = 10^7$"),
        )
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )

    plt.ylabel("Events")
    plt.yscale("log")
    plt.title("First interaction counts")
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()


def plot_linearity(results, n_events):
    """
    Plot accepted counts: MC vs expected with binomial errors.

    Parameters
    ----------
    results : dict
        Output of run_length_scan.
    n_events : int
        Number of trials per target length.
    """
    lam = CrossSections.lambda_cm()
    L = results["L_cm"]
    acc_mc = results["accepted_mc"]

    p_mc = acc_mc / n_events
    err_bin = np.sqrt(n_events * p_mc * (1.0 - p_mc))

    filename = PLOT_DIR / "linearity.pdf"
    plt.figure(figsize=(12, 5), dpi=1200)
    plt.errorbar(
        L,
        acc_mc,
        yerr=err_bin,
        fmt="o",
        label="MC interaction counts - binomial errors",
        markersize=3,
        elinewidth=1,
        capsize=4,
        capthick=1,
    )
    x = np.linspace(0, L.max(), 1000)
    plt.plot(
        x,
        n_events * (x / lam),
        label=(
            "Interaction counts expected (linear approx):"
            r" $\frac{L}{\lambda}$"
        ),
    )

    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$\lambda= {lam:.0f}\,\mathrm{{cm}}$, $N = 10^7$"),
        )
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )

    plt.xlim(-0.5, 1.1 * L.max())
    plt.xlabel(r"Target length $L$ [cm]")
    plt.ylabel("Events")
    plt.yscale("log")
    plt.title("First interaction counts")
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()


def poisson_garwood_interval(n, cl=0.68):
    """
    Garwood confidence interval for Poisson mean mu, given observed n.

    Parameters
    ----------
    n : array-like or scalar
        Observed counts (>=0).
    cl : float
        Confidence level, e.g. 0.68 or 0.90.

    Returns
    -------
    low, up : np.ndarray
        Lower and upper limits for the Poisson mean mu.
    """
    n = np.asarray(n, dtype=float)
    if np.any(n < 0):
        raise ValueError("Poisson counts must be >= 0.")
    alpha = 1.0 - cl

    low = np.zeros_like(n)
    up = np.zeros_like(n)

    # lower bound defined only for n>0
    mask = n > 0
    low[mask] = 0.5 * chi2.ppf(alpha / 2.0, 2.0 * n[mask])

    # upper bound defined for all n, including n=0
    up = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (n + 1.0))

    return low, up


def plot_rare_channels(results, cl=0.68):
    """
    Plot rare channels (pi0, eta) with Garwood confidence intervals.

    Parameters
    ----------
    results : dict
        Output of run_length_scan.
    cl : float
        Confidence level for Garwood interval (e.g. 0.68 or 0.90).
    """
    lam = CrossSections.lambda_cm()
    L = results["L_cm"]
    pi0 = results["pi0_mc"].astype(float)
    eta = results["eta_mc"].astype(float)

    # Garwood intervals for mean mu, given observed counts
    pi0_low, pi0_up = poisson_garwood_interval(pi0, cl=cl)
    eta_low, eta_up = poisson_garwood_interval(eta, cl=cl)

    # Convert to asymmetric error bars around the observed count
    pi0_yerr = np.vstack([pi0 - pi0_low, pi0_up - pi0])
    eta_yerr = np.vstack([eta - eta_low, eta_up - eta])

    filename = PLOT_DIR / "rare_channels.pdf"
    plt.figure(figsize=(12, 5))
    plt.errorbar(
        L,
        pi0,
        yerr=pi0_yerr,
        fmt="o",
        markersize=4,
        capsize=2,
        elinewidth=1,
        label=rf"$\pi^0n$ - {int(cl * 100)}% CL",
    )
    plt.errorbar(
        L,
        eta,
        yerr=eta_yerr,
        fmt="o",
        markersize=4,
        capsize=3,
        elinewidth=1,
        label=rf"$\eta n \rightarrow 2\gamma$ - {int(cl * 100)}% CL",
    )

    ax = plt.gca()
    minor = np.array([0.1])
    major = np.arange(0.5, 6, 0.5)
    xticks = np.concatenate([minor, major]) * lam
    ax.set_xticks(xticks)
    labels = [rf"${m:g}\lambda$" for m in np.concatenate([minor, major])]
    ax.set_xticklabels(labels)
    ax.set_xlim(0, 5.2 * lam)
    ax.set_xlabel(r"Target length $L$ in units of $\lambda$")
    plt.ylabel("Counts")
    plt.title("Rare channels counts with Garwood confidence intervals")
    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(rf"$\lambda= {lam:.0f}\,\mathrm{{cm}}$, $N = 10^7$"),
        )
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()


def plot_depth_histogram(rng, L_cm, n_samples, n_bins):
    """
    Plot the histogram of accepted depths x, compare to truncated exponential.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    L_cm : float
        Target length used for the depth histogram.
    n_samples : int
        Number of sampled depths.
    n_bins : int
        Number of histogram bins.
    """
    lam = CrossSections.lambda_cm()

    x = sample_depth(rng, size=n_samples)
    x_acc = x[x < L_cm]

    filename = PLOT_DIR / "depth_check.pdf"

    plt.figure(figsize=(12, 5))
    plt.grid()
    plt.hist(
        x_acc,
        bins=n_bins,
        range=(0.0, L_cm),
        density=True,
        histtype="step",
        label=r"Accepted depths: $x<\lambda$",
    )

    xs = np.linspace(0.0, L_cm, 400)
    norm = 1.0 - np.exp(-L_cm / lam)
    pdf_trunc = (1.0 / lam) * np.exp(-xs / lam) / norm
    plt.plot(
        xs,
        pdf_trunc,
        label=(
            r"Truncated exponential pdf: "
            r"$\frac{1}{\lambda}\,\frac{e^{-x/\lambda}}{1 - e^{-L/\lambda}}$"
        ),
    )

    extra_line = [
        Line2D(
            [],
            [],
            color="none",
            label=(
                rf"$L = \lambda= {lam:.0f}\,\mathrm{{cm}}$, $N = {n_samples}$"
            ),
        )
    ]
    plt.xlabel("First interaction depth $x$ [cm]")
    plt.ylabel("Probability density")
    plt.title(
        "First interaction sampled depth $x$ distribution: MC vs expected"
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + extra_line,
        labels + [extra_line[0].get_label()],
    )
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(filename, dpi=1200)
    plt.close()


def main(linear_approx=False):
    """
    Execute the main workflow.

    - Create a single RNG with a fixed seed.
    - Run the scan over target lengths.
    - Produce the requested plots.

    Parameters
    ----------
    linear_approx : bool
        If True, run only the linear approximation lengths.
    """
    seed = 42
    rng = np.random.default_rng(seed)
    n_events = 10_000_000
    if linear_approx is not True:
        lengths = np.linspace(
            CrossSections.lambda_cm() / 10,
            5 * CrossSections.lambda_cm(),
            50,
        )  # cm
    else:
        lengths = np.linspace(0.5, 21.5, 21)  # cm

    results = run_length_scan(
        rng=rng,
        lengths_cm=lengths,
        n_events=n_events,
    )

    print(f"Mean free path lambda = {results['lambda_cm']:.6f} cm")

    if linear_approx is not True:
        plot_accepted_mc_vs_expected(results, n_events=n_events)
        plot_rare_channels(results)

    else:
        plot_linearity(results, n_events=n_events)

    plot_depth_histogram(
        rng=rng,
        L_cm=CrossSections.lambda_cm(),
        n_samples=200_000,
        n_bins=80,
    )


if __name__ == "__main__":
    for linear_approx in [False, True]:
        main(linear_approx=linear_approx)
