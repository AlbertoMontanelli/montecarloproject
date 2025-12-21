"""
pi- on polyethylene target (simple MC scaffold).

This script simulates the depth of the first interaction of an incoming
pi- in a target of thickness L, assuming an exponential law with mean
free path lambda.

Workflow
--------
1) Sample u ~ Uniform(0, 1) and transform to x = -lambda * ln(u).
2) Accept the event if x < L (interaction happens inside target).
3) For accepted events, choose the interaction channel according to
   cross-section ratios.
4) Produce three plots:
   (1) Accepted counts: MC vs expected with binomial errors.
   (2) Rare channels (pi0, eta): MC counts with Poisson errors, with an
       optional scaled reminder of accepted counts.
   (3) Histogram of accepted depths x and comparison with the expected
       truncated exponential pdf on [0, L].

Note
----
This is a counting-level model. No photon transport or detector effects
are included at this stage.
"""

import matplotlib.pyplot as plt
import numpy as np


class Physics:
    """Static container for physical constants and derived quantities."""

    # --- Cross sections (cm^2) ---
    SIGMA_TOT_CM2 = 25.0e-27
    SIGMA_PI0_CM2 = 3.2e-30
    SIGMA_ETA_CM2 = 0.33e-30

    # --- Material properties ---
    AVOGADRO = 6.02214076e23
    DENSITY_G_CM3 = 0.93
    MOLAR_MASS = 16.0
    NR_PROTONS = 2.0

    @staticmethod
    def mean_free_path_cm():
        """
        Compute the mean free path lambda in the target material.

        Returns
        -------
        float
            Mean free path in cm.
        """
        return Physics.MOLAR_MASS / (
            Physics.NR_PROTONS
            * Physics.AVOGADRO
            * Physics.DENSITY_G_CM3
            * Physics.SIGMA_TOT_CM2
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
        p_pi0 = Physics.SIGMA_PI0_CM2 / Physics.SIGMA_TOT_CM2
        p_eta = Physics.SIGMA_ETA_CM2 / Physics.SIGMA_TOT_CM2
        p_other = 1.0 - (p_pi0 + p_eta)
        return {
            "pi0n": p_pi0,
            "etan_2g": p_eta,
            "other": p_other,
        }


def sample_depth_from_uniform(rng, size):
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
    x_cm = -Physics.mean_free_path_cm() * np.log(u)
    return x_cm


def expected_acceptance_prob(L_cm):
    """
    Compute the expected probability to have an interaction inside the target.

    Parameters
    ----------
    L_cm : float
        Target length in cm.

    Returns
    -------
    p_acc : float
        Probability of acceptance.
    """
    return 1.0 - np.exp(-L_cm / Physics.mean_free_path_cm())


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
    labels = np.array(list(Physics.channel_probabilities().keys()))
    p = np.array(
        [Physics.channel_probabilities()[k] for k in labels], dtype=float
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
    accepted_exp = np.zeros(nL, dtype=float)

    pi0_mc = np.zeros(nL, dtype=int)
    eta_mc = np.zeros(nL, dtype=int)
    other_mc = np.zeros(nL, dtype=int)

    p_acc_exp = np.zeros(nL, dtype=float)

    for i, L_cm in enumerate(lengths_cm):
        x = sample_depth_from_uniform(rng, size=n_events)
        mask_acc = x < L_cm

        n_acc = int(mask_acc.sum())
        accepted_mc[i] = n_acc
        rejected_mc[i] = n_events - n_acc

        p_acc = expected_acceptance_prob(L_cm)
        p_acc_exp[i] = p_acc
        accepted_exp[i] = n_events * p_acc

        channels = sample_channels(rng, n=n_acc)
        pi0_mc[i] = int(np.sum(channels == "pi0n"))
        eta_mc[i] = int(np.sum(channels == "etan_2g"))
        other_mc[i] = int(np.sum(channels == "other"))

    results = {
        "lambda_cm": Physics.mean_free_path_cm(),
        "L_cm": lengths_cm,
        "accepted_mc": accepted_mc,
        "rejected_mc": rejected_mc,
        "accepted_exp": accepted_exp,
        "p_acc_exp": p_acc_exp,
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
    L = results["L_cm"]
    acc_mc = results["accepted_mc"]
    acc_exp = results["accepted_exp"]

    p_mc = acc_mc / n_events
    err_bin = np.sqrt(n_events * p_mc * (1.0 - p_mc))

    plt.figure(figsize=(12, 5))
    plt.errorbar(
        L,
        acc_mc,
        yerr=err_bin,
        fmt="o",
        label="Accepted (MC, binomial error)",
    )
    plt.plot(L, acc_exp, label="Accepted (expected)")
    plt.xlabel("Target length L (cm)")
    plt.ylabel("Accepted events")
    plt.title("Accepted counts: MC vs expectation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rare_channels(results, show_accepted_trend=True):
    """
    Plot rare channels (pi0, eta) with Poisson errors.

    Parameters
    ----------
    results : dict
        Output of run_length_scan.
    show_accepted_trend : bool
        If True, also plot a scaled accepted trend for visual reference.
    """
    L = results["L_cm"]
    pi0 = results["pi0_mc"].astype(float)
    eta = results["eta_mc"].astype(float)

    err_pi0 = np.sqrt(pi0)
    err_eta = np.sqrt(eta)

    plt.figure(figsize=(12, 5))
    plt.errorbar(
        L, pi0, yerr=err_pi0, fmt="o", label="pi0n (MC, Poisson error)"
    )
    plt.errorbar(
        L, eta, yerr=err_eta, fmt="o", label="etan_2g (MC, Poisson error)"
    )

    if show_accepted_trend:
        acc = results["accepted_mc"].astype(float)
        denom = max(pi0.max(), eta.max(), 1.0)
        scale = max(acc.max(), 1.0) / denom
        plt.plot(L, acc / scale, label=f"Accepted scaled (÷{scale:.1f})")

    plt.xlabel("Target length L (cm)")
    plt.ylabel("Counts")
    plt.title("Rare channels vs target length")
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    lam = Physics.mean_free_path_cm()

    x = sample_depth_from_uniform(rng, size=n_samples)
    x_acc = x[x < L_cm]

    plt.figure(figsize=(12, 5))
    plt.hist(
        x_acc,
        bins=n_bins,
        range=(0.0, L_cm),
        density=True,
        histtype="step",
        label=f"Accepted depths (x < {L_cm} cm)",
    )

    xs = np.linspace(0.0, L_cm, 400)
    norm = 1.0 - np.exp(-L_cm / lam)
    pdf_trunc = (1.0 / lam) * np.exp(-xs / lam) / norm
    plt.plot(xs, pdf_trunc, label="Truncated exponential pdf (expected)")

    plt.xlabel("First interaction depth x (cm)")
    plt.ylabel("Probability density")
    plt.title("Depth check: accepted x follows truncated exponential")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Execute the main workflow.

    - Create a single RNG with a fixed seed.
    - Run the scan over target lengths.
    - Produce the requested plots.
    """
    seed = 42
    rng = np.random.default_rng(seed)

    lengths_cm = np.linspace(1, 100, 100)
    n_events = 1_000_000

    results = run_length_scan(
        rng=rng,
        lengths_cm=lengths_cm,
        n_events=n_events,
    )

    print(f"Mean free path lambda = {results['lambda_cm']:.6f} cm")

    plot_accepted_mc_vs_expected(results, n_events=n_events)
    plot_rare_channels(results, show_accepted_trend=True)

    L_hist = 100.0
    n_hist = 200_000
    n_bins = 80
    plot_depth_histogram(
        rng=rng,
        L_cm=L_hist,
        n_samples=n_hist,
        n_bins=n_bins,
    )


if __name__ == "__main__":
    main()
