# Usage

This repository implements a toy Monte Carlo for a 100 GeV $\pi^-$ beam on a polyethylene ($\mathrm{CH}_2$) target.
The signal channels are:

- $\pi^- + p \rightarrow \pi^0 + n$, with $\pi^0 \rightarrow \gamma\gamma$
- $\pi^- + p \rightarrow \eta + n$, with $\eta \rightarrow \gamma\gamma$

The main study is the optimisation of the target thickness $L$ (in cm) to maximise discovery/measurement
performance, given a fixed detector model.

## Main components

- `scattering.py`: samples the first interaction depth using total cross sections.
- `diff_cross_section.py`: samples the scattering angle using binned $d\sigma/dt$.
- `kinematics.py`: builds event kinematics and generates the two photons.
- `detector.py`: applies acceptance, energy threshold, cluster separation, and smearing.
- `generate_events.py`: runs the full chain and fills invariant-mass histograms (ROOT output + metadata).
- `optimize_target.py`: scans over $L$ and extracts metrics (significance, resolution, efficiency, purity).

## Quick start

### 1) Generate events for single/multiple target thickness

Example:

```bash
python generate_events.py \
  --L_values 20.5 \
  --N_pions 100000 \
  --N_gen_eta 100000 \
  --N_gen_bkg 100000 \
  --N_gen_pi0 100000 \
```

Typical outputs:

- a ROOT file with the histograms you later scan (e.g. `histograms.root` for signal and background)
- three metadata dictionaries saved in a JSON file for signal and backgrounf (e.g. `metadata_bkg.json`, `metadata_eta.json`, `metadata_pi0.json`)

### 2) Scan target thickness and produce optimisation plots

Example:

```bash
python optimize_target.py \
  --scan True
```

Perform the target length scan and save results in a JSON file (default `metrics.json`).

To use the found results from the scan and visualize the plots, run:

```bash
python optimize_target.py \
  --plot True
```

Typical outputs (PDFs) include:

- `significance_pi0.pdf`, `significance_eta.pdf`: significance and signal/background vs $L$
- `sigma_pi0.pdf`, `sigma_eta.pdf`: mass resolution and efficiency vs $L$
- optional fit/control plots at specific $L$ values (signal-only fits, signal+background shapes)
