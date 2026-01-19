<!-- docs/rep_structure.md -->
# Repository structure

This document provides an overview of the **montecarloproject**
repository layout and the role of each main component. 
It also explains the integrated **Continuous Integration (CI)** system that ensures
the documentation remain consistent across updates.

---

## Directory Layout

```
montecarloproject/
│
├── LICENSE
├── README.md                 # Main project description
├── pyproject.toml            # Build configuration (dependencies, optional extras)
│
├── src/                      # Core source code package
│   ├── scattering.py
│   ├── diff_cross_section.py
│   ├── detector.py
│   ├── kinematics.py
│   ├── generate_event.py
│   ├── optimize_target.py
│   ├── data/                # data files directory
│   └── plots/               # plots directory
│       ├── bkg/             # background oversampled histograms for various L
│       ├── fits/            # signal oversampled spectra fits for various L
│       ├── normalized_spectra/  # normalized spectra for sig+bkg for various L
|       └── ...
│
├── docs/                     # Sphinx + MyST documentation
│   ├── index.rst
│   ├── installation.md
│   ├── usage.md
│   ├── repository_structure.md
│   └── _api/                 # Auto-generated API reference by sphinx-apidoc
│
└── .github/workflows/        # Continuous Integration (CI) configurations
    └── docs.yml              # Builds and deploys documentation to GitHub Pages
```

---

## Continuous Integration (CI)

### `docs.yml`

- Builds the Sphinx documentation using `myst-parser` and `sphinxawesome-theme`.
- Deploys the generated HTML site to **GitHub Pages**.
- Ensures that the public documentation is always in sync with the latest code.

The live documentation is automatically published at:

**[https://albertomontanelli.github.io/montecarloproject/](https://albertomontanelli.github.io/montecarloproject/)**

