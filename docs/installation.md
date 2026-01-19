<!-- docs/installation.md -->
# Installation

This guide explains how to install and set up the **montecarloproject** package,
a Monte Carlo simulation of a high-energy pion beam interacting with a fixed plastic target.

## Prerequisites
Before installation, ensure to have the following dependencies:

- **Conda**
- **Git**

---

## Step 1: Clone the repository
```bash
git clone https://github.com/AlbertoMontanelli/montecarloproject.git
cd montecarloproject
```
---

## Step 2: Create and activate a Conda environment

It is necessary to create a new Conda environment and activate it.

```bash
conda create -n montecarloproject python=3.9 -y
conda activate montecarloproject
```

Then install **ROOT** via conda

```bash
conda install -c conda-forge root
```

---

## Step 3: Install the Package via `pyproject.toml`

The repository uses a modern Python packaging system managed through the `pyproject.toml` file.
Install the package in *editable* mode with:

```bash
pip install -e .
```

This command will:

- install the `montecarloproject` package locally (editable mode);
- automatically install the dependencies defined in `pyproject.toml` ('numpy', 'loguru', 'matplotlib', etc.).

To confirm installation, run:

```bash
pip list | grep montecarloproject
```

---

## Step 4: Optional — Developer and Documentation Dependencies

If it is planned to contribute or build the documentation, install the
additional optional dependencies defined in
`pyproject.toml`.

### Developer tools

```bash
pip install -e .[dev]
```

This will install the developer dependencies such as:
- `ruff` : code style and linting.

### Documentation tools

```bash
pip install -e .[docs]
```

This will install the documentation dependencies such as:
- `sphinx`, `myst-parser`, `sphinxawesome-theme`, `sphinx-autodoc-typehints`.

### Everything together (recommended)

```bash
pip install -e .[dev,docs]
```

This installs all dependencies (base + dev + docs) and is recommended for full development environments.

---

## Step 5: Verify the Installation

Check that all modules import correctly:

```python
import ROOT
import montecarloproject
```

If no errors appear, the setup is complete.

---

## Step 6: Build the Documentation (optional)

To build the documentation locally using **Sphinx** and **MyST Parser**,
from the repository root, run:

```bash
sphinx-build -b html docs docs/_build/html
```

The generated HTML site will be available at:

```
docs/_build/html/index.html
```

If GitHub Pages is configured, it will update automatically after pushing to `main`.

---

## Continuous Integration

The repository includes automated **Continuous Integration (CI)** workflows:

- **docs.yml** : builds and deploys the HTML documentation to GitHub Pages.

These workflows ensure that the package and documentation are always
consistent with the latest version of the code.


