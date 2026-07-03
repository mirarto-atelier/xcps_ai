# xcps-ai

Python project scaffold for reusable XPCS analysis code.

The first maintained workflow computes one-time XPCS correlation curves from the existing
two-time correlation maps, then fits per-temperature/per-ROI relaxation parameters.

## Environment

Create or refresh the virtual environment with `uv` using the local Python installation:

```bash
uv venv --python /export/apps/python/3.11/bin/python3.11
uv sync
```

Run Python through `uv run`:

```bash
uv run python -c "import xcps_ai; print(xcps_ai.__version__)"
```

Run the CLI:

```bash
uv run xcps-ai --help
```

## Current Structure

- `src/xcps_ai/`: maintained analysis package
- `tests/`: regression tests
- `data/`: local raw experiment data, intentionally ignored by Git
- `analysis/`: intended generated-output folder, ignored by Git

## First-Pass Analysis

Build a metadata inventory:

```bash
uv run xcps-ai inventory
```

Reduce TTCF maps to one-time `g2(tau)` curves:

```bash
uv run xcps-ai reduce --uid 165012 --roi 1 --force
```

Fit reduced curves with a KWW model:

```bash
uv run xcps-ai fit
```

Fit the same curves with one shared `tau_s` across all ROI curves at each temperature,
while keeping contrast and KWW beta ROI-specific:

```bash
uv run xcps-ai fit-shared-tau
```

Fit the same curves with one shared KWW beta across all ROI curves at each
temperature, while keeping `tau_s` and contrast ROI-specific:

```bash
uv run xcps-ai fit-shared-beta
```

Summarize fit parameters by temperature:

```bash
uv run xcps-ai summarize
```

Plot filled intermediate scattering function maps by ROI and delay time:

```bash
uv run xcps-ai plot-isf-contours
```

For a linear tau axis with a restricted color range:

```bash
uv run xcps-ai plot-isf-contours --yscale linear --vmin 0.5 --vmax 1.05
```

For q/energy axes, with ROI 0 as q=0 and 0.0051 r.l.u. between ROI bins:

```bash
uv run xcps-ai plot-isf-contours --x-axis q --y-axis energy --yscale linear --vmin 0.5 --vmax 1.05 --roi-max 8
```

Add `--yscale log --plot-points` to show the energy axis logarithmically and overlay every measured point.
Use `--point-size` and `--point-alpha` to tune marker visibility.
Use `--y-min` and `--y-max` to restrict the displayed tau or energy range.

Plot fit-parameter figures that support the below-50 K spin-order interpretation:

```bash
uv run xcps-ai plot-spin-signature
```

Plot fitted-contrast figures:

```bash
uv run xcps-ai plot-contrast-signature
```

Fit the per-temperature tau-Q power law `tau = k * Q^-n` and plot `k(T)` and
`n(T)` separately:

```bash
uv run xcps-ai fit-sub-ballistic
```

Fit the same tau-Q power law with one shared exponent `n` across temperatures
and temperature-specific prefactors `k(T)`. By default this excludes 250 K:

```bash
uv run xcps-ai fit-sub-ballistic-common-n
```
