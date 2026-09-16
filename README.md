# NV ODMR TrackBench

A reproducible benchmark for comparing realtime ODMR resonance and linewidth
estimators under matched acquisition budgets.

The project targets an NV ensemble with four crystallographic orientations and
two electronic spin transitions per orientation. It asks which causal strategy
best tracks the center frequency, FWHM, and Q factor of all eight resolved ODMR
resonances when measurement time, fluorescence samples, or photon budget is
held fixed.

Two benchmark modes are in scope:

- recorded playback, preserving the original causal acquisition order; and
- closed-loop emulation, in which an estimator selects the next microwave
  frequency to interrogate.

The scientific contracts and fairness requirements are specified in
[docs/scientific_spec.md](docs/scientific_spec.md).

## Playback and synthetic-emulation proof of concept

The package exposes three deterministic JSON commands:

```bash
odmrbench dataset-info
odmrbench playback --path /explicit/local/ODMR_data.dat --max-observations 100
odmrbench simulate --config bundled:drift
```

`dataset-info` reports the checked optional Figshare record. `playback` parses
only an explicit local raw file and preserves its row-major order; it does not
download data, infer timestamps by default, or assign resolved raw units.
`simulate` runs the bundled fixed schedule through a seeded, synthetic
eight-resonance virtual instrument. Neither command invokes an estimator.

The installed wheel includes the deterministic `bundled:drift` scenario, so
`odmrbench simulate --config bundled:drift` works from an arbitrary current
directory. By contrast, paths under `configs/` and `examples/` refer to files
in a source checkout and are not wheel-contained. The checkout retains its
human-readable configuration at `configs/drift.yaml`; from the repository
root, use `odmrbench simulate --config configs/drift.yaml` when changing or
inspecting that file.

## Offline, two-point, and sparse-linewidth estimators

The package now includes a constrained eight-component Lorentzian and
pseudo-Voigt fitter plus two completed-sweep wrappers.
`RepeatedFullSweepEstimator` independently cold-starts each sweep;
`WarmStartedFullSweepEstimator` may seed from only the latest earlier
successful public fit and records warm rejection, recovery, source, stale age,
CPU, and evaluation diagnostics without duplicating acquisition resources. Fit
success is conditional on the model, initializer, and configured quality
thresholds; it does not prove the presence of eight physical resonances.

See [docs/estimators.md](docs/estimators.md), including its
[warm-started completed-sweep guidance](docs/estimators.md#warm-started-completed-sweeps),
and [calibrated two-point guidance](docs/estimators.md#calibrated-two-point-center-tracking),
and [sparse-linewidth guidance](docs/estimators.md#composite-center-and-sparse-linewidth-tracking),
for model, initialization, calibration provenance, mandatory budget treatment,
policy, timing, failure, resource, ordering, and recording-interpretation
guidance. From the repository root of a source checkout with the package
installed, run these download-free generated diagnostics:

```bash
python examples/fit_synthetic_sweep.py
python examples/fit_warm_started_sweeps.py
python examples/track_two_point_centers.py
python examples/track_sparse_linewidth.py
```

They are software fixtures, not benchmark results or evidence of a universal
warm-start speedup, two-point superiority, or sparse-linewidth accuracy. The
tracking examples use `conditional_free_precalibration` and report only public
policy/resource/timing diagnostics. The sparse example completes eight fast
pairs and one five-point scan, then prints its asynchronous live Q, scan-local
Q, separate source epochs, evaluator timing labels, and resource ledgers.
There is **no Stage 6.5 matched-budget superiority result** and **no Stage 6.6**
benchmark artifact, plot, or automated report yet.
The `examples/` scripts are source-tree files, are not contained in the wheel,
and require either the repository-root commands above or explicit paths to the
checkout.

Read [docs/datasets.md](docs/datasets.md) before obtaining or replaying the
optional CC BY data, including its explicit download URL, size, checksum, and
metadata limitations. [docs/emulator.md](docs/emulator.md) describes the
validated synthetic drift configuration and its virtual-time/photon summary.

## Planned matched-budget milestone

The implemented composite can causally alternate two-point center updates with
lower-rate five-point linewidth updates on all eight identities. The next
matched-budget result will compare it with repeated and warm-started full-sweep
fitting on a virtual eight-resonance spectrum. Comparisons will report center,
FWHM, and Q errors together with acquisition and compute resources; none of
those comparative conclusions is implied by the current generated example.

## Synthetic spectrum demonstration

From a source checkout after installing the package, generate the deterministic
eight-resonance spectrum with:

```bash
python scripts/plot_spectrum.py \
  --config configs/static.yaml \
  --output artifacts/spectrum_demo.png
```

The line-shape APIs use FWHM in Hz explicitly. The generated image is an
illustrative synthetic fixture, not experimental data or a benchmark result.
Its YAML loading and numerical curve generation are reusable package helpers,
so the eight identities, centers, and resolved dip structure are tested without
depending on PNG pixel hashes.

## Scientific caution

The Q factor used by this project is the fitted resonance center divided by its
FWHM. Q alone is not equivalent to magnetometric sensitivity. Contrast,
fluorescence or photon rate, spectral slope, acquisition bandwidth, and the
noise model must also be considered.

## Project status

See [PROJECT_STATE.md](PROJECT_STATE.md) for completed work, limitations, and
the next planned actions.
