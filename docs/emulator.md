# Synthetic virtual-instrument CLI

`odmrbench simulate --config configs/drift.yaml` runs the fixed query schedule
in the supplied YAML file. It is a deterministic synthetic-emulation scenario,
not a fitted tracker or an experimental result. Estimators are a later stage.

The bundled configuration contains eight explicit pseudo-Voigt resonances, a
reference-centered baseline, a common linear center slew, seeded Poisson noise,
a nominal photon rate, frequency-setting overhead, and a fixed sequence of
frequency/integration queries. Frequencies are Hz and all time quantities are
seconds.

The command validates the complete configuration in package code before any
query runs. It accepts only the declared Poisson-noise schema at this proof of
concept, exactly eight uniquely identified resonances, an integer seed, and a
non-empty fixed query schedule. Invalid physical values are rejected by the
same model and instrument validation used by library callers.

The sorted JSON summary records the seed, mode, query count, virtual endpoint,
total integration, nominal exposure, signal-conditioned expected photons,
realized photons, and first/last measured sample. Expected photons are an
evaluator-side resource total; this CLI summary is not an estimator observation
interface. Virtual time includes the configured frequency overhead and no wall
clock delay is used.

Identical configuration bytes, query order, and seed reproduce the same
summary. That reproducibility does not establish tracking accuracy, realtime
performance, physical fidelity, or agreement with the optional recorded data.
