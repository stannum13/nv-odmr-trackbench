# Synthetic virtual-instrument CLI

`odmrbench simulate --config bundled:drift` runs the wheel-packaged fixed query
schedule from any working directory. It is a deterministic synthetic-emulation
scenario, not a fitted tracker or an experimental result. Estimators are a
later stage. The identical, editable source-checkout example remains at
`configs/drift.yaml` and can be invoked with
`odmrbench simulate --config configs/drift.yaml`.

The bundled configuration contains eight explicit pseudo-Voigt resonances, a
reference-centered baseline, a common linear center slew, seeded Poisson noise,
a nominal photon rate, frequency-setting overhead, and a fixed sequence of
frequency/integration queries. Frequencies are Hz and all time quantities are
seconds.

The command validates the complete configuration in package code before any
query runs or a virtual instrument is constructed. It accepts only the
declared Poisson-noise schema at this proof of concept, exactly eight uniquely
identified resonances, an integer seed, and a non-empty fixed query schedule.
Every query frequency and integration duration is canonicalized to a finite,
positive Python float, and the entire prospective virtual schedule must remain
finite. Invalid physical values are rejected by the same model and instrument
validation used by library callers.

Malformed YAML, unreadable paths, invalid configuration values, and malformed
local playback files are reported as concise `odmrbench: error: ...` messages
on stderr with exit status 2; expected user input errors do not print a Python
traceback.

The sorted JSON summary records the seed, mode, query count, virtual endpoint,
total integration, nominal exposure, signal-conditioned expected photons,
realized photons, and first/last measured sample. Expected photons are an
evaluator-side resource total; this CLI summary is not an estimator observation
interface. Virtual time includes the configured frequency overhead and no wall
clock delay is used.

Identical configuration bytes, query order, and seed reproduce the same
summary. That reproducibility does not establish tracking accuracy, realtime
performance, physical fidelity, or agreement with the optional recorded data.
