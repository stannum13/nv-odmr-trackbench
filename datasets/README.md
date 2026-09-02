# External dataset registry

This directory records public provenance for optional external data. It does not
contain a copy of the Figshare source file, and ordinary package use never
downloads it. To load the checked Figshare file, obtain the exact version
yourself and pass its local path to `load_figshare_28788437`.

The source values are preserved as raw analog values. Although its header says
`count data (counts/s)`, its `/Dev1/AI0` channel and value range leave physical
units unresolved; the registry therefore labels the quantity
`unknown_analog_signal` and its unit status `conflicted_unverified`.

Use `run_playback(dataset, on_observation)` for estimator evaluation. It keeps
the recording traversal evaluator-owned and supplies one frozen observation per
callback. `iter_playback_for_analysis` retains offline source state by design,
so it is explicitly trusted evaluator tooling for diagnostics and must never be
passed to estimator code. Neither boundary is a security sandbox for adversarial
Python introspection; use process isolation for that threat model.
