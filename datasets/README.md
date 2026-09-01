# External dataset registry

This directory records public provenance for optional external data. It does not
contain a copy of the Figshare source file, and ordinary package use never
downloads it. To load the checked Figshare file, obtain the exact version
yourself and pass its local path to `load_figshare_28788437`.

The source values are preserved as raw analog values. Although its header says
`count data (counts/s)`, its `/Dev1/AI0` channel and value range leave physical
units unresolved; the registry therefore labels the quantity
`unknown_analog_signal` and its unit status `conflicted_unverified`.
