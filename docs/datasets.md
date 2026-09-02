# Recorded dataset playback

The optional real-data anchor is Figshare DOI
[`10.6084/m9.figshare.28788437.v1`](https://doi.org/10.6084/m9.figshare.28788437.v1),
the CC BY 4.0 dataset by Liu. Its explicit versioned file download URL is
`https://ndownloader.figshare.com/files/53646563`. The expected file is
18,974,276 bytes (about 19 MB) and has MD5 checksum
`df03ef2385cdd64d2f0e117ecd9d6c7e`.

The package neither downloads nor redistributes this file. Download it only if
you choose to work with it, verify the checksum locally, and preserve the
source attribution and CC BY 4.0 license. For example, on systems with an MD5
utility:

```bash
md5 20241224-1651-50_ODMR_data_ch0_raw.dat
```

The command below prints the checked registry metadata, including the expected
size and checksum:

```bash
odmrbench dataset-info
```

To structurally parse and summarize an explicit local path without changing
the stored row or frequency order:

```bash
odmrbench playback --path 20241224-1651-50_ODMR_data_ch0_raw.dat
```

Use `--max-observations N` for a prefix of the causal row-major sequence.
`playback` does not infer timestamps and does not turn the file's values into
photons, volts, or normalized fluorescence. Its JSON output accordingly marks
the values as `conflicted_unverified` and timing as
`nominal_without_timestamps`.

The Figshare file has no per-sweep timestamps, current, field, field direction,
or exact resonance trajectory. It therefore cannot supply exact center, FWHM,
Q, photon-budget, adaptive-query, or timing comparisons. Those comparisons
require synthetic emulation; combining real-file residual preparation with
synthetic dynamics must be labelled semi-empirical rather than experimental
ground truth.
