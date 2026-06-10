# Data quality notes (position + clusterless)

Findings from auditing position tracking and clusterless waveform features
for the registry datasets
(`configs/datasets.yaml`). All verified against the live Spyglass DB on
2026-06-08. These are the non-obvious gotchas that the registry fields and
`load_position` defaults were designed around.

## Cross-dataset gotchas

- **DLC position is keyed by `epoch` (an int), not `interval_list_name`.**
  `DLCPosV1`'s primary key has no `interval_list_name`, so a Trodes-style
  `{nwb_file_name, interval_list_name}` restriction silently fails. The
  registry carries `epoch:` for DLC datasets; `load_position` branches on
  `source`.
- **`"pos N valid times"` intervals do not always correspond to run epochs.**
  For Jacob, all six `pos N` intervals map (confirmed via Spyglass
  `get_position_interval_epoch`) to the six *sleep* epochs; the W-track runs
  have no LED/Trodes position at all. Always map `pos N` -> epoch before
  trusting it.
- **DLC can have multiple configs per epoch.** L14 has 4 `DLCPosV1` rows for a
  single epoch (different models/params). `load_position` warns and uses the
  first; pin a specific model if it matters.
- **`load_position` filtering applies in this order:** drop NaN (position_x/y
  only — `orientation` is entirely NaN for some Trodes single-LED epochs and
  would otherwise empty the frame) -> drop origin sentinels -> flag/drop
  high-speed -> optional span-based interpolation.

## Per-dataset

### SC10_2025_06_28 (`SC10420250628_.nwb`) — Trodes, clean
- Source `TrodesPosV1`, all 9 epochs have position. Run epochs (task
  `content_discrimination`): 2, 4, 6, 8. Registry uses `pos 3 valid times`.
- Clean: no `(0,0)` sentinels, max speed ~107 cm/s, no filtering needed.
- Clusterless only in **CgR** (7 sort groups); CA1 is sorted-only.

### Jacob_2025_06_18 (`Jacob20250618_.nwb`) — DLC, good after filtering
- W-track run position is **DLC**, keyed by epoch. Runs: 2(r1), 4(r2), 6(r3),
  8(r4), 10(r5). Registry uses epoch 2.
- **Epoch 8 (r4) is degenerate** — 8x11 cm extent, ~0% moving (DLC failed).
  Do not use.
- Dropout sentinel is **exactly `(0,0)`** (224 samples in ep2). Tracking jumps
  up to ~800 cm/s. Cleaned with `max_speed: 120` + `interpolate: true`
  (origin gone, span interpolation bridges short gaps).
- Both sorted (141 units) and clusterless (8 probes) present in CA1.

### L14_2024_06_04 (`L1420240604_.nwb`) — DLC, partially clean
- DLC, keyed by epoch. Runs: 2, 4, 6 (epoch 4 = most movement). 4 DLC configs
  per epoch.
- **Dropout sentinel is NOT exactly `(0,0)` — it is ~`1e-14`** (numerical
  noise from the centroid transform). An exact `== 0` test misses all 229;
  `load_position` uses a tolerance (`origin_tol = 1e-6`).
- **Residual low-speed off-track mistracks remain** (~1.5% of samples, >8 cm
  off the track centerline at a median ~6 cm/s). These are genuine DLC
  mistracks in the source output; NOT removable by speed (even `max_speed=20`
  leaves ~270) or origin tests. **Decision: left in for now** — track
  decoding linearizes position to a track graph, which projects these onto the
  nearest segment and absorbs most of them. Revisit with a DLC-likelihood
  threshold or density filter only if linearization proves insufficient.
- Both sorted and clusterless present in right CA1 (4 sort groups each).

### SC10_2025_06_16 (`SC10420250616_.nwb`) — BLOCKED
- **No position data at all** (no `PositionSource`, no Trodes pos intervals,
  no DLC). `forkTrack` task. Also has **no clusterless** features. Sorted
  spikes only (PFC + hippocampus). `load_position` raises a clean `ValueError`.
- Needs position processing AND clusterless extraction before use.

## Clusterless waveform amplitude quality

Per-spike extremum amplitude = `max(|feature|)` across channels, from the
`ampl_*` waveform features (see `notebooks/waveform_amplitudes.png`).

| Dataset / area | units | spikes | median | p95 | max |
| --- | --- | --- | --- | --- | --- |
| SC10_06_28 / CgR | 7 | 5.9M | 145 | 324 | **11454** |
| Jacob / Left CA1 | 4 | 13k | 109 | 205 | 403 |
| Jacob / Right CA1 | 4 | 133k | 142 | 218 | 492 |
| L14 / right CA1 | 4 | 308k | 187 | 627 | **2808** |

- **Amplitude scales differ ~20-50x across datasets.** Jacob tops out ~500,
  L14 ~2800, SC10 CgR ~11000. This is almost certainly different probe gain /
  feature units, NOT biology. **Verify amplitude units before any
  cross-dataset clusterless comparison** — the mark space is not shared.
- **Jacob is the cleanest** clusterless: unimodal (Left CA1) / bimodal (Right
  CA1) with a clear ~100-unit detection-threshold edge and a well-behaved tail
  to ~400-500. Left CA1 fires sparsely (13k spikes), Right CA1 is richer.
- **SC10_06_28 CgR has a heavy, gappy high-amplitude tail** out to 11454 with
  isolated clusters above ~8000. Combined with 5.9M threshold crossings over 7
  probes, the extreme tail is likely artifact/saturation, not units — consider
  an amplitude cap before encoding.
- **L14 has a long tail to ~2808** beyond a secondary bump at ~500; the bulk is
  < ~800. Same caution: likely a noise tail worth capping.
- These distributions are the clusterless "marks"; tails of obvious artifacts
  inflate the mark space and should be capped per dataset before decoding.

## Cross-probe coincident artifacts

Events appearing on many spatially-separate probes within ~1 ms are artifacts
(chewing/movement/shared electrical noise), not real spikes. Detected by
binning spike times at 1 ms and counting distinct probes active per bin, vs a
per-probe-jittered null. Scripts: `notebooks/coincident_artifacts.py`,
`notebooks/coincident_amplitude.py`.

| Dataset | probes | all-probe bins (obs/null) | spikes removed (all-probe / >=n-1) |
| --- | --- | --- | --- |
| SC10_06_28 / CgR | 7 | 3223 / 650 | 2.7% / 4.3% |
| Jacob | 8 | 7 / 0 | 0.15% / 0.5% |
| L14 | 4 | 1328 / 33 | 3.0% / 9.8% |

- **Features are SIGNED**; the extreme outliers are large **negative** peaks
  (ranges: SC10 [-11454, 2821], L14 [-2808, 584], Jacob [-492, 194]).
  Coincident spikes' signed peaks are almost entirely negative (Jacob/L14
  100%, SC10 87% — SC10 also has a large-positive component).
- **Coincident spikes populate the extreme-amplitude tail** (confirmed):
  fraction of high-|amp| spikes that are coincident —
  SC10: p99 88%, **p99.9 100%**; L14: p99 34%, **p99.9 99.7%**;
  Jacob: ~0% (its high-amp spikes are real; its few coincident spikes sit at
  the ~100 detection threshold). So amplitude-cap and coincidence detection
  converge on the *same* artifact population at the extreme tail, but diverge
  at moderate amplitudes (coincidence catches synchronous mid-amplitude events
  an amplitude cap misses; an aggressive cap removes real large spikes).
- **L14 and SC10 CgR carry real coincident-artifact populations.**

### Filter implementation (ported from continuum-swr-replay)

`detect_coincident_spikes` (in `data_loaders/spikes.py`, ported from
`continuum_swr_replay`) is wired into `_load_clusterless` /
`load_spikes_by_area` (`remove_coincident=True` by default). It pools all
probes, clusters spikes within `spike_closeness_threshold` (40 us) via
`scipy.ndimage.label`, and drops clusters spanning >
`max_coincident_fraction` (0.33) of probes. It runs across ALL probes in the
NWB (artifacts span the whole probe), and returns kept-indices so waveform
features filter in lockstep. 12 unit tests in `tests/test_coincident_spikes.py`.

**Validated thresholds (jitter-null = chance removal):**

| dataset (40us) | frac>0.33 obs/null/excess | frac>0.5 obs/null/excess |
| --- | --- | --- |
| SC10 CgR | 5.9 / 1.9 / **+4.0%** | 3.0 / 0.2 / +2.9% |
| Jacob | 2.6 / **0.0** / +2.6% | 1.3 / 0.0 / +1.3% |
| L14 | 6.4 / 1.7 / **+4.7%** | 1.7 / 0.0 / +1.7% |

- The **40 us window is essential**: at 1 ms, chance coincidence removes ~98%
  of SC10 (350 Hz/probe); 100 us already admits meaningful chance.
- **Jacob's removal is entirely real** (null 0) — it has a genuine ~2.6%
  coincident population at >=3-of-8 probes (the earlier "Jacob clean" was only
  true at the strict all-8 threshold).
- Default `0.33` captures more real artifacts but incurs ~1.9% chance removal
  on the high-rate datasets (SC10, L14). `0.5` is the conservative option:
  near-zero chance removal, still positive real excess on all three. Tune
  per-dataset via `max_coincident_fraction` if chance removal matters.
- These coincident spikes are the same large-negative-amplitude artifacts as
  above, so coincidence removal and an extreme-amplitude cap are complementary
  (and largely redundant at the extreme tail).
