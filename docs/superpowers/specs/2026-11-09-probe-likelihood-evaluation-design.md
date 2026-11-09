# Do our likelihoods work on high-contact probes? Evaluation design

**Date:** 2026-11-09 · **Owner:** Eric Denovellis · **Status:** design, awaiting approval
**Home:** `/cumulus/edeno/clusterless_high_d` (this repo)
**Package under test:** `non_local_detector` (`main`, currently `11d9b3de`)

Items marked **(user)** were decided by the owner in conversation. Items marked **(my call)**
were chosen by the author of this document and are open to change.

## Why

We want clusterless decoding to work on tetrodes, 32-contact polymer shanks and Neuropixels.
We want sorted-spike and clusterless likelihoods to respect track shape, boundaries and uneven
occupancy. We want analyses to finish in reasonable time.

Today we cannot say whether any of that is true. Three earlier rounds of work reached
conclusions that later rounds overturned:

- June 2026 (this repo's research notes): the KDE fails at 9–10 mark dimensions; the fix is to
  group contacts.
- October 2026: after a package change to the background mark term (PR #79), a rerun on
  simulated polymer marks showed no collapse up to 32 contacts. The June cliff was attributed
  to a numerical floor.
- October 2026: on the one real high-contact recording (RS58), a held-out score put the mark
  GLM slightly ahead of the KDE and a mixture density network ahead of both. An earlier
  version of that score had been inflated fourfold by a mis-built reference.

Each round was done by an AI assistant, and none was independently checked. The owner's main
fear for this round is an incorrect analysis. So the first product of this project is a
yardstick that can be trusted, and only then a verdict.

## What we want to be able to say at the end

1. Whether each current clusterless likelihood works on high-contact shanks, and up to how many
   contacts. Clearing 10 and 20 contacts per shank are the marks that matter **(user)**.
2. Which current likelihood is best, on accuracy and on analysis time.
3. Whether something better exists: other estimators, joint forms of the mark GLM, other
   waveform features including the whole waveform, biophysical priors, and simple models we
   have not tried **(user)**.
4. How sorted-spike and clusterless rate estimates behave at boundaries and at reward wells,
   where occupancy is high, with particular attention to estimators that form a rate as spike
   density divided by occupancy **(user)**.

## Non-goals

- Real-time latency. Speed here means wall-clock time for an analysis.
- Changing package defaults. All new models and options stay opt-in **(user, 2026-10-19)**.
- Any change to `non_local_detector` during steps 1 and 2. They are measurement only
  **(my call)**.
- Processing a real Neuropixels recording. The lab has none. This round uses simulation, plus
  one bounded search of DANDI (see Data).
- Chronic multi-day recordings. They are not ready to analyze; simulated long sessions stand in.
- Spike sorting, non-spatial decoding targets, and changes to Spyglass.

## Anti-goals: what would make this a failure even if it runs

- A headline number that turns out to be wrong.
- A conclusion drawn from simulation that the simulation cannot support (see Simulation).
- Tuning anything on test folds.
- A fourth written record that disagrees with the other three.

## What exists, and how far to trust it

| Thing | Where | State |
|---|---|---|
| Spyglass loaders, position cleaning, cross-probe coincident-spike filter, dataset registry | this repo, `src/clusterless_high_d/data_loaders/`, `configs/datasets.yaml` | committed |
| Nested cross-validation with embargo; evaluation panel (decode error, true-position log density, coverage, agreement with a sorted posterior); mark transforms; their tests | this repo, `src/clusterless_high_d/{cv,evaluation,marks}.py`, `tests/` | **untracked**, plus uncommitted edits to `decode.py` |
| This repo's environment | `.venv` | pins `non_local_detector` `0.6.10.dev330+g2d730c3a8`, the build that had the numerical floor |
| June research notes | this repo, `.claude/research/clusterless-high-d/` | untracked; conclusions partly superseded |
| Flat-referenced held-out score harness | `/cumulus/edeno/lowocc/jacob/jacob_detsel.py` (`--dataset RS58`) | scratch script; read on 2026-11-09, not rerun. Results: `/cumulus/edeno/lowocc/flatref/RS58_{kde,markglm}_fold{0-4}.json` and `.log`. The package commit it ran against was not recorded |
| RS58 cache (marks, position, track graph) | `/cumulus/edeno/lowocc/rs58/rs58_ep6.npz` | 12 CA1 shanks in the recording; 11 have waveform features (sort groups 0–6 and 8–11; group 7 is missing). 26–32 amplitude features each, 1,034,677 spikes, 20 min |
| RS58 and j16 waveform snippets, PCA features, noise covariance | `/cumulus/edeno/lowocc/waveforms/` (README there) | read-only extraction, centre samples equal cached marks |
| October roadmap and literature review | laptop clone of `non_local_detector`, `.claude/docs/plans/clusterless-research-roadmap/` | untracked there; items 0a, 0b, 6c never started |
| Probe simulator using MEArec templates (NEURON for template generation): tetrode, 4×32 polymer, Neuropixels layouts; known unit identity and place-field centres | laptop clone of `non_local_detector`, commit `467cb2ca` (2026-06-02) | **no branch points to it**; not on GitHub; not on `/cumulus` |
| Older dense-probe simulator with spike localization | `origin/dense-probe-simulation` (2026-04-14) | on GitHub, unmerged, no MEArec |
| Database survey of candidate recordings | `/cumulus/edeno/lowocc/dbsurvey/` | CSVs, read-only queries |

What reading the October harness showed:

- Held-out spikes get zero weight in the fit, including the background mark model. The log
  likelihood is summed in float64. Subtracting a flat refit of the same model is the right
  reference. These were checked by reading code, not by test.
- KDE bandwidths were compared on test folds. On RS58 fold 0 the mark bandwidth mattered a lot
  (24: 185 nats/s, 48: 154, 12: 38, on the older hand-built reference).
- The score is the full detector's marginal. On RS58 the held-out posterior spends 21–41% of
  the time in the Non-Local Continuous state.
- No simulation with a known true intensity has ever been scored this way.
- Only RS58 has high-dimensional marks. Jacob and L14 are polymer probes whose marks were
  extracted on 4–6 channels per group.

One June result has no verified explanation: on Jacob, held-out median error was 22–24 cm,
against about 4 cm when fitting on all data. The notes attribute it to drift.

## Scoring

**Primary score (user).** Held-out log likelihood of the full detector, minus that of the same
model refit with no spatial tuning, in nats per second and nats per spike. The hippocampus can
represent positions other than the animal's, and this score allows for that.

- The flat refit keeps every non-spatial setting of the spatial model. For the KDE that means
  `position_std=1e4`; for GLM-type models, `log_rate_prior_std=1e-3`. Every new model must
  define its own flat refit before it is scored.
- The detector, bin width (2 ms) and state set match the October harness unless a reason to
  change is recorded **(my call)**. That harness used `NonLocalClusterlessDetector` at
  500 Hz with a plain `fit` then `predict` on each held-out block, and no EM.
- The primary score is computed without EM **(my call)**, so models are compared on their
  likelihoods at equal cost. EM of the initial conditions and discrete transitions, without
  encoding refits, is a secondary arm on the leading models to check that rankings hold.
  October's note was that EM added about 10 nats/s for sorted and 1 for clusterless without
  changing rankings; that note is untested here.

**Supporting scores (user).** Log density at the animal's position and median decode error on
held-out movement bins, plus coverage of the highest-density region. The code is local much of
the time, so these are evidence, but they do not decide a comparison.

**Cross-check (my call).** When primary and supporting scores disagree about which of two
models is better, the disagreement is reported and investigated.

**Absolute bar on real data (user).** Information captured by a clusterless model relative to
`sorted_spikes_glm` on curated units from the same shanks, same time bins, same folds, each
against its own flat refit. This applies only to shanks with many hippocampal contacts.

An October note says the 100 µV clusterless threshold on RS58 misses 8–76% of curated sorted
spikes, depending on the shank. That figure is unverified. To keep spike detection from being
mistaken for a likelihood failure, sorted units are scored two ways **(my call)**: with all
curated spikes, and with only the spikes the clusterless threshold also caught. The gap between
those two is the detection loss. The gap from the second to the clusterless model is the
likelihood loss. Lowering the threshold is in scope (see Detection threshold).

**Absolute bar in simulation (user: simulation builds trust; the form is my call).** The
fraction of the true model's information that a fitted model recovers, where the true model is
scored through the same detector code.

**Breakdowns.**
- By contacts per shank (see Contact sweep).
- By location: reward-well bins, bins next to a boundary, interior. On real data, time bins
  are grouped by where the animal is. In simulation, the estimated rate is also compared with
  the true rate in each class directly.
- By movement and immobility.

**Cost.** Fit time, decode time, time per EM iteration and peak memory are recorded with every
score **(user: speed matters because it sets how long an analysis takes)**.

**Folds and tuning (my call, following the October protocol).**
- Contiguous-time folds with an embargo; at least 5. Comparisons are paired by fold: mean and
  SD of the difference, and the number of folds better.
- Tuning parameters are chosen inside each fold's training data, using this repo's nested
  cross-validation, with the primary score as the criterion.
- The encoding model is fit on all training time, immobility included **(user)**. One optional
  arm excludes ripple times so its effect at the wells is measured.

## Guarding against a wrong analysis

1. **Reproduce before changing.** The ported harness must first reproduce October's RS58
   numbers, then be rerun on current `main` **(details are my call)**.
   - Candidates were `kde:24:2.0` (mark bandwidth 24, position bandwidth 2.0) and
     `markglm:32`, on 5 blocked folds with no embargo.
   - Per-fold flat-referenced scores to match, folds 0–4, in nats/s: KDE 167.2, 188.4, 166.9,
     183.2, 140.4; mark GLM 165.9, 200.4, 174.6, 188.7, 159.8.
   - The package commit was not recorded. The runs are dated 2026-10-18, one day before the
     mark GLM merged as `f66c42dc`, so reproduce on `f66c42dc`.
   - Reproduced means each KDE fold within 1 nat/s, and each mark GLM fold within 5 nats/s
     (its mixture fit changed in review before the merge).
   - If the KDE does not reproduce after a bounded effort, record the mismatch in the ledger
     as unexplained and continue with current `main` as the baseline. Do not let this block
     the simulation work, which is the stronger check.
   - Differences between `f66c42dc` and current `main` are reported with their cause.
2. **Second computation.** Every headline number is recomputed by a calculation written
   independently from the definition, without reading the first implementation. For the KDE
   that can be a float64 NumPy evaluation on a subset of held-out time.
3. **Gates that run with every result.**
   - The flat model scores about zero against itself.
   - Shifting position against spikes by whole blocks scores about zero.
   - Shuffling marks within a shank scores no better than the same shank with marks removed.
   - In simulation, no fitted model beats the true model beyond fold noise.
   - With a fixed spike set, adding contacts does not lower the score beyond fold noise (more
     features cannot remove information, so a fall is estimator failure).
4. **One ledger.** `docs/claims.md` lists each claim with its evidence path, package commit and
   a status: verified, superseded or untested. The June and October claims are entered first,
   with their current status. Later reports cite the ledger.
5. **Provenance on every row.** Package commit, repo commit, dataset cache, parameters, seed
   and host.
6. **Independent review.** Each step's report is reviewed by someone, or a fresh agent, who is
   given the data and code but not the conclusion.

## The work, in order

### Step 0. Secure what exists

- Commit the untracked evaluation, cross-validation and mark-transform code and tests in this
  repo. Committing needs the owner's go-ahead.
- Pin the simulator commit `467cb2ca` with a branch in the laptop clone and make it reachable
  from `/cumulus`. Pushing to GitHub needs the owner's approval; a `git bundle` copied to
  `/cumulus` is the fallback.
- Point this repo's environment at current `non_local_detector` `main`, pinned by commit.
- Start the claims ledger.

### Step 1. Trust the yardstick

**Goal:** show that the scoring ranks models correctly when the answer is known, and that the
real-data harness is computed correctly.

Done when:
- The October RS58 numbers are reproduced (safeguard 1).
- The simulator runs against current `main`, and the true model is scored through the
  detector.
- All gates in safeguard 3 pass on simulation and on RS58, or each failure is explained.
- The simulation covers the cases under Simulation, including non-local events and tracks with
  wells.
- The June Jacob gap (22–24 cm held-out against 4 cm) is explained with evidence, or recorded
  as unexplained.

### Step 2. Verdict on what we have

**Goal:** a table of every current likelihood against probe type and contacts per shank.

Models: `clusterless_kde`, `clusterless_mark_glm`, `clusterless_gmm`, `clusterless_diffusion`;
and for the boundary question `sorted_spikes_kde`, `sorted_spikes_glm`,
`sorted_spikes_diffusion`, `sorted_spikes_mrf`.

Done when:
- Each clusterless model has the primary and supporting scores, the sorted-unit ratio where
  sorted units exist, and cost, across the contact sweep, in simulation and on every real
  recording in the panel.
- Each sorted and clusterless model has its rate bias by location class in simulation, and its
  score by location class on real data.
- The report states, for each model, the largest contact count at which it works.

**"Works" (targets, fixed before running; a miss is reported, never loosened):**
- In simulation it recovers at least 80% of the true model's information at every contact
  count up to 32 **(my call on the number)**.
- On real shanks it beats its flat refit on every fold.
- On real shanks with at least 10 contacts in the estimated hippocampal span it captures at
  least as much information as sorted units on the same shanks **(user)**. The comparison that
  decides this uses the sorted spikes the clusterless threshold also caught; the all-spikes
  comparison is reported beside it.

### Step 3. Look for better

Step 2 decides the order. If current likelihoods already hold at 20 or more contacts, the
search narrows to speed and boundaries. Candidates are prototyped in this repo; anything that
earns promotion becomes a separate proposal for the package.

**Joint forms of the mark GLM (user).** The mark GLM fits a mixture to marks first, then a
rate map per component, so component membership ignores position.
- Alternate the two: membership uses both the mark density and the fitted rate at the spike's
  position.
- Position-dependent mixture weights (the mixture density network on `feat/clusterless-mdn`;
  +26 nats/s on RS58 in October, ties on tetrodes, fit 6–10× slower).
- The existing joint position-and-mark mixture (`clusterless_gmm`) as the contrast case.

**Waveform features (user).**
- Amplitudes; PCA of snippets at several sizes (PCA-12 beat amplitudes on RS58 and j16 in
  October); peak-centred channel neighbourhoods; the whole waveform; shape summaries such as
  width and asymmetry; spike location as an added feature (location alone lost 25–40%).
- The whole waveform exceeded 4 hours per fold in mixture models. It is tested here with
  estimators that tolerate many features.
- Snippets exist only for RS58 and j16.

**Simple models not tried before (user).** A ladder, each rung fast and able to reuse the
sorted-spike estimators and their boundary handling:
- shank multiunit with no marks (the floor that shows what marks add);
- one pseudo-unit per peak channel;
- pseudo-units from peak channel and amplitude bin;
- pseudo-units from k-means on marks;
- a per-channel product of one-dimensional mark densities;
- a nearest-neighbour histogram of position given mark.

**Avoiding the ratio (user).**
- A forest or boosted-tree classifier over position bins given the mark. Decode cost does not
  grow with stored spikes.
- A classifier-based density ratio: real (position, mark) pairs against occupancy-sampled
  pairs. A neural version is on `feat/clusterless-ratio` (+7 on RS58, −7 on j16).
- Poisson boosted trees for sorted-unit rate maps.

**Neural density estimators (user).** A conditional normalizing flow for marks given position.

**Fast kernels (my call).** Fixed-size approximations of the KDE; locally adaptive kernels.

**Biophysical priors (user).** Noise-covariance whitening, low-rank-plus-noise component
covariances, channel-neighbourhood structure. In October a measured noise covariance added
about 1 nat/s on RS58.

**Bar for calling a candidate better (my call, from the October protocol):** it beats the best
current model by more than the fold-to-fold spread on at least two high-contact recordings,
does not lose on tetrodes, and its analysis time is stated beside the gain. A candidate that
matches accuracy at clearly lower cost also counts.

Left out: full-waveform autoencoders (fit time) and coreset compression (no better than
uniform subsampling in October).

## Which contacts are in hippocampus

The owner does not have a ready answer to this, and asked for a best effort. So it is an
estimate, stated as one **(my call on the method)**.

- Check first whether the database labels region per contact. If it labels per shank, every
  contact on a CA1 shank reads as CA1 and the label is no help.
- Otherwise estimate from signals along each shank's depth: detected spike rate and amplitude
  per contact, ripple-band power during immobility, and how much each contact's multiunit
  activity says about position. The hippocampal span is the contiguous run of contacts around
  the peak of those profiles.
- Produce one figure per shank showing the profiles and the chosen span, for the owner to look
  at. The span is recorded in the ledger as an estimate.
- The sweep is built so the answer does not hinge on the span's exact edges (next section).
- Contacts outside hippocampus still appear in the marks of a real recording. How much they
  hurt each likelihood is reported, because a likelihood that needs contacts hand-picked is
  less useful than one that tolerates them.

## Detection threshold

Lowering the clusterless detection threshold is in scope **(user)**.

- Verify the October figure first: per shank, the fraction of curated sorted spikes that the
  current 100 µV threshold catches.
- Re-extract marks at lower thresholds for RS58, and for added recordings where the raw data
  allow. Read each recording's current threshold from its sorter parameters. Choose the lower
  values relative to each shank's noise level, and state them **(values left to the builder)**.
- Extraction is read-only on the database and writes to `/cumulus` scratch, as the RS58 and
  j16 waveform extractions did. Check that analysis files exist before reading them.
- **Hard check:** re-extraction at the current threshold reproduces the cached spike times and
  marks.
- Report, per threshold: spikes per second, the primary score, the sorted-unit ratio and cost.
  More spikes cost more time, so the report states where added spikes stop adding information.
- Each threshold's score is against its own flat refit on the same spikes.

## Contact sweep

- Real shanks: contiguous blocks of contacts at increasing sizes, nested so smaller blocks sit
  inside larger ones. Blocks start at the contact with the highest spike rate, taken from
  training data, and grow outward. The uncertain edges of the hippocampal span then matter only
  for the largest blocks, and the report marks which block sizes extend past the estimated
  span.
- Two arms **(my call)**:
  - **Fixed spike set.** The spikes are those whose peak contact lies in the smallest block.
    Only the number of mark features changes. This isolates dimensionality and feeds the
    monotonicity gate.
  - **Realistic.** The spikes are those whose peak contact lies in the block, so more contacts
    bring more spikes. This answers the practical question.
- Simulation runs the same sweep on the tetrode, polymer and Neuropixels layouts.
- **Which arm decides (my call).** "Works up to N contacts" is read from the realistic arm,
  because that is the practical question. The fixed-spike-set arm is the diagnostic that says
  whether a failure is caused by dimensionality.
- **Sorted comparison inside the sweep (my call).** At each block size the sorted units are
  those whose peak contact lies in the block, so both decoders see the same contacts.

## Budget

Shared GPUs set the limit, so the design is staged **(my call)**.
- Tune at anchor contact counts (4, 16 and the full shank) and reuse the chosen settings at
  the sizes between, checking one in-between size to confirm the reuse is safe.
- Size each round before launching; about 100 jobs per round has been workable.
- Step 3 candidates are screened on simulation and one RS58 fold first. Only candidates that
  pass the screen get full folds and the contact sweep.

## Data

- **Have:** RS58 epoch 6 (high-contact); j16 tetrodes, Jacob and L14 (4–6 features per group)
  as low-dimension controls.
- **To verify first:** which shanks the curated sorted group
  (`RS5820251203_.nwb_06_r3_shanks_0-11_ca1_default_exclude`) covers, how many units each has,
  and the estimated hippocampal span per shank (see Which contacts are in hippocampus). Not
  every contact on a 32-contact shank is in hippocampus **(user)**.
- **To add:** at least two more recordings with 10 or more hippocampal contacts per shank,
  preferring ones with curated sorted units. Candidates are listed in the October roadmap's
  Data section and the survey CSVs. If fewer than two qualify, say so, and the sorted-unit bar
  rests on RS58 and simulation.
- **Neuropixels:** simulation only. One bounded task searches DANDI for a hippocampal
  Neuropixels recording with raw data and tracked position, and reports what exists with
  dataset details checked on the site. No extraction without the owner's decision.
- Database access is read-only. `SpikeSortingRecording.get_recording` regenerates a missing
  file, which is a write; check the file exists first.

## Simulation

**What it must contain.**
- Tetrode, 32-contact polymer shank and Neuropixels layouts, with the contact sweep.
- Known place fields and a known joint intensity, so the true model can be scored.
- A linear or W track with reward wells where the animal dwells, and a 2-D arena, with place
  fields that stop at boundaries.
- Non-local events with a known represented trajectory, so the primary score is tested on the
  case that makes real data ambiguous **(my call)**.
- Marks that are not Gaussian by construction: amplitude variation within a unit, heavy tails,
  overlapping units, background multiunit, optional drift **(my call)**.
- Long sessions, for the scaling curves.

**What it can and cannot support (user's rule from June).**
- It can test the scoring, the dimension at which an estimator fails, boundary and occupancy
  bias, and cost.
- It cannot rank mixture models against kernel models while its marks are near-Gaussian. A
  ranking between model families needs real data.

The template generator needs NEURON below 9 in a Python 3.11 environment
(`mearec_gen_dense.py` at `467cb2ca` records the command).

## Speed

- Wall-clock fit, decode and per-EM-iteration time, and peak memory, with every result.
- Two curves per likelihood: cost against recording length using simulated long sessions, and
  cost against EM iterations, noting any likelihood that must be recomputed each iteration.
- No optimization work in steps 1 and 2. In step 3, cost is part of the bar.

## Constraints

- Hosts: breeze and zephyr, 10 A100s each, shared with others. Check `nvidia-smi`, use a GPU
  at 0% utilization, pin `CUDA_VISIBLE_DEVICES`, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
  Heavy jobs go to breeze first. Run in named tmux sessions; never kill a window with a job in
  it. Smoke-test one fold and size a sweep before launching it.
- Environments use `uv`.
- Benchmarks use full float32 matmul precision, not the A100's TF32 default.
- Committing to `main` of either repo, pushing, and opening pull requests need the owner's
  approval. Commits on a dedicated feature branch do not.
- Method and model names describe their contents; no letter codes.

## Left to the builder

- Exact fold count above five, embargo length, and the distance thresholds that define well
  and boundary bins. State them in the report.
- How the true model is plugged into the detector for scoring.
- How per-location scores are computed on real data, including whether the package exposes
  per-time-bin evidence.
- Whether the simulator is used in place or moved into this repo.
- The order of step 3 candidates after step 2, and which to drop early.
- How the second, independent computation is organized.

## Deliverables

- After each step: a short report in `docs/`, the ledger updated, result tables with
  provenance.
- At the end of step 2: the verdict table (model × probe type × contacts per shank: score,
  fraction of true or sorted information, cost).
- At the end of step 3: the same table extended with candidates, and a recommendation.
