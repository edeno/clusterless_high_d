# Probe likelihood evaluation: orchestrator plan

**Status:** Not started.
**Date:** 2026-11-09 · **Owner:** Eric Denovellis
**Design:** [`../specs/2026-11-09-probe-likelihood-evaluation-design.md`](../specs/2026-11-09-probe-likelihood-evaluation-design.md).
The design is the source for what is being built and why: scores, the "works" targets, the
sanity gates, the simulation rules and the constraints. This plan does not restate them. It
says how an orchestrating session runs the work through other agents.

**Scope of this plan:** design steps 0, 1 and 2 (secure, trust the yardstick, verdict).
Step 3, the search for better models, is planned after the verdict, because the verdict decides
where to look (see "After the verdict").

## How this plan is meant to be run

One orchestrating session (Claude Fable) owns the goal, the decisions and the record. It hands
bounded work packages to other agents and checks what comes back. It writes little code itself.

### Roles

| Role | Does | Does not |
|---|---|---|
| Orchestrator | Chooses the next packages, writes briefs, checks returns against files, keeps the status file and the claims ledger, talks to the owner | Relay a number it has not traced to a file; do a worker's package itself in parallel with the worker |
| Worker | Implements one package on its own branch and reports | Change scope, tolerances or targets; touch another package's files |
| Verifier | Recomputes or re-derives a result from its definition | Read the worker's implementation or be told the expected value |
| Reviewer | Reads a package's diff or report for errors | Receive the conclusion the report reaches |

### Which model does what

| Work | Model | Reason |
|---|---|---|
| Orchestration, briefs, the verdict report | Fable | Holds the whole design; judges conflicting evidence |
| Scoring code, the true-model likelihood, simulator ground truth, the contact sweep, any statistical judgment | Opus | A subtle error here silently changes every result |
| Independent recomputation and review | Opus, in a fresh agent | Independence matters more than speed; never reuse the implementing agent |
| Ports, data caching, extraction, running sweeps, aggregation, figures | Sonnet | Routine work against a clear contract |
| File moves, bundles, existence checks, log collection | Haiku | Mechanical |

### State lives on disk, not in the conversation

A long orchestration outlives one context window. Three files in this repo hold the state, and
the orchestrator reads all three at the start of every session and updates them after every
package.

- `docs/probe-eval/STATUS.md`: one row per work package: state (not started, briefed, returned,
  verified, blocked), branch, last commit, result path, who verified it and how.
- `docs/claims.md`: the claims ledger the design requires.
- `docs/probe-eval/decisions.md`: dated decisions, each with its reason and who made it.

Results go in files under the results root (below). Chat is never the only home of a number.

### Briefing a worker

A worker knows only its brief. Each brief states:

1. The goal in one or two sentences, and what it is for.
2. What to read: the design sections that apply, and exact file paths.
3. Done-when conditions.
4. Fixed constraints, each with its reason, so the worker can handle cases the brief missed.
5. Validation, with each threshold marked hard (a miss blocks) or target (a miss is reported).
6. A suggested approach, labelled as a suggestion.
7. What to return (next section).

Briefs do not script steps. Agents at this level do better with the goal and the constraints
than with a recipe.

### What a worker returns

- Branch and commits.
- The commands it ran and where their output is.
- Every number it reports, with the path of the file that holds it.
- What it did not do, and anything that surprised it.
- For anything it believes but did not check: say so.

### How the orchestrator checks a return

- Open the result files and confirm the numbers in the report are in them.
- Rerun the cheapest check itself (a unit test, one fold, one gate).
- Read the diff before accepting it. Reviews have twice caught half-done edits here.
- For any number that will appear in a report to the owner, commission a verifier.
  - The verifier's brief contains the definition from the design, the data paths and the
    fitted inputs it may take as given. It does not contain the worker's code path or the
    value to expect.
  - The orchestrator compares the two results afterwards.
- A hard validation miss stops every package that depends on it. Write up what failed and ask
  the owner. Do not work around it.
- A target miss is recorded in the ledger and the work continues.

### Parallelism

- Packages in the same wave can run together unless the table says otherwise.
- At most four worker agents at once **(suggested)**. More than that and the orchestrator
  cannot check returns as fast as they arrive.
- Each package has its own branch and its own git worktree. Agents sharing one working tree
  have overwritten each other's uncommitted work in this project before.
- GPU jobs: one job per GPU, on a GPU at 0% utilization, heavy work on breeze first.

### Where code is edited and where it runs

The repo lives on a shared drive reached over ssh, and it has no git remote. Editing files
through ssh is slow and error-prone for an agent. The suggested arrangement, to be confirmed in
the first package:

- A local clone on the orchestrator's machine, with `breeze:/cumulus/edeno/clusterless_high_d`
  as its remote. Workers edit in local worktrees with ordinary file tools.
- Workers push their package branch to that remote. Jobs on breeze or zephyr run from a
  worktree of that branch under `/cumulus/edeno/probe-eval/worktrees/<package>`.
- The existing checkout at `/cumulus/edeno/clusterless_high_d` is never edited in place. It
  holds uncommitted June work, and October's scripts ran from it.
- Results root: `/cumulus/edeno/probe-eval/results/<package>/`, with a README per folder saying
  what ran, how to check progress and how to aggregate.
- Tests that need no database or GPU run locally. Anything that touches Spyglass, real data or
  a GPU runs on the hosts, inside a named tmux session, so a dropped VPN does not stop it.

### What needs the owner

Ask before any of these. Approval for one does not carry to the next.

- Pushing anything to GitHub, including the simulator branch.
- Merging into `master` here or `main` in `non_local_detector`.
- Any change to `non_local_detector`. Steps 0–2 are meant to need none; if a package finds it
  cannot proceed without one, stop and ask.
- Changing a target or tolerance that the design fixed.
- Downloading a public dataset larger than a few gigabytes.
- Anything that could write to the lab database. The answer will be no.

Tell the owner, without waiting, when a hard gate fails, when a recording is added to or
dropped from the panel, and when a June or October claim changes status in the ledger.

### Known traps on this project

- The Bash tool is zsh: an unquoted variable holding several flags is passed as one argument.
- `ssh host "pkill -f PATTERN"` kills its own session. Kill by PID or tmux window.
- Killing a queue runner's tmux window kills the job inside it.
- In a fresh worktree, `uv run pytest` can import the package from another checkout. Print
  `non_local_detector.__file__` and the commit before trusting a run.
- `/tmp` is per host; `/cumulus` is shared. File locks do not work across hosts.
- JAX on the A100s defaults to TF32 matmul precision. Benchmarks and scores use full float32.
- GPU runs are not bitwise reproducible (about 1e-6 relative). Compare against a rerun, not a
  stored number, when testing for equality.
- `SpikeSortingRecording.get_recording` rewrites a missing analysis file. Check it exists.
- The simulator's marks are near-Gaussian unless its realism options are on.
- j16 sorted units were double-counted in caches made before 2026-11-02.

## Order of work

```
Wave 0  foundation ──────────────┐
                                 ├─► Wave 1a  real-data yardstick ─┐
                                 ├─► Wave 1b  simulation yardstick ─┼─► yardstick review ─┐
                                 └─► Wave 2   data ─────────────────┘                     ├─► Wave 3  verdict
                                                                                          │
                                              (data needs only wave 0) ───────────────────┘
```

Waves 1a, 1b and 2 run in parallel once wave 0 is verified. Wave 3 starts only after the
yardstick review passes.

## Wave 0: foundation

### `workspace-and-environment`
- **Goal:** a working arrangement for editing and running, and an environment on current
  `non_local_detector`.
- **Model:** Sonnet.
- **Done when:**
  - The clone, remote and worktree arrangement above works end to end for a trivial change,
    or a better arrangement is recorded in `decisions.md` with its reason.
  - The repo's environment installs `non_local_detector` at a recorded `main` commit
    (currently `11d9b3de`), replacing the uncommitted pin to `2d730c3a` in `pyproject.toml`.
  - A second way to run against commit `f66c42dc` exists, for the October reproduction.
    October's harness did this with a package worktree on `PYTHONPATH`.
  - The 30 existing tests pass in the new environment, and a script prints the imported
    package's path and commit.
- **Fixed:** `uv` for environments. Do not alter the existing checkout's working tree.
- **Returns:** the commands a later worker uses to get a worktree and run a job on a host.

### `simulator-transfer`
- **Goal:** the probe simulator available in this repo and running against current `main`.
- **Model:** Sonnet.
- **Inputs:** branch `clusterless-curse-of-dimensionality` (commit `467cb2ca`) in the laptop
  clone of `non_local_detector`. Files: `src/non_local_detector/simulate/{mearec_placefield_simulation,probe_layouts,_sim_contract}.py`,
  their tests, and `research/curse_of_dim/*.py`. Entry point:
  `make_mearec_placefield_run_data`, which already has options for a shared per-spike gain,
  overlapping units and an untuned multiunit population.
- **Done when:**
  - The simulator modules and their tests live in this repo (suggested:
    `src/clusterless_high_d/simulate/`), with a header naming the source commit.
  - Their tests pass against current `main`, or each failure is listed with its cause.
  - A copy of the source branch is on `/cumulus` as a git bundle, so the history is not held
    by one laptop.
- **Fixed:** no change to `non_local_detector`. The simulator is copied, not installed from
  the old branch, because that branch is 5 months behind `main`.
- **Note:** generating new MEArec templates needs NEURON below 9 in a Python 3.11 environment;
  `research/curse_of_dim/mearec_gen_dense.py` records the command. Existing template files
  may be enough; check before regenerating.

### `claims-ledger`
- **Goal:** one record of what has been claimed and what is established.
- **Model:** Sonnet, with the orchestrator reviewing every status.
- **Inputs:** `.claude/research/clusterless-high-d/*.md` in the existing checkout (June);
  `.claude/docs/plans/clusterless-research-roadmap/PLAN.md` and `literature-review.md` in the
  laptop clone of `non_local_detector` (October); the design's "What exists" section.
- **Done when:** `docs/claims.md` lists each claim with its evidence path, package commit and
  status (verified, superseded, untested), and the two October documents are copied into
  `docs/reference/` so later agents can read them.
- **Fixed:** a claim is "verified" only if this project has rechecked it. Everything inherited
  starts as untested or superseded.

### `flat-referenced-score`
- **Goal:** the primary score as tested library code, replacing the scratch script.
- **Model:** Opus.
- **Inputs:** `/cumulus/edeno/lowocc/jacob/jacob_detsel.py` (the October harness);
  `src/clusterless_high_d/cv.py` (folds with embargo, nested selection);
  `src/clusterless_high_d/evaluation.py` (supporting scores).
- **Done when:**
  - A function takes a model specification, data and a fold, and returns the primary score in
    nats/s and nats/spike, the supporting scores, state occupancy, cost, and a provenance
    record (package commit, repo commit, data cache, parameters, seed, host).
  - Each model family declares its own flat refit in one place. A model with no declared flat
    refit cannot be scored.
  - Nested selection uses the primary score as its criterion and is covered by a test. The
    existing nested function has none.
  - Unit tests on small synthetic data cover: held-out spikes do not influence the fit; a flat
    model scores zero against itself; the score is unchanged when the held-out block is split
    into two decode calls.
- **Fixed:**
  - Target the current `main` API. The old script stays untouched as the reference for the
    October reproduction.
  - The fit uses all training time. Ripple exclusion is an option, off by default.
  - Float64 accumulation of the evidence. Full float32 matmul precision.
- **Validation (hard):** on one RS58 fold with `kde:24:2.0`, the library score equals the old
  script's score when both run against the same package commit, to within GPU run-to-run noise.
  If the old script no longer runs on `main`, compare on the newest commit where it does.

## Wave 1a: real-data yardstick

### `reproduce-october-rs58`
- **Goal:** know whether October's numbers can be regenerated, and what has changed since.
- **Model:** Sonnet to run; orchestrator checks.
- **Done when:** the per-fold table in the design's safeguard 1 is reproduced on `f66c42dc`
  with the original script, then the same folds are run on current `main` with the library
  score, and every difference larger than the stated tolerance has a cause.
- **Fixed:** tolerances and the bounded-effort rule are in the design. Do not tune anything.
- **Smoke first:** one fold, one model; record time and memory before queuing the rest.

### `sanity-gates-rs58`
- **Goal:** the design's gates, run on real data, as reusable code.
- **Model:** Sonnet.
- **Done when:** on RS58, for the KDE and the mark GLM: the flat model scores about zero; the
  block-shifted-position control scores about zero; shuffled marks score no better than no
  marks. "About zero" is judged against the fold-to-fold spread, and the numbers are reported.
- **Fixed:** the gates are functions any later result can call, not one-off scripts.

### `independent-recomputation`
- **Goal:** a second calculation of the primary score that shares no code with the first.
- **Model:** Opus, fresh agent, verifier role.
- **Brief contains:** the score's definition; the model's mathematical definition for the KDE
  (the kernel form, the background mark term, the bin width); the RS58 cache; a fitted
  detector's transition matrices and initial conditions, which are taken as given. It does not
  contain the library code path or any expected value.
- **Done when:** a float64 NumPy calculation on a short held-out slice gives its own primary
  score, and the orchestrator has compared it with the library's.
- **Validation (target):** agreement within 0.5 nats/s on the slice. A larger gap is
  investigated until explained; the explanation goes in the ledger.

### `explain-june-jacob-gap`
- **Goal:** find out why June's held-out decode error on Jacob was 22–24 cm against about 4 cm
  when fitting on all data.
- **Model:** Opus.
- **Brief:** states the observation and the files (`cv-full-session-results.md`,
  `notebooks/cv_parallel.py`, `cv.py`). It does not suggest a cause.
- **Done when:** a cause is demonstrated with evidence, or the gap is recorded as unexplained
  with what was ruled out. Use the current package, since June's build is superseded.

## Wave 1b: simulation yardstick

### `simulated-tracks-with-wells-and-boundaries`
- **Goal:** simulated sessions where the true rate at boundaries and wells is known.
- **Model:** Opus for the design of the ground truth; Sonnet may implement under it.
- **Done when:**
  - A linear or W track on a track graph, with reward wells where the animal dwells for a
    realistic share of the session, and a 2-D arena with walls.
  - Place fields that stop at boundaries, including fields centred near a wall or well.
  - The output carries the true rate map of every unit on the decoding grid, the true joint
    intensity as a callable, the animal's position, and unit labels.
  - Session length is a parameter, so long sessions can be made for the scaling curves.
- **Fixed:** spike times are not aligned to bin centres. Jitter within the sample interval, or
  recovery tests show a false bias.
- **Validation (hard):** empirical rates from a long simulation match the true rate maps within
  Poisson error, in interior, boundary and well bins separately.

### `simulated-non-local-events`
- **Goal:** sessions where the hippocampus represents a position other than the animal's, with
  the represented trajectory known.
- **Model:** Opus.
- **Done when:** replay-like events during immobility and sweep-like excursions during running
  can be switched on, and the output carries the represented position at every time.
- **Fixed:** events are off by default, so the simple case stays simple.

### `simulated-non-gaussian-marks`
- **Goal:** marks that do not favour mixture models by construction.
- **Model:** Sonnet, after an Opus-written specification.
- **Done when:** the existing realism options (shared gain, overlap, untuned multiunit) are
  confirmed working in the transferred simulator, heavy-tailed amplitude variation and slow
  drift are added, and a diagnostic reports how Gaussian each unit's marks are.
- **Fixed:** the true joint intensity must stay computable for every option, or the true model
  cannot be scored.

### `true-model-likelihood`
- **Goal:** the true simulated model scored through the same detector as every fitted model.
- **Model:** Opus.
- **Inputs:** the package registers likelihoods in `_CLUSTERLESS_ALGORITHMS` and
  `_SORTED_SPIKES_ALGORITHMS` (`non_local_detector.likelihoods`); the contract is in the
  package's `CONTRIBUTING.md`, "Adding a likelihood algorithm".
- **Done when:** a likelihood whose "fit" stores the true intensity can be selected by a
  detector from this repo, at run time, without editing the package; and its flat counterpart
  is defined.
- **Fixed:** the log likelihood is absolute, in the form the contract requires. A likelihood
  correct only up to a per-row constant moves probability between states and corrupts the
  score.
- **Validation (hard):** on a case small enough to compute by hand in float64, the registered
  likelihood matches.

### `sanity-gates-simulation`
- **Goal:** the gates on simulated data, where the answer is known.
- **Model:** Sonnet to run; orchestrator checks.
- **Done when:** for the KDE and the mark GLM, on the tetrode and polymer layouts:
  - no fitted model beats the true model beyond fold noise;
  - flat, block-shift and mark-shuffle gates pass;
  - with a fixed spike set, the true model's score does not fall as contacts are added (this
    checks the gate itself before it is used on fitted models);
  - with non-local events on, the primary score for the true model stays above its flat
    counterpart, and the report shows how the supporting scores behave in the same runs.
- **Hard:** the first two. A failure means the scoring or the simulation is wrong.

### `yardstick-review`
- **Goal:** an independent read of everything in waves 1a and 1b before it is relied on.
- **Model:** Opus, fresh agent, reviewer role.
- **Brief contains:** the code, the result files and the design. It does not contain the
  orchestrator's summary.
- **Done when:** findings are addressed or recorded, and the orchestrator has sent the owner a
  short yardstick report: what was checked, what passed, what remains unexplained.
- **This is a stopping point.** The owner reads the report before wave 3 starts.

## Wave 2: data

These need only wave 0, and all database access is read-only.

### `rs58-sorted-units`
- **Goal:** sorted-unit decoding on RS58, and the size of the detection loss.
- **Model:** Sonnet.
- **Done when:** curated units from the group named in the design are cached with their shank
  and peak contact; the per-shank fraction of sorted spikes that the 100 µV clusterless
  threshold also caught is measured; and `sorted_spikes_glm` scores on the same folds and bins.
- **Fixed:** matching a sorted spike to a clusterless spike uses a stated time window and the
  same shank. State the window and show the match rate is not sensitive to it.

### `hippocampal-span-estimate`
- **Goal:** a per-shank estimate of which contacts are in hippocampus.
- **Model:** Opus for the criterion; Sonnet for the figures.
- **Done when:** the design's method is carried out on RS58: a per-contact check of database
  labels, then depth profiles, a chosen span per shank, and one figure per shank for the owner.
- **Fixed:** it is recorded as an estimate. No result may depend on its exact edges.

### `lower-threshold-extraction`
- **Goal:** clusterless marks at lower detection thresholds.
- **Model:** Sonnet.
- **Inputs:** `/cumulus/edeno/lowocc/waveforms/` (`extract.py`, `spike_sets.py`, README).
- **Done when:** RS58 marks exist at the current threshold and at two or more lower ones chosen
  relative to each shank's noise, with the coincident-spike filter applied the same way.
- **Validation (hard):** extraction at the current threshold reproduces the cached spike times
  and marks.
- **Smoke first:** one shank, timed, before the rest.

### `panel-expansion`
- **Goal:** at least two more high-contact recordings.
- **Model:** the orchestrator shortlists from the survey; one Sonnet worker per recording.
- **Inputs:** `/cumulus/edeno/lowocc/dbsurvey/*.csv`; the candidate list in
  `docs/reference/` (October roadmap, Data section).
- **Done when:** each added recording has cached marks, position, track graph, contacts per
  shank, and sorted units where they exist; an entry in `configs/datasets.yaml`; and the two
  baseline models scored on it.
- **Fixed:** the clusterless spike set matches the database's waveform-feature entries exactly,
  including the artifact filter. Prefer recordings with curated sorted units and clean
  position.
- **If fewer than two qualify:** say so. The sorted-unit bar then rests on RS58 and simulation.

### `dandi-neuropixels-search`
- **Goal:** know whether a usable public Neuropixels recording exists.
- **Model:** Sonnet, with web access.
- **Done when:** a short note lists candidates with raw data, tracked position on a track or
  maze, and hippocampal placement, with identifiers, sizes and licences read from the site.
- **Fixed:** no downloads beyond metadata. Dataset identifiers are never recalled from memory.

## Wave 3: verdict

Starts after the yardstick review. Depends on waves 1 and 2.

### `contact-sweep`
- **Goal:** the design's contact sweep as code, for simulated and real shanks.
- **Model:** Opus.
- **Done when:** given a shank, it produces nested blocks grown from the highest-rate contact,
  in both arms (fixed spike set; realistic), and the matching sorted-unit subset for each block.
- **Fixed:** the starting contact is chosen from training data. Each spike belongs to one block
  arm at a time and is never counted twice.

### `scores-by-location`
- **Goal:** scores and rate bias broken out by wells, boundary bins and interior.
- **Model:** Opus.
- **Done when:** in simulation, estimated against true rate per location class for every
  sorted and clusterless model; on real data, the primary score per class, grouped by where the
  animal is. The thresholds that define the classes are stated.
- **Note:** the detector returns the evidence as one total. Per-bin evidence exists inside the
  package's filter. If it cannot be reached from this repo without editing the package, stop
  and bring the owner the smallest change that would expose it.

### `cost-and-scaling`
- **Goal:** the cost half of the verdict.
- **Model:** Sonnet.
- **Done when:** fit time, decode time, time per EM iteration and peak memory are recorded by
  the scoring function for every run, and the two curves in the design's Speed section exist
  for every current likelihood, with compile time reported separately from warm time.

### `verdict-runs`
- **Goal:** the numbers behind the verdict table.
- **Model:** Sonnet workers per recording and per simulated layout; the orchestrator sizes and
  queues the rounds.
- **Done when:** every current clusterless likelihood is scored across the sweep on every panel
  recording and the three simulated layouts, at each detection threshold where marks exist;
  every sorted likelihood has its boundary and well results; tuning followed the design's
  staged budget; and the gates ran with every result.
- **Fixed:** tuning never sees a test fold. Each round is sized from a smoke run before launch.

### `verdict-report`
- **Goal:** the answer to the owner's first two questions.
- **Model:** the orchestrator writes it. A fresh Opus reviewer reads it with the data and
  without the conclusion. Headline numbers get a verifier.
- **Done when:** `docs/probe-eval/verdict.md` holds the table (model × probe type × contacts per
  shank: primary score, fraction of true or sorted information, cost), the largest contact
  count at which each model works under the design's targets, the boundary and well findings,
  every disagreement between primary and supporting scores, and what remains untested.
- **Fixed:** targets are applied as written in the design. A miss is reported as a miss.
- **This is a stopping point.** The owner reads the verdict before anything further.

## After the verdict

The orchestrator writes a short addendum to this plan for design step 3. It names which
candidates from the design to screen first, given where the current likelihoods failed or were
slow, and applies the same package shape: screen on simulation and one real fold, then full
folds for what passes. Candidates are prototyped here through the same run-time registration
the true-model likelihood uses.

## Open at the start

- Whether the simulator branch should also be pushed to GitHub, or stay as a bundle on
  `/cumulus`. The plan assumes a bundle.
- Whether pushing package branches from the orchestrator's machine to the `/cumulus` repo is
  acceptable as the working arrangement. `workspace-and-environment` confirms it.
