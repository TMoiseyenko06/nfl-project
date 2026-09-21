# NFL Game Prediction System

Ingests nflverse data, engineers leak-free contextual features, trains models,
backtests them walk-forward, and retrains weekly.

The point of this repo is **honest measurement**. Every headline number is
printed next to the baseline it has to beat, and the leakage audit runs in CI
because a prediction system that quietly sees the future is worse than no
system at all.

---

## Quick start

```bash
make setup           # venv + editable install
nflpred fetch        # download nflverse data into a versioned cache (~220 MB)
nflpred build-features
nflpred backtest --by-season
nflpred predict-week
```

Everything is config-driven via `configs/default.yaml`. No season, path, or
hyperparameter is hardcoded.

---

## What the numbers mean

Read this section before reading any result.

### The only evaluation protocol used here

**Walk-forward.** For each (season, week) in chronological order: train on every
completed game that kicked off strictly *before* that week's first kickoff,
predict the week, roll forward. 196 folds, refit every week.

There is no random k-fold split anywhere in this project, and there never
should be. Randomly splitting NFL games lets a model train on Week 12 and test
on Week 3 of the same season, which inflates every metric and measures nothing
you could have actually done on a Tuesday in November.

`tests/test_backtest.py::test_training_data_never_reaches_into_the_test_fold`
captures the actual training frame handed to every fold with a spy model and
asserts `max(train kickoff) < min(test kickoff)`. The protocol is tested, not
assumed.

### The baselines are the story

| Baseline | What it is | Why it's there |
|---|---|---|
| `home_team` | always pick the home team | the floor; anything below this is noise |
| `elo` | sequential Elo with MOV multiplier | what you get from results alone, no box score |
| `vegas` | the market's closing line | **the real bar** |

A model that beats `home_team` has done nothing. A model that beats `elo` has
learned something from play-by-play. A model that beats `vegas` out-of-sample,
consistently, over hundreds of games, is extraordinarily rare — and the honest
prior is that any system claiming to do so has a bug.

### Against the spread: 52.38% is the number that matters

At standard -110 juice you must win **52.38%** of spread bets to break even.
Not 50%. A model at 51% ATS loses money. A model at 53% over 200 games is
indistinguishable from luck (the standard error on 200 picks is ~3.5pp).

`ats_units` in the results table is net units risking 1 unit per game, which is
the only number that answers "would this have made money".

### Calibration

Accuracy alone is not enough. A model that says 70% should be right ~70% of the
time. The `ece` column is expected calibration error (lower is better) and
`nflpred backtest` prints the full curve.

---

## Data

All from [nflverse](https://github.com/nflverse/nflverse-data), free:

| Source | Used for |
|---|---|
| `games.csv` (nfldata) | schedule, results, rest days, roof/surface, closing lines, QBs, coaches, referees |
| `play_by_play_{season}.parquet` | EPA, success rate, PROE, explosive plays — all team form features |

**Cache versioning.** Every download records url, sha256, byte size and fetch
time in `data/cache/raw/manifest.json`. The feature table is keyed by a hash of
those checksums, so a backtest is reproducible against an exact data vintage
and results are allowed to change only when the vintage does.

### What is NOT in here, and will not be faked

**True player-vs-player coverage assignments** (which corner covered which
receiver on a given snap) are charting data behind PFF and similar paid
products. They are not in nflverse and this repo does not invent a proxy and
dress it up as matchup data.

---

## No data leakage

Every feature for a game must be computable strictly before that game's
kickoff. This is enforced by `tests/test_leakage.py`, which runs in CI, and it
tests **two independent directions** — both are necessary, and neither catches
the other's failures:

**1. Future-blindness** (`test_truncation_invariance`)
Rebuild the entire feature pipeline on data truncated to an earlier date.
Every feature for every game before that date must be bit-identical to the
full-data build. This catches season-long aggregates and anything pooled over
the whole dataset.

> This test earned its keep during development: it caught a weather climatology
> fallback that averaged over *all* seasons, including future ones. Subtle,
> real, and invisible to inspection.

**2. Own-game blindness** (`test_own_game_outcome_does_not_affect_its_own_features`)
Corrupt a game's play-by-play and rebuild. That game's own features must not
move. This catches an unshifted rolling window.

Truncation invariance does **not** catch an unshifted window — a within-game
leak is invariant to what comes after it. Both tests are required. Both were
verified against a deliberately injected leak (`shift(1)` → `shift(0)`) to
confirm they fail when they should; a test that cannot fail is decoration.

Also asserted: no outcome column reaches the feature set, no single feature
correlates with the result at |r| > 0.9, Elo is sequential, and unplayed games
carry NaN targets so they can never be trained on.

### Two honest caveats about "pre-kickoff"

**Closing lines.** `spread_line` in nflverse is the **closing** line, not the
opening line. Opening lines are not in any free source I could find. The
closing line is fixed at kickoff, so it is legitimately pre-kickoff information
and not leakage — but it is the *sharpest* version of the market and therefore
the hardest possible benchmark. Where a model is given the line as a feature it
is named `*_plus_vegas` and reported separately, because a model that has
anchored on the closing line is not doing independent work.

**Weather.** Recorded `temp`/`wind` only exist for completed outdoor games —
they are 0% populated for future games, so training on them and serving NaN
would be train/serve skew. Default mode is `climatology`: roofed games get
fixed indoor constants (70°F, 0 mph — true and known in advance), outdoor games
get the venue's historical mean for that month **computed from strictly earlier
seasons**. Set `features.weather.mode: observed` to measure what the recorded
conditions would have been worth; it is backtest-optimistic and not for
production.

---

## What it predicts

Six targets, all from the same pre-kickoff feature set, all backtested the same way:

| target | what it is | market benchmark |
|---|---|---|
| `win` | moneyline / win probability | yes (line → probability) |
| `spread` | final margin (home perspective) | yes (closing line) |
| `total` | final total points | yes (closing total) |
| `h1_win` | leading at halftime | **no** |
| `h1_spread` | halftime margin | **no** |
| `h1_total` | halftime total points | **no** |

Adding a market means adding a `TargetSpec` in `nflpred/models/targets.py`;
the models, backtester and metrics all iterate over that registry rather than
hardcoding targets.

### Halftime targets

Halftime scores are not in the schedule file — they are derived from
play-by-play. The subtlety worth knowing about: **`total_home_score` holds the
score *after* the play it sits on.** So the obvious approach — read the first
play of the second half — silently folds in anything scored on the opening
kickoff of the second half. One 2024 game has a 100-yard kickoff return
touchdown there, which would have put 6 phantom points into the halftime score.

The correct rule is the maximum over first-half plays, which is also robust to
play ordering since a score never decreases. Validated by applying the same
rule to the whole game: it reproduces the official final score for **all 3,052
games** in the dataset.

> **Halftime has no market benchmark.** nflverse carries no halftime lines, so
> there is nothing to check halftime predictions against except the naive
> baselines. The `vegas` row for halftime targets is a *derived* benchmark —
> a fitted map from the full-game line to the halftime outcome — and is
> labelled as derived everywhere it appears. Treat halftime numbers as
> "better than guessing", not as "beats the market", because there is no
> market here to beat.

### Coherent win probability

Fitting the moneyline as its own classifier lets it contradict the spread
model: LightGBM's moneyline pick disagreed with its own spread pick on
**12.7%** of out-of-sample games. That is incoherent in output a human reads.

The `*_coherent` models instead derive the probability from the predicted
margin, via a single-parameter logistic `P = sigmoid(b · margin)` fit on the
training fold. With **no intercept** the curve crosses 0.5 exactly at margin
zero, so the two can never disagree — coherence by construction, not by
tuning. Disagreement drops to 0.0%.

---

## Features

Built in tiers. Tier 1 only, so far.

**Tier 1 (built, backtested):**
- Rolling team EPA/play, offense and defense, split pass/rush
- Success rate, early-down EPA, explosive-play rate, PROE, sack rate, turnovers
- Garbage time excluded (win probability outside 0.05–0.95)
- EWMA (half-life 6 games) and 8-game flat windows, carried across seasons
- **Opponent-adjusted ratings** — a ridge decomposition of every team-game into
  `offense effect + opponent defense effect + home effect`, refit at every
  (season, week) boundary on prior games only, with exponential recency decay.
  Raw EPA conflates a team with its schedule; this separates them.
- Rest days, short week, off-bye, Thursday/Monday, primetime
- Travel: great-circle distance venue-to-venue (relocations and international
  games handled), time-zone delta wrapped across the date line, direction of
  travel, trailing 3-game travel load
- Weather (see caveat above), roof, surface, divisional/conference, neutral site
- Elo rating and Elo-implied probability
- Explicit matchup terms: each offense against the other's defense

**Tier 2 / 3 (not built):** coordinator tendencies, QB-change flags, injury
availability, O-line continuity, embeddings, referee crews. See "What's next".

---

## Module layout

```
nflpred/
  config.py            YAML config, dotted access
  ingest/nflverse.py   versioned download cache + manifest
  features/
    reference.py       stadium coordinates, time zones, divisions
    schedule.py        rest, travel, venue, weather
    epa.py             play-by-play -> team-game box -> rolling form
    adjust.py          opponent-adjusted ridge ratings
    build.py           assembly + feature-group registry
  models/
    elo.py             Elo ratings
    registry.py        baselines + linear + LightGBM, one interface
  backtest/
    walkforward.py     the fold loop
    metrics.py         accuracy, log loss, Brier, MAE, ATS, calibration
    report.py          calibration curves, gain + exact TreeSHAP
  predict/
    week.py            train on everything completed, predict next week
    log.py             append-only, hash-verified prediction log
  cli.py               fetch / build-features / train / backtest / predict-week / score / audit
```

---

## The prediction log

`artifacts/prediction_log.csv` is the most important artifact here. It is the
only evidence the system works on games it had not seen.

Three properties make it credible:

1. **Append-only.** Re-running `predict-week` will not overwrite an existing
   (model, game) row.
2. **Timestamped before kickoff.** Rows logged after kickoff are excluded from
   the track record by `logged_predictions_before_kickoff` — that is the
   difference between a prediction and a retrodiction.
3. **Hash-verified.** Every row carries a content hash. `nflpred audit` reports
   any row edited after the fact. A log you can quietly rewrite is not a track
   record.

"Upcoming" means **kickoff is still in the future**, not merely "unplayed".
A week that is already underway yields only its remaining games, so the system
cannot log a prediction for a game that has already started.

Weekly loop (automated by `.github/workflows/weekly.yml`, Tuesdays 13:00 UTC):

```bash
nflpred fetch          # pull new data
nflpred build-features # recompute features for completed games
pytest tests/test_leakage.py   # gate: no predictions from a leaking vintage
nflpred score          # score last week, append to the performance log
nflpred predict-week   # predict upcoming week, log BEFORE kickoff
```

The weekly job runs the leakage audit **before** it logs anything. A prediction
produced from a vintage that fails the audit does not get written.

---

## Reproducibility

- Seeds fixed in config (`models.seed`), LightGBM runs single-threaded with
  `deterministic=True` and `force_row_wise=True`
- Feature cache keyed by data vintage hash
- No randomness in any split — folds are chronological

---

## Phase 1 results

> **Note:** the table below is the original **three-target, seven-model** run
> (win / spread / total). The halftime targets and the `*_coherent` models were
> added afterwards and are being re-backtested across all six targets; this
> section will be replaced with those numbers. The conclusions below still
> hold for the models they describe.

Walk-forward out-of-sample, **2,524 games** (2017 wk1 – 2026 wk2), 196 folds,
refit every week. Data vintage `e1e882d2217b`.

| model | accuracy | log loss | Brier | ECE | spread MAE | ATS% | ATS units | total MAE |
|---|---|---|---|---|---|---|---|---|
| home_team | .5461 | .6893 | .2481 | .0095 | 11.087 | .5124 | −53.7 | 11.090 |
| elo | .6391 | .6354 | .2223 | .0246 | 10.286 | .4986 | −118.6 | 11.090 |
| **vegas** | **.6657** | **.6077** | **.2102** | .0246 | **9.878** | .5083 | −72.8 | **10.512** |
| linear | .6455 | .6404 | .2236 | .0384 | 10.337 | .5132 | −49.9 | 11.116 |
| lightgbm | .6415 | .6441 | .2251 | .0407 | 10.439 | .4953 | −133.9 | 11.244 |
| linear_plus_vegas | .6729 | .6206 | .2148 | .0309 | 10.018 | .5225 | −6.0 | 10.675 |
| lightgbm_plus_vegas | .6570 | .6263 | .2171 | .0298 | 10.177 | .5026 | −99.5 | 10.814 |

ATS break-even at −110 juice: **.5238**. Every model is below it.

> **Read the `vegas` ATS row carefully.** That model predicts the line exactly,
> so `pred_spread > vegas_spread` is never true and it silently becomes
> "always pick the away team". Its .5083 is just the rate at which away teams
> covered (a real but tiny effect), **not** the market beating itself. It is
> included for completeness, not as a strategy.

### What this actually says

**1. Nothing here beats the market.** Vegas has the best log loss, Brier,
spread MAE and total MAE. `linear_plus_vegas` shows higher raw accuracy
(.6729 vs .6657) but **that gap is not significant** — McNemar p = 0.32 — and
the same model is significantly *worse* than the line on spread MAE
(+0.140, paired p = 0.003). It is a model that has been handed the closing line
and has made it slightly worse. That is the honest reading.

**2. No model clears the betting threshold.** Best ATS is `linear_plus_vegas`
at 52.25% (95% CI 50.3–54.2%), below the 52.38% break-even, p = 0.56 against
it. Every model loses units over 2,463 graded games. Nothing here is a betting
system, and the backtest is what says so.

**3. The gradient-boosted tree lost to the linear model** on every single
metric: accuracy, log loss, Brier, calibration, spread MAE, total MAE, ATS.
With ~3,000 training rows and ~60 correlated features, the extra capacity is
spent on noise. This is the sample-size constraint showing up exactly where it
was predicted to.

**4. Elo has a better log loss than either learned model** (.6354 vs .6404 and
.6441). All of the play-by-play feature engineering — rolling EPA, success
rate, opponent-adjusted ridge ratings, travel, matchup terms — buys about
**0.6 accuracy points over Elo and slightly worse calibration**. Tier 1 is
close to the information ceiling of pure team-strength-from-results.

**5. Per-season variance dwarfs the model differences.** Vegas accuracy ranges
from .6232 (2021) to .7053 (2024). Any single season "proves" whatever you
want it to.

### The sample-size arithmetic

To distinguish a genuine **1 percentage point** ATS edge (53.4% true) from
break-even at 80% power and p < 0.05 requires roughly:

| true ATS | games needed | NFL seasons |
|---|---|---|
| 53.4% | ~18,800 | **69** |
| 55.0% | ~2,850 | 10.5 |
| 57.0% | ~910 | 3.4 |

This entire backtest is 2,463 graded games. It is **not large enough to
establish a small edge even if one existed.** Any weekly track record will be
far smaller still. Treat every ATS number here, including the good-looking
ones, as consistent with zero.

### Calibration

Vegas is well calibrated (gaps mostly under 0.04). The learned models are
overconfident at the extremes — `lightgbm` predicts .910 in its top bin where
the true rate is .643 (n=28), and is +0.069 too high in the .7–.8 bin. This is
why ECE is reported alongside accuracy.

### What the tree model leans on (mean |SHAP|, exact TreeSHAP)

```
elo_prob_home               0.165
d_margin_r5                 0.153
d_adj_off_success_rate_net  0.149
elo_diff                    0.141
ctx_wind                    0.131
d_off_success_rate_ewma     0.101
d_off_sack_rate_ewma        0.097
d_def_explosive_rate_ewma   0.079
```

Elo carries the most weight, followed by recent scoring margin and the
opponent-adjusted success-rate rating — so the opponent adjustment is earning
its place.

**`ctx_wind` at #5 is a red flag, not a finding.** In the default
`climatology` mode wind is a venue×month historical mean, so it cannot be
carrying game-specific weather information. It is acting as a proxy for
*"outdoor stadium, late in the season"* — a stadium/season identity feature
wearing a weather label. That is worth an ablation before it is trusted, and it
is the first thing I would cut.

---

## What I'd do next, and what I'm skeptical of

### Next
1. **Ablate the weather block** and the `ctx_wind` proxy specifically. Compare
   `weather.mode: climatology` / `observed` / `none` on identical folds.
2. **QB-change flag (Tier 2).** The single highest-value feature not yet built.
   The models currently treat a team as its rolling average regardless of who
   is playing quarterback — an obvious, large, known gap.
3. **Prune, don't add.** The tree lost to the linear model, which says the
   feature set is already too wide for the sample. Test a deliberately small
   model (Elo + opponent-adjusted net + rest/travel) before adding anything.
4. **Model the spread directly and derive the win probability from it**, rather
   than fitting a separate classifier. Margin is the richer target.

### Player props — assessed, not built

The data exists and is good: `stats_player_week` carries ~19,000 player-weeks
per season (~200,000 across 2015–2026) with passing/rushing/receiving yards,
receptions, targets, carries, touchdowns, target share and air-yards share.

That is a **fundamentally better sample-size regime than game prediction** —
200,000 rows instead of 3,000 — which is the single strongest argument for
building it. It is also where sportsbooks are least sharp, because they post
thousands of prop lines with far less attention per line than the main markets.

Two real obstacles, neither fatal:

1. **Usage has to be projected before production can be.** A receiver's yards
   depend mostly on whether he plays and how many snaps and targets he gets.
   That needs the injury report, depth charts and snap counts — all free in
   nflverse, all currently unused by this repo.
2. **No free prop lines.** Same problem as halftime, but it bites harder here:
   props are interesting precisely *because* the lines may be soft, and
   without the lines there is no way to measure whether an edge exists. The
   model can be scored on raw accuracy (MAE on receiving yards, say), but
   "beats the book" would be unverifiable.

So props are worth building as a *projection* system, and the honest framing
is that its value would be measurable in MAE and calibration, not in ATS.

### Skeptical of
- **Any model given `vegas_*` features.** It is anchoring on the closing line
  and adding noise. The honest framing is that `vegas` is the benchmark, not an
  input.
- **The opponent-adjustment ridge alpha (12.0) and recency half-life (400d)**
  were set by judgement, not tuned. Tuning them on this data would be
  fitting the test set; they need a nested walk-forward.
- **`d_margin_r5` ranking second.** Recent point margin is famously noisy and
  mean-reverting. Its importance may be the model latching onto variance.
- **A neural network is not justified by these results.** Per the brief, Phase 3
  is gated on beating Phase 1 out-of-sample. Phase 1's best learned model is a
  ridge/logistic pair that still loses to the market, and the *smaller* model
  beat the larger one. Adding capacity is the opposite of what the evidence
  points to. **Not building it yet.**
