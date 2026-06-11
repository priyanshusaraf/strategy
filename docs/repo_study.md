# Repository Study — Prior Research Program (Spread Mean-Reversion, main → orthogonal-mr)

Audit date: 2026-06-11. Sources: `README.md`, `SUMMARY.md` (read in full), and code verification of
`mrlab.py`, `hourly_lab.py`, `book.py`, `broad_book.py`, `duka_cracks.py`, `daily_frozen.py`,
`cracks_seasonal.py`, `calendars.py`, `validate.py`, `hourly4y.py`, `final.py`,
`ou_spread_mr_hourly.pine`, plus the diagnostic/falsification scripts (`diag*.py`, `ic.py`,
`xsec.py`, `anchor.py`, `run*.py`, `sweep.py`, `hourly.py`).

Purpose: infer what was already tested, what was assumed, what was concluded, what failed and why —
and what the documented claims do and do not match in the code — so the orthogonal program does not
re-derive, re-break, or silently inherit flaws.

---

## 1) Hypotheses tested

| # | Hypothesis (statement) | Where | Result |
|---|---|---|---|
| H1 | Daily-bar OU-gated binary z-score MR works on ~40 economically-linked spreads | `mrlab.py` (`strat_binary` + `ou_halflife` gate), `run.py` | **Falsified**: 1/28 spreads survive IS+OOS |
| H2 | Continuous/linear sizing, lookback ensembles (10–250d), trade bands rescue daily MR | `mrlab.strat_linear`, `strat_linear_ens`, `run.py`, `sweep.py` (48-config grid) | **Falsified**: mean OOS Sharpe ≈ −0.3 to −0.5 across the grid |
| H3 | Detrending the signal level (cumsum of drift-adjusted changes) fixes roll-carry bleed | `mrlab.backtest(detrend=...)`, `run.py` `lin_dt` configs | Helps (3/28 survive) but **no edge** |
| H4 | Seasonal (day-of-year norm) anchoring extracts crack/calendar MR daily | `mrlab.deseasonalize`, `cracks_seasonal.py`, `calendars.py` | **Falsified for cracks**: best HO–Brent Sharpe 0.15, 0 smooth of 11. Calendars: works only in-era (see H12) |
| H5 | Fade-the-k-day-move (k=5…252), with/without trend filters; long-horizon margin reversion | `diag2.py`, `diag3.py` | **Falsified**: negative at every horizon post-2000 |
| H6 | Asymmetric floor fade (long-cheap only; storage/run-cut floor) — pre-registered | `anchor.py` (hypothesis stated in docstring before running) | **Failed its pre-registered bar** (5/40 OOS-positive) |
| H7 | Cross-sectional MR within economic families (family trend cancels in demeaning) | `xsec.py` (7 families) | **Falsified**: all families negative |
| H8 | Trailing-Sharpe meta-filter creates edge / picking recent winners improves the book | `mrlab.meta_filter`, `final.py` `meta_gate`, `book.py` `SELECT` variant | Filter only cuts bleed; **selection actively destroys** (book −0.74 vs +0.68 naive) |
| H9 | Daily spreads contain forward-predictive information at all (IC map, delay-1) | `ic.py`, `diag.py` (AR1/variance-ratio map) | Cracks ≈ random walks daily (VR≈1); weak/unstable IC — consistent with H1–H7 |
| H10 | The same 2–3 week reversion is alive on HOURLY bars (entries at intrabar extremes) | `hourly.py` (3.3y CL−BRN TV export), `hourly2.py` | **Validated**: IS 0.78 / OOS 1.28; rule frozen here |
| H11 | A vol gate (5d vol > 6m rolling median) is the robustness/smoothness lever | validated on independent CL−BRN set; implemented in every hourly engine | **Validated**: worst-half Sharpe sum +0.6 → +2.5; book OOS 1.05→1.84 |
| H12 | Storage-anchored calendar spreads were the smooth MR vehicle in the in-edge era | `calendars.py`, `daily_frozen.py` (1990–2009 window) | **Confirmed in-era only**: 10 smooth names 1990–2009, book Sharpe 2.02; decays after ~2008 |
| H13 | The frozen hourly rule generalizes to a 19–20 spread Yahoo panel (2y) | `hourly_lab.py` + `final.py` | 11/20 positive both halves; curated family-parity book 0.78 full / 1.84 OOS |
| H14 | Diversification (not per-spread quality) is the smoothness engine | `book.py` (37-spread universe, inverse-vol, 35% family cap) | **Confirmed**: maxDD halved (−12.6% → −6.6%), gated book 0.68/1.09 |
| H15 | Lack of hourly history is the only blocker to 10 smooth cracks (data-length hypothesis) | `hourly4y.py` (TV+Yahoo merged, ~3–4y) vs `duka_cracks.py` (12.3y) | **Refuted as a general claim**: BRENT–WTI smooth at 3y AND 12y, but diesel cracks/metals stay choppy at 12y. Data length is not the binding constraint for most spreads |
| H16 | How many commodity spreads truly revert smoothly over 12y hourly | `duka_cracks.py` (Dukascopy) | **~1 unambiguous (BRENT–WTI: Sh 1.93, eqR² 0.98, G/P 17.6) + 2 borderline BTU**; rest choppy. Structural ceiling of the asset class |
| H17 | Broadening to tightly-cointegrated FX crosses reaches "10 smooth securities" | `broad_book.py` (cracks anchor + ~40 FX candidates) | 10 pass the screen (3 cracks + 7 FX); book Sharpe 1.26, eqR² 0.99, G/P 14.5 — **with material caveats, see §2/§3** |

### Documented claims vs. code — verified divergences

1. **"Fills: NEXT BAR OPEN (never same-close)" is NOT true of the headline engines.**
   - Honest next-open fills with day/overnight decomposition exist in `mrlab.backtest`
     (exec_mode="open"), `hourly_lab.run` (used by `final.py`/`book.py`), and the Pine script.
   - But `duka_cracks.one`, `hourly4y.one`, and `broad_book.one` all compute
     `pnl = u.shift(1)*dS − 0.05*sig.shift(1)*du`, where `u[i]` is decided from `z[i]` (which
     includes close i). The entry price is therefore **the same bar close the signal observed**.
     The 12-year "definitive" crack result, the BRN–WTI Sharpe 2.90 (3.1y), and the 10-security
     deliverable all use same-close hourly fills. Both legs come from the same Dukascopy feed
     (synchronous closes), so the cross-venue async artifact (the GO_crack +2.9→−0.7 failure mode)
     does not apply — but a zero-delay fill at the observed extreme is still optimistic, and it
     contradicts the frozen-rule box in README/SUMMARY. The honest-fill confirmations of the same
     edges are smaller: BZ_CL at 2y under `hourly_lab` next-open fills = 0.78 full / 1.18 OOS.
2. **"Costs 1.5 ticks/leg/turn (stress 3.0)" applies only to `hourly_lab`/`final.py`.**
   `duka_cracks`, `hourly4y`, and `broad_book` charge a vol-proportional cost (0.05 × bar σ per
   unit turned, the daily-lab λ) with **no tick mapping and no 2× stress run** in their mains.
   The 2×-slip column exists only in `final.py`. The headline 10-security table is unstressed.
3. **FX carry/swap is ignored.** `broad_book` trades FX crosses (incl. EURPLN/EURCZK/EURHUF
   candidates) holding up to 10 trading days, with no financing/swap PnL. Detrending removes
   price-drift but not the actual carry cash flow. For positive-carry-fade pairs this can be a
   first-order omission. Not flagged in README/SUMMARY.
4. **The "smoothness bar" drifts across studies**: eqR²≥0.90/GP≥3.0 (`daily_frozen`),
   ≥0.80/2.5 (`calendars`, `duka_cracks`), ≥0.75/2.0 (`broad_book`, the deliverable),
   ≥0.60/2.5 (`hourly4y`). The deliverable uses the loosest bar; "fixed in advance, not tuned to
   reach 10" is asserted but not enforceable from the code history.
5. **The book of 10 is formed ex-post from screen survivors.** `broad_book` screens ~40 candidate
   securities on full-sample criteria (both halves +, eqR², G/P) and then builds the equal-risk
   book **only from the 10 winners**. Book Sharpe 1.26 / eqR² 0.99 therefore embeds winner-selection
   bias (the both-halves requirement mitigates but does not remove it). The honest no-selection
   analogue is `book.py` NAIVE (0.68/1.09, −6.6% DD).
6. **The "all 36 pairwise spreads scanned" claim is not reproducible from committed code** —
   `duka_cracks.SPREADS` contains 10 spreads; the 36-pair scan was evidently run ad hoc.
7. **Stale APIs in the falsification record**: `sweep.py` calls `mrlab.backtest(..., delay=1)`,
   a kwarg the committed `mrlab.backtest` no longer accepts — parts of the daily falsification
   record won't re-run as committed.
8. Minor: vol-matched leg weights are estimated on the **first ~¼ of the sample** and applied from
   bar 0 (mild in-sample contamination early); metric vol-rescaling to 10%/yr uses **full-sample**
   std (presentation-only lookahead in DD/eqR² figures); `broad_book.SECURITIES` has duplicate
   dict keys (EURSEK/EURNOK/EURGBP/NZDCAD/GBPCHF defined twice — harmless, last wins).

Verified as documented: detrending (cumsum of dX minus causal 28td rolling mean of dX) everywhere;
causal z (rolling mean/std exclude current bar via `shift(1)`); the vol gate (5d EWM σ > rolling
126d median); per-trade risk-frozen sizing (units = sign/σ_at_entry, no intra-trade resizing);
the {9,12,18,24}td × {2.5,3.0} ensemble; roll censoring in `hourly_lab` (day-boundary |diff| >
8×1.4826×MAD ⇒ bar contributes 0 to signal AND PnL, re-roll cost charged on held positions);
`validate.py` positive (synthetic OU, HL=10d) and negative (random walk) controls under both exec
modes; family risk caps (35%/30%); meta-gate at trailing-6m Sharpe > −0.5; Pine port decides on
confirmed bars and fills next open (`process_orders_on_close=false`, commission 0.01% + 1 tick).

---

## 2) Assumptions

**Execution**
- Bar-level fills, no book/impact model; capacity never modeled. Slippage = fixed ticks
  (`hourly_lab`) or 0.05×σ per unit (everywhere else; docstring equates this to ≈1 tick on
  liquid markets, stress 0.10 — stress only run in the daily lab).
- Same-close execution deemed legitimate for SYNC same-venue spreads (exchange-listed spread
  instruments fill at settle difference) — a defensible but unproven institutional assumption;
  silently extended to hourly Dukascopy CFD closes in the headline engines (§1.1).
- One position per spread, no pyramiding; exits are z-based + disaster stop |z|≥4 + time stop.
- FX treated as costlessly holdable (no swap/financing), CFD prices treated as tradeable.

**Data**
- Back-adjusted continuous daily contracts: levels meaningless (years of negative prices),
  $-changes are true PnL ⇒ everything built from leg $-changes; z-scores location-invariant.
- Yahoo hourly front-months are NOT adjusted ⇒ roll censoring required (8×MAD day-boundary rule);
  livestock/softs excluded for ≤6 bars/day and censor misfires.
- Dukascopy hourly CFDs (12.3y commodities, ~10y FX) accepted as clean continuous series, bars
  floored to the hour and intersected across legs (drops non-overlapping sessions; assumes
  synchronicity of the joint feed).
- Fixed economic weights (42 gal/bbl, crush ratios) preferred; otherwise vol-matched on the first
  quarter of the sample. No estimated/rolling betas anywhere (deliberate anti-overfit stance).
- A data-bug class was found and fixed (epoch-second timestamps silently dropped by the date
  parser, deleting the CL/HO crack history) — parser robustness is a standing assumption.

**Statistical**
- Robustness = positive in BOTH chronological halves (mrlab daily uses 60/40); no
  cross-validation, no block bootstrap, no multiplicity correction across the ~40-security ×
  multiple-study search (the both-halves + eqR² + G/P joint screen is the only guard).
- Smoothness = eqR² of the cumulative curve against a straight line + gain/pain (Calmar-like),
  argued (correctly) to be horizon-fair where absolute maxDD is not.
- Parameters frozen on an independent set (3.3y CL−BRN), chosen by best *worst-half* Sharpe, then
  applied unchanged; ensembles over windows/thresholds instead of per-spread fitting.
- Warm-up zero-return stretches trimmed before metrics (they fake smoothness).
- Apparatus credibility from synthetic controls (OU detected at Sharpe ~+0.9–1.4 under full
  costs; random walks ~0) — also establishing a **Sharpe ceiling ~1.2–1.4 even on a planted
  edge**, used to argue Sharpe-3 "no-downside" claims would indicate bugs.

---

## 3) Empirical conclusions, ranked by strength of evidence

1. **Daily-bar spread MR is dead post-~2008** (strongest). Converges from three independent
   samples (37y daily book IS 1.40 / OOS −0.51; 2y hourly where only intraday works; the
   cross-section) and ~9 falsified strategy families (§4), with a validated apparatus that
   detects planted OU edges under identical costs. Cracks specifically are ~random walks at the
   daily horizon (VR≈1, `diag.py`).
2. **Same-close fills on async-settle spreads fabricate alpha** (GO_crack: +2.9 same-close →
   −0.7 with one bar delay). Demonstrated directly; drove the SYNC/next-open architecture.
3. **The apparatus itself is sound** (positive/negative controls pass, both exec modes), so the
   daily kills are kills of the edge, not of the code.
4. **Hourly MR on tightly-cointegrated spreads is real, led by BRENT–WTI**: robust at 3.3y (TV,
   next-open, IS 0.78/OOS 1.28), at 2y (Yahoo panel, next-open, BZ_CL 0.78/1.18; 11/20 positive
   both halves), and at 12.3y (Dukascopy, Sh 1.93, eqR² 0.98 — but same-close fills, §1.1).
   Direction is highly credible; the 12y/4y magnitudes are upper bounds.
5. **The vol gate (trade only in dislocation regimes) is the single validated regime lever**
   (independent-set worst-half improvement +0.6→+2.5; book OOS 1.05→1.84; DD −12.5%).
6. **Diversification, not per-name quality, produces smoothness**: 37-spread no-selection book
   halves drawdown to −6.6%; and **trailing-Sharpe selection backfires** (−0.74) — strategy-level
   performance is not autocorrelated at 6-month horizon.
7. **The smooth-reverter population in commodities is structurally tiny** (~1 clean + 2–3
   borderline of 36 pairs at 12y) — not a data-length artifact (H15 refuted).
8. **Storage-anchored calendars were the historical smooth vehicle (1990–2009), not cracks** —
   an explicit redirection of the original thesis; that edge decayed with electronification.
9. **Weakest headline: the 10-security broadened book** (Sharpe 1.26, eqR² 0.99). Real signal
   likely present (each name positive both halves over 10–12y), but the reported magnitude rests
   on same-close fills, unstressed vol-proportional costs, ignored FX carry, the loosest smooth
   bar in the repo, and ex-post winner selection for the book (§1.1–1.5). Treat as optimistic.

---

## 4) Failures and why (the falsification record)

| Failure | Why it failed |
|---|---|
| Binary OU-gated daily z-MR (1/28 survive) | No daily edge to find: spreads ≈ random walks at daily horizon post-2000; OU half-life gate cannot manufacture reversion |
| Linear/ensemble sizing, lookbacks 10–250d, bands (mean OOS −0.3…−0.5) | Same null edge; extra parameters only added cost turns |
| Detrend-only fix (3/28) | Removes roll-carry bleed (a real artifact) but exposes no underlying signal |
| Seasonal day-of-year anchoring (daily cracks: 0 smooth of 11) | The seasonal is priced; deviation from seasonal norm has no net forecast power after costs |
| Patient tail-fade / fade-the-move k=5…252 (+ trend filters) | Negative at every horizon post-2000 — moves are information, not noise, at daily frequency in the modern regime |
| Pre-registered asymmetric floor fade (5/40) | The carry/run-cut floor logic is economically right but already enforced by arbitrageurs faster than daily bars |
| Cross-sectional family MR (all 7 families negative) | Demeaning removes family trend but the residual is not mean-reverting either |
| Trailing-Sharpe SELECTION of spreads (book −0.74 vs +0.68 unselected) | **Key negative result**: 6-month strategy Sharpe does not predict the next period; chasing recent winners systematically buys decayed dislocation episodes. Do not add "drop the laggards" overlays |
| Same-close evaluation of cross-venue spreads (GO_crack +2.9 → −0.7) | Fabricated alpha from asynchronous settles — an apparatus failure mode, caught and institutionalized |
| Epoch-timestamp parsing bug (CL/HO legs silently dropped) | Date parser assumed ISO strings; cracks lost 27–36y of history until fixed — motivated loader hardening |
| "More data smooths everything" (H15) | Refuted: 12y leaves diesel cracks/metals ratios choppy. The constraint is the number of genuinely cointegrated relationships, not sample length |
| Why the daily edge died at all | Structural: electronic/HFT/stat-arb capital post-2008 arbitraged 2–3-week settle-to-settle reversion; the residual edge migrated intraday (entries at intrabar extremes) |

---

## 5) Constraints / standards the orthogonal program must inherit

1. **Honest fills, uniformly.** Decide on closed bars, fill next bar open. Same-close only with an
   explicit synchronous-settle argument — and given §1.1, the new program should NOT copy the
   `pnl = u.shift(1)*dS` shortcut from `duka_cracks`/`broad_book`; use the open-based
   day/overnight decomposition of `hourly_lab.run`/`mrlab.backtest`.
2. **Cost realism + stressing.** Tick-based per-leg costs where ticks are known (1.5 ticks/leg/turn,
   report 2×–3× stress alongside, as `final.py` does); vol-proportional λ only as fallback with
   the 0.10 stress actually run. Model FX swap/carry if FX instruments are used.
3. **Apparatus controls before trusting any kill or any find** (the standing gate): positive
   control (synthetic planted edge detected under full costs/execution) and negative control
   (random walk scores ~0). Remember the control also calibrates the believable Sharpe ceiling
   (~1.2–1.4 on a clean planted OU) — results far above it are bug-priors.
4. **Roll censoring** on any non-back-adjusted intraday series (8×MAD day-boundary rule, censored
   bars contribute 0 to signal AND PnL, re-roll cost charged); never drop negative-price eras;
   build from $-changes with location-invariant signals.
5. **Regime honesty: assume post-2008 decay of daily settle-frequency MR.** Pre-2010 daily results
   are demonstrations, not forward claims. **MR is intraday-alive only** in this universe — new
   hypotheses should be tested at hourly (or finer) frequency, and 1990–2009 daily success must
   not be extrapolated.
6. **Anti-overfit discipline**: freeze rules on an independent set selected by *worst-half*
   performance; ensembles over windows/thresholds instead of per-name fits; fixed economic
   weights; pre-register screens ONCE (do not let the smoothness bar drift per study, §1.4);
   report failures explicitly; both-halves-positive + eqR² + gain/pain as the standard report.
7. **Book construction**: inverse-vol with family risk caps, no performance-chasing selection
   (validated negative result); if a screened subset is reported, also report the no-selection
   universe book as the honest baseline.
8. **Causality hygiene**: all rolling stats through t−1; trim warm-up zeros before metrics; no
   intra-trade resizing once risk is set at entry.

---

## 6) What was NOT explored — the orthogonal space

The prior program is one idea probed exhaustively: *fade a causal z-score of a detrended
cointegrated spread, conditioned on a single binary vol gate.* Everything below is untouched:

1. **Liquidity / exhaustion events.** Volume is loaded by the fetchers and never used anywhere.
   No event studies of volume spikes, range expansion/contraction, gap behavior, or
   capitulation/exhaustion bars as MR triggers. No scheduled-event anchoring (EIA/USDA/WASDE
   inventory releases, OPEC, FOMC for FX) despite trading inventory-driven spreads.
2. **Vol-state conditioning beyond one binary gate.** The 5d-vs-6m-median gate is the only regime
   model. Unexplored: vol-of-vol, vol term structure, conditioning *exit/sizing/thresholds* (not
   just entry) on vol state, multi-state regimes (HMM/clustering), gate hysteresis, and whether
   the gate's edge comes from vol level vs. vol *change*.
3. **Session / microstructure structure.** No hour-of-day, day-of-week, or session (Asia/EU/US,
   settle window, open auction) analysis — even though the core thesis ("the edge is intraday,
   at intrabar extremes") is microstructural. The overnight/day-session return decomposition was
   built in `xsec.py` loaders and never used as a signal. Dukascopy tick/bid-ask data exists
   upstream and was aggregated away.
4. **Leader–laggard / lead-lag.** All spreads are contemporaneous symmetric combinations. No
   cross-correlation or lead-lag studies (e.g., crude leading products intra-day, Brent vs WTI
   information flow, FX majors leading crosses), no asymmetric leg treatment, no using one leg's
   move to time the other's reversion.
5. **Regime-conditional nonlinearity.** The response is globally linear-in-z with fixed
   thresholds. Unexplored: convex/saturating response curves, interaction of z with vol state or
   trend state, asymmetric long/short thresholds at intraday frequency (only the daily
   pre-registered floor fade was tried), conditional holding periods, and ML/nonparametric
   conditional-expectation maps with honest validation.
6. **Other untouched directions** (smaller): time-varying OU estimation (Kalman/MLE) intraday;
   Hurst/half-life as continuous sizing input rather than binary gate; spread-of-spread or basket
   residual MR beyond pairwise; trade-level analytics (MFE/MAE, holding-time distributions) to
   shape exits; capacity/participation modeling.

These six axes are mutually orthogonal to the prior z-fade program and to each other, and items
1–5 map directly to the new program's mandate.
