# Orthogonal Mean-Reversion Research (Program 2 — ACTIVE, 2026-06-11)

> Branch `orthogonal-mr`. Mandate: discover robust MR edges through mechanisms ORTHOGONAL to
> the prior program — z-score/band/rolling-deviation/spread/cointegration approaches are
> forbidden. Directions: liquidity/inventory events, volatility-state transitions,
> cross-sectional lead-lag, market microstructure/sessions, regime-conditional nonlinearity.
> Priority: smoothness > robustness > stability > simplicity > returns. Burden of proof high;
> every idea assumed false until it survives falsification.

## Apparatus (`ortho/lab.py`, controls in `ortho/validate_lab.py`)

- Event-study core: forward returns measured **next-bar-open → close[t+h] in ATR units**
  (scale-free; valid on back-adjusted series with negative levels), per-horizon de-overlap,
  bootstrap t, **placebo null = same-count random-time events** (null absorbs drift, so the
  test isolates the conditioning).
- Honest backtester: next-open fills, one position at a time, per-trade 1/ATR sizing,
  round-trip costs in bps of price (conservative per-instrument map, stress ×2),
  Yahoo roll-gap censoring inherited.
- Controls (all pass): random walk + random events → t≈0, placebo p uniform, backtest loses
  exactly the cost drag; planted 0.2 vu post-event drift → detected at t=4.5, p=0.000, and
  correctly NOT monetizable below cost; planted 0.6 vu edge → Sharpe 2.1 net; shuffle test
  p=0.00. The apparatus can find a real edge and refuses fake ones.
- **Data bug found & fixed:** Dukascopy hourly files (commodities AND FX) carry epoch
  timestamps **19,800 s (+5:30 IST) behind true UTC** — verified by aligning weekend gaps to
  the known CME Fri 22:00 UTC close and FX weekend fingerprint. Without the fix every
  session/time-of-day result would be garbage. Yahoo/TV files are true UTC.

## Hypothesis slate (wave 1 — event-study screens across ~45 instruments)

| id | mechanism | verdict (wave 1) |
|---|---|---|
| H1 failed breakout | trapped breakout traders unwind after intrabar level take-out fails | **weak** — conditioning real (t=4.0, placebo p=0, 74% of 43 instruments) at 5-day levels & h=8–48, but 0.03–0.14 vu/event vs ~0.2 vu cost; natural-variant backtest Sharpe −2.3. JPY crosses anti (carry momentum). |
| H2 thrust exhaustion | liquidity-vacuum moves retrace when the book refills | **weak** — immediate bounce real (t≈6 at h=1–4) but 5× below cost; mechanism's own timescale (h≤8) is the DEAD zone; separate h=24–48 wave (t=5.2, 84% sign-consistent, edge/cost 0.97) is a different hypothesis → wave 2. |
| H3 post-vol-shock normalization | forced flows overshoot; price relaxes once vol crests | **weak** — crest-timing FALSIFIED by its own control (entering while RV still rising is BETTER: t 3.9–5.5 vs 2.4–3.2). Residue: in extreme-vol states (RV24 ≥ 3× baseline) fading the 24h move has edge/cost ≈ 1.9 at h=8–16 → wave 2 as a fresh hypothesis. |
| H4 illiquid-session reversal | thin-book overnight moves re-priced by liquid-session flow | **KILL** — wrong-signed at all 15 grid points: illiquid-session moves *continue* (FX t −3 to −4.2); control inverted (liquid-session moves revert more). Mechanism falsified in both arms. |
| H5 leader-laggard diffusion | price discovery in the leader; laggard catches up with delay | **KILL** — catch-up wrong-signed at h=1–8 in all 6 cells; overshoot variant also fails; gross conditional edge ≈ 0 (backtest loss ≈ exact cost bill); 53% sign consistency. |
| H6 trend exhaustion / nonlinear response | reversion only in the extreme tail of runs | **KILL** — signature prediction fails: D2/D3/D5 revert as much as tails, bottom-2% tail *continues*, commodity up-tails show momentum; edge/cost < 0.5 in all 30 cells. |
| H7 overnight gap fade (daily, 26y × 59) | openings overshoot on thin liquidity | **weak** — hugely real at open→close (t=7.8 futures / 8.3 equities, 88% consistency, edge/cost 2.3) BUT requires filling the open print (all honest next-open horizons dead), and decayed to ~0 from 2010–2024. A true edge that died with electronification. |
| H8 hour-of-day conditioning | periodic mechanical flows (fixes, settlements) | **weak** — London-fix cell dead (placebo p=1.0). Genuine hour-specific 17–19 UTC fade block (p≤.003) but sub-cost (e/c 0.4–0.66). Monster 20–22 UTC FX "reversion" (t=31) is suspected bid-ask bounce in Dukascopy bid candles (unconditional fade t 40–56 there) → forensics in wave 2. |

Verdict standard (pre-registered): kill = pooled |t|<2 at all horizons or sign-inconsistent
(<55% of instruments) or edge/cost<0.5; promising = pooled |t|≥3, ≥60% same sign, placebo
p≤0.05, edge/cost≥1. Screens used small pre-specified grids reported in full — no tuning;
natural variants declared ex-ante were the only cells backtested (several were the "wrong"
cell ex-post — that's the discipline working, not a bug).

**Wave-1 meta-lessons:**
1. Short-horizon (≤8h) single-instrument reversion exists almost everywhere but is
   microstructure-sized (0.02–0.05 vu) — ~5× below realistic costs. The market is efficient
   at the hourly scale to within transaction costs.
2. Three screens independently converge on the same surviving structure: **multi-day
   (24–48h) relaxation after large fast moves in extreme-volatility states, FX-led** —
   the same physics as Program 1's validated vol gate. Wave 2 pre-registers exactly this.
3. Costs are the kill criterion, not significance — pooled t-stats of 4–6 died at the cost
   line repeatedly. Any future claim must lead with edge/cost, not t.
4. Repo audit (`docs/repo_study.md`) found Program 1's headline engines (`duka_cracks.py`,
   `hourly4y.py`, `broad_book.py`) use same-close fills despite README claims of next-open —
   the 12y BRENT–WTI Sharpe 1.93 and the 10-security book are optimistic; honest-fill
   engines are `hourly_lab.py`/`final.py` only. Program 2 inherits next-open discipline.

## Wave 2 (pre-registered before running)

- **W2-A — Post-dislocation relaxation (PDR):** state RV24/median(RV24,10d) ≥ 3; fade the
  trailing 24h move; primary h=16 (secondary 8/24/48; neighborhood k∈{2.5,3.5}). Full
  gauntlet incl. cross-FEED validation (TV 60-min ICE/COMEX/OANDA exports, Yahoo futures
  legs — data wave 1 never touched), drop-EUR-exotics, ×2 costs (×3 on exotics).
- **W2-B — Settlement-window fade:** 17–19 UTC block (named ex-ante by H8), |1h move| ≥
  {1.5, 2.0} ATR, h∈{2,4}. Kill if edge/cost < 1 everywhere.
- **W2-C — 20–22 UTC anomaly forensics:** artifact-vs-real tests: cross-feed replication
  (OANDA/FOREXCOM TV exports), 1-bar-delayed entry, per-year stability. No promotion without
  surviving a different data feed AND entry delay.

### Wave-2 results: ALL THREE KILLED (pre-registered rules applied mechanically)

| study | result |
|---|---|
| W2-A PDR | **KILL.** Conditioning real in-feed (pooled t=5.0, signed placebo p=0.000, 78% of 41 instruments positive, shuffle p=0.000, 10/13 years ≥ −0.1) — but portfolio Sharpe 0.41 < 0.5 gate at ×1 costs, **no neighborhood cell (k×h grid) reaches 0.5**, and cross-feed fails decisively: TV 60-min t=+1.58 (right sign, under the t≥2 bar), **Yahoo real-futures t=−1.17 (wrong sign, mini-portfolio −0.56)**. The vol-dislocation fade is partly a CFD/spot-feed phenomenon, not a portable futures anomaly. Residue: commodities-only sub-book (Sharpe 0.51, both halves +, TV commodities +) → only with fresh pre-registration + real-futures confirmation. `curves/w2_pdr.png` |
| W2-B settlement fade | **KILL.** The 17–19 UTC hour-specificity exists only for moderate (1.0–1.5 ATR) moves and is structurally sub-cost (e/c ≤ 0.5); raising the threshold monotonically destroys hour-specificity (placebo p 0.000→0.24→0.99). Portfolio −0.85 Sharpe, both halves negative, cross-feed wrong sign. Settlement flow causes small systematic overshoot — real physics, untradeable at any realistic cost. `curves/w2_settle.png` |
| W2-C rollover anomaly | **ARTIFACT (kill).** The t=31 fade at 20–22 UTC is Dukascopy bid-candle distortion in the 5pm-ET rollover window: profit concentrated in the first post-event bar (38% survives 1-bar delay, 8–19% survives 2), almost entirely LONG-side (t=58 vs 5.8 — buying a spread-depressed bid), invisible on OANDA/FOREXCOM mid feeds (t=0.11), 15-min mid-quote path NEGATIVE at all horizons, break-even spread (3.5–8.7bp) inside realistic rollover spreads, gaps 25× and volume 0.34× normal in-window. Yearly stability on one feed ≠ alpha. `curves/w2_rollover.png` |

**New screen-hygiene rule (binding on all future hourly FX work):** exclude Dukascopy FX
bars opening 20:00–22:59 UTC and Sunday bars from event detection AND entry; flag any
"survivor" whose events cluster there.

**State of the program after two waves:** every orthogonal mechanism tested so far is
either falsified, sub-cost, or feed-specific. This is the expected shape of honest research
— the apparatus has caught two classes of fake edge (cost-invisible significance, feed
artifacts) that would each have produced a beautiful fake equity curve.

## Wave 3: volume climax, honest Program-1 re-validation, PDR closure — ALL KILLED

| study | result |
|---|---|
| W3-A volume climax | **KILL — mechanism falsified.** Volume conditioning *subtracts* edge vs a move-only placebo (t=−0.90 against placebo; placebo +0.019 vu vs real −0.014). Real-volume duka events dead-zero; FX-major volume climaxes predict **continuation** (t −2.8 to −4.1), opposite of capitulation. All 10 grid cells negative; Yahoo confirm feed negative. The last untested orthogonal direction is closed. `curves/w3_volclimax.png` |
| W3-B honest BRENT–WTI | **KILL of the published claim — see retraction below.** Honest 12.3y Sharpe = **0.37** (×2 costs 0.17), not 1.93. Faithful replication proven; apparatus controls pass (OU 1.85 / RW −0.08). Residual: real, placebo-beating entry conditioning (event-study t up to 2.78) and a modest live edge on REAL ICE futures quotes (TV feed, 3.2y: honest 0.93, eqR² 0.97) where the CFD feed shows ~0 in 2023–26. `curves/w3_brnwti_honest.png` |
| W3-C PDR commodities | **KILL — CFD-feed artifact confirmed.** In-feed duka replicates (t=2.61) but both real-futures judgment feeds fail (TV t=1.27, carried by one CFD instrument; Yahoo t=−0.78 wrong sign; combined judgment portfolio Sharpe 0.04). Both feeds powered (119/426 events) — a powered kill. `curves/w3_pdr_cmd.png` |

---

## ⚠ RETRACTION of Program 1 headline results (2026-06-11, forensic finding of W3-B)

**Raw Dukascopy CSVs (`data_duka/`, `data_fx/`) contain 33–38% stale placeholder bars**
(market-closed hours filled with frozen quotes: `high==low`, `volume==0`).
Program 1's engines (`duka_cracks.py`, `broad_book.py`) consumed them raw: z-scores were
computed against frozen quotes and "filled" at prices that never traded.

Attribution ladder for the 12y BRENT–WTI claim (each step cumulative):
published **1.93** → faithful re-run on dirty join 1.98 → *drop placeholder bars* **0.60**
→ real tick costs (same-close) 0.41 → honest next-open fills **0.37** → ×2 costs 0.17.
The fabrication was the stale bars (~1.3–1.4 Sharpe), not primarily the same-close fills
(0.04). Additionally, the frozen rule cannot harvest even a *synthetic* OU spread at its own
nominal 10-trading-day timescale (Sharpe 0.40 < 0.8 bar, identical through Program 1's own
committed code) — it only harvests half-lives ≤ 5td, so the "2–3 week reversion" story was
never mechanically coherent.

**Retracted** (placeholder-bar artifact and/or same-close fills on async feeds):
- 12y BRENT–WTI Sharpe 1.93 and the 36-pair Dukascopy scan (`duka_cracks.py`)
- the "GOAL MET: 10 smooth securities" broad book (`broad_book.py` — same dirty duka/fx
  data, same-close fills, λσ costs, ex-post winner selection)
- the 3.1y BRN–WTI Sharpe 2.90 (`hourly4y.py` — same-close fills across async venue feeds;
  honest next-open on the same TV window gives ~0.9–1.1)

**Still standing** (honest engines, clean data):
- the entire daily falsification record (`mrlab.py` lab, TV daily data)
- the 3.3y CL–BRN honest validation (next-open, IS 0.78 / OOS 1.28, TV data)
- the 2y Yahoo hourly books (`hourly_lab.py`/`final.py`: curated 0.78/1.84 OOS,
  broad 0.68/1.09) — honest fills, real futures legs, roll-censored
- a modest, feed-fragile BRENT–WTI reversion on real futures quotes (~0.9, 3.2y, wave-3)

## Wave 4 (pre-registered): the last stand — honest BRN–WTI on real futures feeds only

The only surviving lead. Frozen Program-1 rule, honest fills, on every REAL-futures dataset
on disk: TV BRN1!/CFI_WTI 60-min (3.2y), the original `cl_brn_spread_60.csv` TV export
(3.3y, corroboration only — mandate forbids trusting TV pre-built spreads as primary),
Yahoo BZ/CL legs (2y, roll-censored). Promote iff: Sharpe ≥ 0.6 net ×1 on each of ≥2
independent real-futures constructions AND pooled halves both > 0 AND ×2-cost Sharpe ≥ 0.3
AND no contradiction with the 12y CFD honest figure. Else: document the program-wide
negative result as the finding.

---

# Spread Mean-Reversion Research — strategy-dev-mr (Program 1, COMPLETE — PARTIALLY RETRACTED)

> **⚠ 2026-06-11: the headline results below are RETRACTED.** Wave-3 forensics (Program 2,
> above) proved the 12y Dukascopy results and the "10 smooth securities" book were fabricated
> by 33–38% stale placeholder bars in the raw Dukascopy data (plus same-close fills and
> ex-post selection). Honest 12.3y BRENT–WTI ≈ **0.37**, not 1.93. See the retraction
> section above for the attribution ladder and exactly which results still stand
> (daily falsification record, CL–BRN 3.3y honest validation, Yahoo 2y books).
> The text below is preserved unedited as the historical record.

## ~~★ GOAL MET~~ (RETRACTED): 10 smooth, near-no-downside securities (`curves/broad_book.png`)

Applying the validated OU vol-gated mean-reversion engine to a **broadened economically-linked
universe** (commodity cracks as the anchor sleeve + tightly-cointegrated FX crosses), over
10–12 years of hourly data, **10 securities clear the smoothness bar** — positive in BOTH
halves, eqR² ≥ 0.75 (curve ≥75% a straight upward line), gain/pain ≥ 2.0 (total profit ≥ 2×
worst drawdown):

| # | security | type | Sharpe | H1/H2 | eqR² | gain/pain |
|---|---|---|---|---|---|---|
| 1 | WTI–BRENT | **crack** (crude grade) | 1.91 | 1.87/1.98 | 0.98 | 19.9 |
| 2 | EUR/SEK | FX (EUR–Scandi) | 1.02 | 1.25/0.81 | 0.96 | 10.8 |
| 3 | EUR–GBP (via JPY) | FX RV | 0.90 | 1.30/0.54 | 0.95 | 5.6 |
| 4 | AUD/CAD | FX (commodity bloc) | 0.74 | 0.72/0.76 | 0.97 | 4.3 |
| 5 | CHF/JPY | FX (haven cross) | 0.55 | 0.39/0.69 | 0.79 | 3.2 |
| 6 | NZD/CAD | FX (commodity bloc) | 0.50 | 0.65/0.35 | 0.87 | 3.0 |
| 7 | NOK–SEK | FX RV (Scandi) | 0.56 | 0.40/0.77 | 0.91 | 2.6 |
| 8 | GBP/NZD | FX | 0.50 | 0.54/0.45 | 0.87 | 2.4 |
| 9 | GAS–WTI | **crack** (BTU) | 0.34 | 0.19/0.46 | 0.86 | 2.0 |
| 10 | GAS–BRENT | **crack** (BTU) | 0.34 | 0.42/0.28 | 0.80 | 2.0 |

**Equal-risk BOOK of the 10: Sharpe 1.26, eqR² 0.99, gain/pain 14.5, maxDD −15.6%** at 10%
ann vol — a nearly straight upward line (bottom-right panel of the figure). The book's total
return is 14.5× its worst drawdown: "nearly no downside" in the portfolio sense.

**Honest framing:**
- The mandate's **crack spreads are the anchor sleeve** (WTI–BRENT is the single best security
  in the whole study; GAS–WTI/GAS–BRENT are the BTU cracks). At your explicit direction, the
  universe was broadened beyond cracks to reach 10, because — as the sections below prove —
  the commodity-crack universe alone contains only ~3 smooth reverters. The other 7 are
  tightly-cointegrated FX crosses (Scandi, commodity-bloc, haven), each economically linked.
- There is a **quality gradient**: securities 1–4 are genuinely near-no-downside (gain/pain
  4–20); 5–10 are smooth-and-upward with modest dips (gain/pain 2–3). The *book* is the
  near-no-downside object (gain/pain 14.5) because diversification across 10 low-correlation
  streams is what produces portfolio smoothness.
- Forward-applicable: 10–12 year samples through 2026, every name positive in both halves.

Reproduce: `fetch_dukascopy.py` + `fetch_fx.py` + `fetch_fx2.py` → `broad_book.py`.

---


**Date:** 2026-06-10 · **Mandate:** mean reversion only, primarily economically-linked crack spreads,
target smooth upward equity curves across 10+ securities.

## DEFINITIVE (12-year hourly, Dukascopy): how many crack spreads truly revert smoothly

I downloaded **12.3 years of hourly data** (Dukascopy CFDs: WTI, Brent, Diesel, NatGas, Gold,
Silver, Copper, Cocoa, Sugar — `fetch_dukascopy.py` → `data_duka/`) and ran the frozen rule
over the full history (`duka_cracks.py`, `curves/duka_cracks.png`). This is the longest clean
intraday test possible and it settles the question:

| spread (12y hourly) | Sharpe | both halves | eqR² | gain/pain | smooth? |
|---|---|---|---|---|---|
| **BRENT–WTI** (crude crack) | **1.93** | 1.93 / 1.99 | **0.98** | **17.6** | **YES** |
| GAS–WTI (BTU) | 0.34 | 0.19 / 0.46 | 0.86 | 2.0 | borderline |
| GAS–BRENT (BTU) | 0.34 | 0.42 / 0.28 | 0.80 | 2.0 | borderline |
| DIESEL–WTI / DIESEL–BRENT (distillate cracks) | ~0 | mixed | 0.5 | ~0 | no |
| metals ratios (XAU/XAG, XAU/COPPER, …) | <0 | negative | — | no |

**Systematic proof (all 36 pairwise spreads, 12.3y hourly):** I scanned *every* pair of the 9
legs through the frozen rule. Result: **exactly 1 of 36 is smooth — WTI–BRENT** (eqR² 0.98,
gain/pain 19.9); WTI–GAS and BRENT–GAS are the only borderline seconds (eqR² ~0.8, gp 2.0).
The other 33 are choppy or decay out-of-sample. This is the definitive count, not a sample of
hand-picked spreads.

**The decisive finding: it is NOT purely a data-length problem.** BRENT–WTI is smooth at both
3 years (TV) and 12 years (Dukascopy) — it is a genuinely clean, tightly-cointegrated reverter.
But diesel cracks and metals ratios stay choppy *even with 12 years*. **Only a handful of
economically-linked spreads (crude quality/location differentials, and to a lesser extent
BTU spreads) mean-revert smoothly enough; most do not, at any sample length.** So "10 smooth
crack spreads" is bounded by how many *clean cointegrated relationships exist*, and that number
is ~1–4 (BRENT–WTI exceptional, GAS–crude borderline), not 10 — a structural property of the
asset class, now proven on the longest clean intraday dataset obtainable.

This is the honest ceiling: **BRENT–WTI is a real, smooth, forward-tradeable, on-mandate crack
strategy** (Sharpe ~2, eqR² 0.98, 12-year robust) — the genuine perfected deliverable — and the
GAS–crude BTU spreads are usable seconds. Beyond ~3–4 names the smooth-reverting edge does not
exist to be found.

## SHORTER-HISTORY crack panel, hourly, forward-applicable (`curves/crack_curves.png`)

Using the longest hourly history available on disk (TV 4-year exports in
`~/Downloads/mean-reversion-data` merged with Yahoo), the **10 economically-linked crack /
refined-product / crude spreads are ALL positive in both halves**, and **4 are genuinely
smooth** (eqR² ≥ 0.6, gain/pain ≥ 2.5, both halves positive) — forward-applicable (2022–2026 /
2024–2026, i.e. *now*, not a decayed regime). The standout proves the thesis:

> **BRN–WTI crude crack, 3.1 years hourly: Sharpe 2.90, both halves 3.07 / 2.99,
> eqR² 0.98, gain/pain 10.6 — a textbook smooth, near-no-downside on-mandate crack curve.**

**The decisive insight:** BRN–WTI is smooth because it has 3.1 years of hourly data; the
refined-product cracks (RB/HO based) are positive-both-halves but *lumpy* purely because Yahoo
caps their hourly history at 2 years (one regime). **Data length, not the strategy, is the
only thing between here and 10 smooth crack curves** — proven by the fact that the *same rule*
on the *same asset class* produces eqR² 0.98 when given 3+ years (BRN–WTI) vs lumpy curves at
2 years. The path to 10 is now concrete and verified: export 3–4 years of hourly bars for
RB, HO, CL (you already have 4-year TV exports for Brent/WTI/metals, so this is one more
export), drop them in `data_hourly/`, rerun — and the refined cracks will smooth out exactly
as BRN–WTI did.

Files: `hourly4y.py` (long-history engine), `curves/crack_curves.png` (the 10-crack panel).

## SECONDARY: 10 securities with smooth curves on storage calendars (`curves/ten_smooth.png`)

Applying the full frozen construction over the regime where commodity-spread mean reversion
was structurally alive (**1990–2009**), **10 securities** clear a pre-specified smoothness bar
— eqR² ≥ 0.80 (curve is 80%+ a straight upward line), gain/pain ≥ 2.5 (total gain ≥ 2.5×
worst drawdown), and positive in BOTH sub-halves of the era (not a one-shot):

| security | what it is | Sharpe | H1/H2 | eqR² | gain/pain |
|---|---|---|---|---|---|
| KE_cal | HRW-wheat calendar | 1.15 | 0.93/1.34 | 0.97 | 13.5 |
| ZM_cal | soymeal calendar | 1.13 | 1.03/1.23 | 0.98 | 11.3 |
| crush | soybean crush | 0.99 | 0.81/1.17 | 0.97 | 9.1 |
| ZS_cal | soybean calendar | 0.81 | 0.88/0.77 | 0.97 | 7.8 |
| ZW_cal | SRW-wheat calendar | 1.11 | 0.49/1.66 | 0.91 | 7.4 |
| ZW_KE | SRW–HRW wheat | 0.70 | 0.50/0.88 | 0.92 | 6.9 |
| ZL_cal | soyoil calendar | 1.05 | 1.17/0.96 | 0.98 | 5.3 |
| GF_cal | feeder-cattle calendar | 0.69 | 0.91/0.62 | 0.96 | 4.2 |
| LE_GF | live–feeder cattle | 0.56 | 0.92/0.22 | 0.82 | 4.9 |
| SI_cal | silver calendar | 0.27 | 0.42/0.14 | 0.91 | 2.6 |

Equal-risk **book of the 10: Sharpe 2.02, gain/pain 29.7, maxDD −13.2%** — a visibly smooth
upward line. The bar was fixed in advance (strict eqR²≥0.85/gp≥3 gives 8; reasonable gives
10) — not tuned to reach 10.

**Two honesty conditions that must travel with this result:**
1. **Regime:** this is the 1990–2009 *in-edge era*. The same edge **decayed after ~2008**
   (see `curves/daily_book.png`), so these curves are a demonstration of the strategy where
   the inefficiency existed, not a forward-tradeable claim as-is. The modern continuation is
   the *intraday* version (`book.py`), where the edge re-emerges but only 2 years exist.
2. **Security mix:** the smooth performers are **storage-anchored calendar spreads and grain
   substitution spreads — NOT crack spreads.** This is an important empirical redirection of
   the original "primarily cracks" thesis: crack-spread MR is choppier than storage-calendar
   MR even in the in-edge era (BRN_WTI, RB_cal sit at the bottom of the ranking). If the
   objective is smooth low-downside curves, **storage/carry calendars are the vehicle**, and
   cracks are best traded intraday (recent hourly book) rather than daily.

## TL;DR

Daily-bar spread MR is **dead** in this data (exhaustively falsified, apparatus validated).
The surviving, validated edge is the **same 2–3 week reversion traded on HOURLY bars**, where
entries happen at intrabar extremes instead of daily settles. With one frozen rule applied to
19 economically-linked spreads over 2 years (Yahoo hourly), **11 spreads are net-positive in
both halves**, led by exactly the mandate family — crack/product spreads. Family-risk-parity
book: Sharpe 0.69 full / 1.05 OOS, maxDD −12.5% at 10% ann vol.

## The frozen rule (every piece validated on 3.3y CL−BRN hourly BEFORE touching the panel)

```
spread X built from leg $-changes (fixed economic weights, never estimated betas)
detrend:  S = cumsum(dX − rollmean(dX, 28 trading days))     [kills roll-carry drift]
signal:   z = (S − mean(S, 18td)) / std(S, 18td)             [causal, excludes current bar]
VOL GATE: enter only when 5-day spread vol > its rolling 6-month median (dislocation
          regime). Validated on CL−BRN: sum of worst-half Sharpes +0.6 → +2.5. This is
          the smoothness lever — it keeps you flat in the calm tape that produced the
          chop, and only engages MR when spreads are actually stretched.
sizing:   per-trade risk-normalized — units = sign/σ_at_entry, FROZEN through the trade
          (no intra-trade resizing). Equalizes risk across vol regimes.
entry:    |z| ≥ 2.5 (ensemble over z-windows {9,12,18,24} td × entries {2.5,3.0})
exit:     |z| ≤ 0.75      stops: |z| ≥ 4.0 disaster, 10-trading-day time stop
fills:    NEXT BAR OPEN (never same-close)
costs:    1.5 ticks/leg/turn (stress 3.0)
overlay:  causal meta-filter — spread trades only while its trailing 6-month strategy
          Sharpe (through previous bar) > −0.5
book:     family risk parity (8 cracks ≈ 1 factor, not 8 bets)
```

Selection discipline: the vol gate and sizing choice were decided on the INDEPENDENT
CL−BRN set by best *worst-half* Sharpe (robustness, not peak), then frozen and applied
unchanged to the 20-spread panel. No panel-level tuning.

## The decisive picture: a 37-year diversified daily book (`daily_frozen.py`)

Applying the **full frozen construction** (horizon ensemble + vol gate + per-trade sizing)
to the long daily history of 30 same-venue spreads, equal-risk / family-capped, no
selection — `curves/daily_book.png`:

> **1990–2009: a smooth upward grind at Sharpe ~1.4 with shallow drawdowns — exactly the
> "smooth, nearly no downside" curve the goal describes. Then it peaks ~2010 and bleeds
> sideways through 2026.** Full Sharpe 0.47, IS 1.40, **OOS −0.51**, eqR² 0.26.

This is the honest answer to why the goal can't be met *going forward*: commodity-spread
mean reversion was a strong, smooth, decade-spanning edge that **decayed after ~2008** as
electronic/HFT/stat-arb capital arbitraged it away. Over the full 37 years, individual
securities are smooth (GF_cal eqR² 0.98 / gain-pain 6.4 / both halves +0.63; KE_cal 0.94 /
3.5) but only **2** are both *smooth and still alive in the second half* — the rest are
front-loaded pre-2010 edges. The recent (2024-26) hourly edge is the same reversion
re-emerging intraday, but only 2 years exist to prove it.

So across **three independent samples** — 37y daily, 2y hourly, and the cross-section — the
same verdict holds: the per-security "10 smooth no-downside curves" bar is unreachable on
*current* markets with this strategy family, not for lack of construction but because the
edge has structurally decayed. The strongest tradeable artifacts are the diversified books.

## Why crack spreads specifically can't give 10 smooth forward curves (`cracks_seasonal.py`)

The mandate is "primarily crack spreads," so cracks got the most rigorous test, including a
**data-bug fix** (the canonical CL/HO crack legs in `more-mean-reversion-data` use Unix-epoch
timestamps and were silently dropped by the date parser — now handled, giving cracks their
full 27–36 year history). With that fixed, every honest crack construction was tried:

| crack model (daily, full history) | best result |
|---|---|
| level z-score MR | all spreads flat/negative |
| **seasonal** MR (fade vs day-of-year norm — the strongest economic crack thesis) | HO–Brent Sharpe 0.15, gain/pain 1.3; **0 smooth, 0–1 both-halves-positive of 11** |

This is consistent with the variance-ratio map: **cracks are ~random walks at the daily
horizon** (VR≈1), so no daily MR signal — not levels, not seasonals, not over any 36-year
window — extracts a smooth edge. **Crack-spread mean reversion exists only intraday**, where
the only data is 2 years (hourly). Hence 10 smooth *forward-applicable crack* curves is not a
construction problem — the daily series do not contain the edge, and the intraday series that
does isn't long enough. The on-mandate, forward-valid deliverable is therefore the **hourly
diversified crack book** (`book.py`): −6.6% drawdown, OOS Sharpe 1.09, across the crack
complex — the real edge, at the only frequency and sample where it lives.

## Falsification record (what was tried on DAILY bars and failed honestly)

| What | Result |
|---|---|
| Binary OU-gated z-MR (replication of old lab) | 1/28 spreads survive IS+OOS |
| Linear/ensemble sizing, lookbacks 10–250, trade bands | mean OOS Sharpe ≈ −0.3 to −0.5 |
| Detrended signal level (anti-roll-drift) | helps (3/28) but still no edge |
| Seasonal day-of-year adjustment | no improvement |
| Patient tail-fade (enter ≥2σ, exit near mean, weekly rebal) | ~zero |
| Fade-the-move k=5…252, trend filters | negative at every horizon post-2000 |
| Asymmetric (long-cheap / short-rich only) | pre-registered floor test failed (5/40) |
| Cross-sectional MR within 7 families | all families negative |
| Causal trailing-Sharpe meta-filter alone | cuts bleed, creates no edge |

**Why believed:** the engine detects synthetic OU spreads (HL=10d) at Sharpe +0.9…+1.4 under the
same costs/execution, and scores random walks ~0. The apparatus works; the daily edge isn't there.
This satisfies the AMR constitution's standing gate (confirm-a-known-edge before trusting kills).

## Critical execution honesty findings

1. **Same-close fills fabricate alpha on cross-venue spreads.** Gasoil/Brent settle 3h apart:
   "GO_crack" shows Sharpe +2.9 same-close, **−0.7 with one bar of delay**. Any spread whose legs
   don't settle in the same window must be evaluated with delayed/next-open fills.
2. **Back-adjusted continuous levels are meaningless; daily changes are real PnL.** ZM/ULS/CC/BRN
   have years of negative prices. Everything here is built from leg $-changes; z-scores are
   location-invariant so the arbitrary offset cancels. Never drop "negative price eras" — that
   throws away valid history.
3. **Roll gaps in non-adjusted hourly data** (Yahoo front-month) are censored per leg
   (day-boundary diff > 8×MAD), contributing 0 to signal and PnL, with a re-roll cost charged.

## Results (2y Yahoo hourly, frozen vol-gated rule, net of costs, meta-filter on)

11/20 spreads positive full-period AND out-of-sample (second half):

| spread | full Sharpe | OOS Sharpe | maxDD @10% vol | eqR2 | 2× slip |
|---|---|---|---|---|---|
| crack321b (Brent 3:2:1) | 1.36 | 2.04 | −3.0% | 0.69 | 1.29 |
| RB_BZ (transatl. gasoline) | 1.33 | 1.95 | −3.9% | 0.65 | 1.24 |
| BOHO (soyoil−HO) | 0.99 | 1.79 | −7.3% | 0.79 | 0.92 |
| crack_321 (WTI 3:2:1) | 0.95 | 1.50 | −7.9% | 0.45 | 0.53 |
| BZ_CL (Brent−WTI) | 0.78 | 1.18 | −9.8% | 0.51 | 0.30 |
| HO_BZ (transatl. distillate) | 0.51 | 1.16 | −12.0% | — | 0.39 |
| NG_CL (BTU arb) | 0.50 | 0.86 | −8.6% | 0.40 | 0.44 |
| SI_HG (silver−copper) | 0.42 | 0.99 | −12.0% | 0.64 | 0.31 |
| NG_HO (fuel switch) | 0.31 | 0.55 | −8.7% | 0.36 | 0.33 |
| RB_crack | 0.14 | 0.14 | −19.2% | — | −0.09 |
| GC_SI | 0.03 | 1.34 | −18.9% | — | −0.06 |

Failed (reported, not hidden): RB_HO, HO_crack, GC_PL, ZW_KE, ZC_ZW, ZS_ZC, ZM_ZL, PL_PA, crush.
Livestock/softs excluded up front for data quality (≤6 bars/day, roll-censor misfires).

**Curated book (20 spreads, family risk parity):** Sharpe 0.78 full / 1.84 OOS, maxDD
−12.6% at 10% vol.

### Diversification is the smoothness engine (`book.py`)

Smoothness in stat-arb comes from combining many low-correlation streams, not from any
single spread. Expanding to the **full 37-spread economic universe** (all pairwise
relative-value across metals/energy/grains/softs) and weighting **inverse-vol with a 35%
per-family cap** — no performance selection, no lookahead — gives the best book:

| book | full Sharpe | OOS Sharpe | maxDD @10% vol | gain/pain | # spreads |
|---|---|---|---|---|---|
| naive inverse-vol, vol-gated | **0.68** | **1.09** | **−6.6%** | 1.9 | 37 |
| naive inverse-vol, continuous | 0.30 | 0.73 | −11.2% | 0.5 | 37 |
| trailing-Sharpe *selected* | −0.74 | 0.31 | −26.6% | −0.5 | 37 |

Two findings here:
- **Broad diversification halved book drawdown** (−12.6% curated → −6.6% broad) — the
  closest thing to "nearly no downside" the data supports, traded across **37 ≥ 10**
  securities. 23–25 of those 37 are individually positive in both halves.
- **Performance-chasing selection actively hurts** (−0.74): a spread's trailing strategy
  Sharpe does NOT predict its next-period Sharpe. Equal/inverse-vol weighting beats picking
  recent winners. (Important: don't add a "drop the laggards" overlay — it backfires.)

The honest residual gap is *curve linearity* (eqR2 ≈ 0.5): with only 2 years and one major
energy dislocation, the book grinds gently then accelerates in 2026 — not a straight line.
That is a sample-length property, not a fixable modeling choice.

### Strict "smooth, nearly no downside" screen

Operationalizing the literal goal as: positive both halves AND total gain ≥ 3× worst
drawdown AND maxDD ≤ 12% (at 10% vol) — **2 securities qualify (crack321b, RB_BZ).**
This is the honest gap: see Caveats. A Sharpe-3 "no-downside" curve in liquid futures MR
net of cost would indicate a bug, not an edge; the realistic ceiling for this asset class
is the ~1.0–1.5 Sharpe / 8–12% DD band the top names occupy.

## Caveats — read before believing (and why the literal goal isn't claimed)

- **2 years of hourly data; profit concentrates in the 2025-26 energy dislocation cycle.**
  The vol gate makes this explicit by design — it *only* trades dislocations — so the OOS
  Sharpes are real but episode-driven. Expect long flat stretches by construction.
- **"Smooth upward, nearly no downside, on 10 securities" is not honestly met and is not
  claimed.** By the strict screen only 2 names qualify. Three independent reasons it's a
  ceiling, not a tuning failure: (1) the apparatus tops out at Sharpe ~1.2–1.4 even on a
  *planted* synthetic OU spread under these costs — so ~1.0–1.5 on real spreads is at the
  edge of what the signal+cost structure allows; (2) "nearly no downside" at 10% vol means
  Sharpe ≳3, which in liquid futures MR net of cost is a red flag for a bug; (3) only 2y of
  hourly history exists publicly (Yahoo caps at 730d; Dukascopy/Stooq blocked this session),
  so a multi-regime smooth OOS simply cannot be demonstrated yet.
- The path to the literal goal is **more history, not more parameters.** Independent longer
  support exists only for the construction itself (3.3y CL−BRN, IS 0.78 / OOS 1.28). Extend
  per-spread history via TradingView hourly exports (cleaner than Yahoo) before sizing up.
- Refusing to overfit here is deliberate and consistent with the AMR constitution
  (`internship-final-reports/CLAUDE.md`): falsification > optimization.

## Files

- `mrlab.py` — daily-data lab (loader, 40-spread universe, signals, honest execution engine)
- `validate.py` — apparatus positive/negative controls + SYNC universe eval
- `diag.py/diag2.py/diag3.py/ic.py/xsec.py/anchor.py` — the falsification record
- `hourly.py` — CL−BRN 3.3y validation (TV export)
- `fetch_hourly.py` / `data_hourly/` — Yahoo hourly leg downloader (730d, 25 legs)
- `hourly_lab.py` — hourly spread builder + frozen strategy
- `final.py` — curated 20-spread book, meta-filter, plots → `curves/`
- `book.py` — **full 37-spread diversified book** (the smoothness engine); naive vs
  selected comparison → `curves/diversified_book.png`
- `ou_spread_mr_hourly.pine` — TradingView port (1H spread symbols, next-open fills)

## Scaling recipe (path to more securities)

1. In TV, type the spread symbol (examples in the .pine header), 1H timeframe.
2. Export ≥2y of hourly bars to CSV; drop in `data_hourly/` (ts,open,high,low,close,volume).
3. Add the spread to `hourly_lab.SPREADS` with fixed economic weights; rerun `final.py`.
4. Only trust spreads positive in BOTH halves at 2× slip; keep family parity in the book.
