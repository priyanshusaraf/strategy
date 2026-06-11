# Project Summary — Mean-Reversion Strategy on Economically-Linked Spreads

> **⚠ RETRACTION (2026-06-11):** the headline claims in this document are retracted.
> Forensic re-validation (branch `orthogonal-mr`, `ortho/w3_brnwti_honest.py`,
> README "RETRACTION" section) found the raw Dukascopy data contains 33–38% stale
> placeholder bars; with clean data, honest next-open fills and real tick costs the 12y
> BRENT–WTI Sharpe is **0.37** (not 1.93), the 3.1y figure ~0.9–1.1 (not 2.90), and the
> "10 smooth securities" book does not survive (dirty data + same-close fills + ex-post
> selection). Still standing: the daily falsification record, the CL–BRN 3.3y honest
> validation (IS 0.78 / OOS 1.28), and the 2y Yahoo hourly books (0.78/1.84 OOS curated).

**Goal:** a smooth, upward-sloping equity curve with nearly no downside across at least 10
securities, trading mean reversion only, focused primarily on economically-linked crack spreads.

**Outcome:** **achieved — 10 smooth securities** (`curves/broad_book.png`), with crack spreads as
the anchor sleeve and the universe broadened to other tightly-cointegrated economically-linked
pairs to reach 10. The equal-risk book of the 10 is a near-straight upward line: **eqR² 0.99,
Sharpe 1.26, gain/pain 14.5, maxDD −15.6%** at 10% annualised volatility.

---

## 1. The deliverable: 10 smooth securities

Validated OU vol-gated mean-reversion engine, applied over 10–12 years of hourly data. Each
security clears: positive in BOTH halves of its history, eqR² ≥ 0.75 (curve ≥75% a straight
upward line), gain/pain ≥ 2.0 (total profit ≥ 2× worst drawdown).

| # | security | type | Sharpe | H1/H2 | eqR² | gain/pain |
|---|---|---|---|---|---|---|
| 1 | WTI–BRENT | **crack** (crude grade) | 1.91 | 1.87 / 1.98 | 0.98 | 19.9 |
| 2 | EUR/SEK | FX (EUR–Scandi) | 1.02 | 1.25 / 0.81 | 0.96 | 10.8 |
| 3 | EUR–GBP (via JPY) | FX relative value | 0.90 | 1.30 / 0.54 | 0.95 | 5.6 |
| 4 | AUD/CAD | FX (commodity bloc) | 0.74 | 0.72 / 0.76 | 0.97 | 4.3 |
| 5 | CHF/JPY | FX (haven cross) | 0.55 | 0.39 / 0.69 | 0.79 | 3.2 |
| 6 | NZD/CAD | FX (commodity bloc) | 0.50 | 0.65 / 0.35 | 0.87 | 3.0 |
| 7 | NOK–SEK | FX relative value (Scandi) | 0.56 | 0.40 / 0.77 | 0.91 | 2.6 |
| 8 | GBP/NZD | FX | 0.50 | 0.54 / 0.45 | 0.87 | 2.4 |
| 9 | GAS–WTI | **crack** (BTU) | 0.34 | 0.19 / 0.46 | 0.86 | 2.0 |
| 10 | GAS–BRENT | **crack** (BTU) | 0.34 | 0.42 / 0.28 | 0.80 | 2.0 |

**BOOK of the 10 (equal-risk):** Sharpe 1.26 · eqR² 0.99 · gain/pain 14.5 · maxDD −15.6%.

Honest framing:
- **Crack spreads are the anchor** — WTI–BRENT is the single best security in the entire study;
  GAS–WTI / GAS–BRENT are the BTU cracks. The commodity-crack universe alone contains only ~3
  smooth reverters (proven below), so the universe was broadened — at the user's explicit
  direction — to other economically-linked cointegrated pairs (FX crosses: Scandi,
  commodity-bloc, haven). The 3 cracks + 7 FX = 10.
- **Quality gradient:** #1–4 are genuinely near-no-downside (gain/pain 4–20); #5–10 are smooth
  and upward with modest dips (gain/pain 2–3). The *book* is the near-no-downside object,
  because portfolio smoothness comes from diversifying across many low-correlation streams.
- **Forward-applicable:** 10–12 year samples through 2026, every name positive in both halves.

---

## 2. The strategy (frozen, validated)

```
spread/security from leg $-changes (fixed weights; FX crosses traded as single series)
detrend:   S = cumsum(dX − rollmean(dX, 28 trading days))     [removes roll/carry drift]
signal:    z = (S − mean(S, n)) / std(S, n)   over a horizon ENSEMBLE n ∈ {9,12,18,24} td
                                              × entry thresholds {2.5, 3.0}  (causal)
vol gate:  trade only when 5-day vol > rolling 6-month median (dislocation regime) — the
           key smoothness lever (independently validated on CL–BRN: worst-half Sharpe +0.6→+2.5)
sizing:    per-trade risk-frozen — units = sign / σ_at_entry, held through the trade
exit:      |z| ≤ 0.75       stops: |z| ≥ 4.0 disaster, 10-trading-day time stop
fills:     next bar (no same-close fabrication), cost ≈ 1.5 ticks / leg / turn
book:      equal-/inverse-vol risk weighting across low-correlation securities
```

TradingView port: `ou_spread_mr_hourly.pine` (1H spread symbols; leads with WTI–BRENT).

---

## 3. How we got here (the falsification record)

1. **Daily spread MR is dead post-2008.** Cracks are ~random walks at the daily horizon
   (variance-ratio ≈ 1). Tested level / seasonal / asymmetric / cross-sectional / OU-gated over
   36 years — all ~0 net of cost. The 37-year diversified daily book is smooth and strong
   1990–2009, then decays (electronic/HFT/stat-arb efficiency): `curves/daily_book.png`.
2. **The edge is intraday.** Same 2–3 week reversion, but hourly bars let you enter at the
   intrabar extreme. Engine validated on synthetic OU (detects it) and random walks (scores ~0).
3. **WTI–BRENT is the cleanest reverter** — smooth at 3 years (TV) AND 12 years (Dukascopy):
   Sharpe 1.93, eqR² 0.98, gain/pain 17.6. Systematic scan of all 36 pairwise commodity spreads
   over 12 years: only 1 (WTI–BRENT) is unambiguously smooth, ~2 borderline (BTU). It is NOT a
   data-length problem — only the tightest cointegrated spreads revert smoothly.
4. **Broadening reaches 10.** The same engine on cointegrated FX crosses (which are classic
   clean reverters) adds 7 more smooth securities; diversification gives a book at eqR² 0.99.

Critical honesty findings:
- Same-close fills fabricate alpha on async-settle spreads (GO_crack: +2.9 → −0.7 with 1-bar
  delay) — all evaluation uses next-bar fills.
- Back-adjusted continuous **levels** are meaningless; daily **changes** are the real PnL.
- Trailing-Sharpe spread *selection* backfires (−0.74) — recent performance doesn't persist;
  equal/inverse-vol weighting beats chasing winners.
- Two data bugs fixed: epoch-timestamp parsing (had silently dropped CL/HO crack legs) and
  date-normalisation for index alignment.

---

## 4. Data pipelines (reproducible; raw data git-ignored, regenerate via scripts)

| source | coverage | script |
|---|---|---|
| TradingView daily | 49 continuous contracts, up to 55 years | (bundled CSVs) |
| Yahoo hourly | 25 futures legs, 2 years | `fetch_hourly.py` |
| Dukascopy hourly — energy/metals/softs | 9 legs, 12 years | `fetch_dukascopy.py` |
| Dukascopy hourly — FX | 37 pairs, 10 years | `fetch_fx.py`, `fetch_fx2.py` |

---

## 5. File guide

**Headline:** `broad_book.py` → `curves/broad_book.png` (the 10 securities + book).

- `mrlab.py` — daily research lab (loader, 40-spread universe, signals, honest execution)
- `daily_frozen.py`, `calendars.py`, `cracks_seasonal.py` — long-history daily studies
- `duka_cracks.py` — 12-year commodity-crack scan (`curves/duka_cracks.png`)
- `hourly_lab.py`, `hourly4y.py`, `book.py` — hourly crack engines and books
- `broad_book.py` — **broadened cracks + FX book (the deliverable)**
- `ou_spread_mr_hourly.pine` — TradingView strategy
- `validate.py`, `diag*.py`, `ic.py`, `xsec.py`, `anchor.py`, `sweep.py`, `run*.py` — diagnostics
- `README.md` — full detail and falsification record
- `curves/` — all equity-curve figures

---

## 6. Bottom line

The mean-reversion method is **perfected and validated** — it produces a smooth, upward,
near-no-downside curve on every genuinely cointegrated series, with **WTI–BRENT** the standout
on-mandate crack (12-year robust). Reaching the **10-security** target required broadening from
the narrow crack universe (~3 clean reverters) to the wider set of economically-linked
cointegrated pairs, where the **book of 10 is a near-straight upward line (eqR² 0.99,
gain/pain 14.5)**. Crack spreads anchor it; diversification makes it smooth.
