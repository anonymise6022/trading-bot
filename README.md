# btc-quant-suite

BTC/USDT quantitative trading system: direction prediction, volatility
forecasting, options pricing, backtesting, and risk-managed execution.

## Why this layout

Structured the way most mature open-source data/ML projects are (the
pattern you'll see in things like `scikit-learn`, `statsmodels`, or any
serious `src/`-layout Python package):

- **`src/<package>/` not a flat pile of scripts.** Code that gets
  imported (features, models, risk logic) lives in an installable
  package. `scripts/` is the thin "run this" layer on top of it. This
  is what makes the same feature/model code usable from a backtest, a
  live executor, AND a Jupyter notebook without copy-pasting.
- **One subpackage per concern** (`data`, `features`, `models`, `risk`,
  `backtest`, `live`) so you can reason about (and test) each piece in
  isolation. `models/` is further split into `direction/`,
  `volatility/`, `pricing/` because those are genuinely three separate
  problems (see mapping below) even though they feed into one pipeline.
- **`tests/` mirrors `src/`** — this is the standard convention (Redis'
  `tests/unit` / `tests/integration` split is the same idea) so it's
  always obvious where a test for a given module belongs.
- **`configs/*.yaml` instead of hardcoded constants.** All the
  "constants you define yourself" from risk management (RISK_PER_TRADE,
  MIN_CONFIDENCE, stop multipliers) live here, not buried in scripts —
  makes it possible to version, diff, and swap configs without touching
  code.
- **`artifacts/` is gitignored** — trained models and backtest outputs
  are generated, not source. Keeps the repo itself lightweight and
  diffable.
- **`docs/` for the "why", code for the "how".** `docs/model_notes.md`
  is where model assumptions/flaws (e.g. Black-Scholes' constant-vol
  assumption) get written down in plain language — the kind of context
  that's easy to lose once it's only in your head.

## Directory structure

```
btc-quant-suite/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml            # package metadata + dependencies
├── requirements.txt          # pinned runtime deps
├── requirements-dev.txt      # + pytest, linters, etc.
├── Makefile                  # make train / make backtest / make test
├── .gitignore
│
├── src/btc_quant/
│   ├── data/                 # everything that touches the exchange/API
│   │   ├── fetch_ohlcv.py            # mirrors btc_data.py
│   │   ├── fetch_open_interest.py    # mirrors latest_data.py (OI half)
│   │   ├── fetch_options_chain.py    # NEW: needed for options work
│   │   └── loaders.py
│   │
│   ├── features/             # feature engineering, source-of-truth
│   │   ├── lag_returns.py
│   │   ├── ema_trend.py
│   │   ├── volatility_features.py
│   │   ├── interactions.py
│   │   └── pipeline.py               # chains the above into one call
│   │
│   ├── models/
│   │   ├── direction/                # "which way" — your existing work
│   │   │   ├── xgboost_classifier.py     # current main model
│   │   │   ├── neural_net_classifier.py  # Regime_serious.py, kept for ensembling
│   │   │   └── ensemble.py               # combine the two
│   │   │
│   │   ├── volatility/               # "how much" — the genuinely predictive
│   │   │   │                         #  half of the options problem
│   │   │   ├── realized_vol.py
│   │   │   ├── garch.py
│   │   │   └── implied_vol.py
│   │   │
│   │   └── pricing/                  # deterministic math, not prediction
│   │       ├── black_scholes.py
│   │       ├── greeks.py
│   │       └── heston.py             # add once BS's flaws are actually felt
│   │
│   ├── risk/                 # the guardrails, separate from any model
│   │   ├── position_sizing.py        # risk_amount -> position_size -> leverage
│   │   ├── should_trade.py           # confidence/margin/volatility gate
│   │   └── limits.py                 # daily loss limit, max exposure
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── costs.py                  # fees/slippage/funding — non-negotiable
│   │   ├── metrics.py                # Sharpe, drawdown, calibration
│   │   └── walk_forward.py           # rolling retrain, regime-shift check
│   │
│   ├── live/
│   │   ├── executor.py
│   │   └── monitor.py                # live-vs-backtest drift detection
│   │
│   └── utils/
│       ├── logging.py
│       └── config.py
│
├── tests/
│   ├── unit/                 # one test file per src module, same name
│   └── integration/          # full pipeline / backtest-engine tests
│
├── configs/                  # every tunable constant, out of code
│   ├── data.yaml
│   ├── direction_model.yaml
│   ├── volatility_model.yaml
│   ├── risk.yaml
│   └── backtest.yaml
│
├── scripts/                  # thin CLI entry points, no logic lives here
│   ├── run_data_pipeline.py
│   ├── train_direction_model.py
│   ├── train_volatility_model.py
│   ├── run_backtest.py
│   └── run_live.py
│
├── notebooks/                 # exploratory only — nothing here is imported
├── data/
│   ├── raw/                   # gitignored — fetched data lands here
│   └── processed/              # gitignored — feature-engineered output
├── artifacts/
│   ├── models/                # gitignored — trained model files
│   └── backtests/              # gitignored — backtest result dumps
└── docs/
    ├── architecture.md         # how the pieces connect, diagrammed
    ├── model_notes.md          # assumptions + known flaws, per model
    └── risk_policy.md          # the actual numbers you're committing to
```

## How your existing files map in

| Current file | New home |
|---|---|
| `btc_data.py` | `src/btc_quant/data/fetch_ohlcv.py` |
| `latest_data.py` | `src/btc_quant/data/fetch_ohlcv.py` + `fetch_open_interest.py` (split — it currently does both) |
| `training.py` (linear regression) | retired — superseded by `Regime_xgboost.py` |
| `Regime_serious.py` (neural net) | `src/btc_quant/models/direction/neural_net_classifier.py` |
| `Regime_xgboost.py` | `src/btc_quant/models/direction/xgboost_classifier.py` |
| `trading_strategy.py` | split across `scripts/run_live.py` + `src/btc_quant/live/executor.py` |
| the `should_trade()` filters we wrote | `src/btc_quant/risk/should_trade.py` |

## Suggested build order (matches the learning order from our conversations)

1. `data/` + `features/` — get the pipeline producing clean, versioned features
2. `models/direction/` — your XGBoost model (already working) + ensemble with the NN
3. `risk/` — position sizing + `should_trade()` gate, wired to `configs/risk.yaml`
4. `backtest/` — with real costs, before anything touches live capital
5. `models/volatility/` + `models/pricing/` — only once the direction+risk loop above is solid; this is the options half of the project
6. `live/` — paper trade first, `monitor.py` watching for live-vs-backtest drift

## Note

This is a scaffold, not a finished codebase — most files are stubs with
a one-line docstring describing their purpose. Fill them in module by
module, in the order above, rather than all at once.
