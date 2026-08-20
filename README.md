# btc-quant-suite
I-it's not like I built you a whole quant trading system because I wanted to, baka. It just so happens BTC/USDT needed direction prediction, volatility forecasting, options pricing, backtesting, and risk-managed execution, and someone had to do it properly. Don't get the wrong idea.

## Why this layout
Hmph. You probably expected some flat pile of scripts thrown together at 3am, didn't you? Well too bad, that's not how I operate. This is structured the way actually competent open-source projects are — the kind of layout you'd see in `scikit-learn` or `statsmodels`, not that it matters to you.

- **`src/<package>/`** — not a flat pile of scripts, obviously. Code that gets imported (features, models, risk logic) lives in a proper installable package. `scripts/` is just the thin "run this" layer sitting on top. It's so the same feature/model code works from a backtest, a live executor, *and* a notebook without you having to copy-paste like some amateur. I did this for you. Not because I care if your code is DRY. I just... don't like looking at duplication, that's all.

- **One subpackage per concern** (`data`, `features`, `models`, `risk`, `backtest`, `live`) — so each piece can be reasoned about and tested in isolation, obviously. `models/` gets split further into `direction/`, `volatility/`, `pricing/` because — and pay attention, I'm only explaining this once — those are three genuinely different problems that just happen to feed into one pipeline. It's not that complicated. Keep up.

- **`tests/` mirrors `src/`** — this is just the standard convention, like Redis's `tests/unit`/`tests/integration` split. I organized it this way so it's always obvious where a test belongs. Not that I expect you to actually write the tests. But if you do, at least you won't get lost. Not that I'd help you if you did.

- **`configs/*.yaml` instead of hardcoded constants** — all the "constants you define yourself" from risk management (`RISK_PER_TRADE`, `MIN_CONFIDENCE`, stop multipliers) live here, not buried in some script where you'll forget about them in a month. This way you can version, diff, and swap configs without touching code. You're welcome. Don't thank me though, it's annoying.

- **`artifacts/` is gitignored** — trained models and backtest outputs are generated, not source. Keeps the repo lightweight and diffable. Obviously I'm not going to let generated junk clutter up something I actually built carefully. Th-that's just basic hygiene, not perfectionism or anything.

- **`docs/` for the "why", code for the "how"** — `docs/model_notes.md` is where model assumptions and flaws (like Black-Scholes' constant-vol assumption, which, yes, is a real limitation, I'm not pretending it isn't) get written down in plain language. It's the kind of context that's easy to lose once it only lives in your head. I wrote it down so future-you doesn't have to bother past-you. It's not like I did it because I was worried you'd forget everything and blame the repo.

## Anyway
It's not like this structure is going to make your trading strategy actually profitable or anything — that's still on you and your models. I just made sure the *code* wouldn't be the reason it falls apart.
...Don't get used to this kind of effort.

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
