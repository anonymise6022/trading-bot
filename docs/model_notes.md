# Model notes

One entry per model: what it assumes, why that assumption is made,
where it breaks for BTC specifically, and the practical consequence.

## Black-Scholes
- Assumes: constant volatility, smooth continuous price moves
- Breaks: BTC volatility clusters and jumps (news, liquidations)
- Consequence: underprices far-OTM / event-driven options

## GARCH (volatility forecast)
- Assumes: volatility clusters and reverts to a long-run average
- Breaks: doesn't see structural regime shifts coming (regulation, ETF news)
- Consequence: caught flat-footed exactly at the moments that matter most

## XGBoost (direction)
- Assumes: nothing about functional form -- learns splits from data
- Breaks: only as good as feature set + can overfit noisy 1h return data
- Consequence: needs walk-forward validation, not a single static split
