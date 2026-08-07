from dataclasses import dataclass

@dataclass
class ResultStruct:
    leverage: int
    margin: int


# Constants
RISK_PER_TRADE = 0.025
ACC_EQUITY = 1000
MIN_CONFIDENCE = 0.55
MAX_LEVERAGE = 15


risk_amount = ACC_EQUITY * RISK_PER_TRADE
stop_loss_distance_pct = 0.015
position_size = risk_amount / stop_loss_distance_pct
margin_to_use = ACC_EQUITY * 0.20
leverage_required = position_size / margin_to_use


def should_trade():
    should_trade = True
    if leverage_required > MAX_LEVERAGE:
        should_trade = False
    return should_trade

