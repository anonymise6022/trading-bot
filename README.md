# Algorithmic Trading Bot for BTC/USDT
Hmph, so you want to hear about this trading bot for BTC/USDT, huh? It's not like I care or anything! Anyway, there's this fancy algorithmic trading framework in Python and PyTorch that’s supposed to forecast Bitcoin price movements using, ugh, historical data and technical indicators. I-it's just a bunch of computations involving things like Exponential Moving Averages and the Relative Strength Index, okay?! Not that I'm impressed or anything! 

# Step 1: Data Pipeline (btc_data.py & latest_data.py)
First off, there's the data pipeline, where this btc_data.py script, um, ingests all that historical market data to create features. Totally boring, but it generates target signals based on future returns or whatever, and spits out processed data. Don't get any ideas! 

# Step 2: Model Training (training.py)
Then, there's latest_data.py that fetches the latest 1,000 candles and the Open Interest history from Binance—that's just data gathering! You wouldn't understand, but it calculates log returns and relative Open Interest momentum to save it all into historical_data.csv. Who cares?! 

Next comes model training in training.py, where it processes the data sets, using standard scaling to avoid messing things up. And then it’s about some deep neural network or linear models that use PyTorch to make predictions. T-that's just how it works! 

Oh, and of course, they use class weighting to handle imbalances during training. It's not like I’m giving you an explanation because I want to! 

# Step 3: Inference Strategy (trading_strategy.py)
Finally, let’s talk about inference strategy in trading_strategy.py, where it loads all those saved weights and features based on what users... uh, want. It extracts recent arrays from datasets and formats input tensors for predictions. Not that any of this is impressive or anything! Hmph!

                    STOCK
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
   QUANTITATIVE   FUNDAMENTAL      HUMAN
     MODEL          MODEL         RESEARCH
       │              │              │
       ↓              ↓              ↓
 "Numbers say..." "Business says..." "I think..."
       │              │              │
       └──────────────┼──────────────┘
                      ↓
                  AI / MODEL
                      ↓
              FINAL ASSESSMENT
