# trading-bot
Momemtum and mean reversion/ at least 1 hour time frame

# What each files do (in order):

# first step: fetch ✨"le data"✨
- btc_data.py: fetch local btc data to BTCUSDT-1h.csv(40k lines)
- latest_data.py: fetch latest btc data and OI to historical_data.csv (700 - 1000 lines)

# Second step: Train ✨"le trading model"✨
- training.py: uses the data we fetched, and put it in an equation of Y_HAT = XW+b with 6 lags, and get clanker to find the correct weight and bias, AND FINALLY, it gives you ✨le prediction✨

# Third step: Get ✨"le signal"✨
- trading_strategy.py: So~ we save the lags and weight as well as bias into pth, and user picks the X,B,W from either 1st or 2nd set of datas. Then we load it in (6,1) format because its a linear model -6 inputs and 1 output. Finally, we go back to csv and GRABBB the latest 6 lag features [-1] means last row, and -6: means last 6 columns, and we just convert it so that stu.. stupid baka pytorch can read it and give me the damn prediction.
