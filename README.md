# trading-bot
I-I’m only explaining how this trading bot works because I have to, okay?! D-Don’t get the wrong idea or anything! I-it's not like I’m doing this because I like you or something, so just... just listen, okay? Hmph!
(Momemtum and mean reversion/ at least 1 hour time frame)

 What each files do (in order):

# first step: fetch ✨"le data"✨
- btc_data.py: fetch local btc data to BTCUSDT-1h.csv(40k lines)
- latest_data.py: fetch latest btc data and OI to historical_data.csv (700 - 1000 lines)

# Second step: Train ✨"le trading model"✨
- training.py: Hmph! So, like, you take that data we gathered, and you shove it into this equation, okay? It's Y_HAT = XW + b or whatever. Yeah, and you include 6 lags like it's no big deal! Then, you get that clanker to do its thing and find the right weight and bias for you. I-I guess, if you really need it, it *finally* gives you ✨the prediction✨, Tch!

# Third step: Get ✨"le signal"✨
- trading_strategy.py: Ugh, so listen up, alright? We take all those lags and weights and biases and shove them into a pth file, okay? And then, the user gets to choose the X, B, and W from either the first or second set of data, not that it matters that much! Tch! After that, we have to load it in a (6,1) format because it's just a linear model—duh! Six inputs and one output! Like, seriously, is that so hard to understand? Then... we dig through the csv and, um, GRAB the latest 6 lag features. [-1] is for the last row, and -6 means the last 6 columns! It's not rocket science, stupid baka! And we just convert it so that PyTorch can, I don't know, read it or whatever and give me the prediction I need! Not that I care if you understand or anything... idiot!
