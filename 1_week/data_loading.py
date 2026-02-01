import yfinance as yf
import numpy as np
import pandas as pd


appl = yf.download("AAPL", period = '1y')
print(appl['Close'].iloc[-1])

appl['log_returns'] = np.log(appl['Close'] / appl['Close'].shift(1))

historical_vol = appl['log_returns'].std() * np.sqrt(252) * 100

print(f"\n{historical_vol:.2f}%")