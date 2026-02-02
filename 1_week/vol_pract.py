import numpy as np
import pandas as pd
import yfinance as yf

def data_load(name:str, period:str)->pd.DataFrame:
    return yf.download(name,period=period)

def returns(data:pd.DataFrame)->pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].iloc[:, 0]
    else:
        close = data['Close']
    ret = np.log(close / close.shift(1)).dropna()
    return ret

def vol(data:pd.Series)->float:
    return data.std() * np.sqrt(365)

eth = data_load("ETH-USD", "1y")
btc = data_load("BTC-USD", "1y")
eth_ret = returns(eth)
btc_ret = returns(btc)

cor = eth_ret.corr(btc_ret)
vol_btc = vol(btc_ret)
vol_eth = vol(eth_ret)

print(f"Доходность BTC: {btc_ret}\n Доходность ETH: {eth_ret}\n Корреляция: {cor}")
print(f"Волатильность BTC: {vol_btc:.2f}\nВолатильность ETH: {vol_eth:.2f}")