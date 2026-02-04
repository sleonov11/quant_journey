import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(" Загружаем данные глобальных рынков...")
aapl = yf.download("AAPL", period="1y", auto_adjust=True)
tlt = yf.download("TLT", period="1y", auto_adjust=True)
gspc = yf.download("^GSPC", period="1y", auto_adjust=True)

# Надёжное извлечение цены (работает с любым форматом)
def safe_price(data):
    close = data['Close']
    if isinstance(close, pd.DataFrame):  # мультииндекс
        close = close.iloc[:, 0]
    return float(close.iloc[-1])

# Надёжный расчёт волатильности
def safe_vol(data):
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    returns = np.log(close / close.shift(1)).dropna()
    return float(returns.std() * np.sqrt(252)), returns

# Получаем данные
aapl_price = safe_price(aapl)
tlt_price = safe_price(tlt)
gspc_price = safe_price(gspc)

aapl_vol, aapl_ret = safe_vol(aapl)
tlt_vol, tlt_ret = safe_vol(tlt)
gspc_vol, gspc_ret = safe_vol(gspc)

print(f"{'='*70}")
print(f"СРАВНЕНИЕ РЫНКОВ: АКЦИИ vs ОБЛИГАЦИИ")
print(f"{'='*70}")
print(f"Apple (AAPL) — акции:")
print(f"  Текущая цена: ${aapl_price:.2f}")
print(f"  Годовая волатильность: {aapl_vol:.1%}")
print(f"\nTLT — облигации США (долгосрочные):")
print(f"  Текущая цена: ${tlt_price:.2f}")
print(f"  Годовая волатильность: {tlt_vol:.1%}")
print(f"\nS&P 500 (^GSPC) — индекс:")
print(f"  Текущая цена: {gspc_price:.2f} пунктов")
print(f"  Годовая волатильность: {gspc_vol:.1%}")
print(f"\n Ключевой вывод кванта:")
print(f"  Акции (σ={aapl_vol:.0%}) в {aapl_vol/tlt_vol:.1f}x волатильнее облигаций (σ={tlt_vol:.0%})")
print(f"  → Опционы на акции будут в {aapl_vol/tlt_vol:.1f}x дороже")
print(f"{'='*70}")

# Визуализация
plt.figure(figsize=(14, 7))

# Нормируем цены к 100
def normalize(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return (series / series.iloc[0] * 100).values

aapl_norm = normalize(aapl['Close'])
tlt_norm = normalize(tlt['Close'])
gspc_norm = normalize(gspc['Close'])

plt.plot(aapl.index, aapl_norm, label=f"AAPL (акции, σ={aapl_vol:.0%})", linewidth=2.5)
plt.plot(tlt.index, tlt_norm, label=f"TLT (облигации, σ={tlt_vol:.0%})", linewidth=2.5, alpha=0.8)
plt.plot(gspc.index, gspc_norm, label=f"S&P 500 (индекс, σ={gspc_vol:.0%})",
         linewidth=2.5, linestyle='--', alpha=0.7)

plt.title("Акции vs Облигации vs Индекс — волатильность за год", fontsize=14, fontweight="bold")
plt.ylabel("Нормированная цена (старт = 100)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("global_markets_comparison.png", dpi=150, bbox_inches="tight")
print("\n График сохранён: global_markets_comparison.png")
plt.show()

# Корреляция
common_dates = aapl_ret.index.intersection(tlt_ret.index)
if len(common_dates) > 10:
    correlation = float(aapl_ret[common_dates].corr(tlt_ret[common_dates]))
    print(f"\n Корреляция доходностей AAPL/TLT: {correlation:.2f}")
    print(f"   → {'Низкая корреляция' if abs(correlation) < 0.3 else 'Умеренная корреляция'} = {'хорошая' if abs(correlation) < 0.3 else 'ограниченная'} диверсификация")
else:
    print("\n  Недостаточно данных для корреляции")