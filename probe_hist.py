import pandas as pd
import yfinance as yf
for t in ["BTC-USD","ETH-USD","SOL-USD"]:
    for p in ["5y","10y","max"]:
        try:
            d = yf.download(t, period=p, progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            print(f"{t:9s} {p:4s} -> {len(d):5d} filas  {d.index[0].date()} .. {d.index[-1].date()}")
        except Exception as e:
            print(f"{t:9s} {p:4s} ERROR {e}")
