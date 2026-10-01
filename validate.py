"""Harness de validacion walk-forward (rapido, sklearn, semilla fija).
Compara un modelo Ridge contra baselines: naive (manana=hoy) y MA5.
Uso: python validate.py [TICKER]  (default BTC-USD)
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge

SEED = 7
TICKER = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"

df = yf.download(TICKER, period="5y", progress=False, auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna()
close = df["Close"].astype(float)

feat = pd.DataFrame(index=df.index)
feat["ret1"] = close.pct_change(1)
feat["ret5"] = close.pct_change(5)
feat["rsi"] = 100 - 100 / (1 + close.diff().clip(lower=0).rolling(14).mean()
                           / (-close.diff().clip(upper=0).rolling(14).mean()).replace(0, np.nan))
feat["dist_ema20"] = close / close.ewm(span=20, adjust=False).mean() - 1
feat["vol20"] = feat["ret1"].rolling(20).std()
feat["target"] = close.shift(-1)  # cierre de manana
data = feat.dropna()
print("filas:", len(data), "rango:", data.index[0].date(), "->", data.index[-1].date(), flush=True)

Xcols = ["ret1", "ret5", "rsi", "dist_ema20", "vol20"]
test = data.iloc[-365:]  # ultimo anio como OOS rodante
preds, dates = [], []
for i in range(0, len(test), 30):  # reentrena cada mes
    block = test.iloc[i:i + 30]
    train = data.loc[:block.index[0]].iloc[:-1]
    mu, sd = train[Xcols].mean(), train[Xcols].std().replace(0, 1)
    model = Ridge(random_state=SEED)
    model.fit(((train[Xcols] - mu) / sd).values, train["target"].values)
    p = model.predict(((block[Xcols] - mu) / sd).values)
    preds.extend(p)
    dates.extend(block.index)
    print(f"  ventana {block.index[0].date()}... {len(block)} preds", flush=True)

res = pd.DataFrame({"real": test["target"].values[:len(preds)],
                    "modelo": preds}, index=dates)
# reconstruccion correcta: para cada fecha objetivo, naive = cierre dia anterior
allc = close
res["naive"] = [allc.loc[:d].iloc[-2] if len(allc.loc[:d]) > 1 else np.nan for d in res.index]
res["ma5"] = [allc.loc[:d].iloc[-6:-1].mean() for d in res.index]
res = res.dropna()

closes = close.reindex(res.index)
for col in ("modelo", "naive", "ma5"):
    mae = float((res["real"] - res[col]).abs().mean())
    direction = float((((res["real"] - closes.reindex(res.index).values) > 0)
                       == ((res[col] - closes.reindex(res.index).values) > 0)).mean())
    print(f"{col:8s} MAE={mae:9.1f}  acierto_direccion={direction:.1%}")
