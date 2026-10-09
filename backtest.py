"""Backtest real de las reglas de la Sentinel V10 Pro.

Simula exactamente lo que describe el manual operativo y lo que implementa
streamlit-crypto-model.py:

    ENTRADA  (señal evaluada sobre el cierre del día T)
        LONG : Close[T] > EMA(span)[T]  y  ret3D[T] <= -umbral
        SHORT: Close[T] < EMA(span)[T]  y  ret3D[T] >= +umbral
        Ejecución al día siguiente: Open[T+1]

    SALIDA   (time-exit, 24 h ≈ 1 vela diaria)
        Close[T+1]   (cierre del mismo día en que se entró)

    DIMENSIONADO
        nocional = riesgo_pct * capital / (ATR14[T] / Close[T])
        nocional = min(nocional, apalancamiento_max * capital)

No hay stop loss, igual que en la app. Por eso el riesgo "0.5%" NO acota la
pérdida: una vela adversa mayor que 1 ATR la excede. El backtest modela eso en
lugar de ocultarlo, y por eso el nocional se recorta cuando la pérdida supera el
patrimonio (liquidación).

Uso:
    python backtest.py                      # BTC-USD, todos los activos
    python backtest.py ETH-USD
    python backtest.py BTC-USD --fee 0.001
"""
import argparse

import numpy as np
import pandas as pd
import yfinance as yf

FEE_POR_LADO = 0.0005   # 5 bps por lado: comisión + slippage
RIESGO_PCT = 0.005      # 0.5% del balance, igual que el default de la app
APALANCAMIENTO_MAX = 2.0
EMA_SPAN = 50
MOM_PCT = 3.0

# Slippage. Con un edge de unos pocos bps, esto decide si la estrategia existe.
SPREAD_BPS = 2.0              # medio spread que se cruza en cada lado
IMPACTO_BPS_POR_SQRT_PART = 50.0  # coef. de la ley raíz: bps = coef * sqrt(participación)


def descargar(ticker, periodo="max", solo_cerradas=True):
    """Descarga OHLCV.

    Por defecto descarta la última vela si su fecha es hoy (UTC): yfinance la devuelve
    incompleta y hace que los resultados cambien intradía. Un backtest debe usar solo
    velas cerradas para ser reproducible.
    """
    df = yf.download(ticker, period=periodo, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if solo_cerradas and len(df):
        hoy = pd.Timestamp.now(tz=df.index.tz).normalize() if df.index.tz else pd.Timestamp.now().normalize()
        if df.index[-1] >= hoy:
            df = df.iloc[:-1]
    return df


def indicadores(df, ema_span=EMA_SPAN, mom_pct=MOM_PCT):
    out = df.copy()
    out["ema"] = out["Close"].ewm(span=ema_span, adjust=False).mean()
    out["ret3d"] = out["Close"].pct_change(3) * 100
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    # Volumen Advanced Daily en USD, para estimar el impacto de mercado.
    adv_usd = (out["Volume"].rolling(30).mean() * out["Close"]).shift(1)
    out["adv_usd"] = adv_usd
    return out.dropna()


def senales(ind, mom_pct=MOM_PCT):
    """Devuelve la serie de señales: 1 long, -1 short, 0 sin posición."""
    s = pd.Series(0, index=ind.index, dtype=int)
    s[(ind["Close"] > ind["ema"]) & (ind["ret3d"] <= -mom_pct)] = 1
    s[(ind["Close"] < ind["ema"]) & (ind["ret3d"] >= mom_pct)] = -1
    return s


def simular(ind, fee=FEE_POR_LADO, riesgo_pct=RIESGO_PCT,
            apalancamiento=APALANCAMIENTO_MAX, capital_inicial=100_000.0,
            senal=None, rng=None, slippage=True):
    """Itera sobre las velas. Sin lookahead: la señal de T se ejecuta en T+1.

    `senal` permite inyectar una serie de señales alternativa (para los baselines
    aleatorios). Si es None se usan las reglas de la estrategia.

    Con `slippage=True` el coste no es una comisión plana sino medio spread más
    impacto de mercado por la ley raíz sobre la participación diaria (nocional/ADV).
    """
    if senal is None:
        senal = senales(ind)
    capital = capital_inicial
    pico = capital
    dd_max = 0.0
    operaciones = []
    equity_curve = []

    fechas = list(ind.index)
    for i in range(len(fechas) - 1):
        equity_curve.append((fechas[i], capital))
        f_salida, f_entrada = fechas[i], fechas[i + 1]
        lado = int(senal.iloc[i])
        if lado == 0:
            continue

        px_entrada = float(ind["Open"].loc[f_entrada])       # Open[T+1]
        px_salida = float(ind["Close"].loc[f_entrada])        # Close[T+1], 24 h después
        atr = float(ind["atr"].loc[f_salida])
        if not np.isfinite(atr) or atr <= 0 or px_entrada <= 0:
            continue

        # Dimensionado idéntico al de la app.
        nocional = (capital * riesgo_pct) / (atr / px_entrada)
        nocional = min(nocional, apalancamiento * capital)

        # Coste real de ejecución.
        coste = 2 * fee
        adv = float(ind["adv_usd"].loc[f_salida]) if "adv_usd" in ind else np.nan
        participacion = np.nan
        if slippage and np.isfinite(adv) and adv > 0:
            participacion = nocional / adv
            impacto = IMPACTO_BPS_POR_SQRT_PART * np.sqrt(participacion) / 10_000
            coste += 2 * (SPREAD_BPS / 10_000 + impacto)

        bruto = px_salida / px_entrada - 1
        neto = (bruto if lado == 1 else -bruto) - coste
        pnl = nocional * neto
        # Sin stop loss: si la pérdida supera el patrimonio, es liquidación total.
        pnl = max(pnl, -capital)
        capital += pnl

        pico = max(pico, capital)
        dd_max = max(dd_max, (pico - capital) / pico)

        operaciones.append({
            "entrada": f_entrada, "salida": f_entrada, "lado": lado,
            "px_entrada": px_entrada, "px_salida": px_salida,
            "ret_neto": neto, "coste": coste, "nocional": nocional, "pnl": pnl,
            "participacion": participacion,
            "equity": capital,
        })

    equity_curve.append((fechas[-1], capital))
    return pd.DataFrame(operaciones), pd.Series(
        dict(equity_curve)).sort_index(), dd_max


def metricas(ops, equity, dd_max, capital_inicial=100_000.0):
    if ops.empty:
        return {"trades": 0}
    pnl = ops["pnl"]
    Wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    m = {
        "trades": len(ops),
        "win_rate": float((pnl > 0).mean()),
        "payoff": float(Wins.mean() / abs(losses.mean())) if len(losses) else np.nan,
        "media_operacion": float(pnl.mean()),
        "mediana_operacion": float(pnl.median()),
        "mejor": float(pnl.max()),
        "peor": float(pnl.min()),
        "dd_max": float(dd_max),
        "retorno_total": float((equity.iloc[-1] - capital_inicial) / capital_inicial),
        "final_equity": float(equity.iloc[-1]),
        "exposicion_pct": float((ops["ret_neto"].abs() > 0).mean()),
    }
    m["profit_factor"] = float(Wins.sum() / abs(losses.sum())) if len(losses) else np.inf
    m["recovery_factor"] = (
        float(pnl.sum() / (dd_max * capital_inicial)) if dd_max > 0 else np.inf
    )
    años = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    m["cagr"] = float((equity.iloc[-1] / capital_inicial) ** (1 / años) - 1)
    return m


def baseline_buy_hold(ind, capital_inicial=100_000.0):
    px0, px1 = float(ind["Close"].iloc[0]), float(ind["Close"].iloc[-1])
    final = capital_inicial * px1 / px0
    equity = capital_inicial * ind["Close"] / px0
    pico = equity.cummax()
    dd = float(((pico - equity) / pico).max())
    años = max((ind.index[-1] - ind.index[0]).days / 365.25, 1e-9)
    return {"trades": 0, "retorno_total": final / capital_inicial - 1,
            "dd_max": dd, "cagr": (final / capital_inicial) ** (1 / años) - 1}


def baseline_aleatorio(ind, n_operaciones, fee=FEE_POR_LADO, repeticiones=2000,
                       seed=7, capital_inicial=100_000.0):
    """Baseline decisivo: entradas ALEATORIAS con el MISMO número de operaciones y el
    MISMO dimensionamiento que la estrategia.

    Si la estrategia no supera esta distribución, su resultado se explica por el
    dimensionamiento y el límite de operaciones, no por las reglas EMA+momentum.
    """
    rng = np.random.default_rng(seed)
    reales = np.flatnonzero(senales(ind).to_numpy() != 0)
    n = len(reales)
    if n == 0:
        return None
    ret = []
    for _ in range(repeticiones):
        idx = rng.choice(len(ind) - 2, size=n, replace=False)
        s = pd.Series(0, index=ind.index, dtype=int)
        for j in idx:
            s.iloc[j] = 1 if rng.random() < 0.5 else -1
        o, e, _ = simular(ind, fee=fee, senal=s, rng=rng, capital_inicial=capital_inicial)
        ret.append(float(o["pnl"].sum()))
    return np.array(ret)


def linea(nombre, m):
    if m.get("trades", 0) == 0 and "retorno_total" not in m:
        return f"  {nombre:22s} sin operaciones"
    r = m["retorno_total"]
    d = m["dd_max"]
    extra = ""
    if m.get("trades"):
        extra = f" | trades {m['trades']:4d} | win {m['win_rate']:6.1%} | payoff {m['payoff']:.2f}"
    return (f"  {nombre:22s} retorno {r:+8.2%} | dd_max {d:6.2%}"
            f" | cagr {m['cagr']:+7.2%}{extra}")


def significatividad(ops, n_boot=10000, seed=7):
    """Bootstrap y t-estadístico sobre los retornos por trade.

    AVISO: las operaciones son diarias y pueden ser consecutivas, así que sus
    retornos tienen autocorrelación y NO son iid. El intervalo bootstrap asume
    independencia, luego es optimista. Se reporta también la autocorrelación de
    orden 1 para poder juzgarlo.
    """
    r = ops["ret_neto"].to_numpy()
    n = len(r)
    if n < 5:
        return None
    rng = np.random.default_rng(seed)
    medias = np.array([rng.choice(r, size=n, replace=True).mean() for _ in range(n_boot)])
    ic = np.percentile(medias, [2.5, 97.5])
    t_stat = float(r.mean() / (r.std(ddof=1) / np.sqrt(n)))
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if n > 3 else float("nan")
    # Qué fracción del retorno depende de unas pocas operaciones ganadoras.
    top5 = np.sort(r)[-5:].sum()
    return {"n": n, "media": float(r.mean()), "ic": (float(ic[0]), float(ic[1])),
            "t": t_stat, "ac1": ac1, "share_top5": float(top5 / r.sum()) if r.sum() != 0 else float("nan")}


def breakeven_fee(ind, max_fee=0.01, tolerancia=0.00005):
    """Comisión por lado a la que el retorno cae a cero. 0 = nunca es rentable."""
    o, e, d = simular(ind, fee=0.0)
    if metricas(o, e, d).get("retorno_total", -1) <= 0:
        return 0.0
    lo, hi = 0.0, max_fee
    for _ in range(20):
        mid = (lo + hi) / 2
        o, e, d = simular(ind, fee=mid)
        if metricas(o, e, d).get("retorno_total", -1) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerancia:
            break
    return lo


def barrido_costas(ind):
    """¿En qué comisión deja de ser rentable la estrategia?"""
    print(f"\n  BARRIDO DE COSTES (la ventaja es pequeña: los costes importan)")
    print(f"  {'fee/lado':>9s} {'coste/trade':>12s} {'retorno':>10s} {'pnl':>12s}")
    for fee in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01):
        o, e, d = simular(ind, fee=fee)
        m = metricas(o, e, d)
        if m.get("trades", 0) == 0:
            continue
        print(f"  {fee:8.4%} {2 * fee:11.4%} {m['retorno_total']:+9.2%} {m['final_equity'] - 100_000:>+11,.0f}")
    be = breakeven_fee(ind)
    if be == 0.0:
        print("  → Nunca es rentable: ni con comisión cero el retorno bruto es positivo.")
    else:
        print(f"  → Punto de equilibrio: {be:.4%} por lado ({2 * be:.4%} por trade).")
        print(f"    Por debajo de eso la estrategia gana; por encima, pierde dinero.")


def por_ano(ind, fee=FEE_POR_LADO):
    """El edge, ¿es consistente cada año o está concentrado en unos pocos?"""
    print(f"\n  DESGLOSE POR AÑO (indicadores calculados sobre toda la historia)")
    print(f"  {'año':>6s} {'ops':>5s} {'retorno':>10s} {'win':>7s} {'payoff':>7s}")
    for año in sorted(set(ind.index.year)):
        sub = ind[ind.index.year == año]
        if len(sub) < 30:
            continue
        o, e, d = simular(sub, fee=fee)
        m = metricas(o, e, d)
        if m.get("trades", 0) == 0:
            print(f"  {año:6d} {0:5d}   sin operaciones")
            continue
        print(f"  {año:6d} {m['trades']:5d} {m['retorno_total']:+9.2%} {m['win_rate']:6.1%} {m['payoff']:6.2f}x")


def walk_forward(ticker, folds=6, fee=FEE_POR_LADO, df=None,
                 grid_ema=(20, 30, 50, 75, 100), grid_mom=(2.0, 3.0, 4.0, 5.0)):
    """Walk-forward con reoptimización de parámetros.

    En cada fold se elige el par (EMA, momentum) que mejor funcionó en TODOS los datos
    anteriores al fold, y se aplica ese par al fold sin volver a mirar atrás. Es la
    única forma de evaluar el *proceso* de elegir parámetros, no un par elegido a
    posteriori.

    Si el OOS de este procedimiento no supera al azar, elegir parámetros no aporta nada.
    """
    if df is None:
        df = descargar(ticker, "max")
    n = len(df)
    # Fold inicial del 25% para tener historia con la que elegir.
    ini_test = int(n * 0.25)
    paso = (n - ini_test) // folds
    if paso < 60:
        print("  (historia insuficiente para walk-forward)")
        return None

    cache = {}

    def rend(ind_sub, ema, mom):
        k = (id(ind_sub), ema, mom)
        if k in cache:
            return cache[k]
        o, e, _ = simular(ind_sub, fee=fee)
        m = metricas(o, e, _)
        v = m.get("retorno_total", -1.0) if m.get("trades", 0) else -1.0
        cache[k] = v
        return v

    print(f"\n  WALK-FORWARD CON REOPTIMIZACIÓN ({folds} folds)")
    print(f"  grid: EMA{list(grid_ema)} x mom{list(grid_mom)}")
    print(f"  {'fold':>16s} {'params elegidas':>18s} {'ret train':>11s} {'ret test':>10s} {'ops':>5s}")

    resultados = []
    capital = 100_000.0
    pnl_folds = []
    ini_oos = []          # fechas de inicio de cada fold, para verificar no solapan
    for k in range(folds):
        fin_train = ini_test + k * paso
        fin_test = min(fin_train + paso, n)
        if fin_test - fin_train < 30:
            continue
        ind_train = indicadores(df.iloc[:fin_train])
        # Warmup para que los indicadores tengan historia, PERO las operaciones solo
        # pueden empezar en la frontera del fold. Sin este recorte, las barras de
        # warmup son datos in-sample y se contabilizarían como out-of-sample.
        ind_ext = indicadores(df.iloc[max(fin_train - 100, 0):fin_test])
        fecha_fold = df.index[fin_train]
        ind_test = ind_ext[ind_ext.index >= fecha_fold]
        if len(ind_train) < 200 or len(ind_test) < 30:
            continue
        ini_oos.append(ind_test.index[0])
        # Elegir con el train, aplicar al test.
        mejor, mejor_r = None, -np.inf
        for ema in grid_ema:
            for mom in grid_mom:
                r = rend(ind_train, ema, mom)
                if r > mejor_r:
                    mejor_r, mejor = r, (ema, mom)
        o, e, d = simular(ind_test, fee=fee)
        m = metricas(o, e, d)
        rt = m.get("retorno_total", 0.0) if m.get("trades", 0) else 0.0
        capital *= (1 + rt)
        pnl_folds.append(float(o["pnl"].sum()) if m.get("trades", 0) else 0.0)
        ini_f = str(ind_test.index[0].date())
        fin_f = str(ind_test.index[-1].date())
        print(f"  {ini_f}→{fin_f} {f'EMA{mejor[0]} / ±{mejor[1]}%':>18s} "
              f"{mejor_r:+10.2%} {rt:+9.2%} {m.get('trades', 0):5d}")
        resultados.append({"ema": mejor[0], "mom": mejor[1], "train": mejor_r, "test": rt})

    if not resultados:
        return None

    total = capital / 100_000.0 - 1
    ems = [r["ema"] for r in resultados]
    mms = [r["mom"] for r in resultados]
    # Verificación explícita de que las ventanas OOS no se solapan.
    solapan = any(ini_oos[i] > fin for i, fin in enumerate(ini_oos[1:]))
    print(f"\n    Equity acumulado del walk-forward: {total:+.2%}")
    print(f"    Ventanas OOS solapadas: {'SI — BUG' if solapan else 'no (verificado)'}")
    print(f"    Parámetros elegidos: EMA {ems}")
    print(f"                        momentum {mms}")
    n_ema, n_mom = len(set(ems)), len(set(mms))
    print(f"    Estabilidad: {n_ema} EMA distintas, {n_mom} umbrales de momentum distintos "
          f"en {len(resultados)} folds")
    positivos = sum(1 for r in resultados if r["test"] > 0)
    print(f"    Folds con retorno positivo: {positivos}/{len(resultados)}")
    if total <= 0 or positivos * 2 <= len(resultados):
        print("    → El procedimiento de elegir parámetros NO genera valor: la eleccion")
        print("      de parámetros es ruido, no una ventaja.")
    else:
        print("    → El procedimiento produce valor, pero revisa que la eleccion de")
        print("      parámetros sea estable (pocos valores distintos) antes de creértelo.")
    return {"total": total, "resultados": resultados,
            "pnl_folds": pnl_folds, "ema_distintas": n_ema, "mom_distintas": n_mom}


def run_ticker(ticker, fee=FEE_POR_LADO, mom=MOM_PCT, ema=EMA_SPAN, simulaciones=2000,
               periodo="max", analisis_extra=True):
    print(f"\n{'=' * 78}\n{ticker}   EMA{ema} / momentum ±{mom}% / "
          f"fee {fee:.3%} por lado / riesgo {RIESGO_PCT:.1%} / lev {APALANCAMIENTO_MAX}x\n{'=' * 78}")
    df = descargar(ticker, periodo)
    ind = indicadores(df, ema, mom)
    print(f"  datos: {ind.index[0].date()} -> {ind.index[-1].date()}  ({len(ind)} velas, period={periodo})")

    ops, equity, dd = simular(ind, fee=fee)
    m = metricas(ops, equity, dd)
    print("\n  ESTRATEGIA")
    print(linea("Sentinel V10", m))

    bh = baseline_buy_hold(ind)
    print("\n  BASELINES (para no confundir azar con edge)")
    print(linea("buy & hold", bh))
    n = len(ind)
    largos = (ind["Close"].pct_change().fillna(0) > 0).mean()
    print(f"  {'dirección al azar':22s} {largos:6.1%} de velas alcistas -> "
          f"{(2 * largos - 1) * 100:+.2f}% de aciertos esperados")

    # El test decisivo: mismas reglas de dinero, entradas elegidas al azar.
    print(f"\n  BASELINE ALEATORIO ({simulaciones} simulaciones, mismo nº de operaciones")
    print("  y mismo dimensionamiento que la estrategia)")
    ale = baseline_aleatorio(ind, m["trades"], fee=fee, repeticiones=simulaciones)
    if ale is not None:
        pnl_real = float(ops["pnl"].sum())
        pct = float((ale < pnl_real).mean())
        print(f"    pnl estrategia      : ${pnl_real:+,.0f}")
        print(f"    pnl aleatorio       : mediana ${np.median(ale):+,.0f}  "
              f"rango ${ale.min():,.0f} … ${ale.max():,.0f}")
        print(f"    percentil de la estrategia entre entradas aleatorias: {pct:.1%}")
        if pct >= 0.95:
            print("    → La estrategia supera al azar con margen: las reglas aportan valor.")
        elif pct >= 0.80:
            print("    → Supera al azar, pero sin margen amplio. Evidencia débil.")
        else:
            print("    → NO supera al azar: el resultado se explica por el dimensionamiento,")
            print("      no por las reglas EMA+momentum.")

    # Split cronológico: mitad de la muestra para in-sample, resto out-of-sample.
    corte = ind.index[len(ind) // 2]
    print(f"\n  SPLIT CRONOLÓGICO (corte en {corte.date()})")
    for etiqueta, sub in (("in-sample ", ind[ind.index <= corte]), ("out-sample", ind[ind.index > corte])):
        if len(sub) < 60:
            print(f"  {etiqueta}: muestra demasiado corta ({len(sub)} velas)")
            continue
        o, e, d = simular(sub, fee=fee)
        print(linea(etiqueta, metricas(o, e, d)))

    if not ops.empty and "participacion" in ops:
        part = ops["participacion"].replace([np.inf, -np.inf], np.nan).dropna()
        coste_medio = float(ops["coste"].mean())
        print(f"\n  COSTES REALES Y PARTICIPACIÓN")
        print(f"    commission        : {2 * fee:.3%} por operación")
        print(f"    coste medio real  : {coste_medio:.3%} por operación "
              f"(spread {SPREAD_BPS:.0f} bps/lado + impacto)")
        if len(part):
            print(f"    participación ADV : mediana {part.median():.6%} | máximo {part.max():.6%}")
            if part.median() < 1e-4:
                print("    → Participación minúscula: el impacto es despreciable y lo que")
                print("      cuesta dinero es la comisión y el spread, no el tamaño de la orden.")

    if analisis_extra and not ops.empty:
        sig = significatividad(ops)
        if sig:
            print(f"\n  SIGNIFICATIVIDAD (bootstrap 10.000, sobre {sig['n']} operaciones)")
            print(f"    retorno medio por trade : {sig['media']:+.4%}")
            print(f"    IC 95% del retorno medio: [{sig['ic'][0]:+.4%}, {sig['ic'][1]:+.4%}]")
            print(f"    t-estadístico          : {sig['t']:+.2f}")
            print(f"    autocorrelación lag-1  : {sig['ac1']:+.3f}  "
                  f"({'los trades NO son independientes' if abs(sig['ac1']) > 0.1 else 'casi independientes'})")
            print(f"    peso de las 5 mejores   : {sig['share_top5']:.0%} del retorno total")
            if sig["ic"][0] <= 0 <= sig["ic"][1]:
                print("    → El IC 95% incluye el cero: no se puede rechazar que el retorno")
                print("      medio sea nulo. La ventaja no es estadísticamente significante.")
            elif sig["t"] < 2:
                print("    → IC por encima de cero pero |t| < 2: evidencia débil.")
            else:
                print("    → IC excluye el cero y |t| > 2: diferencia significativa,")
                print("      sujeta a la cautela por la autocorrelación de los trades.")

        por_ano(ind)
        barrido_costas(ind)
        wf = walk_forward(ticker, fee=fee, df=df)
        if wf:
            ale = baseline_aleatorio(ind, len(ops), fee=fee, repeticiones=400)
            if ale is not None:
                p_wf = float((ale < sum(wf["pnl_folds"])).mean())
                print(f"    Percentil del walk-forward entre entradas aleatorias: {p_wf:.1%}")
                if p_wf < 0.80:
                    print("    → El proceso completo (elegir parámetros y operar) tampoco")
                    print("      supera al azar de forma convincente.")

    return ind, m


def sensibilidad(ticker):
    """¿El resultado depende de los parámetros? Un edge real sobrevive a vecinos."""
    print(f"\n{'=' * 78}\nSENSIBILIDAD DE PARÁMETROS — {ticker}\n{'=' * 78}")
    print("  Un edge real se mantiene al mover los parámetros. Si solo gana en un")
    print("  punto exacto, es sobreajuste (curve fitting), no una ventaja.\n")
    df = descargar(ticker, "max")
    filas = []
    print(f"  {'EMA':>5s} {'mom':>5s} | {'trades':>7s} {'retorno':>10s} {'dd_max':>8s} {'payoff':>7s}")
    for ema in (20, 50, 100):
        for mom in (2.0, 3.0, 4.0):
            ind = indicadores(df, ema, mom)
            o, e, d = simular(ind)
            m = metricas(o, e, d)
            if m.get("trades", 0) == 0:
                print(f"  {ema:5d} {mom:5.1f} | {'0':>7s}  (sin operaciones)")
                continue
            filas.append((ema, mom, m["retorno_total"]))
            print(f"  {ema:5d} {mom:5.1f} | {m['trades']:7d} {m['retorno_total']:+9.2%} "
                  f"{m['dd_max']:7.2%} {m['payoff']:6.2f}x")
    if filas:
        positivos = [r for _, _, r in filas if r > 0]
        negativos = [r for _, _, r in filas if r <= 0]
        print(f"\n  Configuraciones con retorno positivo: {len(positivos)}/{len(filas)}")
        if len(negativos) == 0:
            print("  → Todas las configuraciones ganan. El resultado no depende de los")
            print("    parámetros: no es curve fitting.")
        elif len(positivos) == 0:
            print("  → Todas las configuraciones PIERDEN, de forma consistente. La")
            print("    estrategia no tiene edge en este activo con estos costes.")
        else:
            print("  → El signo cambia al mover los parámetros: el resultado es frágil y")
            print("    no constituye evidencia de ventaja.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=["BTC-USD", "ETH-USD", "SOL-USD"])
    ap.add_argument("--fee", type=float, default=FEE_POR_LADO)
    ap.add_argument("--simulaciones", type=int, default=2000,
                    help="réplicas del baseline aleatorio")
    ap.add_argument("--period", default="max", help="max | 10y | 5y | 2y")
    ap.add_argument("--rapido", action="store_true",
                    help="omite significatividad, desglose anual y barrido de costes")
    ap.add_argument("--sensibilidad", action="store_true", help="barrido de parámetros")
    args = ap.parse_args()

    tickers = args.tickers or ["BTC-USD", "ETH-USD", "SOL-USD"]
    print(f"Backtest Sentinel V10 Pro | coste {args.fee:.3%} por lado "
          f"(entrada + salida = {2 * args.fee:.3%} por trade) | "
          f"riesgo nominal {RIESGO_PCT:.1%} | apalancamiento máx {APALANCAMIENTO_MAX}x")
    print("Sin stop loss: el riesgo nominal NO acota la pérdida por trade.")

    for t in tickers:
        run_ticker(t, fee=args.fee, simulaciones=args.simulaciones, periodo=args.period, analisis_extra=not args.rapido)

    if args.sensibilidad:
        for t in tickers:
            sensibilidad(t)