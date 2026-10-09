# 🚀 AI Crypto Strategist & Sentinel V10 Pro

Interfaz Streamlit para auditar las **reglas de entrada** de la estrategia Sentinel V10
(EMA 50 + momentum 3D + salida por tiempo a 24 h) sobre BTC, ETH y SOL, calcular el
dimensionamiento por ATR y contrastar un modelo LSTM contra un baseline naive.

**Desarrollado por:** [@Bookbinderr-2026](https://x.com) | **Comunidad:** [QuantumTradeA Telegram](https://t.me)

---

## ⚠️ Estado real del proyecto (léelo antes que nada)

Una versión anterior de esta app mostraba una tabla de Win Rate, Payoff, Drawdown Máximo y
Factor de Recuperación con cifras escritas a mano que el código nunca calculaba, seguida de
un dictamen de "robustez matemática impecable". Ese contenido se ha retirado y sustituido por
mediciones reales.

| Qué | Cómo | Resultado |
|---|---|---|
| Reglas de entrada/salida | `backtest.py` y pestaña 3 | **Sin ventaja demostrable** |
| Predicción del nivel de precio (LSTM) | `validate.py` y pestaña 2 | **No supera al naive** |

### Las reglas de trading: no se ha demostrado ventaja

`backtest.py` simula las reglas exactas de la app (señal sobre el cierre de T, entrada al
`Open[T+1]`, salida a `Close[T+1]`, dimensionado por ATR, 5 bps por lado) sobre **toda la
historia disponible**, y las compara contra entradas aleatorias con el mismo número de
operaciones y el mismo tamaño de posición.

| Activo | Período | Retorno neto | DD máx | Ops | t | IC 95% media/trade | Equilibrio |
|---|---|---|---|---|---|---|---|
| BTC-USD | 2014-2026 | **−4.5%** | 14.2% | 575 | −0.08 | [−0.34%, +0.31%] | **1.2 bps/lado** |
| ETH-USD | 2017-2026 | **−12.1%** | 14.7% | 556 | −0.31 | [−0.46%, +0.32%] | nunca |
| SOL-USD | 2020-2026 | **−6.8%** | 9.7% | 516 | −0.17 | [−0.58%, +0.49%] | nunca |

**En los tres activos el intervalo de confianza al 95% incluye el cero.** No se puede
rechazar que el retorno medio por operación sea nulo. El equilibrio de BTC está en **1.2 bps
por lado**, una fracción de la comisión taker de cualquier exchange real, así que la
estrategia es perdedora a costes ejecutables. ETH y SOL pierden dinero incluso con comisión
cero.

### Walk-forward: elegir parámetros no rescata la estrategia

El test más exigente del repositorio. En cada uno de 6 folds se elige el par
(EMA, momentum) que mejor funcionó **solo con los datos anteriores al fold**, y se aplica al
fold sin volver a mirar atrás. Ventanas OOS verificadas como no solapadas.

| Activo | Equity OOS acumulado | Percentil vs. azar | Folds positivos |
|---|---|---|---|
| BTC-USD | +4.0% | 93.0% | 4/6 |
| ETH-USD | −4.8% | 70.0% | 3/6 |
| SOL-USD | −8.0% | 40.5% | 0/6 |

Dos conclusiones:

1. **El caso más favorable es BTC: +4.0% en 9 años**, en el percentil 93. Es el techo
   razonable de esta estrategia, y exige costes por debajo de 1.2 bps por lado para no perder.
2. **El walk-forward elige siempre EMA20 / ±2%, nunca los parámetros documentados
   (EMA50 / ±3%).** Los parámetros de la app no son los que seleccionaría el dato. La
   elección es estable: los 6 de 6 folds eligen lo mismo. Eso sugiere que el resto del
   grid es ruido y solo ese rincón tiene algo de señal — insuficiente para operar.

### Tres cosas que hay que mirar antes de creer cualquier cifra

1. **El resultado depende de la ventana temporal.** Con los últimos 5 años (2021-2026) BTC
   da +11.3% y queda en el percentil 99.7 frente al azar. Con los 12 años completos da
   −4.5% y percentil 80. El +11.3% era un artefacto de haber mirado solo la década alcista.
2. **Los costes se comen todo el edge.** Con medio spread de 2 bps por lado más impacto de
   mercado, el coste medio real es de 0.14% por operación. El equilibrio de BTC está en
   1.2 bps por lado.
3. **Unas pocas operaciones explican el resultado.** Las 5 mejores operaciones suman más que
   el retorno total; sin ellas el resultado es negativo.

Consistencia por años (BTC): negativo de 2014 a 2020, positivo de 2021 a 2026. ETH y SOL
alternan signo sin patrón. Es dependencia de régimen, no una ventaja estable.

### Sobre el modelo de costes

`backtest.py` modela medio spread (2 bps por lado) más impacto de mercado por la ley raíz
sobre la participación diaria (`nocional / ADV`). La participación medida es de ~1e-6 del
volumen diario, así que **el impacto es despreciable y lo que cuesta dinero es la comisión y
el spread**. El coste medio real por operación es de 0.14%.

Dos salvedades: no modela liquidez en eventos de estrés, y en SOL la participación máxima
llega a 0.58% en las velas ilíquidas de 2020, donde el modelo sí subestima. El modelo es, en
el mejor de los casos, una estimación optimista del coste real.

### Reproducibilidad

`descargar()` descarta la vela en curso, de modo que los resultados no cambian intradía.
Verificado: dos ejecuciones seguidas dan rangos, retornos y punto de equilibrio idénticos.

> Nota sobre método: las operaciones son diarias y pueden ser consecutivas, así que sus
> retornos tienen autocorrelación (−0.10 en BTC) y no son iid. El intervalo bootstrap
> asume independencia, luego es optimista. Aun así, el margen es tan pequeño que la
> conclusión no cambia.

### La predicción de precio tampoco tiene edge

```
python validate.py BTC-USD
modelo            MAE    MAE %
modelo        21132.3   26.71%
naive          1294.9    1.64%   <- "mañana = hoy"
ma5            1994.9    2.52%
VEREDICTO: el modelo NO supera al baseline naive (21132 vs 1295).
```

Un modelo lineal sobre features de precio es ~16x peor que decir "mañana valdrá lo mismo que
hoy". El LSTM de la pestaña 2 hace la misma comprobación y avisa cuando no supera al
baseline.

---

## 🔬 Reglas implementadas (Sentinel V10)

1. **Filtro de dirección macro:** EMA 50. Solo se buscan compras con el precio por encima, ventas por debajo.
2. **Gatillo de momentum:** retorno de 3 jornadas ≥ ±3%.
3. **Salida por tiempo:** liquidación a las 24 h exactas, sin stop loss ni take profit.
4. **Dimensionamiento:** riesgo nominal configurable (0.5% por defecto) escalado por ATR(14), con techo de apalancamiento 2.0x.

### Limitaciones conocidas, documentadas en la propia app

- **La app no despacha órdenes sin consentimiento explícito.** El botón de Telegram arranca
  deshabilitado y exige marcar *"He leído que estas reglas no tienen ventaja estadística
  demostrada"* antes de habilitarse. El mensaje que se publica al canal lleva el encabezado
  *"SEÑAL SIN RESPALDO ESTADÍSTICO"* y un resumen del backtest, en vez de presentar la orden
  como accionable. La decisión es del mantenedor: la funcionalidad sigue ahí, pero ya no
  vende una señal como si fuera una recomendación.
- **El dimensionamiento no acota la pérdida real.** Está calculado asumiendo que el riesgo se
  materializa a 1 ATR, pero la estrategia no tiene stop loss. Un movimiento adverso mayor
  que 1 ATR dentro de la ventana de 24 h supera con holgura el límite de riesgo mostrado.
- **La señal se evalúa sobre la vela en curso.** yfinance entrega el último día incompleto,
  así que precio, retorno 3D y señal cambian durante la jornada. El protocolo exige validar a
  las 23:59 UTC; la app lo advierte con un banner explícito. (`backtest.py` sí descarta la
  vela en curso, para que el backtest sí sea reproducible.)
- **El backtest no modela liquidez en eventos de estrés**, solo spread e impacto en régimen
  normal. Con un equilibrio de 1.2 bps, ese detalle es material.
- **Muestra estadística corta.** 9-12 años y ~550 operaciones dan poco poder estadístico:
  el intervalo de confianza del retorno medio es de ±0.3%, mayor que el propio retorno.

---

## 🤖 Validación del LSTM

- **Split temporal 80/20 sin fuga:** el `MinMaxScaler` se ajusta solo con el tramo de
  entrenamiento; las ventanas de train y test no comparten ninguna barra.
- **Early stopping** sobre `val_loss` con `restore_best_weights`, así que el slider de
  épocas es un máximo, no un objetivo.
- **Contraste honesto contra baseline:** MAE del LSTM frente a "mañana = hoy", más acierto
  direccional junto a la frecuencia de días alcistas como referencia de azar. El acierto
  direccional del baseline naive se omite a propósito: su forecast es el cierre de hoy,
  siempre con signo 0, así que medirlo solo reportaría la frecuencia de días alcistas.
- **Semillas fijas** (`SEED=7`) en `random`, NumPy y TensorFlow.

---

## 💻 Despliegue local

```bash
git clone https://github.com/okonomoclub-ilbtc/streamlit-crypto-model.git
cd streamlit-crypto-model
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit-crypto-model.py
```

La app arranca **sin configurar nada más**. Telegram es opcional; sin credenciales el
botón de despacho se deshabilita con un aviso.

### Telegram (opcional)

Crea `.streamlit/secrets.toml` (no se versiona, está en `.gitignore`) o exporta las
variables de entorno:

```toml
TELEGRAM_TOKEN = "123456:ABC..."
MAINTENANCE_PASSWORD = "tu-clave"
```

| Variable | Entorno alternativo |
|---|---|
| `TELEGRAM_TOKEN` | `TELEGRAM_TOKEN` |
| `MAINTENANCE_PASSWORD` | `MAINTENANCE_PASSWORD` |

### Validación del modelo

```bash
python validate.py BTC-USD   # o ETH-USD, SOL-USD
```

Walk-forward: reentrena cada 30 días sobre el último año y compara Ridge contra los dos
baselines. No necesita TensorFlow, así que corre en segundos.

### Backtest de las reglas de trading

```bash
python backtest.py                                  # los tres activos, toda la historia
python backtest.py BTC-USD --sensibilidad           # barrido EMA 20/50/100 x momentum ±2/3/4%
python backtest.py ETH-USD --period 5y              # acortar la ventana (y ver cómo cambia el veredicto)
python backtest.py BTC-USD --fee 0.001              # otros costes
python backtest.py BTC-USD --rapido                 # sin bootstrap, desglose anual ni barrido de costes
```

Sin lookahead: la señal se evalúa sobre el cierre de T y se ejecuta al `Open[T+1]`, con
salida a `Close[T+1]`, y la vela en curso se descarta. Cada corrida muestra retorno, baselines,
split cronológico, significación (bootstrap + t-estadístico), desglose por año, barrido de
costes con punto de equilibrio, walk-forward con reoptimización, y el baseline aleatorio
(2.000 réplicas por defecto, ajustable con `--simulaciones`), que es la comparación que
realmente importa. La pestaña 3 de la app ejecuta este mismo backtest al vuelo sobre el
activo que selecciones.

Tiempos orientativos: `--rapido` ~10 s por activo, la versión completa ~2-3 min.

### CI

`.github/workflows/ci.yml` corre en cada push y PR:

1. `backtest.py BTC-USD --rapido` — verifica que el harness sigue ejecutando.
2. `validate.py BTC-USD` — el contraste contra el baseline naive.
3. `smoke_test.py` — arranca la app en headless y comprueba que no se ha reintroducido
   ninguna cifra retirada, que el veredicto del LSTM es coherente con sus propias métricas,
   que el backtest con walk-forward corre, y que el dispatch a Telegram sigue bloqueado
   hasta marcar la casilla de lectura previa.

Es el mecanismo que impide que las cifras inventadas vuelvan a colarse sin que nadie se entere.

### Tests

```bash
python smoke_test.py
```

Script standalone (no es un módulo de pytest). Arranca la app en headless con el
framework de testing de Streamlit y comprueba la UI, el entrenamiento LSTM, la
coherencia del veredicto frente a las métricas mostradas, el backtest completo
(incluido el walk-forward), y la aritmética de dimensionamiento por ATR. Devuelve
código de salida 1 si algo falla, así que sirve tal cual en CI.

Abre red (yfinance + RSS), entrena un LSTM y ejecuta el backtest con walk-forward, así
que tarda del orden de 3 minutos con la caché tibia. Para CI, conviene comprobar solo
`python backtest.py BTC-USD --rapido --simulaciones 200`, que tarda ~15 s.

---

## 📰 Noticias

Titulares de feeds RSS públicos reales (Yahoo Finance, Decrypt, Cointelegraph), con
reintento en cascada. Si ninguno responde, la app muestra el detalle del fallo y no
muestra titulares de relleno.

---

## Licencia

GPL-3.0. Ver [LICENSE](LICENSE).

💡 *Software educativo. Ninguna cifra de este repositorio constituye una recomendación
de inversión, y no hay garantías de resultados futuros.*