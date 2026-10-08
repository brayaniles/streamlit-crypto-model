# 🚀 AI Crypto Strategist & Sentinel V10 Pro

Interfaz Streamlit para auditar las **reglas de entrada** de la estrategia Sentinel V10
(EMA 50 + momentum 3D + salida por tiempo a 24 h) sobre BTC, ETH y SOL, calcular el
dimensionamiento por ATR y contrastar un modelo LSTM contra un baseline naive.

**Desarrollado por:** [@Bookbinderr-2026](https://x.com) | **Comunidad:** [QuantumTradeA Telegram](https://t.me)

---

## ⚠️ Estado real del proyecto (léelo antes que nada)

Este repositorio **no contiene un backtest de las reglas de trading**. Una versión
anterior de la app mostraba una tabla de Win Rate, Payoff, Drawdown Máximo y Factor de
Recuperación con cifras escritas a mano que el código nunca calculaba, seguida de un
dictamen de "robustez matemática impecable". Ese contenido se ha retirado porque era
inventado.

Lo que sí está medido automáticamente, y por tanto es reproducible:

| Qué | Cómo | Resultado |
|---|---|---|
| Dimensionamiento por ATR | Aritmética en la pestaña 5 | Verificado |
| Baseline naive vs modelo de precio | `validate.py` | El modelo **pierde** |
| LSTM vs baseline naive | Pestaña 2 de la app | El LSTM **pierde** |

Resultado medido con `validate.py BTC-USD` (último año fuera de muestra, walk-forward):

```
modelo            MAE    MAE %
modelo        21132.1   26.71%
naive          1295.0    1.64%   <- "mañana = hoy"
ma5            1995.1    2.52%
```

Un modelo lineal sobre features de precio es ~16x peor que decir "mañana valdrá lo
mismo que hoy". **No hay edge en la predicción del nivel del precio.** El módulo LSTM
de la pestaña 2 hace la misma comprobación y avisa cuando no supera al baseline.

En consecuencia: la app sirve para estudiar la **aritmética de gestión de riesgo** y para
**desmontar la premisa de predicción de precio**, no para operar.

---

## 🔬 Reglas implementadas (Sentinel V10)

1. **Filtro de dirección macro:** EMA 50. Solo se buscan compras con el precio por encima, ventas por debajo.
2. **Gatillo de momentum:** retorno de 3 jornadas ≥ ±3%.
3. **Salida por tiempo:** liquidación a las 24 h exactas, sin stop loss ni take profit.
4. **Dimensionamiento:** riesgo nominal configurable (0.5% por defecto) escalado por ATR(14), con techo de apalancamiento 2.0x.

### Limitaciones conocidas, documentadas en la propia app

- **El dimensionamiento no acota la pérdida real.** Está calculado asumiendo que el riesgo se
  materializa a 1 ATR, pero la estrategia no tiene stop loss. Un movimiento adverso mayor
  que 1 ATR dentro de la ventana de 24 h supera con holgura el límite de riesgo mostrado.
- **La señal se evalúa sobre la vela en curso.** yfinance entrega el último día incompleto,
  así que precio, retorno 3D y señal cambian durante la jornada. El protocolo exige validar a
  las 23:59 UTC; la app lo advierte con un banner explícito.
- **Sin costes de transacción** en ningún cálculo.

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

### Tests

```bash
python smoke_test.py
```

Script standalone (no es un módulo de pytest). Arranca la app en headless con el
framework de testing de Streamlit y comprueba 27 puntos: arranque sin secrets
configurados, ausencia de cifras y titulares fabricados, etiqueta de vela en curso,
entrenamiento LSTM, coherencia del veredicto frente a las métricas mostradas y la
aritmética de dimensionamiento por ATR. Devuelve código de salida 1 si algo falla, así
que sirve tal cual en CI.

Abre red (yfinance + RSS) y entrena un LSTM con 3 épocas, así que tarda del orden de
30 s con la caché de Streamlit tibia; el primer arranque en frío, con la descarga de
datos y la importación de TensorFlow, es más lento.

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