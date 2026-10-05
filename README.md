# 🚀 AI Crypto Strategist & Dictaminador Sentinel V10 Pro

Entorno web cuantitativo avanzado para la auditoría, análisis estacional y proyección predictiva de criptoactivos (BTC, ETH, SOL) en tiempo real. Este ecosistema de simulación fue diseñado bajo un estándar riguroso para eliminar el sesgo de la fe y validar la robustez matemática de las reglas lógicas antes de arriesgar capital en vivo.

**Desarrollado por:** [@Bookbinderr-2026](https://x.com) | **Comunidad Oficial:** [QuantumTradeA Telegram](https://t.me)

---

## 🔬 Arquitectura Core: Sentinel V10 Pro (Copy Trading Edition)
A diferencia de los sistemas minoristas sobreajustados, esta versión explota la microestructura e ineficiencias del mercado institucional moderno (Post-ETFs) mediante un extractor puro de momentum intradiario:

1. **Filtro de Dirección Macro:** Media Móvil Exponencial (**EMA 50**). Define el sesgo institucional; solo se buscan compras en expansión alcista y ventas en compresión bajista.
2. **Gatillo de Momentum Absoluto:** Retorno de 3 jornadas (**Retorno 3D ≥ ±3%**). Identifica puntos de fatiga estructural y sobreextensión del precio.
3. **Cinturón de Seguridad Rígido:** Liquidación forzada por tiempo a las **24 horas exactas** (1 vela diaria). El bot elimina stop loss tradicionales y targets flotantes; mitiga barridas de liquidez institucional saliendo rápido del mercado.
4. **Gestión Monetaria Eficiente:** Riesgo controlado al **0.5% del balance** indexado dinámicamente mediante la volatilidad real del **ATR(14)**. Techo de apalancamiento seguro limitado a 2.0x.

---

## 🤖 Componentes de Inteligencia Artificial y Estabilización Backend
El sistema integra un motor predictivo basado en Redes Neuronales Recurrentes (**LSTM**) optimizado bajo metodologías de validación científica:

* **Split Temporal Antifuga (80/20):** El `MinMaxScaler` se ajusta exclusivamente con el tramo de entrenamiento, impidiendo que el modelo "vea" precios o mínimos futuros del tramo de testeo.
* **Reproducibilidad Matemática:** Enforzamiento de semillas fijas (`SEED=7`) en NumPy y TensorFlow-CPU para garantizar que las proyecciones de 7 días sean idénticas y fidedignas en cualquier servidor.
* **Filtro Informativo:** Conexión directa mediante canal RSS automatizado hacia Yahoo Finance para el monitoreo de titulares macroeconómicos globales.
* **Webhook de Alertas:** Despacho automatizado de fichas de órdenes recomendadas hacia canales públicos de Telegram mediante pasarela API.

---

## 💻 Instrucciones para Despliegue Local

1. Clona este repositorio de forma local en tu terminal de trabajo:
   ```bash
   git clone https://github.com
   cd tu-repositorio
   ```

2. Instala el paquete de dependencias estandarizado:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la interfaz gráfica interactiva de Streamlit:
   ```bash
   streamlit run app.py
   ```

---
💡 **Aviso Legal de Investigación:** *Este software fue construido con fines puramente educativos e investigativos bajo la metodología walk-forward de la ingeniería cuantitativa. Los rendimientos pasados presentados en la tabla de robustez no garantizan retornos futuros.*
