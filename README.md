# Multi-Stock Machine Learning Benchmark & Portfolio Optimization

We built an end-to-end quantitative ML framework. It benchmarks deep learning against traditional financial models. We use **Walk-Forward Validation (WFV)** across a 7-asset universe to predict price movements. We then apply **Inverse Volatility Weighting** for portfolio optimization and backtesting.

---

## 🌟 Overview

We predict price movements across diverse asset classes. No lookahead bias. No data leakage. We train and evaluate **9 model architectures** across **7 equities**. We auto-select top performers using strict validation guardrails. We then allocate capital through a risk-adjusted portfolio strategy.

### 📊 Asset Universe

| Sector | Tickers |
|---|---|
| Technology | Microsoft (`MSFT`), Amazon (`AMZN`) |
| Healthcare | Eli Lilly (`LLY`) |
| Financials | JPMorgan Chase (`JPM`) |
| Consumer Staples | Procter & Gamble (`PG`) |
| Energy | ExxonMobil (`XOM`) |
| Utilities | NextEra Energy (`NEE`) |

---

## 🏗️ System Architecture & Workflow

```text
Raw Data ──► Feature Engineering ──► WFV Training ──► Guardrail Selection ──► Portfolio Backtest
```

- **Feature Engineering** — We generate technical indicators and lag features for every asset.
- **Walk-Forward Validation (WFV)** — We train and test across rolling time windows. This mirrors real trading conditions.
- **Guardrail Selection** — We score candidates on validation-adjusted R². We drop overfit models with this rule:

  ```
  Train Adj R² − Val Adj R² ≤ 0.20
  ```

- **Portfolio Allocation Engine**
  - We split assets into **Buy (Long)** and **Sell (Short)** based on predicted peaks and troughs.
  - We weight capital by **Inverse Volatility** to minimize variance.
  - We simulate execution with integer shares and leftover cash.

---

## 🚀 Model Architectures

We benchmark nine modeling techniques per asset:

| Category | Model Architecture |
|---|---|
| Deep Learning | CNN-LSTM, Conv1D-LSTM, BiLSTM, Stacked LSTM, GRU |
| Attention-Based | Temporal Transformer |
| Gradient Boosting | LightGBM, XGBoost |
| Baseline | Classical Linear Benchmarks |

---

## 📈 Performance & Output Metrics

We export every backtest result to `portfolio_output.csv`:

- **Portfolio Metrics** — Total Return, Daily ROI, Projected Annualized ROI.
- **Risk Metrics** — Annualized Volatility, Sharpe Ratio (Rf = 3.8%).
- **Execution Logs** — Long/Short flag, allocated weight, share count, price anchors, timestamps.

---

## 💻 Tech Stack & Environment

- **Environment**: Kaggle Notebooks (Dual Tesla T4 / P100 GPU)
- **Core Libraries**: `tensorflow`, `keras`, `xgboost`, `lightgbm`, `pandas`, `numpy`, `scikit-learn`
- **Version Control**: GitHub repository (`multi-stock-ml-benchmark`)

---

## 🔧 Getting Started

### Prerequisites

We need Python 3.10+ and these libraries:

```bash
pip install numpy pandas scikit-learn tensorflow xgboost lightgbm
```

### Running the Notebook

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/multi-stock-ml-benchmark.git
   ```

2. Open the notebook in Kaggle or Jupyter with GPU enabled.
3. Run the pipeline top to bottom. Results export to `portfolio_output.csv`.

---

## 📄 License

Add your license here (e.g., MIT, Apache 2.0).
