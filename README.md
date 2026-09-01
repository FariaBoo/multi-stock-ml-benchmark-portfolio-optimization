# Multi-Stock Machine Learning Benchmark & Portfolio Optimization

This is an end-to-end quantitative ML framework we built to benchmark deep learning against traditional financial models. It uses **Walk-Forward Validation (WFV)** across a 7-asset universe to predict price movements, then applies **Inverse Volatility Weighting** for portfolio optimization and backtesting.

---

## 🌟 Overview

The pipeline predicts price movements across diverse asset classes, with no lookahead bias and no data leakage. It trains and evaluates **9 model architectures** across **7 equities**, auto-selects top performers using strict validation guardrails, then allocates capital through a risk-adjusted portfolio strategy.

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

- **Feature Engineering** — generates technical indicators and lag features for every asset.
- **Walk-Forward Validation (WFV)** — trains and tests across rolling time windows, mirroring real trading conditions.
- **Guardrail Selection** — scores candidates on validation-adjusted R² and drops overfit models with this rule:

  ```
  Train Adj R² − Val Adj R² ≤ 0.20
  ```

- **Portfolio Allocation Engine**
  - Splits assets into **Buy (Long)** and **Sell (Short)** based on predicted peaks and troughs.
  - Weights capital by **Inverse Volatility** to minimize variance.
  - Simulates execution with integer shares and leftover cash.

---

## 🚀 Model Architectures

Nine modeling techniques, benchmarked per asset:

| Category | Model Architecture |
|---|---|
| Deep Learning | CNN-LSTM, Conv1D-LSTM, BiLSTM, Stacked LSTM, GRU |
| Attention-Based | Temporal Transformer |
| Gradient Boosting | LightGBM, XGBoost |
| Baseline | Classical Linear Benchmarks |

---

## 📈 Performance & Output Metrics

Every backtest result exports to `portfolio_output.csv`:

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

Python 3.10+ and these libraries:

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

