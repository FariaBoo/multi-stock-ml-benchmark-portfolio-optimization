# Multi-Stock Machine Learning Benchmark & Portfolio Optimization

An end-to-end quantitative machine learning framework built to evaluate deep learning architectures against traditional financial models. The project utilizes Walk-Forward Validation (WFV) across a 7-asset market universe to predict price movements, followed by an Inverse Volatility Weighting strategy for portfolio optimization and backtesting.

---

## 🌟 Overview

Predicting price movements across diverse asset classes requires robust modeling that avoids lookahead bias and data leakage. This pipeline trains and evaluates **9 model architectures** across **7 diverse equities**, automatically selecting top-performing models using strict validation guardrails before allocating capital through a risk-adjusted portfolio strategy.

### 📊 Asset Universe
* **Technology**: Microsoft (`MSFT`), Amazon (`AMZN`)
* **Healthcare**: Eli Lilly (`LLY`)
* **Financials**: JPMorgan Chase (`JPM`)
* **Consumer Staples**: Procter & Gamble (`PG`)
* **Energy**: ExxonMobil (`XOM`)
* **Utilities**: NextEra Energy (`NEE`)

---

## 🏗️ System Architecture & Workflow
