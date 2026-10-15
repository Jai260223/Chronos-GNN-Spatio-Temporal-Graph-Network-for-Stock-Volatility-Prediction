\# ⚡ Chronos-GNN: Spatio-Temporal Graph Network for Stock Volatility Prediction



An institutional quantitative research framework replacing trivial 1-day lagged autoregressive models (LSTMs) with \*\*Cross-Asset Graph Convolutions\*\*, \*\*Unsupervised Macro Volatility Regimes\*\*, and \*\*Triple-Barrier Trade Labeling\*\*.



\---



\## 🎯 Key Innovations Over Traditional Baselines

1\. \*\*Relational Graph Topology:\*\* Instead of modeling a single stock in isolation, this system models supply-chain shock transmission across connected market nodes (`TSM`, `AAPL`, `MSFT`, `GOOGL`, and `NVDA`).

2\. \*\*Triple-Barrier Method:\*\* Replaces non-stationary point-price forecasting with dynamic breakout classification (Take-Profit vs. Stop-Loss vs. Consolidation).

3\. \*\*Macro Regime Filter:\*\* Conditions the neural architecture on dynamic cross-asset volatility states (Low-Vol Calm vs. High-Vol Crisis).

4\. \*\*Interactive Quant Terminal:\*\* Built-in walk-forward backtesting, dynamic Sharpe/Max Drawdown metrics, and an active supplier stress-testing laboratory.



\---



\## 🏗️ Architecture



Synchronized Multi-Asset Feed (NVDA, AAPL, MSFT, GOOGL, TSM)

│

┌────────────────┴────────────────┐

▼                                 ▼

Dynamic Adjacency Matrix          Macro Volatility Filter

(Rolling Correlation > 0.35)       (Unsupervised Regime State)

│                                 │

└───────────────┬─────────────────┘

▼

Spatio-Temporal GNN + LSTM Head

│

▼

Triple-Barrier Directional Classification

\-

