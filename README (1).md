# 📦 Inventory Optimization Elite

> **Two-Echelon Inventory Optimization with Multi-Model Demand Forecasting, Monte Carlo Simulation, and Risk Pooling Analysis**

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
  - [Demand Forecasting](#demand-forecasting)
  - [Two-Echelon Inventory Policy](#two-echelon-inventory-policy)
  - [Risk Pooling](#risk-pooling)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Policy Optimization](#policy-optimization)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Output Figures](#output-figures)
- [Backtest Validation](#backtest-validation)
- [Project Structure](#project-structure)

---

## Overview

**Inventory Optimization Elite** is a production-grade, end-to-end supply chain optimization system built on the Walmart Sales dataset. It models a real-world **two-echelon retail network** — one central warehouse supplying 45 stores — and determines optimal reorder points and order quantities using a pipeline of demand forecasting, stochastic simulation, and combinatorial grid search.

The system demonstrates a **51.5% cost reduction** over a naive single-echelon baseline while maintaining a **97.5% fill rate**, primarily driven by centralized risk pooling of safety stock.

**Dataset:** Walmart Weekly Sales — 45 stores, 143 weeks (February 2010 – October 2012)

---

## Key Results

| Metric | Value |
|---|---|
| **Winner Forecast Model** | Prophet |
| **Two-Echelon Total Cost** | ₹1,538,481,171 |
| **Service Level (Fill Rate)** | 97.5% |
| **Single-Echelon Baseline Cost** | ₹3,174,690,779 |
| **Cost Saving vs Baseline** | ₹1,636,209,609 **(51.5%)** |
| **Risk Pooling Safety Stock Reduction** | **54.1%** |
| **Safety Stock Cost Saving / Week** | ₹8,102,715 |
| **WH Reorder Point (s\*)** | 6,020,780 units (≈ 3.9 weeks of aggregate demand) |
| **WH Order Quantity (Q\*)** | 2,340,203 units (≈ 1.5 weeks of aggregate demand) |
| **Store Reorder Point (s\* per store)** | 34,669 units |
| **Store Order Quantity (Q\* per store)** | 69,339 units |

---

## System Architecture

The supply chain modelled is a classic **two-echelon (s, Q) system**:

```
  Supplier
     │
     ▼  (Lead time: l_wh weeks, stochastic)
  Warehouse  ──── reorder at s_wh, order Q_wh
     │
     ├──▶ Store 1  ──── reorder at s_store, order Q_store
     ├──▶ Store 2
     │    ...
     └──▶ Store 45
              (Lead time: l_store weeks)
```

- **Echelon 1 — Warehouse:** Aggregates demand from all 45 stores. Replenished from an external supplier with stochastic lead time.
- **Echelon 2 — Stores (×45):** Each store independently follows an (s, Q) reorder policy. Replenished from the warehouse.

---

## Methodology

### Demand Forecasting

Three forecasting approaches are evaluated and benchmarked on a 75/25 train/test split:

| Model | Description | Residual σ (Aggregate) |
|---|---|---|
| **Prophet** | Facebook Prophet with weekly + yearly seasonality | 130,899 units |
| **XGBoost** | Gradient boosted trees with lag/calendar features | 104,585 units |
| **Rolling Mean** | Trailing rolling average (baseline) | 161,675 units |

Each model produces a **full aggregate demand forecast** (train + test horizon) used as the mean demand input to the simulation engine.

### Two-Echelon Inventory Policy

Both echelons use a continuous-review **(s, Q) policy** — also known as a reorder-point / order-quantity policy:

- When inventory position drops to or below **s** (reorder point), an order of size **Q** is placed.
- Reorder points incorporate **safety stock** computed from a Normal approximation:

```
ss_wh = z_α × √( l_wh × σ²_agg + μ²_agg × σ²_l )

ss_store = z_α × σ_store × √l_store
```

where `z_α` corresponds to the target service level (97.5%).

### Risk Pooling

A core motivation for the two-echelon structure is the **risk pooling effect**: centralizing safety stock at the warehouse reduces total safety stock significantly compared to holding it independently at each of the 45 stores.

| Safety Stock Mode | Units |
|---|---|
| Decentralised (45 stores, independent) | 1,242,820 |
| Centralised (warehouse) | 570,019 |
| **Saving** | **672,800 (54.1%)** |

This is explained by the square-root law of risk pooling: the standard deviation of aggregate demand scales as `√N` of per-store demand, not linearly with `N`.

### Monte Carlo Simulation

The simulation engine runs **300 independent replications** per policy configuration. Each replication:

1. Initializes warehouse and store inventories at the given (s, Q) values.
2. Runs a warm-up period to reach steady state.
3. Simulates the test horizon week-by-week: generating stochastic demand, processing orders, computing holding costs, stockout costs, and ordering costs.
4. Accumulates fill rate and total cost per run.

The inner simulation loop is **Numba JIT-compiled** (`@njit`) for high-throughput numerical performance. Parallelism across policy grid points uses Python's `multiprocessing` module (12 cores).

### Policy Optimization

Grid search over (s_wh, Q_wh, s_store, Q_store) parameter space — **480 combinations per forecast model** — evaluated via the Monte Carlo engine. The best policy minimizes expected total cost subject to the service level constraint.

```
minimize  E[Holding Cost + Stockout Cost + Ordering Cost]
subject to  Fill Rate ≥ 97.5%
```

---

## Configuration

All model parameters are centralized in the `InventoryConfig` dataclass for full reproducibility:

| Parameter | Value | Description |
|---|---|---|
| `unit_price_usd` | $30.00 | Unit price in USD |
| `inr_per_usd` | 83.5 | USD → INR conversion |
| `annual_carry_rate` | 25% | Annual holding cost as fraction of unit value |
| `stockout_margin` | 5% | Lost margin per stockout unit |
| `holding_cost_store` | ₹12.04 / unit / week | Derived store holding cost |
| `holding_cost_wh` | ₹7.23 / unit / week | Derived warehouse holding cost (60% of store rate) |
| `stockout_cost` | ₹125.25 / unit | Derived stockout penalty |
| `l_total` | 4 weeks | Total supply chain lead time |
| `l_wh` | 2 weeks | Warehouse replenishment lead time |
| `l_store` | 2 weeks | Store replenishment lead time |
| `l_wh_std` | 0.5 weeks | Lead time variability (std dev) |
| `z_alpha` | 1.96 | z-score for 97.5% service level |
| `sigma_buffer_ratio` | 3.0× | Safety multiplier on aggregate σ |
| `n_sims` | 300 | Monte Carlo replications |
| `warm_up_weeks` | 52 | Steady-state warm-up period |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations and stochastic simulation |
| `pandas` | Data loading and manipulation |
| `matplotlib` | All visualizations (11 figures) |
| `prophet` | Facebook Prophet time series forecasting |
| `xgboost` | Gradient boosted demand forecasting |
| `numba` | JIT compilation of simulation inner loop |
| `scipy` | Statistical distributions and KDE |
| `multiprocessing` | Parallel policy grid search |
| `dataclasses` | Configuration management |
| `cmdstanpy` | Stan backend for Prophet |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/inventory-optimization-elite.git
cd inventory-optimization-elite

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install numpy pandas matplotlib prophet xgboost numba scipy

# 4. Install CmdStan (required by Prophet)
pip install cmdstanpy
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

---

## Usage

1. **Prepare the dataset** — Place the Walmart Sales CSV at the path specified in `InventoryConfig.data_path`:

```python
data_path: str = "/path/to/Walmart_Sales.csv"
```

The CSV must contain at minimum the columns: `Store`, `Date` (format `%d-%m-%Y`), `Weekly_Sales`.

2. **Open and run the notebook** — Execute all cells sequentially in Jupyter:

```bash
jupyter notebook inventory_optimization_elite_.ipynb
```

3. **Outputs** — All figures are saved to the `outputs/` directory automatically. The final summary is printed to the notebook output.

> **Runtime note:** The full optimization sweep (480 combos × 3 models × 300 sims) runs in approximately **10–20 minutes** on a 12-core machine. Numba JIT compilation adds a one-time warm-up cost on first run.

---

## Output Figures

| File | Description |
|---|---|
| `fig0_dashboard.png` | Executive KPI card summary |
| `fig1_forecast.png` | Demand forecast vs actuals (all 3 models, test period) |
| `fig2_architecture.png` | Two-echelon system flow diagram |
| `fig3_risk_pooling.png` | Safety stock reduction deep dive (3-panel) |
| `fig4_model_comparison.png` | Cost, service level, and cost-breakdown comparison |
| `fig5_wh_trajectory.png` | Warehouse inventory trajectories — mean ± 90% CI, 300 sims |
| `fig6_monte_carlo.png` | Monte Carlo cost distributions with KDE overlays |
| `fig7_pareto.png` | Cost vs. service level Pareto frontier |
| `fig8_tornado.png` | Sensitivity / tornado chart (parameter perturbation analysis) |
| `fig9_store_heatmap.png` | Store-level normalized weekly demand heatmap (45 stores × 143 weeks) |
| `fig10_feature_importance.png` | XGBoost feature importances (bar + gain decomposition) |

---

## Backtest Validation

A structured sanity-check suite validates the winning policy against theoretical expectations:

| Check | Criterion | Result |
|---|---|---|
| A — σ ratio | Expect 3.0× buffer | ✅ 3.00× |
| B — Inventory weeks on hand | Reasonable coverage | ✅ 2.0 weeks |
| C — Theoretical vs MC service level | Δ < 2% | ✅ Δ = 0.4% |
| D — Holding cost share | 40–70% of total | ⚠️ 87% (high but consistent with high service level) |
| D — Stockout cost share | 5–25% | ✅ 12% |
| D — Ordering cost share | 1–5% | ✅ 1% |
| D — Annual inventory / revenue | 5–15% | ⚠️ 1.1% (low-price Walmart SKU mix) |

---

## Project Structure

```
inventory-optimization-elite/
│
├── inventory_optimization_elite_.ipynb   # Main Jupyter notebook
├── README.md                             # This file
│
├── outputs/                              # Generated figures (created at runtime)
│   ├── fig0_dashboard.png
│   ├── fig1_forecast.png
│   ├── fig2_architecture.png
│   ├── fig3_risk_pooling.png
│   ├── fig4_model_comparison.png
│   ├── fig5_wh_trajectory.png
│   ├── fig6_monte_carlo.png
│   ├── fig7_pareto.png
│   ├── fig8_tornado.png
│   ├── fig9_store_heatmap.png
│   └── fig10_feature_importance.png
│
└── data/
    └── Walmart_Sales.csv                 # Source dataset 
```

---


*Built with Python · Prophet · XGBoost · Numba · Matplotlib*
## Screenshots

![App Screenshot](https://dummyimage.com/468x300?text=App+Screenshot+Here)

