# 项目说明 (Instruction)

本文档用简明语言说明本仓库中**所有文件与代码的用途、使用方式**，并展示项目架构。随项目推进会持续更新。

---

## 一、项目概述

本项目实现**多因子风险模型与约束组合优化管理**，分为 4 个阶段：

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 数据处理和特征工程 | 已完成 |
| 2 | 机器学习预测 | 进行中 |
| 3 | 风险模型构建 | 待开始 |
| 4 | 组合优化与回测 | 待开始 |

---

## 二、项目架构（文件与目录）

```
项目根目录/
├── data/                                    # 数据存储目录（不上传 git）
│   ├── stock_data.db                        # SQLite 数据库（自动生成）
│   ├── prices.parquet                       # 日线行情 + 复权因子 + 可交易标志
│   ├── meta.parquet                         # 每日基本面 + 申万一级行业
│   ├── factors_raw.parquet                  # 原始因子值（保留 NaN）
│   ├── factors_clean.parquet               # 清洗后因子值（百分位排名，不可交易为 NaN）
│   ├── ml_alpha.parquet                     # 第二阶段输出的合成 alpha 信号（ml_analyze_main.py 生成）
│   └── index.parquet                        # CSI 300 指数每日收盘价（data_preparation_main.py 生成）
├── plots/                                   # 图表输出目录
├── src/
│   ├── data_preparation/                    # 第一阶段
│   │   ├── __init__.py
│   │   ├── data_loader.py                   # DataEngine：数据下载与读取
│   │   ├── factors.py                       # FactorEngine：因子计算（A/B/C 三组）
│   │   ├── preprocessor.py                  # FactorCleaner：因子清洗
│   │   └── data_preparation_main.py        # 第一阶段总脚本
│   ├── LightGBM/                            # 第二阶段
│   │   ├── __init__.py
│   │   ├── targets.py                       # calc_forward_return：T+d 前向收益率
│   │   ├── ml_data_prep.py                  # WalkForwardSplitter：滚动窗口 CV
│   │   ├── lgbm_model.py                    # AlphaLGBM：LightGBM 训练/预测/SHAP
│   │   └── ml_analyze_main.py              # 第二阶段总脚本
│   ├── risk_model/                          # 第三阶段（待开发）
│   └── portfolio/                           # 第四阶段
│       ├── __init__.py
│       ├── backtester.py                    # LayeredBacktester：分层回测
│       ├── net_backtester.py               # NetReturnBacktester：净收益回测（含 benchmark 超额收益）
│       ├── ic_analyzer.py                  # IC 评估：calc_ic / calc_ic_metrics / plot_ic
│       ├── optimizer.py                     # PortfolioOptimizer：cvxpy LP 凸优化求解器
│       ├── optimization_backtester.py      # OptimizationBacktester：优化组合回测（含 benchmark 超额收益）
│       └── optimization_main.py            # 第四阶段总脚本
├── .gitignore
├── requirements.txt
└── Instruction.md
```

---

## 三、各阶段 Pipeline 详解

### 第一阶段：数据处理和特征工程

#### 总体 Pipeline 流程

```
DataEngine.download_data()
  ├── Tushare: daily_price / daily_basic / adj_factor / stock_st
  ├── Tushare: index_daily      (CSI 300 日线行情，用于 IVOL 计算)
  └── Tushare: quarterly_financials  (季度财务：ROE / OCF/Rev / 净利润)
       ↓
DataEngine.load_data()
  → 返回 8 个 DataFrame
       ↓
_compute_tradable()
  → tradable_mask (MultiIndex date×code 的 bool Series)
       ↓
FactorEngine.get_all_factors()
  → MultiIndex (date, code) × 50 个原始因子 (保留 NaN/inf)
       ↓
FactorCleaner.process_all()
  Step 1: ±inf → NaN
  Step 2: NaN 填充（行业截面中位数 → 全局截面中位数 fallback）
  Step 3: 截面百分位排名 → (0, 1]
  ↓ （tradable_mask 覆写非可交易格子为 NaN）
       ↓
导出 4 张 Parquet 宽表
  prices.parquet / meta.parquet / factors_raw.parquet / factors_clean.parquet
```

#### 文件说明

| 文件 | 类 / 函数 | 说明 |
|------|-----------|------|
| `data_loader.py` | `DataEngine` | 数据下载与读取，支持 7 张 SQLite 表 |
| `factors.py` | `FactorEngine` | 39 个因子计算（A/B/C 三组），含数学工具函数库 |
| `preprocessor.py` | `FactorCleaner` | 三步清洗：inf→NaN、中位数填充、百分位排名 |
| `data_preparation_main.py` | `main()` | 第一阶段端到端脚本 |

#### 数据库表结构（SQLite: stock_data.db）

| 表名 | 主键 | 说明 |
|------|------|------|
| `daily_price` | (code, date) | 日线 OHLCV + amount（千元） |
| `daily_basic` | (code, date) | 每日扩展基本面（PE、PB、换手率等 15 字段） |
| `stock_info` | code | 股票元信息：名称、申万一级行业、上市日期 |
| `adj_factor` | (code, date) | 复权因子 |
| `stock_st` | (code, start_date) | ST/\*ST 状态历史区间 |
| `index_daily` | date | CSI 300 每日收盘价 |
| `quarterly_financials` | (code, ann_date, end_date) | 季度财务：ROE |

#### 因子清单

**A 组 — 微观结构与量价因子**（使用复权价格，共 32 个）

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `factor_amihud_illiq` | \(\frac{1}{20}\sum\frac{\|R_t\|}{\text{amount}_t}\times 10^6\) | Amihud 非流动性，值越大流动性越差 |
| `factor_ivol` | \(\text{std}(\varepsilon_t)\)，20 日滚动 OLS 残差 | 特异性波动率，对 CSI300 做回归后的残差标准差 |
| `factor_realized_skewness` | \(\text{skew}(R_t, 20\text{天})\) | 日收益率 20 日偏度（bias-corrected） |
| `factor_vol_price_corr` | \(\text{Corr}(\text{rank}(\text{close}), \text{rank}(\text{vol}), 10)\) | 量价秩相关（Spearman），10 日 |
| `factor_vol_price_corr_20d` | 同上，窗口 20 | 量价秩相关（Spearman），20 日 |
| `factor_vw_return_5d` | \(\sum(R_t \cdot V_t, 5) / \sum(V_t, 5)\) | 成交量加权收益率，5 日 |
| `factor_vw_return_10d` | 同上，窗口 10 | 成交量加权收益率，10 日 |
| `factor_vol_oscillator` | \(\text{MA}(\text{vol}, 5) / \text{MA}(\text{vol}, 20)\) | 量震荡指标：短期/中期成交量比值 |
| `factor_net_buy_proxy_5d` | \(\text{mean}\!\left(\frac{C-O}{H-L+\epsilon}\cdot V,\ 5\right)\) | 净买入代理（方向性成交），5 日 |
| `factor_net_buy_proxy_10d` | 同上，窗口 10 | 净买入代理（方向性成交），10 日 |
| `factor_momentum_1d` | \((C_t - C_{t-1}) / C_{t-1}\) | 1 日价格动量 |
| `factor_momentum_3d` | \((C_t - C_{t-3}) / C_{t-3}\) | 3 日价格动量 |
| `factor_momentum_5d` | \((C_t - C_{t-5}) / C_{t-5}\) | 5 日价格动量 |
| `factor_momentum_10d` | \((C_t - C_{t-10}) / C_{t-10}\) | 10 日价格动量 |
| `factor_momentum_20d` | \((C_t - C_{t-20}) / C_{t-20}\) | 20 日价格动量 |
| `factor_trend_strength` | \(\text{mom}_{20} / \sum(\|R_t\|, 20)\) | 趋势强度（IR 代理），接近 ±1 表示单边趋势 |
| `factor_drawdown_from_high` | \((C - \text{tsmax}(C, 60)) / \text{tsmax}(C, 60)\) | 距 60 日最高点的跌幅，恒 ≤ 0 |
| `factor_bias_5d` | \(C / \text{MA}(C, 5) - 1\) | 5 日均线乖离率 |
| `factor_bias_10d` | \(C / \text{MA}(C, 10) - 1\) | 10 日均线乖离率 |
| `factor_bias_20d` | \(C / \text{MA}(C, 20) - 1\) | 20 日均线乖离率 |
| `factor_rvol_5d` | \(\text{std}(R_t, 5)\) | 已实现波动率，5 日 |
| `factor_rvol_10d` | \(\text{std}(R_t, 10)\) | 已实现波动率，10 日 |
| `factor_rvol_20d` | \(\text{std}(R_t, 20)\) | 已实现波动率，20 日（总波动，非残差） |
| `factor_realized_kurtosis` | \(\text{kurt}(R_t, 20)\) | 20 日已实现超额峰度（bias-corrected，Fisher 定义） |
| `factor_hl_range` | \(\text{mean}((H-L)/C,\ 10)\) | 日内振幅均值，10 日滚动（流动性 / 波动代理） |
| `factor_downside_vol` | \(\text{std}(R_t[R_t < 0],\ 20)\) | 下行波动率（仅负收益日），20 日 |
| `factor_upside_vol` | \(\text{std}(R_t[R_t > 0],\ 20)\) | 上行波动率（仅正收益日），20 日 |
| `factor_gap_return` | \((O_t - C_{t-1}) / C_{t-1}\) | 隔夜收益（开盘相对前收） |
| `factor_turnover_volatility` | \(\text{std}(\text{turnover\_rate},\ 20)\) | 换手率 20 日滚动标准差 |
| `factor_distance_from_low` | \((C - \text{tsmin}(C, 60)) / \text{tsmin}(C, 60)\) | 距 60 日最低点的涨幅，恒 ≥ 0 |
| `factor_momentum_acceleration` | \(\text{mom}_{5d} - \text{mom}_{10d}\) | 动量加速度（短期减中期） |
| `factor_return_consistency` | 20 日内正收益日占比 | 收益一致性，值域 [0, 1] |

**B 组 — 基本面与估值因子**（季度数据 PIT 对齐或日频，共 4 个）

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `factor_ep` | \(1/\text{PE}\) | 盈利收益率 |
| `factor_bp` | \(1/\text{PB}\) | 账面市值比 |
| `factor_roe` | 加权平均 ROE | 来自 `fina_indicator`，按 ann_date PIT 对齐 |
| `factor_log_mv` | \(\log(\text{total\_mv})\) | 对数市值，规模因子 |

**C 组 — 截面相对特征**（共 12 个）

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `factor_industry_rel_turnover` | 换手率 − 行业截面**中位数** | 相对行业换手率偏离（差值） |
| `factor_industry_rel_bp` | BP − 行业截面**中位数** | 相对行业估值偏离（差值） |
| `factor_industry_rel_mv` | \(\log(\text{mv}) - \text{行业中位数}(\log(\text{mv}))\) | 相对行业市值偏离（差值） |
| `factor_industry_rel_ep` | EP − 行业截面**中位数** | 相对行业盈利收益率偏离（差值） |
| `factor_industry_rel_roe` | ROE − 行业截面**中位数** | 相对行业 ROE 偏离（差值） |
| `factor_ts_rel_turnover` | 换手率 / 60 日滚动**均值** | 时序相对换手率（比值） |
| `factor_ind_rel_momentum_5d` | \(\text{mom}_{5d} - \text{行业中位数}(\text{mom}_{5d})\) | 行业相对 5 日动量（差值） |
| `factor_ind_rel_momentum_10d` | \(\text{mom}_{10d} - \text{行业中位数}(\text{mom}_{10d})\) | 行业相对 10 日动量（差值） |
| `factor_ind_rel_momentum_20d` | \(\text{mom}_{20d} - \text{行业中位数}(\text{mom}_{20d})\) | 行业相对 20 日动量（差值） |
| `factor_ind_rel_turnover_ratio` | 换手率 / 行业截面**均值** | 行业相对换手率（比值，尺度不变） |
| `factor_ind_rel_vol` | \(\text{rvol}_{20d} /\text{ 行业均值}(\text{rvol}_{20d})\) | 行业相对已实现波动率（比值，以 1 为中心） |

> **差值 vs 比值**：动量相对因子使用行业中位数差，换手率/波动率相对因子使用行业均值比，两类语义不同，均独立保留。

#### 输出 Parquet 文件

| 文件 | 主键 | 内容 |
|------|------|------|
| `prices.parquet` | (trade_date, ts_code) | 原始 OHLCV + 复权因子 + tradable（bool） |
| `meta.parquet` | (trade_date, ts_code) | 15 个每日基本面字段 + 申万一级行业 |
| `factors_raw.parquet` | (trade_date, ts_code) | 48 列原始因子值，保留 NaN |
| `factors_clean.parquet` | (trade_date, ts_code) | 48 列百分位排名 (0,1]，非可交易格子为 NaN |

#### 重要特殊细节

**1. 复权逻辑（A 组价量因子）**
所有价格类因子（含 Amihud 中的 |R_t|、IVOL 中的个股收益率）使用**前复权价格**（`adj_type='forward'`），即 `P_adj = P_raw × adj_factor / adj_factor_latest`。`vol` 和 `amount` 在任何复权模式下均不调整。

**2. Amihud 的量纲处理**
数据库中 `amount` 单位为**千元**，需乘以 1000 转换为元再计算非流动性比率，结果再乘以 10⁶ 以获得可读数量级（每百万元成交量对应的价格冲击）。

**3. PIT（Point-in-Time）财务数据对齐**
季度财务因子（ROE）必须严格遵守公告日期（`ann_date`）对齐原则：对任意交易日 t，只使用满足 `ann_date ≤ t` 的最新数据（`pandas.merge_asof` 实现）。使用报告期（`end_date`）而非公告日会引入未来函数。`fina_indicator` 同一 `end_date` 可能存在多条记录，数据库存储时保留最晚的 `ann_date`（处理修正报告）。

**4. 可交易性掩码（tradable_mask）**
清洗后的因子表中，满足以下任一条件的股票-日期格子被覆写为 NaN：停牌（vol=0）、退市（close 为 NaN 或 0）、涨跌停、ST/\*ST 状态、上市未满 180 天（准新股）。下游模型 `dropna` 时自动排除。

**5. 清洗顺序：先填充再排名**
NaN 填充在百分位排名之前执行，确保每个交易日所有可交易股票都能获得有效排名值，不因财务数据缺失而丢失横截面信息。

**6. IVOL 向量化计算**
为提升性能，IVOL 的滚动 OLS 在时间维度上循环（T-d 次），但在股票截面方向**向量化**（一次批量矩阵乘法处理所有 300 只股票），避免双重循环。

---

### 第二阶段：LightGBM 机器学习预测

目录：`src/LightGBM/`

#### 总体 Pipeline 流程

```
载入 factors_clean.parquet + prices.parquet + meta.parquet
   ↓
calc_forward_return(prices, d=1)
   → forward_return = open_{T+2} / open_{T+1} - 1  （开盘到开盘，无前视偏差）
   ↓
行业中性化：ind_neutral_return = forward_return − groupby(industry, date).mean()
   （同日同行业仅1只股 → NaN，自动被 dropna 排除）
   ↓
合并因子 + ind_neutral_return + forward_return，dropna
cs_rank_return = pct_rank(ind_neutral_return)  ← 训练 target（双重中性化）
   ↓
WalkForwardSplitter(train=18m, val=3m, test=3m, embargo=1d).split()
   ↓
各 Fold：
  AlphaLGBM.train(X, cs_rank_return, early_stopping)
  → predict(X_test) → 收集预测值
   ↓
拼接所有 fold 预测 → 3 日滚动均值平滑 ml_alpha（min_periods=3）
   ↓
IC(ml_alpha, raw forward_return)  → IC 均值/标准差/ICIR + 图（ml_alpha_ic.png）
   ↓
LayeredBacktester(ml_alpha, forward_return)  → 分层净值图（ml_alpha_layered.png）
NetReturnBacktester(ml_alpha, prices)        → 净收益图（ml_alpha_net.png）
   ↓
各 fold 平均特征重要度 → 柱状图（feature_importance.png）
SHAP beeswarm（最后一折，top-10 特征，n=300 采样）→ shap_beeswarm.png
   ↓
输出文字报告 result_ml.txt
```

#### 文件说明

| 文件 | 类 / 函数 | 说明 |
|------|-----------|------|
| `src/LightGBM/targets.py` | `calc_forward_return` | 计算前向收益率（开盘到开盘，见下方执行假设说明） |
| `src/LightGBM/ml_data_prep.py` | `WalkForwardSplitter` | 滚动窗口 CV 分割器，防止时序泄漏 |
| `src/LightGBM/lgbm_model.py` | `AlphaLGBM` | LightGBM 回归器封装（early stopping + SHAP） |
| `src/LightGBM/ml_analyze_main.py` | `main()` | 第二阶段端到端执行脚本 |

#### 关键设计决策

**1. 执行假设与 forward_return 定义（无前视偏差）**

全流程统一使用**开盘到开盘**的执行假设，避免以 T 日收盘价作为入场价所产生的前视偏差：

| 时间点 | 操作 |
|--------|------|
| T 日收盘 | 计算因子、生成 alpha 信号 |
| T+1 日开盘 | 集合竞价成交（入场） |
| T+2 日开盘 | 换仓/平仓（出场） |

因此 `forward_return_T = open_{T+2} / open_{T+1} - 1`。  
`NetReturnBacktester` 和 `OptimizationBacktester` 的持仓收益同样使用 `open_wide.pct_change()`，三者收益定义完全一致，可直接比较输出结果。

**2. 训练 target vs 评估 target 分离**
- **训练**：`cs_rank_return = pct_rank(ind_neutral_return)`，去除市场 beta + 行业 beta，模型学习纯股票选择信号。
- **IC 评估 & 回测 P&L**：使用原始 `forward_return`（即开盘到开盘），反映实际可获得的绝对收益，结果更真实可比。

**2. 行业中性化逻辑**
同日同行业内，至少需要 2 只可交易股票才能计算有意义的行业均值。仅 1 只的行业赋予 NaN，由下游 `dropna` 自动排除，不做特殊处理。

**3. embargo_days = 1**
`FORWARD_DAYS=1`，因此 embargo 最小设置为 1（防止 val 集最后一行的 target 与 test 集首行的 feature 重叠）。

**4. 3 日滚动均值平滑**
预测值在股票维度滚动平均，降低日间信号噪声与组合换手率。`min_periods=3` 确保每只股票前两个预测日自动为 NaN 并被排除。

---

### 第三阶段：风险模型构建

（待开始，预留）

---

### 第四阶段：组合优化与回测

#### 当前已迁移组件

| 文件 | 类 | 说明 |
|------|----|------|
| `src/portfolio/backtester.py` | `LayeredBacktester` | 分层回测：支持全截面与行业中性两种分组模式，计算等权收益、绩效指标与累计净值图 |
| `src/portfolio/net_backtester.py` | `NetReturnBacktester` | 净收益回测：支持全截面与行业中性两种选股模式的纯多头重叠组合，含摩擦成本、换手率、盈亏平衡换手率 |
| `src/portfolio/ic_analyzer.py` | `calc_ic / calc_ic_metrics / plot_ic` | 截面 Spearman IC 评估：计算 IC 序列、均值/标准差/ICIR 指标、绘制时序图 |

#### `LayeredBacktester` 行业中性分层回测（方案 B）

`LayeredBacktester` 新增可选参数 `industry_df`，传入时自动切换为**行业中性分层**模式，与训练 target 的行业中性化设计保持对称。

**分组逻辑（三步）：**

1. **行业内排名**：在每个 `(trade_date, industry)` 截面内，按因子值的百分位秩将股票映射到 G1\~G5。采用 `ceil(pct_rank × N)` 映射，对任意股票数（≥ 2）均成立；仅 1 只股票的行业返回 NaN 并被自动排除。

2. **行业内组均值**：对每个 `(trade_date, industry, group)` 三元组，计算该格子内所有成员股票的**等权平均**前向收益率。

3. **跨行业等权合并（Plan B）**：对每个 `(trade_date, group)`，将各行业的组均值再做**等权平均**。每个行业对最终组收益的贡献相同，规避大行业（如银行）主导组合收益的问题。

**接口示例：**

```python
from backtester import LayeredBacktester

industry_df = meta_df[["trade_date", "ts_code", "industry"]].drop_duplicates()

bt = LayeredBacktester(
    final_alpha_df,       # [trade_date, ts_code, ml_alpha]
    target_flat,          # [trade_date, ts_code, forward_return]
    industry_df=industry_df,   # 新增参数，传入时启用行业中性模式
    num_groups=5,
    rf=0.03,
    forward_days=1,
    plots_dir=PLOTS_DIR,
)
perf_table = bt.run_backtest()
```

`industry_df=None`（默认值）时退化为原来的全截面 qcut 模式，保持向后兼容。

#### `NetReturnBacktester` 行业中性纯多头回测（方案 B）

`NetReturnBacktester` 同样新增可选参数 `industry_df`，传入时切换为**行业中性选股**模式，与 `LayeredBacktester` 的方案 B 保持对称。

**权重构造（两步）：**

1. **行业内选股**：在每个 `(trade_date, industry)` 截面内，按因子值百分位选出前 `top_pct`（默认 20%）的股票。少于 2 只股票的行业当日不参与选股。

2. **行业等权合成（Plan B）**：每只被选中的股票权重为：
   \[
   w_s = \frac{1}{N_{\text{ind\_with\_top}} \times N_{\text{top\_in\_industry}(s)}}
   \]
   其中 $N_{\text{ind\_with\_top}}$ 为当日至少贡献一只股票的行业数，$N_{\text{top\_in\_industry}(s)}$ 为股票 $s$ 所在行业当日被选中的股票数。整个组合权重合计为 1，每个行业贡献相同。

后续重叠权重、换手率、净收益的计算逻辑与标准模式完全相同。

**接口示例：**

```python
from net_backtester import NetReturnBacktester
import pandas as pd

index_prices = pd.read_parquet("data/index.parquet").set_index("trade_date")["close"]

nb = NetReturnBacktester(
    final_alpha_df,        # [trade_date, ts_code, ml_alpha]
    prices_df,             # [trade_date, ts_code, open, close, ...]  需包含 open 列
    industry_df=industry_df,   # 传入时启用行业中性选股
    forward_days=1,
    cost_rate=0.002,
    rf=0.03,
    plots_dir=PLOTS_DIR,
    benchmark_prices=index_prices,  # 可选；传入时生成超额收益指标+双面板图
)
summary = nb.run_backtest()
# summary 额外包含: Bench Ann Return, Excess Ann Return,
#                   Tracking Error, Information Ratio, Max Relative DD
```

#### 凸优化组合管理（第四阶段新增）

| 文件 | 类 / 入口 | 说明 |
|------|-----------|------|
| `src/portfolio/optimizer.py` | `PortfolioOptimizer` | cvxpy LP 求解器：每日求解行业中性、带换手成本的最优权重向量 |
| `src/portfolio/optimization_backtester.py` | `OptimizationBacktester` | 逐日调用优化器的纯多头回测，计算净收益与绩效指标 |
| `src/portfolio/optimization_main.py` | `main()` | 第四阶段独立入口脚本 |

##### 优化问题

每个交易日 $t$ 求解如下 LP：

$$\max_{w_t}\ w_t^\top\hat\alpha_t - \lambda\cdot\tfrac{1}{2}\|w_t - w_{t-1}\|_1$$

其中 $\hat\alpha_t = \alpha_t - \bar\alpha_t$（截面去均值后的信号）；各符号含义：

- $\hat\alpha_t$：截面去均值后的 ML alpha 信号，范围约 $[-0.5,\,0.5]$。**去均值不改变截面排序**，仅将信号中心化为 0，使 $\lambda$ 的调参量级稳定。
- $w_{t-1}$：前一日持仓权重（新股赋 0，退市股强制清仓）
- $\lambda$（`lambda_turnover`）：换手惩罚系数，**无量纲的策略偏好参数**，非交易费率。根据回测报告中的 `Avg Daily Turnover` 手动调整：
  - $\lambda = 0.05$\~$0.1$：高换手，日均换手率约 10\~20%
  - $\lambda = 0.2$\~$0.5$：适中，日均换手率约 2\~8%（推荐起始值）
  - $\lambda \ge 1.0$：持仓极稳定，信号追踪滞后
- $X_{\text{ind}}$：行业 dummy 矩阵（$n \times K$）
- $w_{\text{bench}}$：基准行业权重，每日由 `meta.parquet` 中全部 CSI 300 股票的 `total_mv` 加权计算
- $\delta = 0.01$：行业偏离容差（不可行时自动逐步放宽至 ±5%）

约束：$\sum_i w_i = 1$，$w_i \ge 0$，$w_i \le 0.05$，$|X_{\text{ind}}^\top w_t - w_{\text{bench}}| \le \delta$

**净收益计算**（独立于优化器）：$r_t^{\text{net}} = r_t^{\text{gross}} - \text{turnover}_t \times c_{\text{real}}$，其中 $c_{\text{real}} = 0.002$（实际交易费率，仅用于 P&L 扣费，与 $\lambda$ 完全分离）。

**可行性**：该问题是凸 LP（线性目标减 L1 范数 = 凹函数最大化），由 cvxpy + CLARABEL 求解，每日 <0.5 秒，全周期约 3~5 分钟。

##### 关键设计决策

1. **执行假设与收益定义（与 targets.py 保持一致）**：T 日收盘计算信号，T+1 日开盘入场，T+2 日开盘出场/换仓。回测器使用 `open_wide.pct_change()` 作为持仓期收益，与 `forward_return = open_{T+2}/open_{T+1} - 1` 完全对齐，确保优化目标与实际 P&L 使用相同的收益定义。
2. **双参数分离设计**：优化目标中的换手惩罚使用无量纲的 `lambda_turnover`（默认 0.2），P&L 扣费使用真实费率 `cost_rate`（0.002）。两者不可混用：ML alpha 是百分位秩（量级 ~0.5），若直接用 0.002 作为惩罚系数，换手项量级比 alpha 项小 250 倍，优化器实际上完全忽略换手，导致每日暴力翻仓。
3. **截面 alpha 去均值**：进入优化器前每日截面减均值，保证信号对称分布在 0 附近，`lambda_turnover` 量级具有跨时间的一致性。
4. **行业约束动态基准**：基准权重每日更新（不使用静态值），反映 CSI 300 真实行业构成变化。
5. **不可行日自动处理**：$\delta$ 依次扩大（0.01 → 0.02 → ... → 0.05），所有容差均失败时去掉行业约束求解，报告中记录发生次数。
6. **首日初始化**：$w_0 = \mathbf{0}$（空仓），第一个有效日完整买入，换手率≈100% 计入成本。
7. **基准超额收益分析**：`NetReturnBacktester` 和 `OptimizationBacktester` 均支持可选的 `benchmark_prices` 参数（CSI 300 每日收盘价 Series）。传入后会：
   - 计算超额日收益 `excess_ret = strategy_ret - bench_ret`；
   - 在 `run_backtest()` 报告中追加五项指标：`Bench Ann Return`、`Excess Ann Return`、`Tracking Error`、`Information Ratio`、`Max Relative DD`；
   - 将图表改为**双面板**：上栏绝对净值（策略蓝色 + 沪深300橙色）、下栏超额净值（绿色，基准为 1.0 水平线）并标注 IR / Tracking Error。
   - `benchmark_prices=None`（默认）时行为与原始版本完全相同（向后兼容）。
   - 基准来源：`data/index.parquet`（由 `data_preparation_main.py` 生成，内含 `[trade_date, close]` 两列）。

##### 运行方式

```bash
# 先确保已运行 ml_analyze_main.py（生成 data/ml_alpha.parquet）
python src/LightGBM/ml_analyze_main.py

# 再运行优化回测
python src/portfolio/optimization_main.py
# 输出: result_optimization.txt, plots/optimization_nav.png
```

---

## 四、推荐使用流程

```bash
# 1. 安装依赖（含 cvxpy）
pip install -r requirements.txt

# 2. 填写 Token（编辑 src/config.py）

# 3. 建库并下载数据
python - <<EOF
import sys
sys.path.insert(0, 'src/data_preparation')
from data_loader import DataEngine
engine = DataEngine()
engine.init_db()
engine.download_data()
EOF

# 4. 运行第一阶段总脚本（因子计算 + 清洗 + 导出 Parquet）
python src/data_preparation/data_preparation_main.py

# 5. 运行第二阶段总脚本（LightGBM 训练 + IC + 回测 + SHAP + 保存 ml_alpha.parquet）
python src/LightGBM/ml_analyze_main.py

# 6. 运行第四阶段总脚本（凸优化组合回测）
python src/portfolio/optimization_main.py
```

---

## 五、后续更新说明

- 新增模块或脚本时，在第二节目录树与第三节各阶段说明中补充对应条目。
- 数据库表结构变动时，更新第三节「数据库表结构」表格。
- 使用流程变化时，更新第四节。
