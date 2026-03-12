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
  → MultiIndex (date, code) × 55 个原始因子 (保留 NaN/inf)
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

| 文件                       | 类 / 函数       | 说明                                          |
| -------------------------- | --------------- | --------------------------------------------- |
| `data_loader.py`           | `DataEngine`    | 数据下载与读取，支持 7 张 SQLite 表           |
| `factors.py`               | `FactorEngine`  | 39 个因子计算（A/B/C 三组），含数学工具函数库 |
| `preprocessor.py`          | `FactorCleaner` | 三步清洗：inf→NaN、中位数填充、百分位排名     |
| `data_preparation_main.py` | `main()`        | 第一阶段端到端脚本                            |

#### 数据库表结构

| 表名                   | 主键                       | 说明                                       |
| ---------------------- | -------------------------- | ------------------------------------------ |
| `daily_price`          | (code, date)               | 日线 OHLCV + amount（千元）                |
| `daily_basic`          | (code, date)               | 每日扩展基本面（PE、PB、换手率等 15 字段） |
| `stock_info`           | code                       | 股票元信息：名称、申万一级行业、上市日期   |
| `adj_factor`           | (code, date)               | 复权因子                                   |
| `stock_st`             | (code, start_date)         | ST/\*ST 状态历史区间                       |
| `index_daily`          | date                       | CSI 300 每日收盘价                         |
| `quarterly_financials` | (code, ann_date, end_date) | 季度财务：ROE                              |

#### 因子清单

**A 组 — 微观结构与量价因子**（使用复权价格，共 34 个）

| 因子                           | 公式                                                         | 说明                                         |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| `factor_amihud_illiq`          | $\frac{1}{20}\sum\frac{\|R_t\|}{\text{amount}_t}\times 10^6$ | Amihud 非流动性，值越大流动性越差            |
| `factor_ivol`                  | $\text{std}(\varepsilon_t)$，20 日滚动 OLS 残差              | 特异性波动率，对 CSI300 做回归后的残差标准差 |
| `factor_beta`                  | 20 日滚动 OLS 回归斜率                                       | 对 CSI300 的市场 beta，与 IVOL 同回归        |
| `factor_realized_skewness`     | $\text{skew}(R_t, 20\text{ days})$                           | 日收益率 20 日偏度（bias-corrected）         |
| `factor_vol_price_corr`        | $\text{Corr}(\text{rank}(\text{close}), \text{rank}(\text{vol}), 10)$ | 量价秩相关（Spearman），10 日                |
| `factor_vol_price_corr_20d`    | $\text{Corr}(\text{rank}(\text{close}), \text{rank}(\text{vol}), 20)$ | 量价秩相关（Spearman），20 日                |
| `factor_vw_return_5d`          | $\sum(R_t \cdot V_t, 5) / \sum(V_t, 5)$                      | 成交量加权收益率，5 日                       |
| `factor_vw_return_10d`         | $\sum(R_t \cdot V_t, 10) / \sum(V_t, 10)$                    | 成交量加权收益率，10 日                      |
| `factor_vol_oscillator`        | $\text{MA}(\text{vol}, 5) / \text{MA}(\text{vol}, 20)$       | 量震荡指标：短期/中期成交量比值              |
| `factor_volume_ratio`          | volume_ratio（daily_basic）                                  | 量比：当日成交量 / 5 日均量                  |
| `factor_net_buy_proxy_5d`      | $\text{mean}\!\left(\frac{C-O}{H-L+\epsilon}\cdot V,\ 5\right)$ | 净买入代理（方向性成交），5 日               |
| `factor_net_buy_proxy_10d`     | $\text{mean}\!\left(\frac{C-O}{H-L+\epsilon}\cdot V,\ 10\right)$ | 净买入代理（方向性成交），10 日              |
| `factor_momentum_1d`           | $(C_t - C_{t-1}) / C_{t-1}$                                  | 1 日价格动量                                 |
| `factor_momentum_3d`           | $(C_t - C_{t-3}) / C_{t-3}$                                  | 3 日价格动量                                 |
| `factor_momentum_5d`           | $(C_t - C_{t-5}) / C_{t-5}$                                  | 5 日价格动量                                 |
| `factor_momentum_10d`          | $(C_t - C_{t-10}) / C_{t-10}$                                | 10 日价格动量                                |
| `factor_momentum_20d`          | $(C_t - C_{t-20}) / C_{t-20}$                                | 20 日价格动量                                |
| `factor_trend_strength`        | $\text{mom}_{20} / \sum(\|R_t\|, 20)$                        | 趋势强度（IR 代理），接近 ±1 表示单边趋势    |
| `factor_drawdown_from_high`    | $(C - \text{tsmax}(C, 60)) / \text{tsmax}(C, 60)$            | 距 60 日最高点的跌幅，恒 ≤ 0                 |
| `factor_bias_5d`               | $C / \text{MA}(C, 5) - 1$                                    | 5 日均线乖离率                               |
| `factor_bias_10d`              | $C / \text{MA}(C, 10) - 1$                                   | 10 日均线乖离率                              |
| `factor_bias_20d`              | $C / \text{MA}(C, 20) - 1$                                   | 20 日均线乖离率                              |
| `factor_rvol_5d`               | $\text{std}(R_t, 5)$                                         | 已实现波动率，5 日                           |
| `factor_rvol_10d`              | $\text{std}(R_t, 10)$                                        | 已实现波动率，10 日                          |
| `factor_rvol_20d`              | $\text{std}(R_t, 20)$                                        | 已实现波动率，20 日（总波动，非残差）        |
| `factor_realized_kurtosis`     | $\text{kurt}(R_t, 20)$                                       | 20 日已实现超额峰度（bias-corrected）        |
| `factor_hl_range`              | $\text{mean}((H-L)/C,\ 10)$                                  | 日内振幅均值，10 日滚动（流动性 / 波动代理） |
| `factor_downside_vol`          | $\text{std}(R_t[R_t < 0],\ 20)$                              | 下行波动率（仅负收益日），20 日              |
| `factor_upside_vol`            | $\text{std}(R_t[R_t > 0],\ 20)$                              | 上行波动率（仅正收益日），20 日              |
| `factor_gap_return`            | $(O_t - C_{t-1}) / C_{t-1}$                                  | 隔夜收益（开盘相对前收）                     |
| `factor_turnover_volatility`   | $\text{std}(\text{turnover}\_  \text{rate},\ 20)$            | 换手率 20 日滚动标准差                       |
| `factor_distance_from_low`     | $(C - \text{tsmin}(C, 60)) / \text{tsmin}(C, 60)$            | 距 60 日最低点的涨幅，恒 ≥ 0                 |
| `factor_momentum_acceleration` | $\text{mom}_ {5d} - \text{mom}_ {10d}$                       | 动量加速度（短期减中期）                     |
| `factor_return_consistency`    | 20 日内正收益日占比                                          | 收益一致性，值域 [0, 1]                      |

**B 组 — 基本面与估值因子**（季度数据 PIT 对齐或日频，共 7 个）

| 因子                   | 公式                     | 说明                                        |
| ---------------------- | ------------------------ | ------------------------------------------- |
| `factor_ep`            | $1/\text{PE}$            | 盈利收益率                                  |
| `factor_bp`            | $1/\text{PB}$            | 账面市值比                                  |
| `factor_roe`           | 加权平均 ROE             | 来自 `fina_indicator`，按 ann_date PIT 对齐 |
| `factor_log_mv`        | $\log(\text{total}\_ \text{mv})$ | 对数市值，规模因子                          |
| `factor_ps`            | $1/\text{ps ttm}$       | 市销率倒数（daily_basic）                   |
| `factor_dv_ratio`      | dv_ratio                 | 股息率（daily_basic）                       |
| `factor_circ_mv_ratio` | circ_mv / total_mv       | 流通市值占比                                |

**C 组 — 截面相对特征**（共 12 个）

| 因子                            | 公式                                                       | 说明                                      |
| ------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| `factor_industry_rel_turnover`  | 换手率 − 行业截面**中位数**                                | 相对行业换手率偏离（差值）                |
| `factor_industry_rel_bp`        | BP − 行业截面**中位数**                                    | 相对行业估值偏离（差值）                  |
| `factor_industry_rel_mv`        | $\log(\text{mv}) - \text{行业中位数}(\log(\text{mv}))$     | 相对行业市值偏离（差值）                  |
| `factor_industry_rel_ep`        | EP − 行业截面**中位数**                                    | 相对行业盈利收益率偏离（差值）            |
| `factor_industry_rel_roe`       | ROE − 行业截面**中位数**                                   | 相对行业 ROE 偏离（差值）                 |
| `factor_ts_rel_turnover`        | 换手率 / 60 日滚动**均值**                                 | 时序相对换手率（比值）                    |
| `factor_ind_rel_momentum_5d`    | $\text{mom}_ {5d} - \text{行业中位数}(\text{mom}_ {5d})$   | 行业相对 5 日动量（差值）                 |
| `factor_ind_rel_momentum_10d`   | $\text{mom}_ {10d} - \text{行业中位数}(\text{mom}_ {10d})$ | 行业相对 10 日动量（差值）                |
| `factor_ind_rel_momentum_20d`   | $\text{mom}_ {20d} - \text{行业中位数}(\text{mom}_ {20d})$ | 行业相对 20 日动量（差值）                |
| `factor_ind_rel_turnover_ratio` | 换手率 / 行业截面**均值**                                  | 行业相对换手率（比值，尺度不变）          |
| `factor_ind_rel_vol`            | $\text{rvol}_ {20d} /\text{ 行业均值}(\text{rvol}_ {20d})$ | 行业相对已实现波动率（比值，以 1 为中心） |

**15 alpha 因子 — 来自 WorldQuant Alpha101**


| <span style="display:inline-block;width:70px">因子</span> | <span style="display:inline-block;width:300px">公式</span>   | <span style="display:inline-block;width:300px">说明</span> |
| --------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| `alpha001`                                                | `rank(ts_argmax(signedpower((ret<0)?stddev(ret,20):close, 2), 5)) - 0.5` | 负收益时用波动率替代收盘价，对5日最高值的时序位置排名      |
| `alpha003`                                                | `-1 * correlation(rank(open), rank(volume), 10)`             | 开盘价排名与成交量排名的滚动相关性取反                     |
| `alpha006`                                                | `-1 * correlation(open, volume, 10)`                         | 开盘价与成交量的滚动相关性取反                             |
| `alpha012`                                                | `sign(delta(volume, 1)) * (-1 * delta(close, 1))`            | 成交量方向乘以收盘价变动反向                               |
| `alpha038`                                                | `(-1 * rank(ts_rank(close, 10))) * rank(close/open)`         | 近期高位且高涨幅的股票做空                                 |
| `alpha040`                                                | `(-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)` | 高价波动率乘以高价-成交量相关性                            |
| `alpha041`                                                | `sqrt(high * low) - vwap`                                    | 高低价几何均值与成交均价之差（精确 vwap）                  |
| `alpha042`                                                | `rank(vwap - close) / rank(vwap + close)`                    | 收盘价相对 vwap 的位置比率（delay-0 均值回归）             |
| `alpha054`                                                | `(-1 * (low-close) * open^5) / ((low-high) * close^5)`       | 基于开/收/高/低价五次幂的日内动量                          |
| `alpha072`                                                | `rank(decay_linear(corr((H+L)/2, adv40, 8), 10)) / rank(decay_linear(corr(ts_rank(vwap,3), ts_rank(vol,18), 6), 2))` | 中价-量相关性与 vwap-量 ts_rank 相关性之比                 |
| `alpha088`                                                | `min(rank(decay_linear((rank(O)+rank(L))-(rank(H)+rank(C)),8)), ts_rank(decay_linear(corr(ts_rank(C,8),ts_rank(adv60,20),8),6),2))` | 价格结构排名差与成交量相关性的最小值                       |
| `alpha094`                                                | `signedpower(rank(vwap - ts_min(vwap,11)), ts_rank(corr(ts_rank(vwap,19), ts_rank(adv60,4), 18), 2)) * -1` | vwap 距历史低点的幂次排名因子                              |
| `alpha098`                                                | `rank(decay_linear(corr(vwap,sum(adv5,26),4),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(O),rank(adv15),20),8),6),8))` | vwap-量相关性与开盘价-量相关性低点时序排名之差             |
| `alpha101`                                                | `(close - open) / (high - low + 0.001)`                      | 日内动量：价格区间归一化的涨跌幅                           |
| `alpha_5_day_reversal`                                    | `(close - delay(close, 5)) / delay(close, 5)`                | 5日收盘价的反转因子                                        |

#### 输出 Parquet 文件

| 文件                    | 主键                  | 内容                                     |
| ----------------------- | --------------------- | ---------------------------------------- |
| `prices.parquet`        | (trade_date, ts_code) | 原始 OHLCV + 复权因子 + tradable（bool） |
| `meta.parquet`          | (trade_date, ts_code) | 15 个每日基本面字段 + 申万一级行业       |
| `factors_raw.parquet`   | (trade_date, ts_code) | 原始因子值，保留 NaN                     |
| `factors_clean.parquet` | (trade_date, ts_code) | 百分位排名 (0,1]，非可交易格子为 NaN     |

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

| 文件                              | 类 / 函数             | 说明                                                         |
| --------------------------------- | --------------------- | ------------------------------------------------------------ |
| `src/LightGBM/targets.py`         | `calc_forward_return` | 计算前向收益率（开盘到开盘，见下方执行假设说明）             |
| `src/LightGBM/ml_data_prep.py`    | `WalkForwardSplitter` | 滚动/扩展窗口 CV 分割器，防止时序泄漏。支持 `expanding=False`（固定长度滚动）与 `expanding=True`（训练窗口扩展，充分利用历史） |
| `src/LightGBM/lgbm_model.py`      | `AlphaLGBM`           | LightGBM 回归器封装（early stopping + SHAP）                 |
| `src/LightGBM/ml_analyze_main.py` | `main()`              | 第二阶段端到端执行脚本                                       |

#### 关键设计决策

**1. 执行假设与 forward_return 定义（无前视偏差）**

全流程统一使用**开盘到开盘**的执行假设，避免以 T 日收盘价作为入场价所产生的前视偏差：

| 时间点     | 操作                      |
| ---------- | ------------------------- |
| T 日收盘   | 计算因子、生成 alpha 信号 |
| T+1 日开盘 | 集合竞价成交（入场）      |
| T+2 日开盘 | 换仓/平仓（出场）         |

因此 `forward_return_T = open_{T+2} / open_{T+1} - 1`。 

`NetReturnBacktester` 和 `OptimizationBacktester` 的持仓收益同样使用 `open_wide.pct_change()`，三者收益定义完全一致，可直接比较输出结果。

**2. 训练 target vs 评估 target 分离**

- **训练**：`cs_rank_return = pct_rank(ind_neutral_return)`，去除市场 beta + 行业 beta，模型学习纯股票选择信号。
- **IC 评估 & 回测 P&L**：使用原始 `forward_return`（即开盘到开盘），反映实际可获得的绝对收益，结果更真实可比。

**2. 行业中性化逻辑**
同日同行业内，至少需要 2 只可交易股票才能计算有意义的行业均值。仅 1 只的行业赋予 NaN，由下游 `dropna` 自动排除，不做特殊处理。

**3. embargo_days = 1**
`FORWARD_DAYS=1`，因此 embargo 最小设置为 1（防止 val 集最后一行的 target 与 test 集首行的 feature 重叠）。

**4. Alpha 平滑（可选）**
预测值在股票维度可选用两种平滑方式，顺序为：先滚动均值，再 EMA。配置项 `ALPHA_ROLLING_WINDOW`（默认 1=关闭）、`ALPHA_EMA_BETA`（默认 1=关闭）。滚动均值：`window>1` 时按股票时间序列做 rolling mean，`min_periods=window`。EMA：`alpha'_t = beta * alpha_t + (1-beta) * alpha'_{t-1}`，`beta<1` 时启用。

**5. WalkForwardSplitter 两种模式**

- **滚动窗口**（`expanding=False`，默认）：训练集固定长度（如 24m），每次 fold 整体向前滑动，适合市场结构变化较快的场景。
- **扩展窗口**（`expanding=True`）：训练集从数据起点开始，随 fold 增长，充分利用历史数据，适合因子相对稳定的场景。可选 `step_months` 控制每 fold 前进步长（默认等于 `test_months`）。

---

### 第三阶段：多因子风险模型构建

#### 总体 Pipeline 流程

```
载入 prices.parquet + meta.parquet + index.parquet
   ↓
RiskFactorEngine.compute()
  ├── Size      = log(total_mv)
  ├── Beta      = 60 日滚动 CAPM beta（对 CSI 300 OLS）
  ├── Momentum  = close_{t-21} / close_{t-252} - 1（跳过近 1 月避免反转）
  ├── Volatility= std(20 日日收益率)
  ├── Value     = EP = 1/PE（PE<=0 置 NaN）
  └── Industry  = 申万一级行业哑变量（one-hot）
  ↓ （各风格因子截面 winsorize ±3σ + z-score 标准化）
→ 输出 data/risk_exposure.parquet
   ↓
CovarianceEstimator.compute()
  Step 1: 每日 WLS 截面回归（权重 = sqrt(total_mv)）
    R_t = X_t f_t + ε_t
    → 因子收益 f_t（K 维），残差 ε_t（n_t 维）
  Step 2: 滚动 60 日因子协方差
    F_t = sample_cov(f_{t-59:t})，加岭正则化 + Cholesky 分解
    → 输出 L_t^T（上三角，K×K）→ data/risk_cov_F.parquet
  Step 3: 滚动 60 日个股残差方差
    Δ_{ii,t} = var(ε_{i,t-59:t})，下界 = (0.5%)^2
    历史 < 30 日时用截面中位数填充
    → 输出 sqrt(Δ_{ii}) → data/risk_delta.parquet
   ↓
RiskModelValidator（RUN_VALIDATION=True 时）
   → 沪深 300 市值加权组合：预测方差 w'Σw vs 已实现方差（20 日滚动）
   → 输出 plots/risk_model_validation.png、result_risk_validation.txt
```

#### 文件说明

| 文件                               | 类 / 函数             | 说明                                                 |
| ---------------------------------- | --------------------- | ---------------------------------------------------- |
| `risk_model/risk_factor_engine.py` | `RiskFactorEngine`    | 计算 5 风格因子 + 行业哑变量，截面 z-score 标准化    |
| `risk_model/cov_estimator.py`      | `CovarianceEstimator` | WLS 截面回归、滚动 60 日 F_t Cholesky 分解、Δ_t 估计 |
| `risk_model/risk_model_validator.py` | `RiskModelValidator` | 预测 vs 已实现波动率验证，输出 R²、相关系数、偏差比、RMSE |
| `risk_model/risk_model_main.py`    | `main()`              | 第三阶段端到端脚本，输出三个 parquet 文件，可选执行验证 |

#### 输出 Parquet 文件

| 文件                    | 格式                                                         | 内容                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `risk_exposure.parquet` | 长表 (trade_date, ts_code, size, beta, momentum, volatility, value, ind_*) | 每日每股因子暴露矩阵 X_t；风格因子已 z-score 标准化，行业哑变量 0/1 |
| `risk_cov_F.parquet`    | 长表 (trade_date, f_i, f_j, value)                           | Cholesky 因子 L_t^T 的每个元素；重建方法：pivot(f_i, f_j) → K×K 矩阵 |
| `risk_delta.parquet`    | 长表 (trade_date, ts_code, delta_std)                        | 个股特异性风险标准差 sqrt(Δ_{ii})，按日按股                  |

#### 协方差矩阵分解（高效 SOCP 形式）

风险模型将全局协方差矩阵 $\Sigma = X_t F_t X_t^\top + \Delta_t$ 分解为：

$$
w^\top \Sigma w = \underbrace{\|F_{\text{half}} \cdot (X_t^\top w)\|^2}_{\text{因子风险}} + \underbrace{\|\delta \odot w\|^2}_{\text{特异性风险}}
$$

其中 $F_{\text{half}} = L_t^\top$（Cholesky 上三角因子），$\delta = \sqrt{\text{diag}(\Delta_t)}$。
这样只需 K 维向量运算（K ≈ 33），无需实例化 N×N 矩阵，cvxpy 可将其表达为 SOCP。

#### 风险模型验证（RiskModelValidator）

对比沪深 300 市值加权组合的**预测方差** $w^\top \Sigma w$ 与**已实现方差**（组合收益的滚动样本方差）：

| 指标 | 计算 | 含义 |
|------|------|------|
| R² | 线性回归 realized ~ predicted 的 $r^2$ | 预测对已实现的解释度，> 0.4 较好 |
| Pearson / Spearman | 相关系数 | 线性/秩相关，> 0.6 较好 |
| Bias ratio | mean(pred) / mean(real) | 接近 1 为无偏校准 |
| RMSE_var / RMSE_vol | 方差/波动率空间的均方根误差 | 越小越好 |

#### 运行方式

```bash
# 先确保已运行第一阶段（生成 prices.parquet / meta.parquet / index.parquet）
python src/data_preparation/data_preparation_main.py

# 运行第三阶段（约 6~14 分钟）
python src/risk_model/risk_model_main.py
# 输出: data/risk_exposure.parquet, data/risk_cov_F.parquet, data/risk_delta.parquet,
#       plots/risk_model_validation.png, result_risk_validation.txt（RUN_VALIDATION=True 时）
```

---

### 第四阶段：组合优化与回测

#### IC 分析与回测组件

| 文件                              | 类                                    | 说明                                                         |
| --------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| `src/portfolio/backtester.py`     | `LayeredBacktester`                   | 分层回测：支持全截面与行业中性两种分组模式，计算等权收益、绩效指标与累计净值图 |
| `src/portfolio/net_backtester.py` | `NetReturnBacktester`                 | 净收益回测：支持全截面与行业中性两种选股模式的纯多头重叠组合，含摩擦成本、换手率、盈亏平衡换手率 |
| `src/portfolio/ic_analyzer.py`    | `calc_ic / calc_ic_metrics / plot_ic` | 截面 Spearman IC 评估：计算 IC 序列、均值/标准差/ICIR 指标、绘制时序图 |

#### `LayeredBacktester` 行业中性分层回测

`LayeredBacktester` 新增可选参数 `industry_df`，传入时自动切换为**行业中性分层**模式，与训练 target 的行业中性化设计保持对称。

**分组逻辑：**

1. **行业内排名**：在每个 `(trade_date, industry)` 截面内，按因子值的百分位秩将股票映射到 G1\~G5。采用 `ceil(pct_rank × N)` 映射，对任意股票数（≥ 2）均成立；仅 1 只股票的行业返回 NaN 并被自动排除。

2. **行业内组均值**：对每个 `(trade_date, industry, group)` 三元组，计算该格子内所有成员股票的**等权平均**前向收益率。

3. **跨行业等权合并**：对每个 `(trade_date, group)`，将各行业的组均值再做**等权平均**。每个行业对最终组收益的贡献相同，规避大行业（如银行）主导组合收益的问题。

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

#### `NetReturnBacktester` 行业中性纯多头回测

`NetReturnBacktester` 新增可选参数 `industry_df`，传入时切换为**行业中性选股**模式，与 `LayeredBacktester` 保持对称。

**权重构造：**

1. **行业内选股**：在每个 `(trade_date, industry)` 截面内，按因子值百分位选出前 `top_pct`（默认 20%）的股票。少于 2 只股票的行业当日不参与选股。

2. **行业等权合成**：每只被选中的股票权重为：

   $$
   w_s = \frac{1}{N_{\text{ind}\_\text{with}\_\text{top}} \times N_{\text{top}\_\text{in}\_\text{industry}(s)}}
   $$
   
   其中 $N_{\text{ind}\_\text{with}\_\text{top}}$ 为当日至少贡献一只股票的行业数，$N_{\text{top}\_\text{in}\_\text{industry}(s)}$ 为股票 $s$ 所在行业当日被选中的股票数。整个组合权重合计为 1，每个行业贡献相同。

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

#### 凸优化组合管理

| 文件                                       | 类 / 入口                | 说明                                                         |
| ------------------------------------------ | ------------------------ | ------------------------------------------------------------ |
| `src/portfolio/optimizer.py`               | `PortfolioOptimizer`     | cvxpy LP/SOCP 求解器：每日求解行业中性、带换手成本、可选风险惩罚项与风险约束的最优权重向量 |
| `src/portfolio/optimization_backtester.py` | `OptimizationBacktester` | 逐日调用优化器的纯多头回测，支持 forward_days 重叠组合逻辑与风险模型集成，计算净收益与绩效指标 |
| `src/portfolio/optimization_main.py`       | `main()`                 | 第四阶段独立入口脚本                                         |

**重叠投资组合逻辑**：当 `forward_days = d > 1` 时，与 `NetReturnBacktester` 采用相同逻辑：每日 optimizer 输出 `daily_w_t`，实际持仓为 `overlap_w_t = mean(daily_w_t, ..., daily_w_{t-d+1})`，即过去 d 天目标权重的滚动均值。每日仅操作约 1/d 仓位，压低换手率。收益与换手均基于 `overlap_w` 计算。`forward_days` 应与 alpha 的预测周期（如 `ml_analyze_main` 中的 `FORWARD_DAYS`）一致。默认 1 表示每日全仓换手。

##### 优化问题

每个交易日 $t$ 求解如下 LP（无风险模型）或 SOCP（有风险模型）：

$$
\max_{w_t}\ w_t^\top\hat\alpha_t - \lambda\cdot\tfrac{1}{2}\|w_t - w_{t-1}\|_1 - \tfrac{1}{2}\mu_{\text{risk}}\cdot w_t^\top\Sigma w_t
$$

各符号含义：

- $\hat\alpha_t$：截面去均值后的 ML alpha 信号，范围约 $[-0.5,\,0.5]$。**去均值不改变截面排序**，仅将信号中心化为 0，使 $\lambda$ 的调参量级稳定。
- $w_{t-1}$：前一日持仓权重（新股赋 0，退市股强制清仓）
- $\lambda$（`lambda_turnover`）：换手惩罚系数，**无量纲的策略偏好参数**，非交易费率。根据报告中的 `Avg Daily Turnover` 手动调整：
  - $\lambda \in [0.05, 0.1]$：高换手，日均换手率约 10~20%
  - $\lambda \in [0.2, 0.5]$：适中，日均换手率约 2~8%（推荐起始值）
  - $\lambda \ge 1.0$：持仓极稳定，信号追踪滞后
- $\mu_{\text{risk}}$（`mu_risk`）：风险惩罚系数。由于 alpha 量级 ~0.5、$w^\top\Sigma w$ 日方差量级 ~0.0001，要使两项可比需 $\mu_{\text{risk}} \approx 5000$，推荐调参范围 1000~10000。
- $\Sigma = X_t F_t X_t^\top + \Delta_t$：由第三阶段风险模型提供；分解为 $\|L_t^\top (X_t^\top w)\|^2 + \|\delta \odot w\|^2$ 进行 SOCP 求解
- $X_{\text{ind}}$：行业 dummy 矩阵（$n \times K_{\text{ind}}$）
- $w_{\text{bench}}$：基准行业权重，每日由 `meta.parquet` 中全部 CSI 300 股票的 `total_mv` 加权计算
- $\delta_{\text{ind}} = 0.01$：行业偏离容差（不可行时自动逐步放宽至 ±5%）

优化约束条件：

* 满仓约束：$\sum_i w_i = 1$
* 纯多头约束：$w_i \ge 0$
* 单股权重上限：$w_i \le 0.05$
* 日频换手率约束：$\tfrac{1}{2}\|w_t - w_{t-1}\|_1 \le \text{max\_turnover}$
* 行业基准约束（追踪指数）：$|X_{\text{ind}}^\top w_t - w_{\text{bench}}| \le \delta_{\text{ind}}$
* （可选）日方差约束：$w_t^\top \Sigma w_t \le 2 \cdot \text{max\_variance}$
* （可选）风格因子暴露约束：$|w_{\text{active}}^\top X_{\text{factor}_k}| \le \text{style\_tol}$

**净收益计算**（独立于优化器）：$r_t^{\text{net}} = r_t^{\text{gross}} - \text{turnover}_t \times c_{\text{real}}$，其中 $c_{\text{real}} = 0.002$（实际交易费率，仅用于 P&L 扣费，与 $\lambda$ 无关）

**求解器**：cvxpy + CLARABEL，原生支持 LP 和 SOCP，无需切换求解器。无风险模型时每日约 <0.5 秒（全周期约 3\~5 分钟）；有风险模型（SOCP）每日约 1\~2 秒（全周期约 15\~25 分钟）。

##### 关键设计决策

1. **执行假设与收益定义（与 targets.py 保持一致）**：T 日收盘计算信号，T+1 日开盘入场，T+2 日开盘出场/换仓。回测器使用 `open_wide.pct_change()` 作为持仓期收益，与 `forward_return = open_{T+2}/open_{T+1} - 1` 完全对齐，确保优化目标与实际 P&L 使用相同的收益定义。

2. **双参数分离设计**：优化目标中的换手惩罚使用无量纲的 `lambda_turnover`（默认 0.2），P&L 扣费使用真实费率 `cost_rate`（0.002）。两者不可混用：ML alpha 是百分位秩（量级 ~0.5），若直接用 0.002 作为惩罚系数，换手项量级比 alpha 项小 250 倍，优化器实际上完全忽略换手，导致每日暴力翻仓。

3. **截面 alpha 去均值**：进入优化器前每日截面减均值，保证信号对称分布在 0 附近，`lambda_turnover` 量级具有跨时间的一致性。

4. **换手率硬约束**：$\tfrac{1}{2}\|w - w_{\text{prev}}\|_1 \le \text{max\_turnover}$（默认 10%）。`max_turnover=None` 时禁用。

5. **行业约束动态基准**：基准权重每日更新（不使用静态值），反映 CSI 300 真实行业构成变化。

6. **不可行日自动处理**：$\delta_{\text{ind}}$ 依次扩大（0.01 → 0.02 → ... → 0.05），所有容差均失败时去掉行业约束求解，报告中记录发生次数。

7. **首日初始化**：$w_0 = \mathbf{0}$（空仓），第一个有效日完整买入，换手率≈100% 计入成本。

8. **重叠组合（forward_days > 1）**：与 `NetReturnBacktester` 一致，`overlap_w = rolling_mean(daily_w, d)`。`optimization_main` 中 `FORWARD_DAYS`因 与 `ml_analyze_main` 保持一致。

9. **风险模型集成**：`optimization_main.py` 中设置 `USE_RISK_MODEL=True` 即可启用。

   `OptimizationBacktester` 自动加载三个风险模型 parquet 文件，每日提取 $(X_t, L_t^\top, \delta_t)$ 传入 `PortfolioOptimizer.solve()`。优化器用 `cp.sum_squares` 实现 SOCP 风险项，无需构造 N×N 矩阵。

   回测报告额外输出 `Avg Daily Variance` 和 `Avg Daily Std (%)` 用于 `mu_risk` 标定。

   前置条件：已运行 `risk_model_main.py`。

   可选 `SOLVER` 配置（如 `"ECOS"`、`"SCS"`）指定 cvxpy 求解器，默认 CLARABEL；ECOS 对不可行问题检测更可靠。

   **约束验证与回退**：求解器返回的解会经 `_validate_solution` 检查（含 cvxpy 约束 violation 及显式 max_variance 校验）；若违反容差 1e-5，则发出警告并退回等权，报告中 `Equal-Weight Fallback Days` 统计此类天数。

10. **风格因子中性化（可选）**：设置 `USE_STYLE_NEUTRAL=True` 可对主动组合施加风格因子暴露约束。设 $w_{\text{active}} = w - w_{\text{benchmark}}$（股票级市值加权基准），约束为 $|w_{\text{active}}' X_{\text{factor}_k}| \le \text{style\_tol}$，对每个启用的风格因子 k。`STYLE_FACTORS` 默认 `["size", "beta", "momentum", "volatility", "value"]`，须为 `risk_exposure.parquet` 中存在的列。`STYLE_TOL` 默认 0.05（z-score 单位）。默认关闭，需 `USE_RISK_MODEL=True`。

11. **基准超额收益分析**：`NetReturnBacktester` 和 `OptimizationBacktester` 均支持可选的 `benchmark_prices` 参数（CSI 300 每日收盘价 Series）。传入后会：

   - 计算超额日收益 `excess_ret = strategy_ret - bench_ret`；
   - 在 `run_backtest()` 报告中追加五项指标：`Bench Ann Return`、`Excess Ann Return`、`Tracking Error`、`Information Ratio`、`Max Relative DD`；
   - 将图表改为**双面板**：上栏绝对净值（策略蓝色 + 沪深300橙色）、下栏超额净值（绿色，基准为 1.0 水平线）并标注 IR / Tracking Error。
   - `benchmark_prices=None`（默认）时行为与原始版本完全相同（向后兼容）。
   - 基准来源：`data/index.parquet`（由 `data_preparation_main.py` 生成，内含 `[trade_date, close]` 两列）。

##### 运行方式

```bash
# 先确保已运行 ml_analyze_main.py（生成 data/ml_alpha.parquet）
python src/LightGBM/ml_analyze_main.py

# 可选：运行第三阶段风险模型（约 6~14 分钟，生成三个 risk_*.parquet）
python src/risk_model/risk_model_main.py

# 再运行优化回测（如已运行风险模型，可在 optimization_main.py 中设置 USE_RISK_MODEL=True）
python src/portfolio/optimization_main.py
# 输出: result_optimization.txt, plots/optimization_nav.png
```

