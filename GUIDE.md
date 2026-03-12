## 各文件使用说明

### 1. requirements.txt

- **用途**：定义项目 Python 依赖。
- **使用**：
  ```bash
  pip install -r requirements.txt
  ```

---

### 2. src/config.py

- **用途**：集中管理所有全局参数，避免在业务代码中硬编码。
- **创建方式**：将 `src/config_template.py` 复制为 `src/config.py`，填入真实 Tushare Token。
- **主要配置项**：

  | 变量 | 说明 | 默认值 |
  |------|------|--------|
  | `TUSHARE_TOKEN` | Tushare Pro Token（必填） | `"your_tushare_token"` |
  | `START_DATE` | 数据开始日期（YYYYMMDD） | `"20190101"` |
  | `END_DATE` | 数据结束日期（YYYYMMDD） | `"20231231"` |
  | `UNIVERSE_INDEX` | 股票池指数代码 | `"000300.SH"` |
  | `DB_PATH` | 数据库路径（相对于 src/） | `"../data/stock_data.db"` |
  | `SLEEP_PER_CALL` | 每次 API 调用间隔（秒） | `0.35` |

- **注意**：此文件含 Token，已在 `.gitignore` 中排除，**切勿提交至 git**。

### 3. src/data_preparation/data_loader.py

- **用途**：项目唯一的数据 I/O 层，封装为 `DataEngine` 类。负责从 Tushare Pro 拉取数据并写入本地 SQLite，或从 SQLite 读取为 pandas DataFrame。

- **数据库表结构**：

  | 表名 | 主键 | 说明 |
  |------|------|------|
  | `daily_price` | (code, date) | 日线 OHLCV + amount（千元） |
  | `daily_basic` | (code, date) | 每日扩展基本面（PE、PB、换手率等 15 字段） |
  | `stock_info` | code | 股票元信息：名称、申万一级行业、上市日期 |
  | `adj_factor` | (code, date) | 复权因子 |
  | `stock_st` | (code, start_date) | ST/\*ST 状态历史区间 |
  | `index_daily` | date | CSI 300 每日收盘价 |
  | `quarterly_financials` | (code, ann_date, end_date) | 季度财务：ROE |

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__()` | 初始化 Tushare Pro API，解析数据库路径；Token 未设置时抛出 ValueError |
  | `init_db()` | 建表 + 模式迁移（幂等，可重复调用） |
  | `download_data()` | 下载 price / daily_basic / adj_factor / stock_st / index_daily / quarterly_financials；六套独立缓存，可断点续传 |
  | `load_data()` | 从 SQLite 读取，返回结构化字典（见下） |

- **`load_data()` 返回值**：
  ```python
  {
    "df_price"    : DataFrame  # MultiIndex (date, code)，列：open, high, low, close, vol, amount
    "df_mv"       : DataFrame  # MultiIndex (date, code)，列：total_mv
    "df_basic"    : DataFrame  # MultiIndex (date, code)，15 个扩展基本面字段
    "df_industry" : DataFrame  # index = code，列：name, industry（申万一级）, list_date
    "df_adj"      : DataFrame  # MultiIndex (date, code)，列：adj_factor
    "df_st"       : DataFrame  # 列：code, start_date, end_date（ST 状态区间）
    "df_index"    : DataFrame  # index = date，列：close（CSI 300）
    "df_financials": DataFrame # 列：code, ann_date, end_date, roe（季度财务）
  }
  ```

- **使用示例**：
  ```python
  from data_loader import DataEngine
  
  engine = DataEngine()
  engine.init_db()        # 建表（幂等）
  engine.download_data()  # 下载并缓存
  data = engine.load_data()
  ```

---

### 4. src/data_preparation/factors.py

- **用途**：因子计算模块，封装为 `FactorEngine` 类。输入为 `DataEngine.load_data()` 返回的数据字典，输出为原始因子值（保留 NaN/inf，清洗由后续模块处理）。

- **因子分组**（A/B/C 三组，共 53 个 + alphas 15 个）：

  - **A 组**：微观结构与量价因子（34 个），使用复权价格。如 amihud_illiq、ivol、beta、momentum_1d~20d、rvol_5d~20d、factor_realized_skewness 等。
  - **B 组**：基本面与估值因子（7 个），季度数据 PIT 对齐或日频。如 ep、bp、roe、log_mv、ps、dv_ratio、circ_mv_ratio。
  - **C 组**：截面相对特征（12 个）。如 industry_rel_turnover、industry_rel_bp、industry_rel_mv、ts_rel_turnover、ind_rel_momentum_5d~20d 等。
  - **alphas**：来自 WorldQuant Alpha101，详见 [Multi-factor-Model-for-Stock-Selection](https://github.com/Parsnip77/Multi-factor-Model-for-Stock-Selection/tree/main)。

- **复权模式**（`adj_type` 参数，默认 `"forward"`）：

  | 模式 | 公式 | 说明 |
  |------|------|------|
  | `"forward"` | `P × adj_factor / adj_factor_latest` | 前复权，最新价等于原始价 |
  | `"backward"` | `P × adj_factor` | 后复权 |
  | `"raw"` | 不调整 | 直接使用数据库原始价格 |

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict, adj_type, latest_adj)` | 接收数据字典；自动计算精确 vwap 并应用复权 |
  | `get_all_factors()` | 计算所有已实现因子，返回 MultiIndex (date, code) × 因子列的 DataFrame |

- **使用示例**：
  ```python
  from data_loader import DataEngine
  from factors import FactorEngine
  
  data = DataEngine().load_data()
  factor_engine = FactorEngine(data, adj_type="forward")
  df_raw = factor_engine.get_all_factors()
  ```

---

### 5. src/data_preparation/preprocessor.py

- **用途**：因子预处理模块，将 `FactorEngine.get_all_factors()` 输出的**原始因子**清洗为**可直接输入模型的因子**，封装为 `FactorCleaner` 类。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict)` | 从数据字典中提取行业映射（`df_industry`） |
  | `fill_industry_median(factor_df)` | 截面 NaN 填充：行业截面中位数 → 全局截面中位数 fallback |
  | `percentile_rank(factor_df)` | 截面百分位排名 → (0, 1] |
  | `process_all(raw_factors_df)` | 完整三步流水线：±inf→NaN、中位数填充、百分位排名；返回清洗后 DataFrame |

- **`process_all` 流水线**：

  | 步骤 | 操作 | 目的 |
  |------|------|------|
  | 1. 异常值初筛 | ±inf → NaN | 防止数学错误 |
  | 2. NaN 填充 | 行业截面中位数 → 全局截面中位数 | 避免信息丢失 |
  | 3. 百分位排名 | 截面 pct_rank → (0, 1] | 分布无关、自动处理极值 |

- **输入**：`raw_factors_df` — MultiIndex (date, code) × 因子列，保留 NaN/inf。
- **输出**：`df_clean_factors` — 同结构，值为 (0, 1] 百分位排名；不可交易格子由调用方覆写为 NaN。

- **使用示例**：
  ```python
  from preprocessor import FactorCleaner
  
  cleaner = FactorCleaner(data)
  df_clean = cleaner.process_all(df_raw)
  ```

---

### 6. src/data_preparation/data_preparation_main.py

- **用途**：第一阶段端到端总脚本，串联 `DataEngine → _compute_tradable → FactorEngine → FactorCleaner`，将全流程处理结果导出为四张 Parquet 文件。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 检查 `data/stock_data.db` 是否存在，缺失则报错退出 |
  | 2 | 调用 `DataEngine.load_data()` 载入全量数据 |
  | 2.5 | 调用 `_compute_tradable()` 计算可交易性掩码（停牌/退市/涨跌停/ST/准新股） |
  | 3 | 调用 `FactorEngine.get_all_factors()` 计算原始因子（前复权） |
  | 4 | 调用 `FactorCleaner.process_all()` 执行三步清洗；之后对不可交易格子覆写为 NaN |
  | 5 | 导出四张 Parquet 至 `./data/` |
  | 6 | 打印汇总信息 |

- **输出文件**（主键逻辑对齐：`trade_date × ts_code`）：

  | 文件 | 列 | 说明 |
  |------|----|------|
  | `prices.parquet` | trade_date, ts_code, open, high, low, close, vol, amount, adj_factor, tradable | 原始行情 + 复权因子 + 可交易性标志 |
  | `meta.parquet` | trade_date, ts_code, industry, turnover_rate, ... | 15 个每日基本面字段 + 申万一级行业 |
  | `factors_raw.parquet` | trade_date, ts_code, factor_* | 原始因子值（保留 NaN） |
  | `factors_clean.parquet` | trade_date, ts_code, factor_* | 清洗后百分位排名；不可交易格子为 NaN |

- **使用**：
  ```bash
  python src/data_preparation/data_preparation_main.py
  ```

### 7. src/LightGBM/targets.py

- **用途**：标签生成模块，计算未来 d 日收益率，为 IC 分析与回测提供 target。

- **执行假设（无前视偏差）**：
  - T 日收盘：计算因子、生成 alpha 信号
  - T+1 日开盘：集合竞价成交（入场）
  - T+2 日开盘（d=1 时）：换仓/平仓（出场）

  因此 `forward_return_T = open_{T+d+1} / open_{T+1} - 1`。

- **核心函数**：

  | 函数 | 说明 |
  |------|------|
  | `calc_forward_return(prices_df, d)` | 输入平表 prices_df（含 trade_date/ts_code/open）和前向天数 d，返回 MultiIndex (trade_date, ts_code) × `forward_return` 的长表 |

- **使用示例**：
  
  ```python
  import pandas as pd
  from targets import calc_forward_return
  
  prices_df = pd.read_parquet("data/prices.parquet")
  target_df = calc_forward_return(prices_df, d=1)
  ```

---

### 8. src/LightGBM/ml_data_prep.py

- **用途**：量化专用时序滚动切分器，封装为 `WalkForwardSplitter` 类。

- **两种模式**：
  - **滚动窗口**（`expanding=False`，默认）：训练集固定长度，每次 fold 整体向前滑动。
  - **扩展窗口**（`expanding=True`）：训练集从数据起点开始，随 fold 增长，充分利用历史数据。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(train_months, val_months, test_months, embargo_days, expanding, step_months)` | 配置窗口参数；`embargo_days` 需 ≥ 预测周期 d |
  | `split(df)` | 生成器：对含 `trade_date` 列的 DataFrame 滚动切分，每次 yield `(train_mask, val_mask, test_mask)` |
  | `n_splits(df)` | 返回给定数据集上预计可生成的折数 |

- **使用示例**：
  ```python
  from ml_data_prep import WalkForwardSplitter
  
  splitter = WalkForwardSplitter(train_months=15, val_months=3, test_months=3, embargo_days=1)
  for train_mask, val_mask, test_mask in splitter.split(df_merged):
      X_train = df_merged[train_mask][feature_cols]
      ...
  ```

---

### 9. src/LightGBM/lgbm_model.py

- **用途**：LightGBM 训练引擎，封装为 `AlphaLGBM` 类。

- **超参数设计**：`objective='regression'`、`max_depth=5`、`num_leaves=20`、`subsample=0.8`、`colsample_bytree=0.8`。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `train(X_train, y_train, X_val, y_val)` | 训练并监控验证集，early stopping |
  | `predict(X_test)` | 使用 best_iteration 预测，返回 ndarray |
  | `get_feature_importance(importance_type)` | 返回按重要性排序的 DataFrame |
  | `plot_feature_importance(fold, save_path)` | 绘制特征重要性条形图 |
  | `plot_shap(X_sample, save_path, max_display)` | 生成 SHAP beeswarm 图（需安装 shap） |

- **使用示例**：
  ```python
  from lgbm_model import AlphaLGBM
  
  model = AlphaLGBM()
  model.train(X_train, y_train, X_val, y_val)
  y_pred = model.predict(X_test)
  imp_df = model.get_feature_importance()
  model.plot_shap(X_test.sample(300), save_path=Path("plots/shap.png"))
  ```

---

### 10. src/LightGBM/ml_analyze_main.py

- **用途**：第二阶段端到端总脚本，串联数据加载 → 行业中性化 target → 滚动训练 → IC 分析 → 双回测 → 图表与报告输出。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 载入 `factors_clean.parquet`、`prices.parquet`、`meta.parquet` |
  | 2 | 调用 `calc_forward_return(prices_df, d=1)` 生成 target |
  | 3 | 行业中性化：`ind_neutral_return = forward_return - groupby(industry, date).mean()` |
  | 4 | 合并因子 + target，dropna；训练 target：`cs_rank_return = pct_rank(ind_neutral_return)` |
  | 5 | 初始化 `WalkForwardSplitter`（train=15m, val=3m, test=3m, embargo=1d） |
  | 6 | 各 Fold：训练 `AlphaLGBM`，预测测试集，记录预测结果 |
  | 7 | 拼接所有 Fold 预测；可选 3 日滚动均值平滑 |
  | 8 | IC 分析 → `plots/ml_alpha_ic.png` |
  | 9 | `LayeredBacktester`（行业中性）→ `plots/ml_alpha_layered.png` |
  | 10 | `NetReturnBacktester`（行业中性）→ `plots/ml_alpha_net.png` |
  | 11 | 特征重要性条形图 → `plots/feature_importance.png` |
  | 12 | SHAP beeswarm → `plots/shap_beeswarm.png` |
  | 13 | 保存 `ml_alpha.parquet`，输出 `result_ml.txt` |

- **可调配置**（脚本顶部常量）：

  | 变量 | 默认值 | 说明 |
  |------|--------|------|
  | `FORWARD_DAYS` | `1` | 未来收益率天数 |
  | `TRAIN_MONTHS` | `15` | 训练窗口（月） |
  | `VAL_MONTHS` | `3` | 验证窗口（月） |
  | `TEST_MONTHS` | `3` | 测试窗口（月） |
  | `EMBARGO_DAYS` | `1` | 静默期天数（需 ≥ FORWARD_DAYS） |
  | `ALPHA_ROLLING_WINDOW` | `1` | 滚动均值窗口；1=关闭 |
  | `ALPHA_EMA_BETA` | `0.5` | EMA 平滑系数；1=关闭 |

- **使用**：
  ```bash
  python src/LightGBM/ml_analyze_main.py
  ```

- **前提**：已运行 `data_preparation_main.py`，`data/factors_clean.parquet` 和 `data/prices.parquet` 已生成。

---

### 11. src/risk_model/risk_factor_engine.py

- **用途**：计算每日风险因子暴露矩阵 X_t，封装为 `RiskFactorEngine` 类。

- **因子定义**：
  
  - **风格因子**（5 个）：Size、Beta、Momentum、Volatility、Value；截面 winsorize ±3σ 后 z-score 标准化。
  - **行业因子**：申万一级行业哑变量（0/1）。
  
- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `compute()` | 计算并返回长表 [trade_date, ts_code, size, beta, momentum, volatility, value, ind_*] |

- **使用**：由 `risk_model_main.py` 调用，一般不单独使用。

---

### 12. src/risk_model/cov_estimator.py

- **用途**：估计每日因子协方差与个股特异性方差，封装为 `CovarianceEstimator` 类。

- **方法**：
  - Step 1：每日 WLS 截面回归 `R_t = X_t f_t + ε_t`，权重 = sqrt(total_mv)
  - Step 2：滚动 60 日因子协方差 F_t，加岭正则化 + Cholesky 分解 → L_t^T
  - Step 3：滚动 60 日个股残差方差 Δ_{ii}，下界 (0.5%)²

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `compute()` | 返回 (exposure_df, cov_F_df, delta_df) 三个 DataFrame |

- **使用**：由 `risk_model_main.py` 调用，一般不单独使用。

---

### 13. src/risk_model/risk_model_validator.py

- **用途**：验证风险模型预测能力，对比沪深 300 市值加权组合的预测波动率与已实现波动率。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `run_validation()` | 返回 (predicted_var, realized_var, metrics_dict) |
  | `plot(predicted_var, realized_var, save_path, show)` | 绘制时序图 + 散点图 |

- **评估指标**：

  | 指标 | 计算公式 | 含义 | 参考标准 |
  |------|----------|------|----------|
  | **R²** | 线性回归 realized ~ predicted 的 $r^2$ | 预测方差对已实现方差的解释度 | > 0.4 较好，0.2~0.4 中等 |
  | **Pearson 相关系数** | $\text{corr}(\sigma^2_{\text{pred}}, \sigma^2_{\text{real}})$ | 线性相关强度 | > 0.6 较好 |
  | **Spearman 相关系数** | 秩相关 | 排序一致性，对异常值稳健 | 通常略高于 Pearson |
  | **Bias ratio** | $\overline{\sigma^2_{\text{pred}}} / \overline{\sigma^2_{\text{real}}}$ | 预测均值与已实现均值之比 | 接近 1 为无偏，0.95~1.05 较好 |
  | **RMSE (variance)** | $\sqrt{\text{mean}((\sigma^2_{\text{pred}} - \sigma^2_{\text{real}})^2)}$ | 方差空间的均方根误差 | 越小越好，结合量级判断 |
  | **RMSE (volatility)** | $\sqrt{\text{mean}((\sqrt{\sigma^2_{\text{pred}}} - \sqrt{\sigma^2_{\text{real}}})^2)}$ | 波动率空间的均方根误差 | 越小越好 |

- **图表解读**：时序图两条曲线越接近表示预测越准；散点图点越靠近 y=x 表示拟合越好；点整体在 y=x 上方为高估，下方为低估。

- **使用**：由 `risk_model_main.py` 在保存 parquet 后自动调用，一般不单独使用。

---

### 14. src/risk_model/risk_model_main.py

- **用途**：第三阶段端到端总脚本，串联 `RiskFactorEngine` 与 `CovarianceEstimator`，输出三个 Parquet 文件；可选执行 `RiskModelValidator` 验证。

- **输出文件**：

  | 文件 | 格式 | 内容 |
  |------|------|------|
  | `risk_exposure.parquet` | 长表 | (trade_date, ts_code, size, beta, momentum, volatility, value, ind_*) |
  | `risk_cov_F.parquet` | 长表 | (trade_date, f_i, f_j, value) — Cholesky 因子 L_t^T 元素 |
  | `risk_delta.parquet` | 长表 | (trade_date, ts_code, delta_std) — sqrt(Δ_{ii}) |
  | `plots/risk_model_validation.png` | 图 | 预测 vs 已实现波动率（RUN_VALIDATION=True 时） |
  | `result_risk_validation.txt` | 报告 | 验证指标（RUN_VALIDATION=True 时） |

- **可调配置**：`COV_WINDOW=60`、`MIN_PERIODS=30`、`DELTA_FLOOR=2.5e-5`、`RIDGE=1e-6`、`RUN_VALIDATION=True`、`REALIZED_WINDOW=20`。

- **使用**：
  ```bash
  python src/risk_model/risk_model_main.py
  ```

- **前提**：已运行 `data_preparation_main.py`，生成 `prices.parquet`、`meta.parquet`、`index.parquet`。
- **耗时**：约 6~14 分钟。

---

### 15. src/portfolio/ic_analyzer.py

- **用途**：因子 IC（信息系数）评估模块，衡量因子值与未来收益率的截面相关性。

- **核心函数**：

  | 函数 | 说明 |
  |------|------|
  | `calc_ic(factors_df, target_df)` | 按 trade_date 计算截面 Spearman 相关，返回 ic_series |
  | `calc_ic_metrics(ic_series)` | 计算 IC 均值、标准差、ICIR，返回字典 |
  | `plot_ic(ic_series, factor_name, show)` | 绘制 IC 时间序列柱状图 + 累计 IC 折线图 |

- **使用示例**：
  ```python
  from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic
  
  ic_series = calc_ic(single_factor_df, target_df)
  metrics = calc_ic_metrics(ic_series)
  plot_ic(ic_series, factor_name="ml_alpha")
  ```

---

### 16. src/portfolio/backtester.py

- **用途**：分层回测模块，封装为 `LayeredBacktester` 类。支持全截面与行业中性两种分组模式。

- **分组逻辑（行业中性模式）**：
  1. 行业内排名：按因子百分位秩映射到 G1~G5
  2. 行业内组均值：等权平均前向收益率
  3. 跨行业等权合并：各行业组均值再做等权平均

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(factor_df, target_df, industry_df=None, num_groups=5, rf=0.03, forward_days=1, plots_dir=None)` | 接收因子与 target；`industry_df` 传入时启用行业中性 |
  | `run_backtest()` | 执行回测，返回绩效指标 DataFrame |
  | `plot(show)` | 绘制 G1..GN + L-S 累计净值图 |

- **使用示例**：
  ```python
  from backtester import LayeredBacktester
  
  bt = LayeredBacktester(final_alpha_df, target_flat, industry_df=industry_df,
                         num_groups=5, rf=0.03, forward_days=1, plots_dir=PLOTS_DIR)
  perf_table = bt.run_backtest()
  bt.plot(show=True)
  ```

---

### 17. src/portfolio/net_backtester.py

- **用途**：纯多头净收益回测模块，封装为 `NetReturnBacktester` 类。支持全截面与行业中性两种选股模式，含摩擦成本、换手率、盈亏平衡换手率。

- **重叠组合逻辑**：`overlap_w_T = mean(daily_w_T, ..., daily_w_{T-d+1})`；净收益 = 毛收益 - 换手率 × cost_rate。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(alpha_df, prices_df, industry_df=None, forward_days=1, cost_rate=0.0035, rf=0.03, plots_dir=None, top_pct=0.20, benchmark_prices=None)` | `benchmark_prices` 传入时生成超额收益指标与双面板图 |
  | `run_backtest()` | 执行回测，返回绩效指标 Series |
  | `plot(show)` | 绘制累计净值图 |

- **使用示例**：
  ```python
  from net_backtester import NetReturnBacktester
  
  nb = NetReturnBacktester(final_alpha_df, prices_df, industry_df=industry_df,
                          forward_days=1, cost_rate=0.002, benchmark_prices=index_prices)
  summary = nb.run_backtest()
  nb.plot()
  ```

---

### 18. src/portfolio/optimizer.py

- **用途**：凸优化组合求解器，封装为 `PortfolioOptimizer` 类。求解 LP（无风险模型）或 SOCP（有风险模型）。

- **优化问题**：
  ```
  max  w' alpha_centered - lambda_turnover/2 * ||w - w_prev||_1 - 1/2 * mu_risk * w^T Σ w
  s.t. sum(w)=1, w>=0, w<=max_weight,
       0.5||w-w_prev||_1 <= max_turnover,
       |X_ind' w - w_bench| <= tol,
       (optional) w^T Σ w <= max_variance
       (optional) |w_active^T X_factor| <= factor_tol
  ```
  
- **双参数设计**：
  
  - `lambda_turnover`：换手惩罚系数，**无量纲策略偏好**，非交易费率。推荐 0.2~0.5。
  - `cost_rate`：实际交易费率，仅用于 P&L 扣费，与 `lambda_turnover` 分离。
  
- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `solve(alpha_t, w_prev, X_industry, w_benchmark, X_risk, F_half, delta_std)` | 返回 (w_star, tol_used, fallback_used) |

- **使用**：由 `OptimizationBacktester` 调用，一般不单独使用。

---

### 19. src/portfolio/optimization_backtester.py

- **用途**：逐日调用 `PortfolioOptimizer` 的纯多头回测，支持重叠组合逻辑与风险模型集成，计算净收益与绩效指标。

- **执行假设**：与 `targets.py` 一致，T 日收盘计算信号，T+1 日开盘入场，T+2 日开盘出场。收益使用 `open_wide.pct_change()`。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(alpha_df, prices_df, meta_df, ...)` | 接收 alpha、价格、meta；可传入 risk_model 相关路径 |
  | `run_backtest()` | 执行回测，返回绩效指标 Series |
  | `plot(show)` | 绘制累计净值图 |

- **使用**：由 `optimization_main.py` 调用，一般不单独使用。

---

### 20. src/portfolio/optimization_main.py

- **用途**：第四阶段端到端总脚本，串联数据加载 → 凸优化回测 → 报告与图表输出。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 载入 `ml_alpha.parquet`、`prices.parquet`、`meta.parquet` |
  | 2 | 可选：载入风险模型 parquet（risk_exposure、risk_cov_F、risk_delta） |
  | 3 | 实例化 `OptimizationBacktester`，逐日求解 LP/SOCP |
  | 4 | 打印并保存绩效报告至 `result_optimization.txt` |
  | 5 | 保存累计净值图至 `plots/optimization_nav.png` |

- **可调配置**（脚本顶部常量）：

  | 变量 | 默认值 | 说明 |
  |------|--------|------|
  | `LAMBDA_TURNOVER` | `0.00` | 换手惩罚系数（无量纲） |
  | `COST_RATE` | `0.002` | 实际交易费率（P&L 扣费） |
  | `USE_RISK_MODEL` | `True` | 是否启用风险模型 |
  | `MU_RISK` | `0.0` | 风险惩罚系数；0=禁用 |
  | `MAX_VARIANCE` | `None` | 日方差硬约束；None=禁用 |
  | `USE_STYLE_NEUTRAL` | `True` | 是否启用风格因子中性化 |
  | `STYLE_FACTORS` | `["value"]` | 启用的风格因子列表 |
  | `STYLE_TOL` | `0.50` | 风格因子偏离容差（z-score 单位） |
  | `FORWARD_DAYS` | `1` | 持仓周期，需与 ml_analyze_main 一致 |
  | `SOLVER` | `"CLARABEL"` | cvxpy 求解器 |

- **使用**：
  ```bash
  python src/portfolio/optimization_main.py
  ```

- **前提**：已运行 `ml_analyze_main.py` 生成 `ml_alpha.parquet`；若 `USE_RISK_MODEL=True`，需先运行 `risk_model_main.py`。

---

### 21. data/stock_data.db

- **用途**：本地 SQLite 数据库，存储 7 张表（见 data_loader 说明）。
- **生成方式**：由 `DataEngine.init_db()` + `download_data()` 自动生成。
- **git 状态**：已在 `.gitignore` 中排除。

---

### 22. data/*.parquet

- **用途**：各阶段导出的 Parquet 宽表/长表，供下游阶段读取。
- **主要文件**：

  | 文件 | 生成阶段 | 说明 |
  |------|----------|------|
  | `prices.parquet` | 第一阶段 | OHLCV + adj_factor + tradable |
  | `meta.parquet` | 第一阶段 | 每日基本面 + 申万一级行业 |
  | `factors_raw.parquet` | 第一阶段 | 原始因子值 |
  | `factors_clean.parquet` | 第一阶段 | 清洗后百分位排名 |
  | `index.parquet` | 第一阶段 | CSI 300 每日收盘价 |
  | `ml_alpha.parquet` | 第二阶段 | ML 合成 alpha 信号 |
  | `risk_exposure.parquet` | 第三阶段 | 每日因子暴露矩阵 |
  | `risk_cov_F.parquet` | 第三阶段 | Cholesky 因子 L_t^T |
  | `risk_delta.parquet` | 第三阶段 | 个股特异性标准差 sqrt(Δ_{ii}) |

---

### 23. plots/, result_*.txt

* 用途：各阶段结果图标、数据分析报告

* 主要文件：

  | 文件                           | 生成脚本             | 说明                  |
  | ------------------------------ | -------------------- | --------------------- |
  | `plots/ml_alpha_ic.png`        | ml_analyze_main.py   | ML 因子 IC 时间序列图 |
  | `plots/ml_alpha_layered.png`   | ml_analyze_main.py   | 分层回测累计净值图    |
  | `plots/ml_alpha_net.png`       | ml_analyze_main.py   | 净收益回测累计净值图  |
  | `plots/feature_importance.png` | ml_analyze_main.py   | 特征重要性条形图      |
  | `plots/shap_beeswarm.png`      | ml_analyze_main.py   | SHAP 蜂群图           |
  | `plots/risk_model_validation.png` | risk_model_main.py | 预测 vs 已实现波动率  |
  | `plots/optimization_nav.png`   | optimization_main.py | 优化组合累计净值图    |
  | `result_ml.txt`                | ml_analyze_main.py   | ML 阶段文字报告       |
  | `result_risk_validation.txt`   | risk_model_main.py   | 风险模型验证报告      |
  | `result_optimization.txt`      | optimization_main.py | 优化回测文字报告      |