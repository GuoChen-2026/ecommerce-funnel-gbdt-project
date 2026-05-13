# 基于时序特征工程与 GBDT 模型的电商转化预测与高价值用户识别

## 1. 项目背景

本项目基于大规模电商用户行为日志，围绕用户从浏览、加购到购买的转化路径进行分析，并进一步构建用户级转化预测模型，用于识别未来短期内更可能购买的高价值用户。

在真实电商运营场景中，平台通常不会对所有用户进行无差别触达，而是希望在预算有限的情况下，优先识别最有购买意向的用户。因此，本项目不仅关注模型的整体预测效果，也重点关注 Top-K 高分用户的购买率、召回率、Lift 和预测排序收益。

本项目构建了以下流程：

- 原始行为日志清洗；
- 电商转化漏斗分析；
- 用户级时序特征工程；
- Logistic Regression、LightGBM、XGBoost、CatBoost 多模型对比；
- AUC、PR-AUC、Top-K Purchase Rate、Recall@K、Lift@K 等指标评估；
- 基于预测排序的 ROI 模拟；
- CatBoost 模型解释与 SHAP 分析；
- 高价值用户分层和运营策略设计。

---

## 2. 项目目标

本项目主要回答以下问题：

1. 用户从 view、cart 到 purchase 的转化路径是否存在明显流失？
2. 哪些用户行为特征能够刻画短期购买意向？
3. 基于过去 7 天用户行为，能否预测用户未来一天是否会购买？
4. 在预算有限的情况下，应该优先触达哪些用户？
5. 模型筛选出的高分用户是否真的具有更高购买率？
6. 模型判断用户购买倾向时，主要依赖哪些行为特征？
7. GBDT 类模型相比线性模型，在高价值用户识别任务中是否更有优势？

---

## 3. 数据说明

本项目使用电商行为日志数据，原始字段包括：

| 字段 | 含义 |
|---|---|
| event_time | 用户行为发生时间 |
| event_type | 行为类型，包括 view、cart、purchase |
| product_id / item_id | 商品 ID |
| category_id | 商品类目 ID |
| category_code | 商品类目编码 |
| brand | 商品品牌 |
| price | 商品价格 |
| user_id | 用户 ID |
| user_session | 用户会话 ID |

为了保证后续漏斗分析和建模样本的数据口径一致，本项目在 `01_data_understanding.ipynb` 中完成统一数据清洗，并保存为 clean 数据集：

```text
data/processed/ecommerce_behavior_2019_10_01_15_clean.csv
```

最终分析区间为：

```text
2019-10-01 00:00:00 至 2019-10-15 23:59:59
```

clean 数据规模如下：

| 指标 | 数值 |
|---|---:|
| 总行为数 | 20,442,805 |
| 用户数 | 1,781,811 |
| 商品数 | 142,434 |
| 类目数 | 583 |

行为分布如下：

| 行为类型 | 行为次数 | 占比 |
|---|---:|---:|
| view | 19,686,918 | 96.30% |
| cart | 400,024 | 1.96% |
| purchase | 355,863 | 1.74% |

从行为分布可以看出，电商场景中浏览行为占绝大多数，加购和购买行为占比较低，数据具有典型的转化漏斗特征。

---

## 4. 项目结构

```text
ecommerce-funnel-catboost-project/
├── data/
│   ├── raw/
│   │   └── 2019-Oct.csv
│   └── processed/
│       ├── ecommerce_behavior_2019_10_01_15_clean.csv
│       ├── model_data_user_level_7d.csv
│       └── test_data_for_model_interpretation.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_funnel_analysis.ipynb
│   ├── 03_user_level_feature_engineering.ipynb
│   ├── 04_modeling_gbdt_compare.ipynb
│   └── 05_shap_interpretation.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 5. 项目流程

本项目按照以下顺序完成：

```text
原始行为日志
→ 数据清洗与时间窗口截取
→ 漏斗分析
→ 用户级 7 日行为特征工程
→ GBDT 多模型转化预测
→ Top-K 业务评估
→ CatBoost 模型解释
→ 用户分层与运营策略
```

---

## 6. Notebook 说明

### 6.1 01_data_understanding.ipynb

该 notebook 主要完成原始数据读取、字段统一、时间处理和 clean 数据保存。

主要步骤包括：

1. 读取原始电商行为日志；
2. 将 `product_id` 统一重命名为 `item_id`；
3. 将 `event_time` 转换为标准时间格式；
4. 构造 `date`、`hour`、`weekday` 等时间字段；
5. 截取 2019-10-01 至 2019-10-15 的完整行为区间；
6. 检查行为类型分布；
7. 保存 clean 行为数据，供后续 notebook 使用。

主要输出：

```text
data/processed/ecommerce_behavior_2019_10_01_15_clean.csv
outputs/tables/event_type_count_summary.csv
outputs/tables/clean_behavior_data_quality_summary.csv
outputs/figures/event_type_count_distribution.png
```

---

### 6.2 02_funnel_analysis.ipynb

该 notebook 基于 clean 行为数据进行电商漏斗分析。

本项目区分三种漏斗口径：

1. 独立行为漏斗；
2. 用户级严格序列漏斗；
3. Session 级严格序列漏斗。

#### 6.2.1 独立行为漏斗

独立行为漏斗分别统计发生过 view、cart、purchase 的用户数，不要求行为顺序。

| 行为 | 用户数 | 相对 view 用户占比 |
|---|---:|---:|
| view | 1,781,733 | 100.00% |
| cart | 158,286 | 8.88% |
| purchase | 186,997 | 10.50% |

需要注意的是，独立行为统计不要求用户满足 view → cart → purchase 的时间顺序，因此 purchase 用户数可能高于 cart 用户数。这并不表示漏斗错误，而是说明部分用户可能直接购买，或者加购行为没有被完整记录。

#### 6.2.2 用户级严格序列漏斗

用户级严格序列漏斗要求用户在观察期内满足：

```text
view → cart → purchase
```

结果如下：

| 漏斗步骤 | 用户数 | 分步转化率 | 相对 view 转化率 |
|---|---:|---:|---:|
| view | 1,781,733 | - | 100.00% |
| cart_after_view | 158,096 | 8.87% | 8.87% |
| purchase_after_cart | 91,951 | 58.16% | 5.16% |

该结果说明：

- 从浏览到加购存在明显流失；
- 一旦用户完成加购，后续购买概率明显提高；
- 加购行为是非常重要的购买意向信号。

#### 6.2.3 Session 级严格序列漏斗

Session 级严格序列漏斗要求用户在同一个 session 内完成：

```text
view → cart → purchase
```

结果如下：

| 漏斗步骤 | Session 数 | 分步转化率 | 相对 view 转化率 |
|---|---:|---:|---:|
| view | 4,382,818 | - | 100.00% |
| cart_after_view | 242,745 | 5.54% | 5.54% |
| purchase_after_cart | 123,376 | 50.83% | 2.82% |

Session 级漏斗比用户级漏斗更加严格，因为它要求完整转化链路必须发生在同一个 session 内，因此整体转化率低于用户级严格序列漏斗。

主要输出：

```text
outputs/tables/independent_funnel_user_count.csv
outputs/tables/user_level_sequential_funnel.csv
outputs/tables/session_level_sequential_funnel.csv
outputs/tables/daily_independent_funnel.csv
outputs/tables/daily_user_level_sequential_funnel.csv
outputs/tables/daily_gmv_summary.csv

outputs/figures/independent_funnel_user_count.png
outputs/figures/user_level_sequential_funnel.png
outputs/figures/session_level_sequential_funnel.png
outputs/figures/daily_independent_behavior_user_count.png
outputs/figures/daily_user_level_sequential_funnel_users.png
outputs/figures/daily_user_level_sequential_funnel_rate.png
outputs/figures/daily_gmv_trend.png
```

---

### 6.3 03_user_level_feature_engineering.ipynb

该 notebook 将原始行为日志转换为用户级监督学习样本。

建模任务为：

```text
使用用户过去 7 天的行为特征，预测用户在 prediction_date 当天是否会购买。
```

样本粒度为：

```text
一行 = 一个用户在某个 prediction_date 的预测样本
```

特征窗口和标签窗口设计如下：

```text
观察窗口：prediction_date 前 7 天
标签窗口：prediction_date 当天
```

例如：

```text
prediction_date = 2019-10-08
观察窗口 = 2019-10-01 至 2019-10-07
标签窗口 = 2019-10-08
```

最终建模样本如下：

| 指标 | 数值 |
|---|---:|
| 样本量 | 8,178,887 |
| 特征数 | 32 |
| 正样本数 | 88,321 |
| 负样本数 | 8,090,566 |
| 正样本比例 | 1.08% |

prediction_date 范围：

```text
2019-10-08 至 2019-10-15
```

构造的特征主要包括以下几类。

#### 行为频次特征

| 特征 | 含义 |
|---|---|
| user_view_cnt_7d | 用户过去 7 天浏览次数 |
| user_cart_cnt_7d | 用户过去 7 天加购次数 |
| user_purchase_cnt_7d | 用户过去 7 天购买次数 |

#### 活跃度特征

| 特征 | 含义 |
|---|---|
| user_active_days_7d | 用户过去 7 天活跃天数 |
| user_active_hours_7d | 用户过去 7 天活跃小时数 |
| user_session_cnt_7d | 用户过去 7 天 session 数 |

#### 商品与类目多样性特征

| 特征 | 含义 |
|---|---|
| user_unique_item_cnt_7d | 用户过去 7 天浏览或交互过的不同商品数 |
| user_unique_category_cnt_7d | 用户过去 7 天交互过的不同类目数 |
| user_unique_brand_cnt_7d | 用户过去 7 天交互过的不同品牌数 |

#### 价格偏好特征

| 特征 | 含义 |
|---|---|
| user_price_mean_7d | 用户过去 7 天交互商品的平均价格 |
| user_price_median_7d | 用户过去 7 天交互商品的价格中位数 |
| user_price_std_7d | 用户过去 7 天交互商品价格标准差 |

#### 最近一次行为间隔特征

| 特征 | 含义 |
|---|---|
| user_last_event_gap_hours | 距离最近一次任意行为的小时数 |
| user_last_view_gap_hours | 距离最近一次浏览的小时数 |
| user_last_cart_gap_hours | 距离最近一次加购的小时数 |
| user_last_purchase_gap_hours | 距离最近一次购买的小时数 |

#### 行为比例特征

| 特征 | 含义 |
|---|---|
| user_cart_rate_7d | 过去 7 天加购次数 / 浏览次数 |
| user_purchase_rate_7d | 过去 7 天购买次数 / 浏览次数 |
| user_purchase_per_cart_7d | 过去 7 天购买次数 / 加购次数 |
| user_view_per_active_day_7d | 过去 7 天日均浏览次数 |

主要输出：

```text
data/processed/model_data_user_level_7d.csv
outputs/tables/user_level_feature_list.csv
```

---

### 6.4 04_modeling_gbdt_compare.ipynb

该 notebook 基于用户级 7 日行为特征进行转化预测建模，并比较多个模型的效果。

本项目比较了以下模型：

- Logistic Regression
- LightGBM
- XGBoost
- CatBoost

其中，LightGBM、XGBoost 和 CatBoost 都属于 GBDT 系列模型。相比线性模型，GBDT 模型更适合捕捉非线性关系、特征交互和复杂用户行为模式。

#### 时间切分方式

为了模拟真实业务中的“用历史预测未来”，本项目采用基于 prediction_date 的时间切分：

```text
训练集：2019-10-08 至 2019-10-12
测试集：2019-10-13 至 2019-10-15
```

数据规模如下：

| 数据集 | 样本数 | 正样本比例 |
|---|---:|---:|
| 训练集 | 5,009,719 | 1.03% |
| 测试集 | 3,169,168 | 1.16% |

由于购买用户占比较低，项目使用类别权重处理类别不平衡问题，并且不以 Accuracy 作为核心指标，而是重点关注：

- AUC
- PR-AUC
- Top-K Purchase Rate
- Recall@K
- Lift@K
- 预测排序收益模拟

#### 模型评估结果

| 模型 | AUC | PR-AUC |
|---|---:|---:|
| XGBoost | 0.813956 | 0.130591 |
| LightGBM | 0.812688 | 0.129152 |
| CatBoost | 0.813681 | 0.127708 |
| Logistic Regression | 0.802465 | 0.100274 |

从整体排序能力看，XGBoost 的 AUC 和 PR-AUC 略高，说明其整体区分购买用户和非购买用户的能力略强。

但在有限预算营销场景中，更重要的是模型是否能把最有购买倾向的用户排在前面，因此项目进一步比较 Top-K 业务指标。

#### CatBoost Top-K 业务指标

| Top-K | Top-K 用户数 | Top-K 购买率 | Recall@K | Lift@K |
|---|---:|---:|---:|---:|
| Top 5% | 158,458 | 9.20% | 39.62% | 7.92 |
| Top 10% | 316,916 | 6.24% | 53.78% | 5.38 |
| Top 20% | 633,833 | 4.02% | 69.22% | 3.46 |
| Top 30% | 950,750 | 2.98% | 77.11% | 2.57 |

测试集整体购买率仅为 1.16%，而 CatBoost Top 5% 用户购买率达到 9.20%，约为整体购买率的 7.92 倍。这说明 CatBoost 在高价值用户筛选方面具有较强业务价值。

因此，本项目最终选择 CatBoost 作为业务排序与用户分层的主模型。

#### 预测排序收益模拟

项目进一步构造了基于预测排序的收益模拟：

```text
触达成本 = 1
单次购买收益 = 100
```

该部分用于衡量模型排序在有限预算投放场景中的业务价值。

主要输出：

```text
outputs/tables/model_eval_results.csv
outputs/tables/all_models_topk_results.csv
outputs/tables/top10_model_compare.csv
outputs/tables/test_prediction_results.csv
outputs/tables/prediction_roi_simulation.csv
outputs/tables/best_prediction_targeting_strategy.csv

outputs/models/catboost_model.cbm
outputs/models/feature_cols.json

outputs/figures/model_auc_pr_compare.png
outputs/figures/roc_curve_compare.png
outputs/figures/pr_curve_compare.png
outputs/figures/topk_purchase_rate_compare.png
outputs/figures/recall_at_k_compare.png
outputs/figures/lift_at_k_compare.png
```

---

### 6.5 05_shap_interpretation.ipynb

该 notebook 对最终选择的 CatBoost 模型进行解释和用户分层。

主要步骤包括：

1. 读取 04 保存的 CatBoost 模型；
2. 读取 04 保存的测试集；
3. 重新预测并检查与 04 保存结果的一致性；
4. 使用 SHAP 解释模型；
5. 对高分用户和低分用户进行画像对比；
6. 按预测分数进行用户分层；
7. 输出分层运营策略。

#### 模型一致性检查

在 05 中重新加载 CatBoost 模型，并与 04 中保存的预测结果进行一致性检查：

| 指标 | 数值 |
|---|---:|
| 最大绝对误差 | 1.11e-16 |
| 平均绝对误差 | 2.62e-17 |

该结果属于浮点数精度误差，说明模型加载、特征顺序和测试集口径完全一致。

#### SHAP 特征重要性

SHAP 分析显示，模型最重要的特征包括：

| 排名 | 特征 |
|---:|---|
| 1 | user_last_event_gap_hours |
| 2 | user_view_cnt_7d |
| 3 | user_last_view_gap_hours |
| 4 | user_purchase_item_nunique_7d |
| 5 | user_active_hours_7d |
| 6 | user_view_per_active_day_7d |
| 7 | user_purchase_rate_7d |
| 8 | user_session_cnt_7d |
| 9 | user_purchase_cnt_7d |
| 10 | user_last_cart_gap_hours |

这说明模型主要依据以下信号判断用户短期购买概率：

- 用户最近是否活跃；
- 用户过去 7 天浏览强度；
- 用户过去是否有购买行为；
- 用户是否有加购行为；
- 用户活跃时段和 session 行为；
- 用户距离最近一次行为的时间间隔。

#### 高分用户 vs 低分用户画像

按 CatBoost 预测分数排序，取前 10% 和后 10% 用户进行对比：

| 用户组 | 购买率 |
|---|---:|
| 高分用户 Top 10% | 6.24% |
| 低分用户 Bottom 10% | 0.20% |
| 整体测试集 | 1.16% |

高分用户显著更活跃，过去 7 天内浏览次数、购买次数、会话数、活跃天数、购买率和加购率都更高，且最近一次行为距离预测日更近。

#### 用户分层结果

根据 CatBoost 预测分数，将用户分为五层：

| 用户层级 | 用户数 | 购买人数 | 购买率 | 平均预测分 |
|---|---:|---:|---:|---:|
| Very Low | 635,022 | 1,422 | 0.22% | 0.154 |
| Low | 632,645 | 2,072 | 0.33% | 0.202 |
| Medium | 633,834 | 2,900 | 0.46% | 0.264 |
| High | 633,833 | 4,929 | 0.78% | 0.389 |
| Very High | 633,834 | 25,466 | 4.02% | 0.701 |

用户分层结果显示，随着模型预测分数升高，真实购买率明显上升，说明模型具有较好的业务排序能力。

#### 分层运营策略

| 用户层级 | 用户特征 | 推荐策略 | 业务目标 |
|---|---|---|---|
| Very High | 近期活跃、浏览或购买信号强、短期购买概率最高 | 重点触达，限时优惠、购物车召回、库存提醒 | 提高短期转化 |
| High | 有较强购买倾向，但不如 Very High 稳定 | 个性化推荐、优惠券测试、相似商品推荐 | 推动潜在高意向用户转化 |
| Medium | 有一定兴趣，但转化信号不强 | 内容种草、低成本推荐、提高兴趣 | 培养用户兴趣 |
| Low | 购买意向较弱，短期转化概率较低 | 降低触达频率，避免过度营销 | 控制成本 |
| Very Low | 几乎没有明显购买信号 | 暂不重点触达，控制营销成本 | 减少无效触达 |

主要输出：

```text
outputs/tables/shap_feature_importance.csv
outputs/tables/high_vs_low_score_user_profile.csv
outputs/tables/user_score_segment_summary.csv
outputs/tables/user_score_segment_strategy.csv

outputs/figures/shap_summary_bar.png
outputs/figures/shap_summary_dot.png
outputs/figures/score_segment_purchase_rate.png
outputs/figures/score_segment_avg_pred_score.png
```

---

## 7. 核心指标解释

### 7.1 AUC

AUC 衡量模型整体排序能力。可以理解为随机抽取一个购买用户和一个未购买用户，模型将购买用户排在未购买用户前面的概率。

AUC 越高，说明模型整体排序能力越强。

---

### 7.2 PR-AUC

PR-AUC 是 Precision-Recall 曲线下的面积，更适合正负样本极度不平衡的场景。

本项目中，购买用户占比约为 1%，如果只看 Accuracy，很容易被大量负样本掩盖模型问题。因此，PR-AUC 比 Accuracy 更有参考价值。

---

### 7.3 Top-K Purchase Rate

Top-K Purchase Rate 衡量模型预测分数最高的前 K% 用户中的真实购买率。

在有限营销预算场景中，该指标非常重要，因为平台通常只会触达一部分高分用户。

---

### 7.4 Recall@K

Recall@K 衡量前 K% 用户覆盖了全部真实购买用户中的多少比例。

例如，CatBoost 在 Top 20% 用户中覆盖了约 69.22% 的真实购买用户，说明模型可以用较小触达范围覆盖大部分潜在购买用户。

---

### 7.5 Lift@K

Lift@K 表示 Top-K 用户购买率相对于整体购买率的提升倍数。

例如，CatBoost 的 Lift@5% 为 7.92，说明模型筛选出的前 5% 用户购买率约为整体平均水平的 7.92 倍。

---

### 7.6 预测排序 ROI

预测排序 ROI 用于模拟在有限营销预算下，模型排序能够带来的业务收益。

本项目假设：

```text
触达成本 = 1
单次购买收益 = 100
```

然后按照模型预测分数从高到低选取 Top-K 用户，比较不同投放比例下的购买人数、收益、成本和净收益。

需要注意的是，本项目中的 ROI 是普通转化预测场景下的排序收益模拟，不是因果推断中的增量 ROI。如果要严格评估营销触达是否真的带来额外购买，需要进一步结合 A/B 测试或 uplift modeling。

---

### 7.7 SHAP

SHAP 用于解释模型预测结果，衡量每个特征对模型输出的贡献。

通过 SHAP 可以理解模型为什么认为某些用户购买概率更高，也可以帮助业务人员理解模型是否符合业务逻辑。

---

## 8. 项目结论

本项目构建了一套完整的电商转化预测与高价值用户识别流程。

主要结论如下：

1. 电商行为存在明显漏斗结构，大量用户停留在浏览阶段，真正加购和购买的比例较低。
2. 用户级严格序列漏斗显示，从浏览到加购存在明显流失，但加购后的购买转化率较高。
3. 基于过去 7 天行为构造的时序特征可以有效刻画用户短期购买意向。
4. GBDT 类模型整体表现优于 Logistic Regression，说明非线性模型更适合刻画复杂用户行为模式。
5. 在整体模型评估中，XGBoost 的 AUC 和 PR-AUC 略高。
6. 在 Top-K 高价值用户筛选场景中，CatBoost 表现最好。
7. CatBoost Top 5% 用户购买率达到 9.20%，约为整体购买率的 7.92 倍。
8. SHAP 分析显示，最近一次行为间隔、浏览次数、购买历史、活跃小时和 session 数是影响预测结果的重要因素。
9. 用户分层结果显示，Very High 用户组购买率达到 4.02%，显著高于 Very Low 用户组的 0.22%。
10. 模型可以支持电商平台进行高价值用户识别、购物车召回、个性化推荐和营销预算分配。

---

## 9. 项目亮点

本项目亮点包括：

1. 使用真实大规模电商行为日志，数据量超过 2,000 万行；
2. 在 01 中统一清洗并保存 clean 数据，保证后续 notebook 数据口径一致；
3. 明确区分独立行为漏斗、用户级严格序列漏斗和 Session 级严格序列漏斗；
4. 构造用户级 7 日观察窗口特征，避免使用未来信息；
5. 使用时间切分模拟真实业务预测场景；
6. 比较 Logistic Regression、LightGBM、XGBoost 和 CatBoost 多类模型；
7. 通过 GBDT 模型捕捉复杂非线性关系和用户行为特征交互；
8. 针对类别不平衡问题使用 AUC、PR-AUC、Top-K、Recall@K 和 Lift@K 进行评估；
9. 引入预测排序 ROI，更贴近有限预算营销触达场景；
10. 使用 SHAP 解释 CatBoost 模型，提升模型可解释性；
11. 输出用户分层和运营策略，形成从模型到业务动作的完整闭环。

---

## 10. 如何运行

### 10.1 安装依赖

```bash
pip install -r requirements.txt
```

### 10.2 准备数据

将原始数据放置在：

```text
data/raw/2019-Oct.csv
```

### 10.3 按顺序运行 notebooks

```text
notebooks/01_data_understanding.ipynb
notebooks/02_funnel_analysis.ipynb
notebooks/03_user_level_feature_engineering.ipynb
notebooks/04_modeling_gbdt_compare.ipynb
notebooks/05_shap_interpretation.ipynb
```

---

## 11. 主要输出文件

### 数据文件

```text
data/processed/ecommerce_behavior_2019_10_01_15_clean.csv
data/processed/model_data_user_level_7d.csv
data/processed/test_data_for_model_interpretation.csv
```

### 模型文件

```text
outputs/models/catboost_model.cbm
outputs/models/feature_cols.json
```

### 表格输出

```text
outputs/tables/event_type_count_summary.csv
outputs/tables/clean_behavior_data_quality_summary.csv
outputs/tables/independent_funnel_user_count.csv
outputs/tables/user_level_sequential_funnel.csv
outputs/tables/session_level_sequential_funnel.csv
outputs/tables/daily_independent_funnel.csv
outputs/tables/daily_user_level_sequential_funnel.csv
outputs/tables/daily_gmv_summary.csv
outputs/tables/user_level_feature_list.csv
outputs/tables/model_eval_results.csv
outputs/tables/all_models_topk_results.csv
outputs/tables/top10_model_compare.csv
outputs/tables/test_prediction_results.csv
outputs/tables/prediction_roi_simulation.csv
outputs/tables/best_prediction_targeting_strategy.csv
outputs/tables/shap_feature_importance.csv
outputs/tables/high_vs_low_score_user_profile.csv
outputs/tables/user_score_segment_summary.csv
outputs/tables/user_score_segment_strategy.csv
```

### 图像输出

```text
outputs/figures/event_type_count_distribution.png
outputs/figures/independent_funnel_user_count.png
outputs/figures/user_level_sequential_funnel.png
outputs/figures/session_level_sequential_funnel.png
outputs/figures/daily_independent_behavior_user_count.png
outputs/figures/daily_user_level_sequential_funnel_users.png
outputs/figures/daily_user_level_sequential_funnel_rate.png
outputs/figures/daily_gmv_trend.png
outputs/figures/model_auc_pr_compare.png
outputs/figures/roc_curve_compare.png
outputs/figures/pr_curve_compare.png
outputs/figures/topk_purchase_rate_compare.png
outputs/figures/recall_at_k_compare.png
outputs/figures/lift_at_k_compare.png
outputs/figures/shap_summary_bar.png
outputs/figures/shap_summary_dot.png
outputs/figures/score_segment_purchase_rate.png
outputs/figures/score_segment_avg_pred_score.png
```
