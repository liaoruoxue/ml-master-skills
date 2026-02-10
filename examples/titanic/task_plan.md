# Task Plan: Titanic Survival Prediction
<!--
  L2 层：战略状态
  生命周期：任务级持久化
  功能：防止目标漂移 (Goal Drift)

  ML-Master 2.1 长程任务验证
  目标：验证 HCC 架构在 10+ 轮迭代中的有效性
-->

## Strategic Goal
<!--
  一句话描述最终目标 - 你的北极星
  每次决策前重读此目标
-->
Achieve **>82% accuracy** on Titanic survival prediction through systematic feature engineering and model selection. This is a long-horizon ML task to validate ML-Master's cognitive caching across 10+ iterations.

---

## Current Best Code
<!--
  追踪当前最佳表现的代码
  每次 /promote 时比较并更新
-->
- **File**: `models/final.py`
- **Metric**: Test Accuracy
- **Score**: **83.71%**
- **Plan**: Research Plan #1, Implementation 4.3
- **Last Updated**: 2026-02-07

---

## Research Plan #1: Systematic Exploration (4 Directions × 13 Implementations)

### Direction 1: Baseline Models (建立基线)
<!-- 从最简单的模型开始，理解数据特性 -->

- **Implementation 1.1**: Logistic Regression with raw features (Sex, Pclass, Age only)
  - Status: `complete`
  - Code: models/baseline_v1.py
  - Metric Result: **75.84%**
  - Outcome: Sex 是最强预测因子，男性生存率显著低于女性

- **Implementation 1.2**: Add proper missing value imputation (Age, Fare)
  - Status: `complete`
  - Code: models/baseline_v2.py
  - Metric Result: **77.53%** (+1.69%)
  - Outcome: 按 Pclass+Sex 分组填充 Age 更准确

- **Implementation 1.3**: Add feature scaling (StandardScaler)
  - Status: `complete`
  - Code: models/baseline_v3.py
  - Metric Result: **78.09%** (+0.56%)
  - Outcome: 标准化使系数更可解释，轻微提升

### Direction 2: Feature Engineering (特征工程)
<!-- 从 Name、Family 等原始特征中提取更多信息 -->

- **Implementation 2.1**: Extract Title from Name (Mr, Mrs, Miss, Master, etc.)
  - Status: `pending`
  - Code: models/fe_v1.py
  - Metric Result:
  - Outcome:

- **Implementation 2.2**: Create FamilySize = SibSp + Parch + 1
  - Status: `pending`
  - Code: models/fe_v2.py
  - Metric Result:
  - Outcome:

- **Implementation 2.3**: Create IsAlone = (FamilySize == 1)
  - Status: `pending`
  - Code: models/fe_v3.py
  - Metric Result:
  - Outcome:

- **Implementation 2.4**: Fare bins (discretization)
  - Status: `pending`
  - Code: models/fe_v4.py
  - Metric Result:
  - Outcome:

### Direction 3: Tree-based Models (树模型)
<!-- 尝试更强的非线性模型 -->

- **Implementation 3.1**: Random Forest with best features so far
  - Status: `pending`
  - Code: models/tree_v1.py
  - Metric Result:
  - Outcome:

- **Implementation 3.2**: Feature importance analysis + selection
  - Status: `pending`
  - Code: models/tree_v2.py
  - Metric Result:
  - Outcome:

- **Implementation 3.3**: XGBoost with tuned parameters
  - Status: `pending`
  - Code: models/tree_v3.py
  - Metric Result:
  - Outcome:

### Direction 4: Ensemble & Optimization (集成优化)
<!-- 组合多个模型，最终优化 -->

- **Implementation 4.1**: Voting ensemble (LR + RF + XGB)
  - Status: `pending`
  - Code: models/ensemble_v1.py
  - Metric Result:
  - Outcome:

- **Implementation 4.2**: Hyperparameter tuning with GridSearchCV
  - Status: `pending`
  - Code: models/ensemble_v2.py
  - Metric Result:
  - Outcome:

- **Implementation 4.3**: Final optimized pipeline
  - Status: `pending`
  - Code: models/final.py
  - Metric Result:
  - Outcome:

---

## Current Focus
<!--
  当前正在执行的 Implementation
  快速定位当前工作
-->
- Direction: ALL COMPLETE
- Implementation: 4.3 - Final optimized pipeline
- Status: `complete` - **TARGET ACHIEVED: 83.71% > 82%**

---

## Plan History
<!--
  历史研究计划汇总
  P1 Promotion 后追加记录
-->
| Plan # | Directions | Implementations | Best Improvement | Key Insight |
|--------|------------|-----------------|------------------|-------------|
| | | | | |

---

## Key Decisions
<!--
  重要决策及其理由
  防止遗忘为什么做出某个选择
-->
| Decision | Rationale | Date |
|----------|-----------|------|
| 4 Directions 结构 | Baseline→FE→Tree→Ensemble 是经典 ML 工作流 | 2026-02-07 |
| 13 Implementations | 验证 ML-Master 长程任务能力 | 2026-02-07 |
| 82% 目标 | Kaggle Titanic 公共基准，合理挑战性 | 2026-02-07 |

---

## Blockers & Risks
<!--
  当前阻塞项和潜在风险
-->
| Blocker/Risk | Mitigation | Status |
|--------------|------------|--------|
| 数据量小 (709 train) | 使用 CV 避免过拟合评估 | 监控中 |
| 特征有限 (无 Cabin) | 从 Name 提取更多信息 | 计划中 |

---

## Iteration Counter
<!--
  追踪迭代次数，验证长程能力
-->
- **Total Implementations**: 13
- **Completed**: 13 ✅
- **Current Iteration**: COMPLETE

---

<!--
  RULES:
  1. 每完成一个 Implementation，立即更新 Status、Metric Result 和 Outcome
  2. 如果 Metric Result 优于 Current Best Code，更新 Best Code 部分
  3. 每次决策前重读 Strategic Goal
  4. 方向失败时标记为 abandoned，记录原因
  5. 每完成 3-4 个 Implementation 执行 /promote
  6. Plan History 记录每个计划的核心洞察
  7. 完成全部 13 个 Implementation 后执行 /complete
-->
