# Task Plan: [Title]
<!--
  L2 层：战略状态
  生命周期：任务级持久化
  功能：防止目标漂移 (Goal Drift)

  ML-Master 2.0 Research Plan 结构:
  - m Directions (探索方向)
  - q Implementations per Direction (具体实施)
-->

## Strategic Goal
<!--
  一句话描述最终目标 - 你的北极星
  每次决策前重读此目标
-->
[One sentence describing the end state]

---

## Current Best Code
<!--
  追踪当前最佳表现的代码
  每次 /promote 时比较并更新
-->
- **File**: `[path/to/best_code.py]`
- **Metric**: [evaluation metric name, e.g., F1, AUC, RMSE]
- **Score**: [current best score]
- **Plan**: [which plan produced this]
- **Last Updated**: [timestamp]

---

## Research Plan #[N]
<!--
  分层计划树结构：m Directions × q Implementations
  每个 Direction 代表一个探索方向
  每个 Implementation 是该方向的具体尝试
-->

### Direction 1: [探索方向名称]
<!-- 高层策略方向，例如：Model Architecture / Feature Engineering / Training Strategy -->

- **Implementation 1.1**: [具体实施步骤]
  - Status: `pending`
  - Code: [file path if applicable]
  - Metric Result: [score or N/A]
  - Outcome: [what was learned]

- **Implementation 1.2**: [具体实施步骤]
  - Status: `pending`
  - Code:
  - Metric Result:
  - Outcome:

### Direction 2: [探索方向名称]

- **Implementation 2.1**: [具体实施步骤]
  - Status: `pending`
  - Code:
  - Metric Result:
  - Outcome:

- **Implementation 2.2**: [具体实施步骤]
  - Status: `pending`
  - Code:
  - Metric Result:
  - Outcome:

### Direction 3: [探索方向名称]

- **Implementation 3.1**: [具体实施步骤]
  - Status: `pending`
  - Code:
  - Metric Result:
  - Outcome:

---

## Current Focus
<!--
  当前正在执行的 Implementation
  快速定位当前工作
-->
- Direction: [direction number and name]
- Implementation: [implementation number]
- Status: `in_progress`

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
| | | |

---

## Blockers & Risks
<!--
  当前阻塞项和潜在风险
-->
| Blocker/Risk | Mitigation | Status |
|--------------|------------|--------|
| | | |

---

<!--
  RULES:
  1. 每完成一个 Implementation，立即更新 Status、Metric Result 和 Outcome
  2. 如果 Metric Result 优于 Current Best Code，更新 Best Code 部分
  3. 每次决策前重读 Strategic Goal
  4. 方向失败时标记为 abandoned，记录原因
  5. 一个 Research Plan 完成后执行 /promote，开始新的 Plan #[N+1]
  6. Plan History 记录每个计划的核心洞察
-->
