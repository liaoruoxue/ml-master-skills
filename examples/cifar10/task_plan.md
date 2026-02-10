# Task Plan: CIFAR-10 Image Classification
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
Achieve >85% test accuracy on CIFAR-10 using PyTorch CNN within reasonable training time (~30min per model).

---

## Current Best Code
<!--
  追踪当前最佳表现的代码
  每次 /promote 时比较并更新
-->
- **File**: `models/resnet_v1.py`
- **Metric**: Test Accuracy
- **Score**: **89.38%**
- **Plan**: Research Plan #1
- **Last Updated**: 2026-02-06

---

## Research Plan #1
<!--
  分层计划树结构：m Directions × q Implementations
  每个 Direction 代表一个探索方向
  每个 Implementation 是该方向的具体尝试
-->

### Direction 1: Basic CNN Architecture
<!-- 从简单 CNN 开始，建立基线 -->

- **Implementation 1.1**: 3-layer CNN baseline
  - Status: `complete`
  - Code: `models/cnn_v1_hf.py`
  - Metric Result: **75.47%** (30 epochs)
  - Outcome: Baseline established. Gap to target: 9.53%
  - Plan: Conv(32) -> Conv(64) -> Conv(128) -> FC -> 10 classes

- **Implementation 1.2**: Add BatchNorm + Dropout
  - Status: `complete`
  - Code: `models/cnn_v2.py`
  - Metric Result: **82.59%** (30 epochs)
  - Outcome: +7.12% over baseline, regularization effective
  - Plan: Add BatchNorm after each Conv, Dropout(0.5) before FC

### Direction 2: ResNet-style Architecture
<!-- 使用残差连接提升性能 -->

- **Implementation 2.1**: Mini-ResNet (simplified ResNet-18)
  - Status: `complete`
  - Code: `models/resnet_v1.py`
  - Metric Result: **89.38%** (30 epochs)
  - Outcome: Skip connections + SGD with LR schedule achieved target
  - Plan: 3 ResBlock groups, [2,2,2] blocks, channels [64,128,256]

- **Implementation 2.2**: ResNet + Data Augmentation
  - Status: `pending`
  - Code: `models/resnet_v2.py`
  - Metric Result: N/A
  - Outcome: Test data augmentation impact
  - Plan: RandomCrop, HorizontalFlip, ColorJitter

### Direction 3: Training Optimization
<!-- 优化训练策略 -->

- **Implementation 3.1**: Learning Rate Schedule + Mixup
  - Status: `pending`
  - Code: `models/resnet_v3.py`
  - Metric Result: N/A
  - Outcome: Test advanced training techniques
  - Plan: CosineAnnealing LR, Mixup alpha=0.2

---

## Current Focus
<!--
  当前正在执行的 Implementation
  快速定位当前工作
-->
- Direction: Direction 2 - ResNet-style Architecture
- Implementation: Implementation 2.1 - Mini-ResNet
- Status: `complete` - **TARGET ACHIEVED: 89.38% > 85%**

---

## Plan History
<!--
  历史研究计划汇总
  P1 Promotion 后追加记录
-->
| Plan # | Directions | Implementations | Best Improvement | Key Insight |
|--------|------------|-----------------|------------------|-------------|
| 1 | 3 (2 executed) | 3 complete | 75.47% → 89.38% (+13.91%) | Skip connections + LR schedule critical for >85% |

---

## Key Decisions
<!--
  重要决策及其理由
  防止遗忘为什么做出某个选择
-->
| Decision | Rationale | Date |
|----------|-----------|------|
| Start with simple CNN | Establish baseline before complex architectures | Session Start |
| Use CIFAR-10 standard splits | 50k train / 10k test for fair comparison | Session Start |
| Target >85% accuracy | Achievable within time constraint, above random (10%) | Session Start |

---

## Blockers & Risks
<!--
  当前阻塞项和潜在风险
-->
| Blocker/Risk | Mitigation | Status |
|--------------|------------|--------|
| Training time on CPU | Use smaller models, fewer epochs | monitoring |
| Overfitting on small dataset | Data augmentation in Direction 2 | planned |

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
