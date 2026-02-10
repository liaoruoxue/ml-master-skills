# Task-specific Wisdom
<!--
  L3 层：任务级智慧存储
  生命周期：跨任务永久存在
  内容：从完成的任务中提炼的可迁移策略

  通过 P2 (Task-level Promotion) 更新：
  任务完成时执行 /complete 命令，总结并追加到此文件
-->

## How to Use This File

1. **任务开始时**: 根据任务类型查找相关 wisdom
2. **任务完成时**: 执行 /complete 添加新 wisdom
3. **格式要求**: 每条 wisdom 必须包含:
   - Task name and date
   - Key insight (what worked)
   - Best approach (recommended strategy)
   - Pitfalls (what to avoid)

---

## Image Classification

### Task: MNIST Handwritten Digit Classification - 2026-02-03
- **Key insight**: MNIST is essentially "solved" - a simple 2-layer CNN achieves 99%+ with minimal tuning. Focus on getting baseline working first.
- **Best approach**: Start with basic Conv-Pool-Conv-Pool-FC architecture using Adam (lr=1e-3), batch_size=64, 5 epochs. BatchNorm/Dropout provide only marginal gains (+0.05%).
- **Pitfalls**: Don't over-engineer for MNIST. Complex architectures, extensive regularization, or data augmentation are unnecessary and waste time.
- **Final Score**: Accuracy: 99.15%

### Task: MNIST Classification (ML-Master 2.1 Test) - 2026-02-06
- **Key insight**: 简单任务的 baseline 即可达标。Conv(32)→Conv(64)→FC(128)→10 在 MNIST 上 5 epochs 即可达到 99.20%。
- **Best approach**: 从最简单架构开始，验证 baseline 后再考虑优化。MPS 加速有效，45.6s 完成 5 epochs。
- **Pitfalls**: 不要在简单数据集上过度设计。MNIST 28x28 灰度图不需要 ResNet 或复杂正则化。
- **Final Score**: Accuracy: 99.20%

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

## Tabular Data

### Task: Titanic Survival Prediction (Long-horizon Test) - 2026-02-07
- **Key insight**: 13 个 Implementation 的系统性探索验证了 ML-Master 长程任务能力。Sex 和 IsAlone 是最强特征，Ensemble 效果最佳。
- **Best approach**: Baseline→Feature Engineering→Tree Models→Ensemble 的经典工作流。从 LR baseline 开始，逐步添加特征，最终用加权 Voting (LR+RF+GB, GB权重2x) 达到最佳效果。
- **Pitfalls**: (1) Title 与 Sex 冗余，检查相关性再添加；(2) RF 过拟合风险高 (CV↑ Test↓)；(3) FamilySize 需转换为 IsAlone 才有效；(4) GridSearchCV 不一定提升 Test 准确率。
- **Final Score**: Accuracy: 83.71% (13 iterations: 75.84% → 83.71%, +7.87%)

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

## NLP / Text

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

## Time Series

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

## Object Detection / Segmentation

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

## Recommendation / Ranking

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

## Other Tasks

### Template
<!--
### Task: [task_name] - [YYYY-MM-DD]
- **Key insight**: ...
- **Best approach**: ...
- **Pitfalls**: ...
- **Final Score**: [metric: score]
-->

---

<!--
  P2 PROMOTION RULES:
  1. 任务完成时调用 /complete 触发 P2
  2. Agent 根据 L1+L2 总结任务级 wisdom
  3. 按任务类型分类追加到对应 section
  4. Wisdom 应该是可迁移的策略，不是具体代码
  5. 这是长期记忆，不会被清空
-->
