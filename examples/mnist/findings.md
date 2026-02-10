# Findings & Strategic Memory
<!--
  L2 层：提炼知识
  生命周期：任务级持久化
  内容：从 L1 提炼的高价值信息，去除噪音
-->

## Key Insights
<!--
  关键判断和发现
  例如："特征 A 导致了过拟合"
  例如："数据集存在类别不平衡问题"
-->
- MNIST 28x28 灰度图非常简单，2-layer CNN (Conv32→Conv64→FC) 即可达到 99%+
- 5 epochs 足够收敛，训练时间 <1 分钟 (MPS 加速)
- 简单任务无需复杂架构，baseline 即可达标

---

## Validated Hypotheses
<!--
  已验证的假设和实验结论
  格式：假设 -> 验证方法 -> 结论
-->
| Hypothesis | Validation | Conclusion |
|------------|------------|------------|
| 简单 CNN 可达 99%+ on MNIST | cnn_v1.py 训练 5 epochs | ✅ 99.20% - 目标达成 |

---

## Failed Attempts
<!--
  失败的尝试及原因
  防止重复犯错
-->
| Attempt | Why Failed | Lesson |
|---------|------------|--------|
| | | |

---

## Experiment Results
<!--
  实验结果汇总 - 结构化记录每个 Research Plan 的结果
  用于跨计划比较和决策参考
-->
| Plan | Direction | Implementation | Metric | Score | Conclusion |
|------|-----------|----------------|--------|-------|------------|
| #1 | Simple CNN | 1.1 - 2-layer CNN | Test Acc | **99.20%** | Target achieved (>99%) |

---

## Best Code History
<!--
  最佳代码演进记录
  追踪性能提升的关键节点
-->
| Plan | Score | Key Change | Date |
|------|-------|------------|------|
| #1-1.1 | 99.20% | Baseline 2-layer CNN | 2026-02-06 |

---

## Technical Decisions
<!--
  技术决策及理由
  从 task_plan.md 的决策中提炼
-->
| Decision | Rationale |
|----------|-----------|
| 简化架构 (2-layer) | MNIST 28x28 灰度图简单，无需复杂模型 |
| 5 epochs | MNIST 收敛快，5 epochs 足够 |

---

## Resources
<!--
  有用的资源链接、文件路径、API 参考
-->
-

---

<!--
  RULES:
  1. 只记录结论和洞察，不记录执行细节（执行细节在 execution_trace.md）
  2. /promote 时从 L1 提取洞察写入此文件
  3. 此文件只增不减（除非信息被证明错误）
  4. /clear 后通过此文件恢复认知状态
-->
*Updated via /promote from execution_trace.md*
