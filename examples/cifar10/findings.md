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
- BatchNorm + Dropout 是小数据集（50K样本）防过拟合的有效组合
- Skip connections 允许训练更深网络，显著提升性能
- SGD + LR schedule 对 ResNet 比 Adam 效果更好
- LR decay 是突破性能瓶颈的关键时刻（epoch 16: 80% -> 89%）
- HuggingFace datasets 在网络不佳时比 torchvision 更可靠

---

## Validated Hypotheses
<!--
  已验证的假设和实验结论
  格式：假设 -> 验证方法 -> 结论
-->
| Hypothesis | Validation | Conclusion |
|------------|------------|------------|
| BatchNorm 能提升泛化 | cnn_v1 vs cnn_v2 | +7.12% (75.47% -> 82.59%) |
| Skip connections 能提升深层网络 | cnn_v2 vs resnet_v1 | +6.79% (82.59% -> 89.38%) |
| LR schedule 对 SGD 必要 | 观察 epoch 15-16 跳跃 | 从 80% 跳到 88.8% |
| SGD+momentum 适合 ResNet | 直接使用 SGD 训练 | 达到 89.38% |

---

## Failed Attempts
<!--
  失败的尝试及原因
  防止重复犯错
-->
| Attempt | Why Failed | Lesson |
|---------|------------|--------|
| torchvision CIFAR-10 下载 | 网络慢（100KB/s）| 使用 HuggingFace datasets |
| 直接 multiprocessing 评估 | `__main__` 模块属性错误 | 创建独立 evaluate.py |

---

## Experiment Results
<!--
  实验结果汇总 - 结构化记录每个 Research Plan 的结果
  用于跨计划比较和决策参考
-->
| Plan | Direction | Implementation | Metric | Score | Conclusion |
|------|-----------|----------------|--------|-------|------------|
| 1 | Basic CNN | 1.1 BasicCNN | Test Acc | 75.47% | 基线建立 |
| 1 | Basic CNN | 1.2 RegularizedCNN | Test Acc | 82.59% | 正则化有效 |
| 1 | ResNet | 2.1 MiniResNet | Test Acc | 89.38% | 目标达成 |

---

## Best Code History
<!--
  最佳代码演进记录
  追踪性能提升的关键节点
-->
| Plan | Score | Key Change | Date |
|------|-------|------------|------|
| 1 | 75.47% | BasicCNN baseline | 2026-02-06 |
| 1 | 82.59% | +BatchNorm +Dropout | 2026-02-06 |
| 1 | 89.38% | +ResNet +SGD +LR schedule | 2026-02-06 |

---

## Technical Decisions
<!--
  技术决策及理由
  从 task_plan.md 的决策中提炼
-->
| Decision | Rationale |
|----------|-----------|
| 使用 HuggingFace datasets | torchvision 下载太慢 |
| SGD 而非 Adam | ResNet 原论文使用 SGD，效果更稳定 |
| MultiStepLR [15,25] | 在 epoch 15, 25 降 LR，平衡训练时间和精度 |
| 30 epochs | 足够收敛且训练时间合理（~100min） |
| Batch size 128 | 在 MPS GPU 上内存和速度的平衡 |

---

## Resources
<!--
  有用的资源链接、文件路径、API 参考
-->
- 数据集: `uoft-cs/cifar10` via HuggingFace datasets
- 最佳模型权重: `models/resnet_v1_best.pth`
- CIFAR-10 标准化参数: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

---

<!--
  RULES:
  1. 只记录结论和洞察，不记录执行细节（执行细节在 execution_trace.md）
  2. /promote 时从 L1 提取洞察写入此文件
  3. 此文件只增不减（除非信息被证明错误）
  4. /clear 后通过此文件恢复认知状态
-->
*Updated via P2 Promotion - Task Complete - 2026-02-06*
