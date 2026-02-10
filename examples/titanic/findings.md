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
- **Sex 是最强预测因子**: 男性生存率显著低于女性 (coef: -2.46)
- **Pclass 其次**: 低等舱生存率低 (coef: -1.19)
- **Title 与 Sex 高度相关**: 提取 Title 无额外增益
- **IsAlone 是强特征**: 独自旅行者生存率明显不同 (+2.81%)
- **FamilySize 需与 IsAlone 配合**: 单独使用反而降低准确率
- **RF 有过拟合风险**: CV 82% 但 Test 79%
- **GradientBoosting 泛化好**: 首个超过 82% 的模型
- **Ensemble 组合最有效**: 加权投票 (LR+RF+GB, GB权重2x) 达到最佳
- **最终: 83.71%** (目标 >82%) ✅

---

## Validated Hypotheses
<!--
  已验证的假设和实验结论
  格式：假设 -> 验证方法 -> 结论
-->
| Hypothesis | Validation | Conclusion |
|------------|------------|------------|
| 简单 LR 可达 75%+ | baseline_v1.py | ✅ 75.84% |
| 更好的缺失值填充可提升 | baseline_v2.py | ✅ +1.69% |
| Title 特征有用 | fe_v1.py | ❌ 与 Sex 冗余 |
| FamilySize 有预测力 | fe_v2.py, fe_v3.py | ⚠️ 需配合 IsAlone |
| Fare 离散化有帮助 | fe_v4.py | ✅ +0.56% |

---

## Failed Attempts
<!--
  失败的尝试及原因
  防止重复犯错
-->
| Attempt | Why Failed | Lesson |
|---------|------------|--------|
| Title extraction (fe_v1) | 与 Sex 高度相关 | 检查特征相关性再添加 |
| FamilySize alone (fe_v2) | 线性关系不足 | 需要转换为 IsAlone |

---

## Experiment Results
<!--
  实验结果汇总 - 结构化记录每个 Research Plan 的结果
  用于跨计划比较和决策参考
-->
| Plan | Direction | Implementation | Metric | Score | Conclusion |
|------|-----------|----------------|--------|-------|------------|
| #1 | Baseline | 1.1 LR raw | Test Acc | 75.84% | 基线建立 |
| #1 | Baseline | 1.2 imputation | Test Acc | 77.53% | 缺失值处理重要 |
| #1 | Baseline | 1.3 scaling | Test Acc | 78.09% | 标准化有帮助 |
| #1 | Feature Eng | 2.1 Title | Test Acc | 78.09% | 无增益 |
| #1 | Feature Eng | 2.2 FamilySize | Test Acc | 76.97% | 负增益 |
| #1 | Feature Eng | 2.3 IsAlone | Test Acc | 80.90% | **关键特征** |
| #1 | Feature Eng | 2.4 FareBin | Test Acc | 81.46% | 轻微提升 |
| #1 | Tree | 3.1 RF | Test Acc | 79.21% | 过拟合风险 |
| #1 | Tree | 3.2 Feature Sel | Test Acc | 80.34% | Top4特征最佳 |
| #1 | Tree | 3.3 GB | Test Acc | 82.58% | **首次达标** |
| #1 | Ensemble | 4.1 Voting | Test Acc | 83.15% | 组合有效 |
| #1 | Ensemble | 4.2 GridSearch | Test Acc | 82.58% | CV高Test低 |
| #1 | Ensemble | 4.3 Final | Test Acc | **83.71%** | **最终结果** |

---

## Best Code History
<!--
  最佳代码演进记录
  追踪性能提升的关键节点
-->
| Plan | Score | Key Change | Date |
|------|-------|------------|------|
| #1-1.1 | 75.84% | LR baseline | 2026-02-07 |
| #1-1.3 | 78.09% | + Scaling | 2026-02-07 |
| #1-2.3 | 80.90% | + IsAlone | 2026-02-07 |
| #1-2.4 | 81.46% | + FareBin | 2026-02-07 |
| #1-3.3 | 82.58% | GradientBoosting | 2026-02-07 |
| #1-4.1 | 83.15% | Voting Ensemble | 2026-02-07 |
| #1-4.3 | **83.71%** | Final Weighted Ensemble | 2026-02-07 |

---

## Technical Decisions
<!--
  技术决策及理由
  从 task_plan.md 的决策中提炼
-->
| Decision | Rationale |
|----------|-----------|
| | |

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
