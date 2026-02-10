# Execution Trace
<!--
  L1 层：工作记忆 (RAM)
  生命周期：极短 - 阶段结束时必须清空
  简化版：4 sections (Phase, Operations Log, Metric Log, Notes)
-->

## Phase: [Name] | Started: [Timestamp]

---

## Operations Log
<!--
  5-Action Rule: 每 5 次 Write/Edit/Bash 后更新此表
  记录重要操作和结果摘要
-->
| Time | Action | Summary |
|------|--------|---------|
| | | |

---

## Metric Log
<!--
  每次训练/评估后记录
  用于追踪 Best Code 和比较实验效果
-->
| Time | Code | Metric | Score | vs Best |
|------|------|--------|-------|---------|
| | | | | |

---

## Notes
<!--
  临时观察、待验证假设、错误堆栈、重要发现
  合并了原 Code Patches / Terminal Outputs / Observations
-->


---

<!--
  RULES:
  1. 只记录执行细节，不记录结论（结论去 findings.md）
  2. 阶段结束时执行 /promote 将洞察迁移到 L2
  3. 迁移后此文件必须清空
  4. 如果此文件过大（>80行），考虑提前 promote
-->
*Cleared after /promote*
