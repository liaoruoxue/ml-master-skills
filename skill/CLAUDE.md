# ML-Master 2.1 Project Context

## What is ML-Master?

Hierarchical Cognitive Caching (HCC) system for ultra-long-horizon ML tasks.
Based on paper: arXiv:2601.10402v3

## Memory Architecture

```
L1: execution_trace.md  → 工作记忆 (阶段结束清空)
L2: findings.md         → 提炼知识 (任务级持久)
L2: task_plan.md        → 战略计划 (任务级持久)
L3: wisdom/             → 永久智慧 (跨任务)
```

## Quick Commands

| Command | Script | Purpose |
|---------|--------|---------|
| `/plan` | `scripts/init-session.sh` | 初始化任务 |
| `/status` | `scripts/status.sh` | 快速状态概览 |
| `/promote` | `scripts/promote.sh` | L1→L2 压缩 |
| `/recover` | `scripts/recover.sh` | /clear 后恢复状态 |
| `/complete` | `scripts/task-complete.sh` | 任务完成，P2 Promotion |

## Critical Rules

1. **5-Action Rule**: 每 5 次 Write/Edit/Bash 后更新 `execution_trace.md`
2. **Dual Write**: 执行细节→L1, 结论洞察→L2
3. **Best Code Tracking**: Metric 提升时更新 `task_plan.md`
4. **Post-Clear Recovery**: `/clear` 后必须 `/recover`

## Workflow Reminder

```
开始任务 → /plan
  ↓
执行实验 → 更新 L1 (5-Action Rule)
  ↓
阶段完成 → /promote (L1→L2)
  ↓
任务完成 → /complete (L2→L3)
```

## After /clear

Always run `/recover` immediately to restore cognitive state from L2 files.

---
*ML-Master v2.1.0 - Reduced hook noise, 5-Action Rule, /status command*
