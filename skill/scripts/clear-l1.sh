#!/bin/bash
# ML-Master: Clear L1 (execution_trace.md)
# Resets execution_trace.md to empty template state

set -e

L1_FILE="${1:-execution_trace.md}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/../templates"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M")

echo "=== ML-Master: Clear L1 ==="
echo ""

if [ ! -f "$L1_FILE" ]; then
    echo "INFO: $L1_FILE not found. Nothing to clear."
    exit 0
fi

# Backup current L1 size for verification
OLD_LINES=$(wc -l < "$L1_FILE" | tr -d ' ')
echo "Current L1 size: $OLD_LINES lines"

# Get current phase from task_plan.md if exists
CURRENT_PHASE="[Next Phase]"
if [ -f "task_plan.md" ]; then
    # Try to extract current focus
    FOCUS=$(grep -A1 "## Current Focus" task_plan.md 2>/dev/null | tail -1 | sed 's/- Direction: //' || true)
    if [ -n "$FOCUS" ]; then
        CURRENT_PHASE="$FOCUS"
    fi
fi

# Reset L1 to template
cat > "$L1_FILE" << EOF
# Execution Trace
<!--
  L1 层：工作记忆 (RAM)
  生命周期：极短 - 阶段结束时必须清空
  内容：代码片段、终端输出、错误堆栈、临时观察
-->

## Current Phase: $CURRENT_PHASE
## Started: $TIMESTAMP

---

## Operations Log
<!--
  记录每个工具调用的摘要
  2-Action Rule: 每 2 次工具调用后必须更新此表
-->
| Timestamp | Action | Output Summary |
|-----------|--------|----------------|
| | | |

---

## Code Patches
<!-- 当前正在执行或测试的代码片段 -->


---

## Terminal Outputs
<!-- 原始终端输出和错误堆栈 -->


---

## Observations
<!--
  单步调试的临时观察
  实验中的临时发现
  待验证的假设
-->


---

<!--
  CRITICAL RULES:
  1. 只记录执行细节，不记录结论（结论去 findings.md）
  2. 阶段结束时执行 /promote 将洞察迁移到 L2
  3. 迁移后此文件必须清空
  4. 如果此文件过大（>100行），考虑提前 promote
-->
*This file is cleared after each phase completion via /promote*
EOF

NEW_LINES=$(wc -l < "$L1_FILE" | tr -d ' ')
echo "New L1 size: $NEW_LINES lines (template)"
echo ""
echo "L1 cleared successfully."
echo "Ready for next phase: $CURRENT_PHASE"
