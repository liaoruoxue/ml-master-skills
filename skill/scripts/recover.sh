#!/bin/bash
# ML-Master: Recover cognitive state after /clear
# Reads L2 files to rebuild context without conversation history

set -e

TASK_PLAN="${1:-task_plan.md}"
FINDINGS="${2:-findings.md}"

echo "=== ML-Master: Context Recovery ==="
echo ""

# Check task_plan.md
if [ -f "$TASK_PLAN" ]; then
    echo "--- STRATEGIC STATE (task_plan.md) ---"
    echo ""

    # Extract Strategic Goal
    echo "[GOAL]"
    sed -n '/## Strategic Goal/,/^##/p' "$TASK_PLAN" | head -5
    echo ""

    # Extract Current Focus
    echo "[CURRENT FOCUS]"
    sed -n '/## Current Focus/,/^##/p' "$TASK_PLAN" | head -10
    echo ""

    # Count status
    COMPLETE=$(grep -c "Status: \`complete\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)
    IN_PROGRESS=$(grep -c "Status: \`in_progress\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)
    PENDING=$(grep -c "Status: \`pending\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)
    ABANDONED=$(grep -c "Status: \`abandoned\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)

    # Ensure variables are valid integers
    COMPLETE=${COMPLETE:-0}
    IN_PROGRESS=${IN_PROGRESS:-0}
    PENDING=${PENDING:-0}
    ABANDONED=${ABANDONED:-0}

    echo "[PROGRESS]"
    echo "  Complete: $COMPLETE"
    echo "  In Progress: $IN_PROGRESS"
    echo "  Pending: $PENDING"
    echo "  Abandoned: $ABANDONED"
    echo ""
else
    echo "WARNING: $TASK_PLAN not found"
    echo ""
fi

# Check findings.md
if [ -f "$FINDINGS" ]; then
    echo "--- STRATEGIC MEMORY (findings.md) ---"
    echo ""

    # Extract Key Insights
    echo "[KEY INSIGHTS]"
    sed -n '/## Key Insights/,/^##/p' "$FINDINGS" | grep "^-" | head -10
    echo ""

    # Extract Failed Attempts
    echo "[FAILED ATTEMPTS]"
    sed -n '/## Failed Attempts/,/^##/p' "$FINDINGS" | grep "|" | head -5
    echo ""

    # Extract Validated Hypotheses
    echo "[VALIDATED]"
    sed -n '/## Validated Hypotheses/,/^##/p' "$FINDINGS" | grep "|" | head -5
    echo ""
else
    echo "WARNING: $FINDINGS not found"
    echo ""
fi

echo "=== RECOVERY COMPLETE ==="
echo ""
echo "You now have full context from L2 memory."
echo "DO NOT rely on conversation history - it has been cleared."
echo ""
echo "Next steps:"
echo "1. Read task_plan.md for full plan details"
echo "2. Read findings.md for all accumulated knowledge"
echo "3. Continue from Current Focus"
echo ""
