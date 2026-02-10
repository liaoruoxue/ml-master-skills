#!/bin/bash
# ML-Master: Check task completion status
# Counts Implementation statuses in task_plan.md

PLAN_FILE="${1:-task_plan.md}"

echo "=== ML-Master: Task Completion Check ==="
echo ""

if [ ! -f "$PLAN_FILE" ]; then
    echo "ERROR: $PLAN_FILE not found"
    echo "Cannot verify completion without a task plan."
    exit 1
fi

# Count by status (using backtick format from template)
COMPLETE=$(grep -c "Status: \`complete\`" "$PLAN_FILE" 2>/dev/null | tr -d '[:space:]' || echo 0)
IN_PROGRESS=$(grep -c "Status: \`in_progress\`" "$PLAN_FILE" 2>/dev/null | tr -d '[:space:]' || echo 0)
PENDING=$(grep -c "Status: \`pending\`" "$PLAN_FILE" 2>/dev/null | tr -d '[:space:]' || echo 0)
ABANDONED=$(grep -c "Status: \`abandoned\`" "$PLAN_FILE" 2>/dev/null | tr -d '[:space:]' || echo 0)

# Ensure variables are valid integers
COMPLETE=${COMPLETE:-0}
IN_PROGRESS=${IN_PROGRESS:-0}
PENDING=${PENDING:-0}
ABANDONED=${ABANDONED:-0}

TOTAL=$((COMPLETE + IN_PROGRESS + PENDING))

echo "Implementation Status:"
echo "  Complete:    $COMPLETE"
echo "  In Progress: $IN_PROGRESS"
echo "  Pending:     $PENDING"
echo "  Abandoned:   $ABANDONED"
echo "  -----------"
echo "  Total:       $TOTAL"
echo ""

# Check if all are complete
if [ "$TOTAL" -eq 0 ]; then
    echo "WARNING: No implementations found in plan."
    echo "Make sure task_plan.md has Implementation entries with Status fields."
    exit 1
fi

if [ "$IN_PROGRESS" -gt 0 ] || [ "$PENDING" -gt 0 ]; then
    echo "TASK NOT COMPLETE"
    echo ""

    # Show current focus
    echo "Current Focus:"
    sed -n '/## Current Focus/,/^##/p' "$PLAN_FILE" | head -6
    echo ""

    echo "Continue working on remaining implementations."
    exit 1
else
    echo "ALL IMPLEMENTATIONS COMPLETE"
    echo ""
    echo "Task finished successfully."
    echo ""
    echo "Recommended: Run /promote one final time to capture any remaining insights."
    exit 0
fi
