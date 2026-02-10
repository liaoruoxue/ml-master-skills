#!/bin/bash
# ML-Master: Task-level Promotion (P2)
# Triggers when a task is complete, prompts Agent to distill wisdom into L3

set -e

TASK_PLAN="${1:-task_plan.md}"
FINDINGS="${2:-findings.md}"
WISDOM_FILE="${3:-wisdom/task_wisdom.md}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== ML-Master: Task-level Promotion (P2) ==="
echo ""

# Check if task plan exists
if [ ! -f "$TASK_PLAN" ]; then
    echo "ERROR: $TASK_PLAN not found"
    echo "Cannot perform P2 promotion without a task plan."
    exit 1
fi

# Check completion status
COMPLETE=$(grep -c "Status: \`complete\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)
PENDING=$(grep -c "Status: \`pending\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)
IN_PROGRESS=$(grep -c "Status: \`in_progress\`" "$TASK_PLAN" 2>/dev/null | tr -d '[:space:]' || echo 0)

# Ensure variables are valid integers
COMPLETE=${COMPLETE:-0}
PENDING=${PENDING:-0}
IN_PROGRESS=${IN_PROGRESS:-0}

echo "Task Status:"
echo "  Complete:    $COMPLETE"
echo "  In Progress: $IN_PROGRESS"
echo "  Pending:     $PENDING"
echo ""

if [ "$IN_PROGRESS" -gt 0 ] || [ "$PENDING" -gt 0 ]; then
    echo "WARNING: Task has incomplete implementations."
    echo "Consider finishing all implementations before P2 promotion."
    echo ""
fi

# Display task summary
echo "=== Task Summary ==="
echo ""
echo "--- Strategic Goal ---"
sed -n '/## Strategic Goal/,/^##/p' "$TASK_PLAN" | head -5
echo ""

echo "--- Current Best Code ---"
sed -n '/## Current Best Code/,/^##/p' "$TASK_PLAN" | head -8
echo ""

# Display findings summary
if [ -f "$FINDINGS" ]; then
    echo "--- Key Insights ---"
    sed -n '/## Key Insights/,/^##/p' "$FINDINGS" | head -10
    echo ""

    echo "--- Best Code History ---"
    sed -n '/## Best Code History/,/^##/p' "$FINDINGS" | head -10
    echo ""
fi

# Check if wisdom file exists
if [ ! -f "$WISDOM_FILE" ]; then
    echo "WARNING: $WISDOM_FILE not found. Creating from template..."
    mkdir -p "$(dirname "$WISDOM_FILE")"
    if [ -f "$SCRIPT_DIR/../wisdom/task_wisdom.md" ]; then
        cp "$SCRIPT_DIR/../wisdom/task_wisdom.md" "$WISDOM_FILE"
    fi
fi

echo "=== P2 PROMOTION INSTRUCTIONS ==="
echo ""
echo "As the Agent, you must now distill task-level wisdom:"
echo ""
echo "1. Review the task summary above"
echo "2. Identify TRANSFERABLE insights (not task-specific details)"
echo "3. Determine the task type (Image Classification, Tabular, NLP, etc.)"
echo "4. Add a new wisdom entry to: $WISDOM_FILE"
echo ""
echo "Wisdom entry format:"
echo "### Task: [task_name] - $(date +%Y-%m-%d)"
echo "- **Key insight**: [what worked best]"
echo "- **Best approach**: [recommended strategy for similar tasks]"
echo "- **Pitfalls**: [what to avoid]"
echo "- **Final Score**: [metric: score]"
echo ""
echo "5. After adding wisdom, the task is complete!"
echo ""
echo "================================="

# P2 Embedding Index Update
WISDOM_DIR="${SCRIPT_DIR}/../wisdom"
EMBEDDING_SCRIPT="$WISDOM_DIR/embedding_utils.py"

echo ""
echo "=== P2 Embedding Index ==="

# Extract task info from task_plan.md
TASK_TITLE=$(grep -m1 "^# Task Plan:" "$TASK_PLAN" 2>/dev/null | sed 's/# Task Plan: //' || echo "unknown")
TASK_ID=$(echo "$TASK_TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-')
TASK_ID="${TASK_ID:-task-$(date +%Y%m%d)}"

# Try to extract strategic goal as descriptor
STRATEGIC_GOAL=$(sed -n '/## Strategic Goal/,/^##/p' "$TASK_PLAN" 2>/dev/null | grep -v "^##" | grep -v "^$" | head -1 || echo "")
DESCRIPTOR="${STRATEGIC_GOAL:-Machine learning task}"

echo "Task ID: $TASK_ID"
echo "Descriptor: ${DESCRIPTOR:0:80}..."
echo ""

# Prompt user for task type if not auto-detected
echo "Please specify the task type for embedding index:"
echo "  1. image_classification"
echo "  2. tabular"
echo "  3. nlp"
echo "  4. time_series"
echo "  5. object_detection"
echo "  6. recommendation"
echo "  7. other"
echo ""

# 检测 Python 环境: 优先使用 uv run，然后 python3
PYTHON_CMD=""
if [ -f "pyproject.toml" ] && command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Check if embedding_utils.py exists and Python is available
if [ -f "$EMBEDDING_SCRIPT" ] && [ -n "$PYTHON_CMD" ]; then
    echo "To add this task to the embedding index, run:"
    echo ""
    echo "  $PYTHON_CMD $EMBEDDING_SCRIPT add \"$TASK_ID\" \"<task_type>\" \"$DESCRIPTOR\" \"wisdom/task_wisdom.md#$TASK_ID\""
    echo ""
    echo "Example:"
    echo "  $PYTHON_CMD $EMBEDDING_SCRIPT add \"$TASK_ID\" \"image_classification\" \"$DESCRIPTOR\" \"wisdom/task_wisdom.md#$TASK_ID\""
    echo ""
    echo "Or let the Agent run the command with the appropriate task type."
else
    echo "WARNING: Embedding utilities not available."
    echo "Wisdom will be stored in task_wisdom.md but not indexed for semantic search."
    if [ -z "$PYTHON_CMD" ]; then
        echo "Install python3 or use 'uv init' to create a virtual environment"
    fi
fi

echo ""
echo "================================="

# Update CLAUDE.md with task completion status
if [ -f "CLAUDE.md" ]; then
    GOAL=$(grep -A1 "## Strategic Goal" "$TASK_PLAN" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
    BEST_FILE=$(grep "^\*\*File\*\*:" "$TASK_PLAN" 2>/dev/null | head -1 | sed 's/.*`\(.*\)`.*/\1/')
    BEST_SCORE=$(grep "^\*\*Score\*\*:" "$TASK_PLAN" 2>/dev/null | head -1 | sed 's/.*: *//')

    cat > CLAUDE.md << EOF
# Project Context (ML-Master)

## Task Complete!
**Goal**: $GOAL
**Result**: ${BEST_FILE:-N/A} achieved ${BEST_SCORE:-N/A}
**Status**: P2 Promotion Done

## Memory Files
- \`task_plan.md\` - Final plan (L2)
- \`findings.md\` - Key insights (L2)
- \`wisdom/task_wisdom.md\` - Task wisdom (L3)

## Next Steps
Start a new task with \`/plan\`

---
*Task completed via ML-Master /complete - $(date +%Y-%m-%d)*
EOF
    echo ""
    echo "Updated: CLAUDE.md (task marked complete)"
fi
