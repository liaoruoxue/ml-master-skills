#!/bin/bash
# ML-Master: Context Promotion (L1 → L2)
# Displays L1 content for Agent to summarize and migrate to L2

set -e

L1_FILE="${1:-execution_trace.md}"
L2_FINDINGS="${2:-findings.md}"

echo "=== ML-Master: Context Promotion (L1 → L2) ==="
echo ""

# Check if L1 exists
if [ ! -f "$L1_FILE" ]; then
    echo "INFO: $L1_FILE not found. Nothing to promote."
    exit 0
fi

# Count lines in L1
L1_LINES=$(wc -l < "$L1_FILE" | tr -d ' ')
echo "L1 ($L1_FILE): $L1_LINES lines"
echo ""

if [ "$L1_LINES" -lt 10 ]; then
    echo "INFO: L1 has minimal content. Consider continuing work before promoting."
    exit 0
fi

echo "--- L1 Content to Summarize ---"
cat "$L1_FILE"
echo ""
echo "--- End of L1 Content ---"
echo ""

echo "=== AGENT INSTRUCTIONS ==="
echo "1. Summarize the above L1 content"
echo "2. Extract KEY INSIGHTS and add to findings.md:"
echo "   - Key Insights section: important discoveries"
echo "   - Validated Hypotheses: confirmed theories"
echo "   - Failed Attempts: what didn't work and why"
echo "   - Experiment Results: any metrics/results"
echo ""
echo "3. Update task_plan.md:"
echo "   - Mark completed Implementation as 'complete'"
echo "   - Update Current Focus to next Implementation"
echo ""
echo "4. CLEAR L1 by running:"
echo "   \${CLAUDE_PLUGIN_ROOT}/scripts/clear-l1.sh"
echo ""
echo "5. CLAUDE.md will be auto-updated with latest Best Code info"
echo ""
echo "==========================="

# Update CLAUDE.md with current state (auto Context Hit sync)
if [ -f "task_plan.md" ] && [ -f "CLAUDE.md" ]; then
    GOAL=$(grep -A1 "## Strategic Goal" task_plan.md 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')
    BEST_FILE=$(grep "^\*\*File\*\*:" task_plan.md 2>/dev/null | head -1 | sed 's/.*`\(.*\)`.*/\1/')
    BEST_SCORE=$(grep "^\*\*Score\*\*:" task_plan.md 2>/dev/null | head -1 | sed 's/.*: *//')
    SYNC_TIME=$(date "+%Y-%m-%d %H:%M")

    if [ -n "$GOAL" ] || [ -n "$BEST_SCORE" ]; then
        # Update CLAUDE.md with current state
        cat > CLAUDE.md << EOF
# Project Context (ML-Master)

## ⚠️ BEFORE YOU START - MUST READ
1. **Read \`task_plan.md\`** - Get current plan and implementation status
2. **Read \`findings.md\`** - Check Key Insights and **Failed Attempts**
3. **THEN continue working** - Don't repeat failed experiments!

---

## Current State
| Field | Value |
|-------|-------|
| **Goal** | $GOAL |
| **Best Code** | \`${BEST_FILE:-N/A}\` |
| **Best Score** | ${BEST_SCORE:-N/A} |
| **Last Sync** | $SYNC_TIME |

> ⚠️ This summary may be STALE if you continued working without /promote.
> Always verify by reading the actual files.

## Memory Files
| File | Layer | Content |
|------|-------|---------|
| \`task_plan.md\` | L2 | Strategic Goal, Research Plan, Best Code |
| \`findings.md\` | L2 | Key Insights, Failed Attempts, Experiment Results |
| \`execution_trace.md\` | L1 | Current phase operations (cleared on /promote) |

## Quick Commands
| Command | Action |
|---------|--------|
| \`/status\` | View current state |
| \`/promote\` | Compress L1→L2, sync this file |
| \`/recover\` | Restore context after /clear |
| \`/complete\` | Finish task, P2 Promotion to L3 |

## Rules
1. **5-Action Rule**: Update \`execution_trace.md\` every 5 Write/Edit/Bash
2. **Best Code Tracking**: Update \`task_plan.md\` when metrics improve
3. **Don't Repeat Failures**: Check \`findings.md\` Failed Attempts before trying new approaches

## ⚠️ After /clear
**IMMEDIATELY run \`/recover\`** before doing anything else!
Your conversation history is gone, but L2 files still have the context.

---
*Auto-synced by ML-Master /promote at $SYNC_TIME*
EOF
        echo ""
        echo "Updated: CLAUDE.md (synced with current Best Code)"
    fi
fi

# Check if --auto-clear flag is passed
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$3" = "--auto-clear" ] || [ "$1" = "--auto-clear" ]; then
    echo ""
    echo "Auto-clearing L1..."
    sh "$SCRIPT_DIR/clear-l1.sh" "$L1_FILE"
fi
