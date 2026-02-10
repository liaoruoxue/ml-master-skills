#!/bin/bash
# ML-Master /status - 快速状态概览
# 用法: ${CLAUDE_PLUGIN_ROOT}/scripts/status.sh

echo "=== ML-Master Status ==="
echo ""

# Strategic Goal & Best Code
if [ -f task_plan.md ]; then
  echo "🎯 Goal:"
  grep -A1 "## Strategic Goal" task_plan.md 2>/dev/null | tail -1 | sed 's/^/   /'
  echo ""

  echo "🏆 Best Code:"
  grep -E "^\*\*(File|Metric|Score)\*\*:" task_plan.md 2>/dev/null | head -3 | sed 's/^/   /'
  echo ""

  echo "📍 Current Focus:"
  grep -A3 "## Current Focus" task_plan.md 2>/dev/null | grep -E "^-" | sed 's/^/   /'
  echo ""
else
  echo "⚠️  No task_plan.md found. Run /plan first."
  echo ""
fi

# L1 Status
if [ -f execution_trace.md ]; then
  LINES=$(wc -l < execution_trace.md 2>/dev/null | tr -d ' ')
  METRICS=$(grep -c "^|" execution_trace.md 2>/dev/null || echo 0)
  echo "📊 L1 (execution_trace.md):"
  echo "   Lines: $LINES"
  echo "   Metric entries: $((METRICS - 4))"  # 减去表头行
  if [ "$LINES" -gt 80 ]; then
    echo "   ⚠️  Consider /promote soon!"
  fi
  echo ""
else
  echo "📊 L1: Not initialized"
  echo ""
fi

# L2 Status
if [ -f findings.md ]; then
  INSIGHTS=$(grep -c "^- " findings.md 2>/dev/null || echo 0)
  EXPERIMENTS=$(grep -c "^| [0-9]" findings.md 2>/dev/null || echo 0)
  echo "📚 L2 (findings.md):"
  echo "   Key insights: $INSIGHTS"
  echo "   Experiment results: $EXPERIMENTS"
  echo ""
else
  echo "📚 L2: Not initialized"
  echo ""
fi

# Quick Actions
echo "💡 Quick Actions:"
echo "   /promote  - Compress L1 to L2"
echo "   /recover  - Restore state after /clear"
echo "   /complete - Finish task (P2 Promotion)"
