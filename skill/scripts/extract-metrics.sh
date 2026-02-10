#!/bin/bash
# ML-Master extract-metrics.sh - 从训练输出提取 metrics
# 用法: ${CLAUDE_PLUGIN_ROOT}/scripts/extract-metrics.sh [log_file]

LOG_FILE="${1:-training.log}"

if [ ! -f "$LOG_FILE" ]; then
  echo "Error: File '$LOG_FILE' not found"
  echo "Usage: extract-metrics.sh [log_file]"
  exit 1
fi

echo "=== Extracted Metrics from $LOG_FILE ==="
echo ""

# 提取 accuracy 相关
echo "📊 Accuracy:"
grep -iE "(accuracy|acc)[^a-z]*[=:][^0-9]*[0-9]+\.?[0-9]*%?" "$LOG_FILE" | tail -5 | sed 's/^/   /'

echo ""

# 提取 loss 相关
echo "📉 Loss:"
grep -iE "(loss)[^a-z]*[=:][^0-9]*[0-9]+\.?[0-9]*" "$LOG_FILE" | tail -5 | sed 's/^/   /'

echo ""

# 提取 epoch 相关
echo "🔄 Epochs:"
grep -iE "(epoch|Epoch)[^0-9]*[0-9]+" "$LOG_FILE" | tail -3 | sed 's/^/   /'

echo ""

# 提取 best 相关
echo "🏆 Best Results:"
grep -iE "(best|Best)[^0-9]*[0-9]+\.?[0-9]*" "$LOG_FILE" | tail -3 | sed 's/^/   /'

echo ""
echo "💡 Tip: Copy relevant metrics to execution_trace.md Metric Log"
