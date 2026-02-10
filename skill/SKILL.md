---
name: ml-master
version: "2.1.0"
description: ML-Master 2.1 - Hierarchical Cognitive Caching (HCC) for ultra-long-horizon ML tasks. Implements L1/L2/L3 memory layers with P1/P2 context promotion. v2.1 reduces hook noise, uses 5-Action Rule, adds /status command.
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - WebFetch
  - WebSearch
hooks:
  PreToolUse:
    - matcher: "Write|Edit|Bash"
      hooks:
        - type: command
          command: |
            # 只在 L1 过大时警告，减少噪音
            if [ -f execution_trace.md ]; then
              LINES=$(wc -l < execution_trace.md 2>/dev/null | tr -d ' ' || echo 0)
              if [ "$LINES" -gt 80 ]; then
                echo "[ML-Master] ⚠️ L1 has $LINES lines. Consider /promote soon."
              fi
            fi
  PostToolUse:
    - matcher: "Write|Edit|Bash"
      hooks:
        - type: command
          command: |
            # 5-Action Rule: 每 5 次操作提醒更新 L1
            COUNT_FILE="/tmp/ml-master-action-count-$$"
            COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
            COUNT=$((COUNT + 1))
            echo $COUNT > "$COUNT_FILE"
            if [ $((COUNT % 5)) -eq 0 ]; then
              echo "[ML-Master] 📝 5 actions done. Update execution_trace.md"
            fi
  Stop:
    - hooks:
        - type: command
          command: |
            SCRIPT_DIR="${CLAUDE_PLUGIN_ROOT:-$(dirname "$0")}/scripts"
            if [ -f "$SCRIPT_DIR/check-complete.sh" ]; then
              sh "$SCRIPT_DIR/check-complete.sh"
            else
              echo "[ML-Master] Warning: check-complete.sh not found"
            fi
---

# ML-Master 2.0: Hierarchical Cognitive Caching

实现超长视界 (Ultra-Long-Horizon) 的自主机器学习工程能力。

## HCC 三层存储架构

```
┌─────────────────────────────────────────────────────┐
│  L1: Execution Trace (execution_trace.md)           │
│      工作记忆 - 阶段结束时清空                        │
│      内容: 代码片段、终端输出、错误堆栈               │
├─────────────────────────────────────────────────────┤
│  L2: Strategic Memory (findings.md + task_plan.md)  │
│      中期战略记忆 - 任务级持久化                      │
│      内容: 关键判断、实验洞察、计划状态               │
├─────────────────────────────────────────────────────┤
│  L3: Prior Wisdom (wisdom/)                         │
│      长期记忆 - 跨任务永久存在                        │
│      内容: 最佳实践、代码模板、常见错误解决方案        │
└─────────────────────────────────────────────────────┘
```

## 核心文件

| 文件 | 层级 | 生命周期 | 用途 |
|------|------|----------|------|
| `execution_trace.md` | L1 | 阶段级 | 执行细节、Metric Log、Notes (简化版 4 sections) |
| `findings.md` | L2 | 任务级 | 提炼的知识和洞察 |
| `task_plan.md` | L2 | 任务级 | 分层计划 (m×q)、Best Code 追踪 |
| `wisdom/global_wisdom.md` | L3 | 永久 | ML 最佳实践 |
| `wisdom/task_wisdom.md` | L3 | 永久 | 任务级智慧 (P2 生成) |
| `wisdom/embeddings.json` | L3 | 永久 | 语义嵌入向量索引 |
| `wisdom/embedding_utils.py` | L3 | 永久 | 嵌入检索工具 |

## Research Plan 结构

每个 Research Plan 包含 **m Directions × q Implementations**：

```
Research Plan #N
├── Direction 1: [探索方向]
│   ├── Implementation 1.1: [具体尝试]
│   ├── Implementation 1.2: [具体尝试]
│   └── ...
├── Direction 2: [探索方向]
│   ├── Implementation 2.1
│   └── Implementation 2.2
└── Direction 3: [探索方向]
    └── Implementation 3.1
```

**Best Code Tracking**: 每次实验后比较 Metric，保留最佳代码路径和分数。

## L3 语义检索 (Embedding-based Retrieval)

使用 `sentence-transformers` 实现智慧的语义检索：

```
任务开始 → Context Prefetching
    q = E(task_descriptor)           # 计算任务描述的嵌入向量
    for h_n in L3_index:
        if cos(q, h_n) > δ:          # δ = 0.4 阈值
            prefetch(wisdom_n)       # 预加载相关智慧

任务完成 → P2 Embedding Index
    h_τ = E(task_descriptor)         # 计算任务嵌入
    L3_index.add(task_id, h_τ, wisdom_ref)  # 更新索引
```

**工具使用**:
```bash
# 搜索相似智慧
python3 wisdom/embedding_utils.py search "image classification plant disease"

# 添加新智慧到索引
python3 wisdom/embedding_utils.py add <task_id> <task_type> <descriptor> <wisdom_ref>

# 列出所有索引条目
python3 wisdom/embedding_utils.py list
```

**降级方案**: 如果 `sentence-transformers` 未安装，自动降级到关键词匹配。

## 依赖安装

### 可选依赖（推荐）

```bash
# 安装 sentence-transformers 以启用语义检索
pip install sentence-transformers

# 或使用 uv
uv add sentence-transformers
```

**注意**:
- 语义检索（cosine similarity）比关键词匹配更准确
- 首次加载模型约需 10-30 秒
- 模型大小约 90MB（all-MiniLM-L6-v2）
- 无此依赖时自动降级到 Jaccard 关键词匹配

### 使用 uv 管理项目

如果项目使用 `uv` 管理虚拟环境：

```bash
# 初始化项目
uv init

# 添加 ML 依赖
uv add torch torchvision sentence-transformers

# 运行脚本时使用 uv run
uv run python your_script.py
```

## 🚨 强制规则

### 1. 双重读写规则 (Dual Read/Write)

```
执行细节 (代码运行、报错、输出) → 只写 L1 (execution_trace.md)
结论洞察 (什么有效、什么无效) → 只写 L2 (findings.md)
```

**绝对禁止**：将执行细节写入 L2，或将结论写入 L1。

### 2. 五步一记 (5-Action Rule)

> 每执行 5 个工具调用 (Bash, Write, Edit)，必须更新 `execution_trace.md`。

这防止信息在上下文中丢失，同时避免过于频繁的更新。

### 3. 禁止上下文堆积

```
❌ 错误：依赖对话历史记忆之前的错误
✅ 正确：依赖 execution_trace.md 和 findings.md 的文件记录
```

### 4. Best Code 追踪规则

每次代码运行产生 Metric 后：
1. 记录到 `execution_trace.md` 的 Metric Log
2. 与 `task_plan.md` 的 Current Best Code 比较
3. 如果更优，更新 Best Code 信息

### 5. 阶段完成时必须 Promote (P1)

当 `task_plan.md` 中一个 Implementation 完成时：
1. 从 L1 提取洞察写入 L2
2. 清空 L1
3. 更新计划状态

## 命令

### `/status` - 快速状态概览

显示当前任务状态的简洁摘要：
- Strategic Goal
- Current Best Code
- Current Focus
- L1/L2 文件状态

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/status.sh
```

### `/plan` - 初始化任务

1. 从模板创建 `task_plan.md`, `findings.md`, `execution_trace.md`
2. 读取 `wisdom/global_wisdom.md` 获取相关知识
3. **Context Prefetching**: 从 L3 检索相似任务智慧 (可选)
4. 开始第一个 Implementation

```bash
# 初始化 (无 Context Prefetching)
${CLAUDE_PLUGIN_ROOT}/scripts/init-session.sh

# 初始化 (带 Context Prefetching)
${CLAUDE_PLUGIN_ROOT}/scripts/init-session.sh "your task description here"
```

### `/promote` - 上下文提升 (L1 → L2)

手动触发阶段性总结：

1. 读取 `execution_trace.md` 全部内容
2. **你来总结**：提取"执行摘要"和"战略洞察"
3. 将洞察追加到 `findings.md`
4. 更新 `task_plan.md` 中对应 Implementation 的状态
5. 清空 L1

```bash
# 辅助脚本 - 显示需要总结的内容
${CLAUDE_PLUGIN_ROOT}/scripts/promote.sh

# 完成总结后，清空 L1
${CLAUDE_PLUGIN_ROOT}/scripts/clear-l1.sh
```

### `/recover` - 恢复认知状态

`/clear` 后执行，从 L2 重建上下文：

1. 读取 `task_plan.md` - 当前阶段和目标
2. 读取 `findings.md` - 已知结论
3. **不读取** 对话历史

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/recover.sh
```

### `/complete` - 任务完成 (P2 Promotion)

任务全部完成时执行，触发 Task-level Promotion (L2 → L3)：

1. 验证所有 Implementation 已完成
2. 显示任务摘要 (Goal, Best Code, Key Insights)
3. **你来总结**: 提取可迁移的任务级智慧
4. 将智慧追加到 `wisdom/task_wisdom.md`
5. 按任务类型分类 (Image Classification, Tabular, NLP, etc.)
6. **更新 L3 嵌入索引** (如有 sentence-transformers)

```bash
${CLAUDE_PLUGIN_ROOT}/scripts/task-complete.sh
```

**智慧格式**:
```markdown
### Task: [task_name] - [date]
- **Key insight**: [what worked best]
- **Best approach**: [recommended strategy]
- **Pitfalls**: [what to avoid]
- **Final Score**: [metric: score]
```

## 工作流程

### 任务开始
```
1. /plan → 创建文件，加载 L3 智慧
2. 阅读 task_plan.md → 确认目标和第一个 Implementation
3. 开始执行
```

### 执行中
```
每 5 个工具调用:
  → 更新 execution_trace.md (5-Action Rule)

每次代码运行后:
  → 记录 Metric 到 execution_trace.md 的 Metric Log
  → 如果优于 Best Code，更新 task_plan.md 的 Current Best Code

遇到关键发现:
  → 记录到 execution_trace.md 的 Observations

遇到错误:
  → 记录到 execution_trace.md 的 Terminal Outputs
```

### Research Plan 完成 (P1 Promotion)
```
1. /promote → L1 压缩到 L2
2. 更新 task_plan.md 的 Plan History
3. 开始新的 Research Plan #[N+1]
```

### 任务完成 (P2 Promotion)
```
1. 确认所有 Implementation 完成
2. /complete → L2 提炼到 L3
3. 添加任务级智慧到 wisdom/task_wisdom.md
```

### 上下文清除后
```
1. /recover → 从文件重建状态
2. 继续任务（无信息丢失）
```

## 5-Question Reboot Test

如果你能回答这 5 个问题，说明认知状态完整：

| Question | Answer Source |
|----------|---------------|
| 我在哪？ | task_plan.md → Current Focus |
| 我要去哪？ | task_plan.md → Plan Tree |
| 目标是什么？ | task_plan.md → Strategic Goal |
| 我学到了什么？ | findings.md |
| 我做了什么？ | execution_trace.md (当前阶段) |

## 验收标准

1. **持久化验证**：`/clear` 后能在 1 分钟内通过读取文件说出当前状态
2. **信息流转**：L1 大小呈锯齿状，L2 大小呈阶梯状
3. **长程推理**：第 50 轮对话仍能引用第 1 轮确立的战略原则

## 模板位置

- `${CLAUDE_PLUGIN_ROOT}/templates/execution_trace.md`
- `${CLAUDE_PLUGIN_ROOT}/templates/task_plan.md`
- `${CLAUDE_PLUGIN_ROOT}/templates/findings.md`
- `${CLAUDE_PLUGIN_ROOT}/wisdom/global_wisdom.md`
- `${CLAUDE_PLUGIN_ROOT}/wisdom/task_wisdom.md`

## 脚本位置

- `${CLAUDE_PLUGIN_ROOT}/scripts/init-session.sh` - /plan
- `${CLAUDE_PLUGIN_ROOT}/scripts/status.sh` - /status
- `${CLAUDE_PLUGIN_ROOT}/scripts/promote.sh` - /promote (P1)
- `${CLAUDE_PLUGIN_ROOT}/scripts/clear-l1.sh` - 清空 L1
- `${CLAUDE_PLUGIN_ROOT}/scripts/recover.sh` - /recover
- `${CLAUDE_PLUGIN_ROOT}/scripts/task-complete.sh` - /complete (P2)
- `${CLAUDE_PLUGIN_ROOT}/scripts/check-complete.sh` - 完成检查
- `${CLAUDE_PLUGIN_ROOT}/wisdom/embedding_utils.py` - L3 嵌入检索工具

## 参考

基于论文 "Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering" (arXiv:2601.10402v3) 的 HCC 架构设计。
