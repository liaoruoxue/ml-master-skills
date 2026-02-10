# ML-Master 2.1

**Hierarchical Cognitive Caching (HCC) for Ultra-Long-Horizon ML Tasks**

基于论文 [arXiv:2601.10402v3](https://arxiv.org/abs/2601.10402v3) 的 Claude Code Skills 实现。通过分层认知缓存架构，让 AI Agent 在超长视界 ML 任务中保持上下文连贯性。

## 核心架构

```
┌─────────────────────────────────────────┐
│           L1: Working Memory            │  execution_trace.md
│           (每 5 次工具调用更新)            │  生命周期: 分钟级
├────────────── P1 Promotion ─────────────┤
│           L2: Strategic Memory          │  findings.md + task_plan.md
│           (关键洞察和实验记录)             │  生命周期: 任务级
├────────────── P2 Promotion ─────────────┤
│           L3: Permanent Wisdom          │  wisdom/*.md + embeddings
│           (跨任务可复用经验)              │  生命周期: 永久
└─────────────────────────────────────────┘
```

## 快速安装

```bash
# 复制 skill 到你的项目
cp -r skill/ your-project/.claude/skills/ml-master/

# 确保脚本可执行
chmod +x your-project/.claude/skills/ml-master/scripts/*.sh
```

安装后，在 Claude Code 中即可使用 `/ml-master` 启动任务。

## 使用方式

### 1. 初始化任务
```
> /ml-master
> 任务描述: CIFAR-10 image classification, target >85% accuracy
```

### 2. 工作流程
- Agent 自动执行 5-Action Rule (每 5 次工具调用更新 L1)
- 阶段完成时执行 P1 Promotion (L1 → L2)
- 任务完成时执行 P2 Promotion (L2 → L3)

### 3. 命令
| 命令 | 说明 |
|------|------|
| `/ml-master` | 启动新 ML 任务 |
| `promote` | P1: 将 L1 洞察提升到 L2 |
| `complete` | P2: 任务完成，提取 wisdom 到 L3 |
| `status` | 查看当前 HCC 状态 |

## 实验验证

| 任务 | Iterations | 结果 | 说明 |
|------|-----------|------|------|
| MNIST | 2 | 99.26% | 基础验证 |
| CIFAR-10 | 3 | 89.38% | 中等任务 |
| **Titanic** | **13** | **83.71%** | **长程验证 (75.84% → 83.71%)** |

## 目录结构

```
ml-master/
├── skill/              # Claude Skill (核心，可直接安装)
│   ├── SKILL.md        # Skill 定义 (Hooks, Commands, 提示词)
│   ├── CLAUDE.md       # 自动上下文模板
│   ├── scripts/        # Shell 脚本 (init, promote, complete...)
│   ├── templates/      # L1/L2 文件模板
│   └── wisdom/         # L3 智慧层 (embedding, wisdom files)
├── docs/               # 文档
│   ├── PRESENTATION.md # 15 分钟分享 PPT
│   ├── PAPER_COMPARISON.md
│   └── design.md       # 详细设计文档
├── examples/           # 实验案例
│   ├── cifar10/        # CIFAR-10 (3 iterations)
│   ├── titanic/        # Titanic (13 iterations, 长程验证)
│   └── mnist/          # MNIST (2 iterations)
└── paper/              # 论文 PDF
```

## 关键概念

- **5-Action Rule**: 每 5 次工具调用后更新 L1，平衡记录频率和噪声
- **P1 Promotion**: L1 → L2，LLM 总结关键洞察，清空 L1
- **P2 Promotion**: L2 → L3，提取跨任务可复用的 wisdom
- **Context Hit**: 通过 CLAUDE.md 自动加载，新 session 立即获得上下文
- **Semantic Search**: L3 使用 sentence-transformers embedding 检索相关 wisdom

## 论文

- **HCC**: [Hierarchical Cognitive Caching for LLM-Based Autonomous Agents](https://arxiv.org/abs/2601.10402v3)

## License

MIT
