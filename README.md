# ML-Master 2.1

**Hierarchical Cognitive Caching (HCC) for Ultra-Long-Horizon ML Tasks**

[中文文档](README.zh-CN.md)

A Claude Code Skills implementation based on [arXiv:2601.10402v3](https://arxiv.org/abs/2601.10402v3). Uses a hierarchical cognitive caching architecture to maintain context coherence for AI Agents across ultra-long-horizon ML tasks.

## Core Architecture

```
┌─────────────────────────────────────────┐
│           L1: Working Memory            │  execution_trace.md
│           (Updated every 5 tool calls)  │  Lifecycle: minutes
├────────────── P1 Promotion ─────────────┤
│           L2: Strategic Memory          │  findings.md + task_plan.md
│           (Key insights & experiments)  │  Lifecycle: task-level
├────────────── P2 Promotion ─────────────┤
│           L3: Permanent Wisdom          │  wisdom/*.md + embeddings
│           (Reusable cross-task wisdom)  │  Lifecycle: permanent
└─────────────────────────────────────────┘
```

## Quick Start

```bash
# Copy the skill to your project
cp -r skill/ your-project/.claude/skills/ml-master/

# Make scripts executable
chmod +x your-project/.claude/skills/ml-master/scripts/*.sh
```

Once installed, use `/ml-master` in Claude Code to start a task.

## Usage

### 1. Initialize a Task
```
> /ml-master
> Task: CIFAR-10 image classification, target >85% accuracy
```

### 2. Workflow
- Agent automatically follows the 5-Action Rule (updates L1 every 5 tool calls)
- Run P1 Promotion (L1 → L2) when a phase completes
- Run P2 Promotion (L2 → L3) when the task completes

### 3. Commands
| Command | Description |
|---------|-------------|
| `/ml-master` | Start a new ML task |
| `promote` | P1: Compress L1 insights into L2 |
| `complete` | P2: Extract wisdom from L2 to L3 |
| `status` | View current HCC state |

## Experiments

| Task | Iterations | Result | Notes |
|------|-----------|--------|-------|
| MNIST | 2 | 99.26% | Basic validation |
| CIFAR-10 | 3 | 89.38% | Medium task |
| **Titanic** | **13** | **83.71%** | **Long-horizon test (75.84% → 83.71%)** |

## Directory Structure

```
ml-master/
├── skill/              # Claude Skill (core, ready to install)
│   ├── SKILL.md        # Skill definition (Hooks, Commands, Prompts)
│   ├── CLAUDE.md       # Auto-context template
│   ├── scripts/        # Shell scripts (init, promote, complete...)
│   ├── templates/      # L1/L2 file templates
│   └── wisdom/         # L3 wisdom layer (embeddings, wisdom files)
├── docs/               # Documentation
│   ├── PAPER_COMPARISON.md  # Paper vs implementation comparison
│   └── design.md       # Detailed design document
├── examples/           # Experiment cases
│   ├── cifar10/        # CIFAR-10 (3 iterations)
│   ├── titanic/        # Titanic (13 iterations, long-horizon test)
│   └── mnist/          # MNIST (2 iterations)
└── paper/              # Paper PDF
```

## Key Concepts

- **5-Action Rule**: Update L1 every 5 tool calls, balancing recording frequency and noise
- **P1 Promotion**: L1 → L2, LLM summarizes key insights, then clears L1
- **P2 Promotion**: L2 → L3, extracts reusable cross-task wisdom
- **Context Hit**: Auto-loaded via CLAUDE.md, new sessions get context immediately
- **Semantic Search**: L3 uses sentence-transformers embeddings to retrieve relevant wisdom

## Paper

- **HCC**: [Hierarchical Cognitive Caching for LLM-Based Autonomous Agents](https://arxiv.org/abs/2601.10402v3)

## License

MIT
