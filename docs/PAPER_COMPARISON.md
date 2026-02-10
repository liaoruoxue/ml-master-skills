# ML-Master vs Paper Implementation Comparison

> Paper: "Toward Ultra-Long-Horizon Agentic Science: Cognitive Accumulation for Machine Learning Engineering"
> arXiv: 2601.10402v3
> ML-Master Version: 2.1.0

---

## 1. Core Architecture: Hierarchical Cognitive Caching (HCC)

### 1.1 Three-Layer Storage Architecture

| Layer | Paper Definition | Paper Purpose | ML-Master Implementation | Status |
|-------|-----------------|---------------|--------------------------|--------|
| **L1** | Evolving Experience | Raw execution data: code, terminal output, error stacks | `execution_trace.md` | ✅ 100% |
| **L2** | Refined Knowledge | Distilled knowledge: insights, hypothesis validation, experiment conclusions | `findings.md` + `task_plan.md` | ✅ 100% |
| **L3** | Prior Wisdom | Cross-task wisdom: best practices, common errors, code templates | `wisdom/` directory | ✅ 95% |

### 1.2 L1 Detailed Comparison

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Storage Content | Code patches, terminal output, error stacks, temp observations | Operations Log + Metric Log + Notes | ✅ |
| Lifecycle | Cleared at phase end | Cleared via `clear-l1.sh` after `/promote` | ✅ |
| Update Frequency | High-frequency updates to prevent info loss | 5-Action Rule (update every 5 tool calls) | ✅ |
| Size Limit | Trigger compression when too large | Hook warns when >80 lines, suggests /promote | ✅ |

### 1.3 L2 Detailed Comparison

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Storage Content | Key judgments, experiment insights, validated hypotheses | findings.md with structured sections | ✅ |
| Plan State | Current goals, progress tracking | task_plan.md Research Plan structure | ✅ |
| Best Code | Track current best code | task_plan.md Current Best Code section | ✅ |
| Lifecycle | Task-level persistence | Persists until task completion | ✅ |

### 1.4 L3 Detailed Comparison

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Static Knowledge | ML best practices, common error solutions | `wisdom/global_wisdom.md` | ✅ |
| Task Wisdom | Transferable knowledge extracted from completed tasks | `wisdom/task_wisdom.md` | ✅ |
| Semantic Index | Embedding vector index h_n = E(d_n) | `wisdom/embeddings.json` | ✅ |
| Semantic Retrieval | cos(q, h_n) > δ threshold retrieval | `embedding_utils.py` search | ✅ |

---

## 2. Context Promotion Mechanism

### 2.1 P1: Phase-level Promotion

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Trigger Condition | Phase/Implementation completion | `/promote` command (manual trigger) | ✅ |
| Compression Process | Agent summarizes L1, extracts insights to L2 | promote.sh displays L1, Agent summarizes | ✅ |
| L1 Clearing | Clear after compression | `clear-l1.sh` script | ✅ |
| L2 Update | Append insights to L2 | Agent updates findings.md | ✅ |

### 2.2 P2: Task-level Promotion

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Trigger Condition | Task fully completed | `/complete` command | ✅ |
| Wisdom Extraction | Distill transferable task-level wisdom from L2 | Agent summarizes and writes to task_wisdom.md | ✅ |
| L3 Update | Add to permanent knowledge base | Appended by task type | ✅ |
| Embedding Update | Update semantic index | embedding_utils.py add | ✅ |

### 2.3 Context Hit

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Auto Retrieval | Automatically query L1/L2 when info needed | CLAUDE.md auto-sync (v2.1) | ✅ 95% |
| L1 Hit | Return raw data | Direct file read | ✅ |
| L2 Hit | Return summarized version | /recover restores from L2 | ✅ |
| Recovery Mechanism | Rebuild state after /clear | `/recover` command | ✅ |

---

## 3. Research Plan Structure

### 3.1 m×q Hierarchical Plan

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| m Directions | Multiple exploration directions | task_plan.md Direction 1/2/3... | ✅ |
| q Implementations | Multiple implementations per direction | Implementation X.1, X.2... | ✅ |
| Status Tracking | pending/in_progress/complete/abandoned | Status field | ✅ |
| Outcome Recording | Results and learnings per implementation | Outcome field | ✅ |

### 3.2 Best Code Tracking

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Code Path | Track best code file | Current Best Code - File field | ✅ |
| Evaluation Metric | Record metrics used | Metric field | ✅ |
| Best Score | Current highest score | Score field | ✅ |
| History | Record score progression | findings.md Best Code History | ✅ |

### 3.3 Metric Tracking

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Real-time Recording | Record metrics after each run | execution_trace.md Metric Log | ✅ |
| Baseline Comparison | vs Best column | Metric Log vs Best field | ✅ |
| Experiment Summary | Cross-experiment comparison | findings.md Experiment Results | ✅ |

---

## 4. Rules & Constraints

### 4.1 N-Action Rule

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Update Frequency | Update L1 every N tool calls | 5-Action Rule (v2.1 optimized) | ✅ |
| Trigger Tools | Write/Edit/Bash and other modification ops | PostToolUse Hook counter | ✅ |
| Reminder Mechanism | Remind Agent to update | Hook output reminder | ✅ |

### 4.2 Dual Read/Write Rule

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| L1 Write | Only write execution details | Code, output, errors → execution_trace.md | ✅ |
| L2 Write | Only write conclusions/insights | Insights, hypotheses, decisions → findings.md | ✅ |
| No Mixing | No cross-writing allowed | SKILL.md rules explicitly forbid mixing | ✅ |

### 4.3 No Context Accumulation

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| File Dependency | Don't rely on conversation history for error recall | Rules explicitly require file reading | ✅ |
| Persistence | All important info written to files | L1/L2/L3 file system | ✅ |

---

## 5. L3 Semantic Retrieval System

### 5.1 Embedding Mechanism

| Paper Requirement | Paper Formula | ML-Master Implementation | Status |
|-------------------|--------------|--------------------------|--------|
| Vector Encoding | h_n = E(d_n) | sentence-transformers all-MiniLM-L6-v2 | ✅ |
| Similarity Calculation | cos(q, h_n) | numpy cosine similarity | ✅ |
| Threshold Filtering | > δ (threshold) | δ = 0.5 default threshold | ✅ |
| Top-K Return | Return K most similar | top_k = 3 default | ✅ |

### 5.2 Context Prefetching

| Paper Requirement | Paper Description | ML-Master Implementation | Status |
|-------------------|-------------------|--------------------------|--------|
| Trigger Timing | At task start | init-session.sh supports description parameter | ✅ |
| Query Generation | q = E(task_descriptor) | Task description passed for retrieval | ✅ |
| Wisdom Loading | Pre-load relevant wisdom | Display matching task_wisdom entries | ✅ |

### 5.3 Fallback Strategy

| Scenario | Paper Requirement | ML-Master Implementation | Status |
|----------|-------------------|--------------------------|--------|
| No sentence-transformers | Should have fallback | Jaccard keyword matching | ✅ |
| No Python | Should still work | Skip retrieval, warn only | ✅ |

---

## 6. Commands & Workflow

### 6.1 Command Mapping

| Function | Paper Description | ML-Master Command | Implementation Script | Status |
|----------|-------------------|--------------------|-----------------------|--------|
| Initialize | Create L1/L2, load L3 | `/plan` | init-session.sh | ✅ |
| Status View | Quick state overview | `/status` | status.sh (v2.1) | ✅ |
| P1 Promotion | L1 → L2 compression | `/promote` | promote.sh | ✅ |
| State Recovery | Rebuild after /clear | `/recover` | recover.sh | ✅ |
| P2 Promotion | L2 → L3 wisdom extraction | `/complete` | task-complete.sh | ✅ |
| Clear L1 | Phase-end cleanup | Internal call | clear-l1.sh | ✅ |

### 6.2 Workflow Mapping

| Phase | Paper Workflow | ML-Master Workflow | Status |
|-------|---------------|-------------------|--------|
| Task Start | Init + Context Prefetching | /plan [description] | ✅ |
| Execution | N-Action Rule updates L1 | 5-Action Rule + Hooks | ✅ |
| Phase Complete | P1 Promotion | /promote | ✅ |
| Task Complete | P2 Promotion | /complete | ✅ |
| Context Cleared | Context Hit recovery | /recover + CLAUDE.md | ✅ |

---

## 7. Hooks Mechanism

### 7.1 PreToolUse Hook

| Paper Requirement | Paper Description | ML-Master v2.0 | ML-Master v2.1 | Status |
|-------------------|-------------------|-----------------|-----------------|--------|
| Context Check | Check state before execution | Full status output each time | Only warn when L1 > 80 lines | ✅ Optimized |
| Trigger Tools | Modification tools | Write\|Edit\|Bash\|Read\|Glob\|Grep | Write\|Edit\|Bash | ✅ Optimized |

### 7.2 PostToolUse Hook

| Paper Requirement | Paper Description | ML-Master v2.0 | ML-Master v2.1 | Status |
|-------------------|-------------------|-----------------|-----------------|--------|
| Update Reminder | Remind to update L1 | Every call | Every 5 calls | ✅ Optimized |
| Best Code Reminder | Update when metric improves | Every call | Removed (reduce noise) | ✅ Optimized |

---

## 8. Acceptance Criteria

| Paper Criterion | Description | ML-Master Implementation | Verification | Status |
|-----------------|-------------|--------------------------|--------------|--------|
| Persistence | Recover state within 1 min after /clear | /recover + CLAUDE.md | Verified with CIFAR-10 test | ✅ |
| Information Flow | L1 sawtooth, L2 staircase pattern | File size changes match expectations | Observed in practice | ✅ |
| Long-horizon Reasoning | Round 50 references round 1 principles | L2 persists Strategic Goal | Verified with CIFAR-10 test | ✅ |
| 5-Question Test | Can answer 5 core questions | All answers derivable from files | /status + /recover | ✅ |

---

## 9. Not Implemented / Platform Limitations

| Paper Feature | Reason | Alternative | Impact |
|---------------|--------|-------------|--------|
| **Parallel Implementation Execution** | Claude Code single-thread limitation | Sequential execution, maintain m×q structure | Low |
| **Initial Code Phase Separation** | Design choice | Agent decides phasing independently | Very Low |
| **Fully Automatic Context Hit** | Hook capability limitation | Approximated via CLAUDE.md | Very Low |

---

## 10. Version History

| Version | Major Changes | Paper Coverage |
|---------|---------------|----------------|
| v2.0.0 | Initial HCC implementation | ~90% |
| v2.1.0 | Hook optimization + /status + CLAUDE.md auto-sync | **~95%** |

---

## 11. Practical Validation: CIFAR-10 Challenge

| Metric | Result |
|--------|--------|
| Target | >85% Test Accuracy |
| Achieved | **89.38%** |
| Model Iterations | 3 Implementations (75.47% → 82.59% → 89.38%) |
| P1 Executions | 3 |
| P2 Executions | 1 |
| L3 Update | task_wisdom.md recorded |
| Features Validated | L1/L2/L3, P1/P2, Best Code Tracking, m×q structure |

---

## 12. File Structure

```
.claude/skills/ml-master/
├── SKILL.md                 # v2.1.0 main config + Hooks
├── CLAUDE.md                # Skill-level context description
├── PAPER_COMPARISON.md      # This document
├── templates/
│   ├── execution_trace.md   # L1 template (simplified 4 sections)
│   ├── task_plan.md         # L2 plan template
│   └── findings.md          # L2 knowledge template
├── wisdom/
│   ├── global_wisdom.md     # L3 static knowledge
│   ├── task_wisdom.md       # L3 task wisdom
│   ├── embeddings.json      # L3 vector index
│   └── embedding_utils.py   # Embedding utility
└── scripts/
    ├── init-session.sh      # /plan
    ├── status.sh            # /status (v2.1)
    ├── promote.sh           # /promote (P1)
    ├── recover.sh           # /recover
    ├── task-complete.sh     # /complete (P2)
    ├── clear-l1.sh          # Clear L1
    ├── check-complete.sh    # Completion check
    └── extract-metrics.sh   # Metrics extraction (v2.1)
```

---

## 13. Summary

```
Paper Coverage: ██████████████████████░░ ~95%

✅ Fully Implemented:
   - HCC three-layer storage architecture (L1/L2/L3)
   - P1/P2 context promotion mechanism
   - Research Plan m×q structure
   - Best Code Tracking
   - Metric Tracking
   - L3 Embedding semantic retrieval
   - N-Action Rule (5-Action)
   - Dual Read/Write Rule
   - Context Hit (via CLAUDE.md)

⚠️ Partially Implemented (Platform Limitations):
   - Parallel Implementation execution → Sequential alternative
   - Initial Code Phase → Agent decides independently

📊 Practical Validation:
   - CIFAR-10: 89.38% (target >85%)
   - Full L1→L2→L3 pipeline verified
```

---

## Appendix A: Claude Code Skills Capability Summary

During the ML-Master (HCC paper) implementation, the following Claude Code Skills capabilities were used:

### A.1 Skill Definition (`SKILL.md`)

```yaml
# Core config file structure
name: ml-master
version: 2.1.0
description: Hierarchical Cognitive Caching for ML tasks
```

**Usage**:
- Define Skill metadata (name, version, description)
- Declare commands and rules
- Configure Hook triggers

### A.2 Hooks System (Core Capability)

#### PreToolUse Hook
```yaml
hooks:
  PreToolUse:
    - matcher: "Write|Edit|Bash"
      hooks:
        - type: command
          command: |
            # Only warn when L1 > 80 lines
            if [ -f execution_trace.md ]; then
              LINES=$(wc -l < execution_trace.md)
              if [ "$LINES" -gt 80 ]; then
                echo "[ML-Master] ⚠️ L1 has $LINES lines"
              fi
            fi
```

**Purpose**: Check state **before** tool calls

#### PostToolUse Hook
```yaml
PostToolUse:
  - matcher: "Write|Edit|Bash"
    hooks:
      - type: command
        command: |
          # 5-Action Rule counter
          COUNT_FILE="/tmp/ml-master-action-count-$$"
          COUNT=$(($(cat "$COUNT_FILE" 2>/dev/null || echo 0) + 1))
          echo $COUNT > "$COUNT_FILE"
          if [ $((COUNT % 5)) -eq 0 ]; then
            echo "[ML-Master] 📝 Update execution_trace.md"
          fi
```

**Purpose**: Remind to update L1 **after** tool calls

### A.3 Custom Commands (Slash Commands)

| Command | Script | Paper Function |
|---------|--------|----------------|
| `/plan` | `init-session.sh` | Init + Context Prefetching |
| `/status` | `status.sh` | Quick state overview |
| `/promote` | `promote.sh` | P1 Promotion (L1→L2) |
| `/recover` | `recover.sh` | Context Hit recovery |
| `/complete` | `task-complete.sh` | P2 Promotion (L2→L3) |

**Definition**:
```yaml
commands:
  - name: promote
    description: Compress L1 to L2
    script: scripts/promote.sh
```

### A.4 Shell Scripts (`scripts/`)

```
scripts/
├── init-session.sh      # Create L1/L2 files + CLAUDE.md
├── status.sh            # Parse files and display status
├── promote.sh           # Display L1 content for Agent to summarize
├── recover.sh           # Recover cognitive state from L2
├── task-complete.sh     # P2 trigger + Embedding command prompt
├── clear-l1.sh          # Reset L1 to template
└── extract-metrics.sh   # Extract metrics from logs
```

**Key feature**: Script output becomes Agent input, enabling **human-AI collaboration**

### A.5 Template System (`templates/`)

```
templates/
├── execution_trace.md   # L1 working memory template
├── task_plan.md         # L2 strategic plan template
└── findings.md          # L2 knowledge base template
```

**Usage**: `init-session.sh` copies templates to project directory

### A.6 CLAUDE.md (Project Context)

```markdown
# Project Context (ML-Master)

## Memory Files
- `task_plan.md` - Strategic Goal & Plan (L2)
- `findings.md` - Key Insights (L2)
- `execution_trace.md` - Current Progress (L1)

## Quick Commands
- `/status` - View current state
- `/promote` - Compress L1→L2
```

**Capability**: Claude Code automatically reads `CLAUDE.md` from the project root, enabling **automatic Context Hit**

### A.7 Wisdom Directory (L3 Permanent Storage)

```
wisdom/
├── global_wisdom.md     # ML best practices (static)
├── task_wisdom.md       # Task-level wisdom (appended via P2)
├── embeddings.json      # Vector index
└── embedding_utils.py   # Semantic retrieval tool
```

**Key feature**: Cross-task persistence with semantic retrieval support

### A.8 Capability Composition Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    SKILL.md (Config)                      │
├─────────────────────────────────────────────────────────┤
│  Hooks                Commands              Templates    │
│  ┌─────────┐         ┌─────────┐          ┌─────────┐  │
│  │PreTool  │─trigger─│/promote │─copy─────│L1/L2    │  │
│  │PostTool │─remind──│/status  │─parse────│templates│  │
│  └─────────┘         │/recover │          └─────────┘  │
│                      └─────────┘                        │
├─────────────────────────────────────────────────────────┤
│  CLAUDE.md (auto-load) ◄─────────────── Scripts (gen)   │
├─────────────────────────────────────────────────────────┤
│  wisdom/ (L3 permanent storage)                          │
│  └── embedding_utils.py (semantic retrieval)            │
└─────────────────────────────────────────────────────────┘
```

### A.9 Unused Skills Capabilities

| Capability | Reason |
|------------|--------|
| `Stop` Hook | Paper requires more complex phase judgment |
| MCP Tools | Not needed for current scenario |
| Multi-Skill Coordination | Single Skill meets requirements |

### A.10 Capability Usage Summary

| Capability | Usage Level | Paper Mapping |
|------------|-------------|---------------|
| **Hooks** | ★★★★★ | N-Action Rule, Context Check |
| **Commands** | ★★★★★ | /promote, /recover, /complete |
| **Scripts** | ★★★★★ | P1/P2 Promotion flow |
| **Templates** | ★★★★ | L1/L2 structured storage |
| **CLAUDE.md** | ★★★★ | Automatic Context Hit |
| **External Scripts (Python)** | ★★★ | L3 Embedding retrieval |

**Core Insight**: The combination of **Hooks + Commands + Scripts** implements the HCC cognitive caching mechanism from the paper, while **CLAUDE.md** provides lightweight automatic context recovery.

---

*Document generated: 2026-02-06*
*ML-Master Version: 2.1.0*
*Paper: arXiv:2601.10402v3*
