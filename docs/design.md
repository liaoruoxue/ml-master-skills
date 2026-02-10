This design document guides the refactoring of the existing `planning-with-files` repository into the full engineering implementation of **ML-Master 2.0**.

---

# ML-Master 2.0 Implementation Design Document
**Target Project**: Refactor `planning-with-files` repository
**Core Theory**: Hierarchical Cognitive Caching (HCC) Architecture
**Design Goal**: Enable ultra-long-horizon autonomous machine learning engineering capability

---

## 1. Core Architecture Refactoring: HCC Layered Storage Design

The original repository has a flat file structure. ML-Master 2.0 requires strict partitioning of storage into three cognitive layers, with redesigned file definitions and lifecycles.

### 1.1 L1 Layer: Evolving Experience
*   **Mapped File**: `execution_trace.md` (refactored from `progress.md`)
*   **Function**: Serves as the Agent's "Working Memory (RAM)".
*   **Content**:
    *   Currently executing code patches.
    *   Raw terminal outputs and error stacks.
    *   Temporary observations from step-by-step debugging.
*   **Lifecycle**: **Very short**. Retained only within the current "exploration phase". Must be **cleared** or archived once a phase ends and "Context Promotion" is triggered.

### 1.2 L2 Layer: Refined Knowledge
*   **Mapped File**: `findings.md` (enhanced version)
*   **Function**: Serves as the Agent's "Mid-term Strategic Memory".
*   **Content**:
    *   **Key Judgments**: e.g., "Feature A caused overfitting".
    *   **Experiment Insights**: e.g., "Learning rate 1e-4 is more stable than 1e-3".
    *   **Phase Summaries**: High-value information distilled from L1, with noise removed.
*   **Lifecycle**: **Task-level persistence**. Retained throughout the entire task, survives multiple Context Clears.

### 1.3 L2 Layer: Strategic State
*   **Mapped File**: `task_plan.md` (structured refactoring)
*   **Function**: Prevents "Goal Drift".
*   **Content**:
    *   **Hierarchical Plan Tree**: Directions → Implementations.
    *   **Status Tracking**: pending / in_progress / complete / abandoned.
*   **Lifecycle**: **Task-level persistence**. Dynamically updated as the task progresses.

### 1.4 L3 Layer: Prior Wisdom
*   **Mapped File**: New `wisdom/` directory or `global_wisdom.md`
*   **Function**: Serves as the Agent's "Long-term Memory".
*   **Content**:
    *   Cross-task code templates (e.g., robust CV validation framework).
    *   Common error solution library.
    *   Best practice checklists.
*   **Lifecycle**: **Permanent**. Persists across tasks, read-only (unless L3 update is triggered at task completion).

---

## 2. Core Mechanism Design: Context Migration

This is the engine of ML-Master 2.0, implemented through scripts or prompt logic for automated data flow.

### 2.1 Mechanism 1: Context Prefetching
*   **Trigger**: Task initialization phase (when `/plan` command is executed).
*   **Logic**:
    1.  Analyze the user's task description.
    2.  Retrieve the most relevant knowledge fragments from L3 (`wisdom/`) (e.g., extract CNN-related wisdom for image classification tasks; extract XGBoost-related wisdom for tabular data).
    3.  Inject extracted wisdom into the initial section of `task_plan.md` or Context as "warm start" information.

### 2.2 Mechanism 2: Context Promotion - Phase Level
*   **Trigger**: When a "sub-phase" in `task_plan.md` is marked as complete.
*   **Logic**:
    1.  **Read** all content from L1 (`execution_trace.md`).
    2.  **Perform cognitive compression**: Call LLM to summarize the phase's "execution summary" and "strategic insights".
    3.  **Write** to L2 (`findings.md`).
    4.  **Clear** L1 (`execution_trace.md`).
    *   *Purpose*: Free context window space, convert "noise" into "signal".

### 2.3 Mechanism 3: Context Hit
*   **Trigger**: When the Agent prepares to generate new code or answer questions.
*   **Logic**:
    1.  Check L1 first: If the question involves a recent error, retrieve directly from L1.
    2.  Fall back to L2: If the question involves previous experiment conclusions, retrieve from L2.
    *   *Implementation*: Force the Agent to consult these two files before acting, via System Prompt.

---

## 3. Interaction Flow & Command Design (SKILL.md Refactoring)

SKILL.md needs to be rewritten, transforming the above logic into a "constitution" the Agent must follow.

### 3.1 System Prompt Core Rules
1.  **Dual Read/Write Rule**:
    *   **Execution details** (code runs, errors) must and can only be written to L1.
    *   **Conclusions** (what works, what doesn't) must and can only be written to L2.
2.  **2-Action Rule**:
    *   Force the Agent to update L1 every 2 tool calls (e.g., `RunCommand`, `EditFile`).
3.  **No Context Accumulation**:
    *   Explicitly forbid the Agent from relying on conversation history to remember past errors; must rely on file records.

### 3.2 New Command Design
*   **`/promote`**: Manually trigger phase-level summarization. Force L1 → L2 transformation and clear L1.
*   **`/recover`**: For recovery after `/clear`. Force the Agent to rebuild cognitive state by reading only L2 (`task_plan.md`, `findings.md`), ignoring previous conversation history.

---

## 4. Templates & Script Requirements

### 4.1 Template Files (`templates/`)
*   **`task_plan.md`**: Pre-set hierarchical structure (Phase / Sub-task / Status / Outcome).
*   **`findings.md`**: Pre-set structured sections (Key Insights / Validated Hypotheses / Failed Attempts).
*   **`execution_trace.md`**: Include placeholders for timestamps, operation types, and output summaries.

### 4.2 Automation Scripts (`scripts/`)
*   **Initialization script**: Automatically create the three files at project startup, copying templates from L3 based on task type.
*   **Cleanup script**: Provide a utility function allowing the Agent to clear `execution_trace.md` in one step (called after Promotion).

---

## 5. Acceptance Criteria (Success Metrics)

The refactored system must meet the following criteria to prove ML-Master 2.0 is achieved:

1.  **Persistence Validation**: After executing `/clear` in Claude Code to wipe context, the Agent can accurately state within 1 minute: "We just completed phase X, discovered conclusion Y, and the next step is Z" — **without** including previous verbose error messages.
2.  **Information Flow Validation**: L1 file size should exhibit a "sawtooth" pattern (grows during execution, resets to zero at phase end), while L2 file size should exhibit a "staircase" pattern (monotonically increases at each phase end).
3.  **Long-horizon Reasoning**: At conversation round 50, the Agent can still reference strategic principles established in round 1 and recorded in L2, without "goal drift".
