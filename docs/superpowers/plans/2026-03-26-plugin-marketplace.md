# Plugin Marketplace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two JSON files to make the `liaoruoxue/ml-master-skills` repo a distributable Claude Code plugin marketplace with zero changes to existing files.

**Architecture:** The `skill/` directory becomes a plugin by adding a `.claude-plugin/plugin.json` manifest that points at the existing `SKILL.md`. The repo root gets `.claude-plugin/marketplace.json` listing that plugin. No existing files are modified.

**Tech Stack:** JSON (no code, no dependencies)

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `skill/.claude-plugin/plugin.json` | Plugin manifest — declares metadata and registers `SKILL.md` as the command |
| Create | `.claude-plugin/marketplace.json` | Marketplace catalog — lists the `ml-master` plugin with its source |

---

### Task 1: Create the plugin manifest

**Files:**
- Create: `skill/.claude-plugin/plugin.json`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p skill/.claude-plugin
```

- [ ] **Step 2: Write `plugin.json`**

Create `skill/.claude-plugin/plugin.json` with this exact content:

```json
{
  "name": "ml-master",
  "version": "2.1.0",
  "description": "Hierarchical Cognitive Caching (HCC) for ultra-long-horizon ML tasks. L1/L2/L3 memory layers with P1/P2 promotion.",
  "author": { "name": "liaoruoxue" },
  "homepage": "https://github.com/liaoruoxue/ml-master-skills",
  "repository": "https://github.com/liaoruoxue/ml-master-skills",
  "license": "MIT",
  "keywords": ["ml", "machine-learning", "hcc", "memory", "long-horizon"],
  "category": "productivity",
  "commands": ["./SKILL.md"]
}
```

Note: `commands` paths are resolved relative to the plugin root (`skill/`), so `"./SKILL.md"` resolves to `skill/SKILL.md`.

- [ ] **Step 3: Validate JSON syntax**

```bash
python3 -c "import json; json.load(open('skill/.claude-plugin/plugin.json')); print('valid')"
```

Expected output: `valid`

- [ ] **Step 4: Commit**

```bash
git add skill/.claude-plugin/plugin.json
git commit -m "feat: add plugin manifest for ml-master"
```

---

### Task 2: Create the marketplace catalog

**Files:**
- Create: `.claude-plugin/marketplace.json`

- [ ] **Step 1: Create the directory**

```bash
mkdir -p .claude-plugin
```

- [ ] **Step 2: Write `marketplace.json`**

Create `.claude-plugin/marketplace.json` with this exact content:

```json
{
  "name": "ml-master-skills",
  "owner": { "name": "liaoruoxue" },
  "metadata": {
    "description": "ML-Master: Hierarchical Cognitive Caching skills for long-horizon ML engineering"
  },
  "plugins": [
    {
      "name": "ml-master",
      "source": "./skill",
      "description": "HCC system for ultra-long-horizon ML tasks. L1/L2/L3 memory with P1/P2 promotion.",
      "version": "2.1.0",
      "author": { "name": "liaoruoxue" },
      "homepage": "https://github.com/liaoruoxue/ml-master-skills",
      "repository": "https://github.com/liaoruoxue/ml-master-skills",
      "license": "MIT",
      "category": "productivity",
      "tags": ["ml", "machine-learning", "hcc", "long-horizon", "agents"]
    }
  ]
}
```

Note: `source: "./skill"` is a relative path from the marketplace root. Requires git-based add (not URL-based).

- [ ] **Step 3: Validate JSON syntax**

```bash
python3 -c "import json; json.load(open('.claude-plugin/marketplace.json')); print('valid')"
```

Expected output: `valid`

- [ ] **Step 4: Run plugin validation (if Claude Code CLI available)**

```bash
claude plugin validate .
```

Expected: no errors. If the CLI is not installed, skip this step.

- [ ] **Step 5: Commit**

```bash
git add .claude-plugin/marketplace.json
git commit -m "feat: add marketplace catalog for ml-master-skills"
```

---

### Task 3: Smoke test the install flow

- [ ] **Step 1: Add the marketplace**

In Claude Code:
```
/plugin marketplace add liaoruoxue/ml-master-skills
```

Expected: marketplace added successfully, `ml-master` plugin visible.

- [ ] **Step 2: Install the plugin**

```
/plugin install ml-master@ml-master-skills
```

Expected: installs without error.

- [ ] **Step 3: Verify the skill command is available**

Open any project in Claude Code and run:
```
/ml-master
```

Expected: skill activates (shows HCC architecture prompt).

- [ ] **Step 4: Verify hooks fire**

Run a Write or Edit action. After 5 such actions, expect:
```
[ML-Master] 📝 5 actions done. Update execution_trace.md
```

---

## Follow-up (out of scope for this plan)

- Update `README.md` Quick Start to include the plugin install path alongside the manual `cp` path.
