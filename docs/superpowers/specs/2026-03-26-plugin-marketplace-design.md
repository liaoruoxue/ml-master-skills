# Plugin Marketplace Design

**Date:** 2026-03-26
**Repo:** liaoruoxue/ml-master-skills
**Status:** Approved

## Goal

Make this repo a distributable Claude Code plugin marketplace so users can install the ML-Master skill via:

```
/plugin marketplace add liaoruoxue/ml-master-skills
/plugin install ml-master@ml-master-skills
```

## Constraints

- Zero changes to existing files (wrap in-place)
- Both install paths must continue to work:
  - Manual: `cp -r skill/ .claude/skills/ml-master/`
  - Plugin: `/plugin install ml-master@ml-master-skills`

## Implementation

Two new files only.

### File 1: `skill/.claude-plugin/plugin.json`

Plugin manifest for the `ml-master` plugin. The **plugin root** is `skill/` (the directory containing `.claude-plugin/`). Per the Claude Code plugin spec, `commands` paths are resolved relative to the plugin root — so `"./SKILL.md"` resolves to `skill/SKILL.md`.

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

### File 2: `.claude-plugin/marketplace.json`

Marketplace catalog at the repo root. `source: "./skill"` is a relative path to the plugin directory. Relative paths require git-based marketplace add (not URL-based).

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

**Version sync:** `marketplace.json` mirrors the version from `plugin.json` manually on each release. These must be kept in sync by the maintainer; there is no automated check. Accepted risk: on divergence, the plugin manifest version wins silently (Claude Code plugin spec behaviour).

## Testing

After creating the two files, verify both install paths:

**Manual install path (unchanged):**
```bash
cp -r skill/ /tmp/test-ml/.claude/skills/ml-master/
ls /tmp/test-ml/.claude/skills/ml-master/scripts/*.sh   # should list 8 scripts
```

**Plugin install path:**
```
/plugin marketplace add liaoruoxue/ml-master-skills
/plugin install ml-master@ml-master-skills
# Then open a project and run:
/status
# Expect: status output from scripts/status.sh
```

**Hook smoke test:**
Run one Write action in a project with the plugin installed. Confirm the PostToolUse counter fires and prints `[ML-Master] 📝 5 actions done...` every 5 actions.

**Validation:**
```bash
claude plugin validate .   # run from repo root
```

## Trade-offs

- **Relative path source** means users must add the marketplace via GitHub (`liaoruoxue/ml-master-skills`), not via a direct URL to `marketplace.json`. Acceptable for a git-hosted repo.
- **Hooks in SKILL.md frontmatter** are not duplicated in `plugin.json` — single source of truth. This relies on the plugin system honoring `SKILL.md` hooks when loaded via `commands`.
- **`CLAUDE_PLUGIN_ROOT` dependency:** The Stop hook in `SKILL.md` uses `${CLAUDE_PLUGIN_ROOT:-$(dirname "$0")}`. The `$(dirname "$0")` fallback is non-functional when invoked by the plugin system (returns the shell binary path). If `CLAUDE_PLUGIN_ROOT` is not set, the check-complete step silently no-ops with a warning — graceful degradation, not a crash.
- **README not updated** as part of this change (zero-changes constraint). The manual `cp` Quick Start in README remains valid; the plugin install path should be added in a follow-up.
