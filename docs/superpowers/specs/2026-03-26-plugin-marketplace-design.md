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

Plugin manifest for the `ml-master` plugin. Points `commands` at the existing `SKILL.md` so the plugin system picks it up without moving the file.

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

`commands: ["./SKILL.md"]` registers `SKILL.md` as the skill/command file. Hooks defined in its YAML frontmatter are loaded automatically.

### File 2: `.claude-plugin/marketplace.json`

Marketplace catalog at the repo root. `source: "./skill"` is a relative path to the plugin directory (requires git-based marketplace add, not URL-based).

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

## Trade-offs

- **Relative path source** means users must add the marketplace via GitHub (`liaoruoxue/ml-master-skills`), not via a direct URL to `marketplace.json`. This is fine for a git-hosted repo.
- **Hooks in SKILL.md frontmatter** are not duplicated in `plugin.json`. This keeps a single source of truth but relies on the plugin system honoring SKILL.md's hooks frontmatter when loaded via `commands`.
