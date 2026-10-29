# Work with an agent in JengaDB

Open this repository in your preferred coding assistant. Start a new chat with:

> Read AGENTS.md, START_HERE.md and docs/INTEGRATIONS.md. Show the current task status, then help me complete the next task in my role. Work locally with mock data first. My role is DATA / VISION / HARDWARE.

Replace the role with yours. That is enough to start; the agent reads the context rather than asking you to repeat the hackathon briefing.

## Pick your lane

| Person (suggested) | Role | First task |
| --- | --- | --- |
| Kazy + agent | data | DATA-01: episode recorder and replay |
| Chris + agent | vision | VISION-01: snapshot perception contract and offline fixture |
| Sam + agent + mentor | hardware | HARDWARE-01: read-only discovery and verified setup notes |

These are proposed assignments, not claims that teammates have accepted them. One agent session per person/branch is the initial workflow. No agent controls the robot through this harness.

## Simple commands

Run from the repository root using Python 3.11+:

```sh
python3 scripts/team.py
python3 scripts/team.py start data
python3 scripts/team.py start vision
python3 scripts/team.py start hardware
python3 scripts/team.py check
python3 scripts/team.py handoff DATA-01
python3 run.py mock-demo
```

The first command shows the board. `start` prints a self-contained prompt to paste into Codex, Claude Code or another coding assistant. It does not launch an agent, send a message, install software, or reserve a shared task. `handoff` prints a template to add to your task card. `check` validates the context files and runs the actual offline tests.

In Codex, open this checkout as a project. In Claude Code, start in this directory. If another assistant does not automatically read instruction files, paste the prompt above. Your existing account/login and tool permissions still apply; no API key belongs in a prompt.

## Team coordination

Each teammate clones the repository and creates a separate feature branch. Use a PR for review when publishing is authorized. Do not have multiple agents editing one checkout concurrently. Update only your task card; keep board edits small, and coordinate changes to shared contracts. A claim written on one laptop is not a lock on another laptop.

At handoff, give your teammate the branch/PR, changed files, checks and next action. If the chat is lost, a fresh agent can resume from the task card. Do not mark tasks complete because a prompt exists.

## What exists today

Working: offline attempt demo, tests, team prompts, task cards, validation command and CI definition.
Not yet built: live episode recorder, replay timeline, segmentation API, application database connection or physical stacking. The harness is merged into main, and its GitHub CI passed on Python 3.11 and 3.12. Atlas management authentication and project discovery succeeded; database reads/writes remain untested. See [integration status](docs/INTEGRATIONS.md) before starting live work.

See docs/HARNESS_DECISION.md for the template evaluation and docs/BUILD_PLAN.md for milestones. The onboarding harness helps humans work with coding agents. The runtime experiment harness (DATA-01) remains the next build.

## First checkout

Clone main, which includes the merged harness:

```sh
git clone https://github.com/JengaDB/jengadb.git
cd jengadb
python3 scripts/team.py doctor
python3 scripts/team.py check
```

Create your own feature branch before editing. No package installation is required. Use your existing editor/assistant; the harness does not require a new agent account.

## Save progress

`tasks/board.json` is the single source of task status. Task cards hold acceptance criteria and detailed handoffs. Record status through the helper:

```sh
python3 scripts/team.py update DATA-01 --status in_progress --owner Kazy
python3 scripts/team.py update DATA-01 --status blocked --owner Kazy --evidence "Waiting for agreed episode schema"
python3 scripts/team.py update DATA-01 --status done --owner Kazy --evidence "Describe acceptance evidence and exact test results here"
```

Use truthful evidence, not the example placeholder. Updating status is local: commit the board/card to your feature branch and share it with the team. It does not assign anyone a GitHub issue or reserve work across laptops. Merge conflicts in task state require human coordination; never silently prefer one person's claim.

## When to use which harness

This harness provides coding-agent context, role prompts, persistent task state, environment checks, handoffs and CI. It does not provide a multi-user chat UI or invoke model APIs. The separate robot experiment harness is DATA-01; finishing coding-agent onboarding does not mark that task done.
