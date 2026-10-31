# JengaDB

A robot that sets up Jenga so humans can play. Each placement becomes a recorded experiment: observations, planned and measured motion, and the observed result.

[![Watch the demo](https://img.youtube.com/vi/5L310rz_KnE/hqdefault.jpg)](https://www.youtube.com/shorts/5L310rz_KnE)

## First milestone

Build two alternating layers (six blocks) from a prepared pickup area and replay the placement records. Full-tower construction and variable pickup are later milestones.

## What works today

The offline Python starter produces 20 deterministic synthetic attempts and an HTML report. Repeating the demo preserves the same IDs. No third-party dependencies, credentials or network are needed.

```sh
python3 -m venv --without-pip .venv
.venv/bin/python run.py mock-demo
.venv/bin/python -m unittest discover -s tests -v
```

Open `reports/mock.html` in a browser. These fixtures illustrate storage and reporting; they are not robot measurements or a stacking simulation.

## Next work

Episode recorder and replay viewer, authorized camera/telemetry integration, supervised placement, and optional MongoDB upload. Live robot and database integrations remain untested. Physical actions are not implemented.

- [Verified integration status and connection steps](docs/INTEGRATIONS.md)
- [Build plan and ownership](docs/BUILD_PLAN.md)
- [Experiment memory design](docs/EXPERIMENT_MEMORY.md)
- [Setup and optional integration commands](docs/SETUP.md)
- [Onsite runbook](docs/RUNBOOK.md)

GitHub organization: JengaDB. Viam organization: believed to be hackathon; assigned resources need verification. Atlas management access to project JengaDB and cluster jengadb is verified; application database connectivity remains pending. Linear workspace Skymoondev was reported by the user; task migration has not been performed.

## Work with coding agents

Start with **[START_HERE.md](START_HERE.md)**. It gives each teammate a role, a copyable agent prompt and one check command. Codex and Claude Code have matching repository context; other assistants can be explicitly asked to read AGENTS.md.

```sh
python3 scripts/team.py status
python3 scripts/team.py start data
python3 scripts/team.py check
```

This onboarding harness does not launch agents or control the robot. See [the template evaluation](docs/HARNESS_DECISION.md).
