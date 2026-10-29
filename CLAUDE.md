# JengaDB agent guide

## Start here
Read START_HERE.md, docs/INTEGRATIONS.md, tasks/board.json, and the selected task card in tasks/. Read docs/BUILD_PLAN.md for product context and docs/EXPERIMENT_MEMORY.md for data design. Inspect git status before editing. Preserve existing work.

Goal: humans play Jenga; the robot builds alternating layers. First physical milestone: two layers, six blocks. Current implemented baseline is a mock attempt logger; the live episode recorder and integrations remain tasks, not completed features.

## Work loop
1. Identify one task and its acceptance criteria. Use `python3 scripts/team.py start <role>` to print a ready-to-copy starting prompt (roles: data, vision, hardware).
2. Use an individual branch/checkout for each human-agent pair. Existing dirty work belongs to its owner; do not reset, overwrite, or automatically rebase it. Do not spawn other agents unless the user explicitly requests delegation.
3. Prefer the smallest offline implementation. Work autonomously on authorized reversible code changes and tests; avoid repeated permission questions. Ask only for missing decisions or consequential actions outside scope.
4. Run `python3 scripts/team.py check` and relevant additional tests. Do not infer integration readiness from mock tests.
5. Update the task card with changes, checks, limitations and next action. `python3 scripts/team.py handoff TASK-ID` prints the format. Task board status changes must match evidence.
6. Report what changed, how to try it and what remains. Publication follows the user's authorization; these files do not grant account, publishing or deployment permissions.

## Boundaries
Mock commands need no SDK imports, network or secrets. Keep optional integrations lazy. No physical movement, live execution implementation, machine configuration or obstacle changes as part of the coding-agent bootstrap. Future live steps require the specific mentor-approved plan and designated operator. Never automatically retry motion or forward model output directly to the arm. Credentials and robot access do not constitute authorization.

Never read real .env contents into chat, log secrets, or send private captures to an API without approval for that data and destination. Use placeholder configuration only. No automatic installation on the shared NUC, cloud provisioning, public tunnels or deployment hooks. Default agent permissions remain unchanged.

## Shared interfaces
Read docs/CONTRACTS.md before changing producer/consumer boundaries. Every observation/proposal/event references an episode and source observation. Record frames, units, timestamps and calibration versions. Preserve planned and measured trajectories separately. Observed outcome is success/failure/unknown, independent of command acceptance and lifecycle state. Do not mix synthetic and live records.

## Ownership
Data: recorder, replay, storage. Vision: snapshots, perception proposals. Hardware: read-only discovery and configuration documentation, not motion control. Shared contract or CLI changes require coordination with the affected human; continue independent work while waiting. Task cards are coordination documents, not distributed locks. Git branches and teammate communication establish ownership across laptops.

## Checks and context
`python3 scripts/team.py check` runs context validation and the offline test suite using the current interpreter, with no installs. Python 3.11+; macOS/Linux for the current store. `python3 run.py mock-demo` generates the report. Keep AGENTS.md and CLAUDE.md byte-identical; other coding agents can be explicitly told to read AGENTS.md. No cross-agent CLI, permissions, or slash-command compatibility is assumed.
