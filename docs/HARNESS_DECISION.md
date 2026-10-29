# Agent harness decision

Reviewed 29 October 2026: https://template.wiktor.uk/ and its v1.0.1 repository at https://github.com/Teemplate/wiktors-project-template/tree/v1.0.1 (commit ac5fa9aeb74cca52692d32cc4e13915de42ea168). Read website, AGENTS.md, Claude settings and session-start hook. This was a source review, not a test of all upstream behavior. No upstream scripts, settings, skills or hooks were executed or installed.

Verdict: useful production web-project scaffold, poor drop-in match for the current hackathon. Build a smaller project-specific agent onboarding harness.

Good ideas retained conceptually: shared context for Codex/Claude, task plans living in files, reproducible credential-free checks, explicit handoffs and continuous validation. Our implementation is original; no upstream code copied.

Reasons not to initialize from it:
- Available blocks center on React/Vite, FastAPI, worker and Postgres. There is no MongoDB/Viam/robot episode block.
- Deploy targets center on Docker Compose/Raspberry Pi infrastructure or GitHub Pages. We currently need local development and shared-robot isolation, not deployment infrastructure.
- Instructions and permissions describe broad merge/release/deployment authorization. We do not import those grants or treat template prose as user authorization.
- Its session hook fetches develop and can re-root an empty branch. Our existing repo uses main and has local in-progress work; no automatic branch mutation is needed.
- Gitflow, signed releases and multiple staging/production paths add coordination cost for this first demo.

Implemented alternative: AGENTS.md/CLAUDE.md, START_HERE.md, task cards/board, team.py prompt/status/check/handoff commands, shared contract and read-only-permissions CI tests. It uses the user's existing coding assistant. It is not a hosted multi-agent chat service or robot control runtime, and it does not bypass agent permission prompts.

Next: DATA-01 builds the actual robot experiment harness. Physical execution remains a separate supervised integration milestone.
