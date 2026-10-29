# DATA-02: Connect the existing Atlas database

Status and owner: see board.json (single source of truth)
Role: data
Branch/PR: not assigned

## Goal and acceptance

Use the existing JengaDB Atlas project and jengadb cluster. Confirm readiness, configure a scoped runtime database connection locally and verify a bounded ping. For the authorized synthetic destination, create the unique attempt_id index, upload the fixture records, verify read-back and repeat without duplicates. Test a failed upload and successful retry while preserving local records. Record actual results separately from offline tests.

## Intended files

docs/INTEGRATIONS.md, docs/SETUP.md and existing MongoDB adapter/configuration if a verified incompatibility requires a fix. Secrets stay outside version control. Do not change the shared episode schema for a connection test.

## Dependencies

Read docs/INTEGRATIONS.md and docs/CONTRACTS.md. Atlas management authentication and project discovery already succeeded on 29 October 2026. The cluster last reported CREATING; readiness and its connection string must be checked again. Database user, network access and exact test destination still need configuration/confirmation. Management credentials are not runtime database credentials. Live episode upload depends on DATA-01; mock fixture upload does not.

## Handoff

Changes: added connection task and verified management-access context only.
Checks: prior management API authentication and read-only discovery succeeded; no database ping or write/read test run.
Limitations: application connection is not configured; service-account credentials are not stored in the harness.
Next: check the existing cluster and establish the scoped database connection, then record evidence for the acceptance criteria.
