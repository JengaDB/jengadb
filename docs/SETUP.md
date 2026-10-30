# Robot attempt logger

A working offline hackathon fallback: 20 deterministic synthetic attempts, durable local JSONL, and an HTML inspector. No robot motion is implemented. Mock mode needs no credentials, network, or third-party packages.

## Run on this Mac

```sh
cd jengadb
.venv/bin/python run.py mock-demo
.venv/bin/python run.py report
open reports/mock.html
.venv/bin/python -m unittest discover -s tests -v
```

The default command (no arguments) is also `mock-demo`. Repeating it preserves exactly 20 fixture IDs. Expected totals: 12 successes, 4 failures, 4 unknown; known-outcome success rate 75%. These are synthetic results, not evidence of real robot performance. The two strategy labels illustrate grouping, not measured improvement.

On a fresh checkout (Mac or Linux with Python 3.11+):

```sh
python3 -m venv --without-pip .venv
.venv/bin/python run.py mock-demo
.venv/bin/python -m unittest discover -s tests -v
```

No package installation is required for these commands. `fcntl` locking makes this local store a macOS/Linux implementation; native Windows is not supported.

## Commands

| Command | Behavior |
| --- | --- |
| `mock-demo` | Generate deterministic fixtures; save JSONL and mock HTML |
| `report` | Read selected mode's local records and generate HTML |
| `viam-check --allow-cloud` | Opt-in, nonmoving resource discovery |
| `mongo-ping --allow-cloud` | Bounded ping only; no index or writes |
| `mongo-index --allow-cloud --confirm-write` | Create unique attempt_id index on authorized test collection |
| `flush --allow-cloud --confirm-write` | Deliver pending local records, verify read-back, acknowledge locally |

`--data-dir` changes the local data root. `--report` changes the HTML destination. `MODE=mock` and `MODE=live` use separate directories and default report files. There is no live recording/action CLI yet. Never copy synthetic records into the live directory.

## Configuration and optional integrations

Defaults: `MODE=mock`, `ENABLE_MOTION=false`, `ALLOW_CLOUD=false`. Invalid booleans fail; enabling motion always fails. Configuration is read from environment variables. **`.env.example` documents names only; `.env` is not automatically loaded.** Atlas service-account authentication was verified separately; it is not configured in the application. See [integration status](INTEGRATIONS.md). The runtime database adapter needs MONGODB_URI, not the Atlas management client ID/secret. No credentials are stored in this repository.

SDKs have not been installed or resolved. No lockfile is supplied because resolution has not occurred. Optional dependency declarations are provisional, and live integration remains untested. Once installation is approved, the intended local-only setup is:

```sh
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install -e '.[viam,mongo]'
.venv/bin/python -m pip freeze --exclude-editable > requirements.lock.txt
```

Run installation only when needed; inspect resolution results before treating SDK compatibility as verified. `requirements.lock.txt` will record the successfully installed versions for this platform; it is not a cross-platform hash lock.

For authorized connections, enter secrets locally at hidden zsh prompts; do not place values in commands, screenshots, or chat:

```sh
read -rs 'MACHINE_ADDRESS?Assigned machine address (hidden): '; echo
read -rs 'API_KEY_ID?Machine key ID (hidden): '; echo
read -rs 'API_KEY?Machine key (hidden): '; echo
export MACHINE_ADDRESS API_KEY_ID API_KEY
ALLOW_CLOUD=true .venv/bin/python run.py viam-check --allow-cloud
```

```sh
read -rs 'MONGODB_URI?Authorized MongoDB URI (hidden): '; echo
export MONGODB_URI
# Set these to organizer-approved test names before any call:
export MONGODB_DATABASE=hackathon MONGODB_COLLECTION=attempts
ALLOW_CLOUD=true .venv/bin/python run.py mongo-ping --allow-cloud
```

Only after separate approval for index creation and synthetic writes on that exact destination:

```sh
ALLOW_CLOUD=true .venv/bin/python run.py mongo-index --allow-cloud --confirm-write
ALLOW_CLOUD=true .venv/bin/python run.py flush --allow-cloud --confirm-write
```

## RGB-D acquisition and Atlas connection

Reused from SKY-66 (see docs/CACHE_INTEGRATION.md for what was and was not carried). Requires `pip install -e '.[rgbd,viam,mongo]'`. Records are written under `data/<mode>/observations/`; live capture requires `MODE=live`. Secrets come from the shell (`set -a; source .env; set +a` loads a git-ignored `.env`; nothing loads it automatically).

| Command | Behavior |
| --- | --- |
| `rgbd-calibration --calibration-version V --alignment verified\|unverified\|unknown --method ... --checked-by ...` | Record the operator's RGB/depth alignment verdict for `SETUP_ID`; `--show` prints the current entry |
| `rgbd-capture --allow-cloud [--episode ID]` | One read-only `camera.get_images` on `CAMERA_RESOURCE`; saves `rgb.png`, 16-bit `depth.png`, `observation.json` |
| `rgbd-overlay --observation ID` | Depth-edge overlay PNG for the alignment check (a picture, not a verdict) |
| `rgbd-status [--orphans --allow-cloud]` | Local observations, per-artifact hash status and acks; `--orphans` lists upload attempts past the age bound |
| `rgbd-verify-embedding --allow-cloud` | Text-only call to the hosted Voyage endpoint through the cached provider; reports model and dimension |
| `rgbd-index --allow-cloud --confirm-write [--status]` | Unique indexes plus the `candidate_context_v1` vector index |
| `rgbd-flush --observation ID\|--all --allow-cloud --confirm-write [--clean-orphans]` | Upload only the named captures: media -> observation -> artifact_sync, idempotent with separate acks. No candidates on this branch |

Local Viam for development (second D435 on this Mac): `brew trust viamrobotics/brews && brew tap viamrobotics/brews && brew install viam-server viam`, download the `viam:realsense` module for `darwin/arm64`, point `configs/viam-local-mac.json` `modules[0].executable_path` at the extracted binary, plug in the camera and run `sudo viam-server -config configs/viam-local-mac.json` (the macOS module exits unless run as root). Then `MODE=live MACHINE_ADDRESS=localhost:8080 MACHINE_AUTH=none SETUP_ID=mac-desk-d435 ALLOW_CLOUD=true`. `MACHINE_AUTH=none` is rejected for any non-loopback address. The embedding cache lives in `data/<mode>/provider-cache/results.sqlite3`; delete the directory to reset it.

The flags express the operator's opt-in; they are not an access-control system. Use a scoped credential. TLS validation remains enabled. Connection, selection, socket, and operation timeouts are bounded. Ping does not establish write permission. No application database commands have been run. The separate Atlas management check authenticated and listed projects/clusters; it did not ping or write to the database.

## Durability and schema

Local writes lock the store, validate records, fsync a temporary file, replace atomically, and fsync the directory. IDs are immutable: identical repeat delivery is accepted; different content under the same ID is rejected. Corrupt JSONL fails closed.

The local log doubles as the spool. Acknowledgments are stored separately per destination hash. Remote writes use an immutable upsert with read-back and require a unique index. A failed or interrupted delivery remains pending, including timeout-after-commit; rerunning flush redelivers the same IDs. Source records are never deleted by flush. The hash identifies host/database/collection, not credentials. If a remote collection is deleted/recreated at the same name, its old acknowledgment file must be retired deliberately before replay.

Schema v1 includes UTC timestamp, mode, object, strategy, observed outcome, duration in milliseconds, optional pose (frame + mm/deg tags), sanitized error category, optional confidence in [0,1], and artifact references. Confidence is null in fixtures; define its measurement meaning before using a real value. There are no raw images. Labels and artifact references must be reviewed before future real-data uploads; arbitrary input is not automatically secret-redacted.

## Cleanup

Clear connection variables after use:

```sh
unset API_KEY API_KEY_ID MACHINE_ADDRESS MONGODB_URI ALLOW_CLOUD
```

Keep source and approved fixtures. To remove this generated environment or demo outputs, move `.venv`, `data/mock`, and `reports/mock.html` to Trash in Finder after closing the report. Recreate the environment with the setup command and regenerate the report with `mock-demo`. Do not delete live records or remote data as part of demo cleanup. Ask credential owners to revoke temporary access after the event.

## References

Connection API checked on 29 October 2026: [Viam RobotClient](https://python.viam.dev/autoapi/viam/robot/client/index.html). MongoDB options checked against [official PyMongo connection options](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/connect/connection-options/). These checks validate API design only, not installed SDK compatibility or live connectivity. Event planning and hardware constraints below come from the supplied preparation document.
