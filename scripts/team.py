#!/usr/bin/env python3
"""Small, dependency-free human-to-agent onboarding helper; no agent execution."""
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('AGENTS.md', 'CLAUDE.md', 'START_HERE.md', 'docs/BUILD_PLAN.md',
            'docs/EXPERIMENT_MEMORY.md', 'docs/CONTRACTS.md', 'docs/HARNESS_DECISION.md',
            'docs/INTEGRATIONS.md')


def load_board(root=ROOT):
    board = json.loads((root / 'tasks/board.json').read_text())
    if not isinstance(board, list) or not board:
        raise ValueError('Board must be a nonempty list')
    seen = set()
    for task in board:
        if not isinstance(task, dict) or any(not isinstance(task.get(k), str) or not task[k].strip()
            for k in ('id', 'role', 'title', 'status', 'card')):
            raise ValueError('Missing task fields')
        if task['id'] in seen or task['status'] not in ('ready', 'in_progress', 'blocked', 'done'):
            raise ValueError('Invalid task ID/status')
        if task['role'] not in ('data', 'vision', 'hardware'):
            raise ValueError('Invalid role')
        if task['status'] != 'ready' and not task.get('owner'):
            raise ValueError('Active tasks require an owner')
        if task['status'] in ('done', 'blocked') and not task.get('evidence'):
            raise ValueError('Done/blocked tasks require evidence')
        seen.add(task['id'])
        card = (root / task['card']).resolve()
        if not card.is_relative_to((root / 'tasks').resolve()) or not card.is_file():
            raise ValueError('Missing or unsafe task card path')
    return board


def context_errors(root=ROOT):
    errors = [f'Missing {name}' for name in REQUIRED if not (root / name).is_file()]
    if not errors:
        if (root / 'AGENTS.md').read_bytes() != (root / 'CLAUDE.md').read_bytes():
            errors.append('AGENTS.md and CLAUDE.md differ')
        if (root / 'AGENTS.md').stat().st_size > 12000:
            errors.append('Agent entry point exceeds 12 KB; move detail into docs')
    try:
        load_board(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'Invalid board: {type(exc).__name__}')
    return errors


def starter(task):
    return (f"Read AGENTS.md, START_HERE.md, docs/INTEGRATIONS.md, docs/CONTRACTS.md and {task['card']}.\n"
            f"Help me implement {task['id']}: {task['title']}. My role is {task['role']}.\n"
            "Inspect existing work first. Use a separate branch and preserve other work. "
            "Build the smallest offline slice that meets the task's acceptance criteria. "
            "Use fixtures when integrations are unavailable. Do not move the robot, change its "
            "configuration, upload images, provision services or publish without applicable authorization.\n"
            "Run python3 scripts/team.py check, update my task card with evidence and a handoff, "
            "then explain what works and the next action. Do not describe mock tests as live verification.")


def update_task(root, task_id, status, owner, evidence):
    # Serialize writers on one checkout. This is not a cross-laptop task lock.
    import fcntl
    board_path = root / 'tasks/board.json'
    with (root / 'tasks/.board.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        board = load_board(root)
        task = next((t for t in board if t['id'] == task_id), None)
        if task is None or not owner.strip() or status not in ('ready', 'in_progress', 'blocked', 'done'):
            raise ValueError('Invalid task, status or owner')
        if status in ('done', 'blocked') and not evidence.strip():
            raise ValueError('Provide evidence for done or blocker details for blocked')
        task.update(status=status, owner=owner.strip(), evidence=evidence.strip(),
                    updated_at=datetime.now(timezone.utc).isoformat())
        fd, temporary = tempfile.mkstemp(dir=board_path.parent, prefix='.board-')
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(board, handle, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, board_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    start = sub.add_parser('start', help='Print a prompt; does not launch an agent')
    start.add_argument('role', choices=('data', 'vision', 'hardware'))
    sub.add_parser('status')
    sub.add_parser('check')
    sub.add_parser('doctor', help='Check local prerequisites without reading secrets')
    update = sub.add_parser('update', help='Update a local task; commit/share to coordinate')
    update.add_argument('task')
    update.add_argument('--status', required=True, choices=('ready','in_progress','blocked','done'))
    update.add_argument('--owner', required=True)
    update.add_argument('--evidence', default='')
    handoff = sub.add_parser('handoff', help='Print a handoff template')
    handoff.add_argument('task')
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            print(f'Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
            errors = context_errors()
            if sys.version_info < (3, 11) or sys.platform == 'win32':
                errors.append('Use Python 3.11+ on macOS/Linux (current store uses fcntl)')
            print('\n'.join(errors) if errors else 'Ready for offline work. No SDKs or credentials required.')
            return int(bool(errors))
        if args.command == 'update':
            try:
                update_task(ROOT, args.task, args.status, args.owner, args.evidence)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print('Local board updated. Share through your branch/PR; this does not reserve work on other laptops.')
            return 0
        if args.command == 'check':
            errors = context_errors()
            if errors:
                print('\n'.join(errors))
                return 1
            print('Agent context valid. Running offline tests...', flush=True)
            return subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT).returncode
        board = load_board()
        if args.command == 'start':
            candidates = [t for t in board if t['role'] == args.role and t['status'] != 'done']
            if not candidates:
                print('No unfinished task for this role. Add a reviewed task card to the board.')
                return 0
            task = sorted(candidates, key=lambda t: {'in_progress':0, 'ready':1, 'blocked':2}[t['status']])[0]
            print(starter(task))
        elif args.command == 'handoff':
            task = next((t for t in board if t['id'] == args.task), None)
            if task is None:
                parser.error('Unknown task ID; use status to list tasks')
            print(f"Update {task['card']} with:\nStatus:\nHuman owner:\nBranch/PR:\nChanges:\nChecks and results:\nLimitations/blockers:\nNext action:\nShared interface changes:")
        else:
            print('JengaDB — team board (local checkout; not a distributed lock)')
            for task in board:
                print(f"{task['id']:12} {task['status']:12} {task['role']:10} {task['title']}")
            print('\nStart: python3 scripts/team.py start data|vision|hardware')
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        print('Could not read team context. Check tasks/board.json and task cards.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
