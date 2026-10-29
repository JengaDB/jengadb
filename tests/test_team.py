import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('team', ROOT/'scripts/team.py')
team = importlib.util.module_from_spec(spec)
spec.loader.exec_module(team)

class TeamContextTests(unittest.TestCase):
    def test_each_role_has_resumable_prompt(self):
        for task in team.load_board():
            prompt = team.starter(task)
            self.assertIn(task['card'], prompt)
            self.assertIn('offline', prompt)
            self.assertIn('Do not move the robot', prompt)

    def test_instruction_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in team.REQUIRED:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('context')
            (root/'CLAUDE.md').write_text('different')
            self.assertIn('AGENTS.md and CLAUDE.md differ', team.context_errors(root))

    def test_board_cannot_read_outside_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tasks').mkdir()
            (root/'outside.md').write_text('not a task')
            (root/'tasks/board.json').write_text(json.dumps([dict(id='X',role='data',title='Test',status='ready',card='outside.md')]))
            with self.assertRaises(ValueError):
                team.load_board(root)

    def test_duplicate_task_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tasks').mkdir()
            (root/'tasks/X.md').write_text('task')
            task = dict(id='X',role='data',title='Test',status='ready',card='tasks/X.md')
            (root/'tasks/board.json').write_text(json.dumps([task, task]))
            with self.assertRaises(ValueError):
                team.load_board(root)
