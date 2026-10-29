import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_team import team

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'tasks').mkdir()
        (self.root/'tasks/X.md').write_text('Acceptance')
        self.path = self.root/'tasks/board.json'
        self.path.write_text(json.dumps([dict(id='X',role='data',title='Recorder',status='ready',card='tasks/X.md')]))

    def test_progress_survives_reload(self):
        team.update_task(self.root,'X','in_progress','Kazy','')
        self.assertEqual(team.load_board(self.root)[0]['owner'],'Kazy')
        team.update_task(self.root,'X','done','Kazy','Acceptance passed; test command result recorded')
        self.assertEqual(team.load_board(self.root)[0]['status'],'done')

    def test_invalid_update_preserves_board(self):
        before = self.path.read_bytes()
        for task, status, owner, evidence in [('X','done','Kazy',''),('X','blocked','Kazy',''),('Y','ready','Kazy',''),('X','in_progress',' ','')]:
            with self.assertRaises(ValueError):
                team.update_task(self.root,task,status,owner,evidence)
            self.assertEqual(self.path.read_bytes(),before)

    def test_failed_atomic_replace_preserves_board(self):
        before = self.path.read_bytes()
        with patch.object(team.os,'replace',side_effect=OSError('full')):
            with self.assertRaises(OSError):
                team.update_task(self.root,'X','in_progress','Kazy','')
        self.assertEqual(self.path.read_bytes(),before)
        self.assertFalse(list((self.root/'tasks').glob('.board-*')))

    def test_completed_role_does_not_restart_finished_work(self):
        team.update_task(self.root,'X','done','Kazy','Verified tests')
        with patch.object(team,'load_board',return_value=team.load_board(self.root)), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(team.main(['start','data']),0)
        self.assertIn('No unfinished task',out.getvalue())
