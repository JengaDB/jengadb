import json
import tempfile
import time
import unittest
from pathlib import Path
from robot_hackathon.live_observer import LiveRunLog, snapshot, safe_image


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / '20261029-190517'
        self.run.mkdir()

    def test_partial_journal_retains_prior_events_and_does_not_claim_live(self):
        (self.run / 'events.jsonl').write_text(json.dumps({'time': 1, 'kind': 'move_requested'}) + '\n{"time":')
        run = snapshot(self.root)['runs'][0]
        self.assertEqual(len(run['events']), 1)
        self.assertFalse(run['active'])

    def test_writer_explicit_frame_and_end_state(self):
        log = LiveRunLog(self.run)
        self.addCleanup(log.close)
        log.emit('observation_received', folder='local/whatever/frame-001')
        log.emit('policy_evaluated', reason='Clear end grip')
        log.close('stopped')
        log.close()  # atexit must preserve the original terminal outcome
        run = snapshot(self.root)['runs'][0]
        self.assertFalse(run['active'])
        self.assertEqual(run['status'], 'stopped')
        self.assertEqual(run['events'][1]['frame'], 'frame-001')
        self.assertEqual(run['events'][1]['frame_link'], 'recorded')

    def test_stale_heartbeat_never_looks_live(self):
        (self.run / 'events.jsonl').write_text(json.dumps({'time': 1, 'kind': 'move_completed'}) + '\n')
        (self.run / 'observer-status.json').write_text(json.dumps({'heartbeat': time.time()-20, 'status': 'running'}))
        self.assertFalse(snapshot(self.root)['runs'][0]['active'])
        (self.run / 'observer-status.json').write_text(json.dumps({'heartbeat': time.time(), 'status': 'running'}))
        self.assertTrue(snapshot(self.root)['runs'][0]['active'])

    def test_legacy_report_does_not_invent_reasons(self):
        (self.run / 'report.json').write_text(json.dumps({'events': [{'time': 1, 'kind': 'sam3_capture', 'folder': 'local/x/frame-001'}, {'time': 2, 'kind': 'pick_selected'}]}))
        event = snapshot(self.root)['runs'][0]['events'][1]
        self.assertEqual(event['frame'], 'frame-001')
        self.assertEqual(event['frame_link'], 'inferred_from_event_order')
        self.assertNotIn('reason', event)

    def test_image_paths_cannot_escape_or_serve_secrets(self):
        outside = self.root.parent / (self.root.name + '-private.png')
        outside.write_bytes(b'private')
        self.addCleanup(outside.unlink)
        (self.run / 'rgb.png').symlink_to(outside)
        for name in ['../' + outside.name, str(self.run.relative_to(self.root) / 'rgb.png'), '.env', 'report.json']:
            with self.assertRaises(ValueError): safe_image(self.root, name)

    def test_snapshot_has_no_motion_api(self):
        self.assertFalse(snapshot(self.root)['can_control_robot'])

if __name__ == '__main__':
    unittest.main()
