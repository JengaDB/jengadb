import asyncio
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch, AsyncMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from robot_hackathon.cli import main
from robot_hackathon.config import Config, ConfigError
from robot_hackathon.local_store import LocalStore, atomic_write
from robot_hackathon.mock_robot import MockRobot
from robot_hackathon.report import summarize, render
from robot_hackathon.viam_adapter import ViamAdapter

class FakeRemote:
    def __init__(self):
        self.rows = {}
        self.fail = None
        self.calls = 0
    def upsert(self, row):
        self.calls += 1
        self.rows[row['attempt_id']] = row.copy()
        if self.fail == self.calls:
            raise TimeoutError('mongodb://fake-secret@host: never display')

class Acceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.store = LocalStore(self.base)
        self.rows = MockRobot().attempts()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict('os.environ', {}, clear=True))
        for name in ('connect', 'connect_ex', 'sendto'):
            self.stack.enter_context(patch.object(socket.socket, name, side_effect=AssertionError('Network blocked')))
        self.stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('Network blocked')))
        self.stack.enter_context(patch('socket.getaddrinfo', side_effect=AssertionError('DNS blocked')))

    def cli(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, output.getvalue()

    def test_offline_blank_secrets_repeat_without_sdk_or_hardware(self):
        with patch.dict(sys.modules, {'viam': None, 'pymongo': None}), patch.object(ViamAdapter, 'discover', side_effect=AssertionError('Hardware forbidden')):
            args = ['mock-demo', '--data-dir', str(self.base), '--report', str(self.base/'report.html')]
            self.assertEqual(self.cli(args)[0], 0)
            first = self.store.path.read_bytes()
            self.assertEqual(self.cli(args)[0], 0)
            self.assertEqual(first, self.store.path.read_bytes())
            self.assertEqual(len(self.store.read()), 20)
            self.assertEqual(len({r['attempt_id'] for r in self.store.read()}), 20)
            self.assertIn('MOCK DATA', (self.base/'report.html').read_text())

    def test_fixture_matches_generator(self):
        fixture = Path(__file__).resolve().parents[1]/'fixtures'/'mock-attempts.jsonl'
        self.assertEqual([json.loads(x) for x in fixture.read_text().splitlines()], self.rows)

    def test_unknown_denominator(self):
        groups = summarize(self.rows)
        self.assertTrue(all(g['rate'] == .75 and g['unknown'] == 2 for g in groups))
        self.assertIsNone(summarize([self.rows[4]])[0]['rate'])

    def test_malformed_configuration(self):
        for env in ({'MODE':'bad'}, {'ALLOW_CLOUD':'yes'}, {'ENABLE_MOTION':'1'}):
            with self.subTest(env=env), self.assertRaises(ConfigError):
                Config.from_env(env)
        self.assertFalse(Config.from_env({'ALLOW_CLOUD':'false'}).allow_cloud)
        # ENABLE_MOTION alone never suffices: live-step also needs MODE=live and a named operator.
        with self.assertRaises(ConfigError):
            Config.from_env({'ENABLE_MOTION':'true'}).require_motion('Kris')
        with self.assertRaises(ConfigError):
            Config.from_env({'ENABLE_MOTION':'true', 'MODE':'live'}).require_motion('')
        with self.assertRaises(ConfigError):
            Config.from_env({'MODE':'live'}).require_motion('Kris')
        Config.from_env({'ENABLE_MOTION':'true', 'MODE':'live'}).require_motion('Kris')

    def test_default_cloud_commands_denied(self):
        for command in ('viam-check', 'mongo-ping', 'mongo-index', 'flush'):
            code, output = self.cli([command])
            self.assertEqual(code, 2)
            self.assertIn('Requires', output)

    def test_write_requires_second_gate(self):
        with patch.dict('os.environ', {'ALLOW_CLOUD':'true'}):
            for command in ('flush', 'mongo-index'):
                code, output = self.cli([command, '--allow-cloud'])
                self.assertEqual(code, 2)
                self.assertIn('--confirm-write', output)

    def test_config_repr_hides_credentials(self):
        config = Config(uri='secret-uri', key='secret-key', key_id='secret-id', address='secret-host')
        self.assertNotIn('secret', repr(config))

    def test_insecure_mongo_config_denied(self):
        for uri in ('bad', 'mongodb://host/?tls=false', 'mongodb://host/?tlsInsecure=true', 'mongodb://host/?tlsAllowInvalidHostnames=true'):
            with self.subTest(uri=uri), self.assertRaises(ConfigError):
                Config(uri=uri).require_mongo()

    def test_failed_remote_keeps_spool_and_sanitizes(self):
        self.store.put_many(self.rows)
        remote = FakeRemote()
        remote.fail = 1
        result = self.store.flush(remote, 'test')
        self.assertEqual(result['pending'], 20)
        self.assertNotIn('secret', json.dumps(result))
        self.assertEqual(self.store.read(), self.rows)

    def test_timeout_after_commit_and_repeat_flush(self):
        self.store.put_many(self.rows)
        remote = FakeRemote()
        remote.fail = 3
        self.assertEqual(self.store.flush(remote, 'test')['sent'], 2)
        remote.fail = None
        self.assertEqual(self.store.flush(remote, 'test')['sent'], 18)
        self.assertEqual(len(remote.rows), 20)
        self.assertEqual(self.store.flush(remote, 'test')['sent'], 0)
        self.assertEqual(self.store.flush(remote, 'different-target')['sent'], 20)

    def test_interruption_before_ack_redelivers(self):
        self.store.put_many(self.rows)
        remote = FakeRemote()
        with patch('robot_hackathon.local_store.atomic_write', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.store.flush(remote, 'test')
        self.assertEqual(self.store.flush(remote, 'test')['sent'], 20)
        self.assertEqual(len(remote.rows), 20)

    def test_auth_failure_is_not_acked(self):
        self.store.put_many(self.rows)
        remote = SimpleNamespace(upsert=lambda row: (_ for _ in ()).throw(PermissionError('secret-password')))
        self.assertEqual(self.store.flush(remote, 'test'), {'sent': 0, 'pending': 20, 'error': 'remote_error'})

    def test_id_conflict_and_mode_mix_rejected(self):
        self.store.put_many(self.rows)
        with self.assertRaises(ValueError):
            self.store.put_many([{**self.rows[0], 'observed_outcome':'failure'}])
        with self.assertRaises(ValueError):
            self.store.put_many([{**self.rows[0], 'mode':'live'}])
        with self.assertRaises(ValueError):
            render([{**self.rows[0], 'mode':'live'}], self.base/'r.html', 'mock')
        self.assertEqual(self.store.read(), self.rows)

    def test_atomic_failure_preserves_previous_file(self):
        self.store.put_many(self.rows)
        before = self.store.path.read_bytes()
        with patch('os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                atomic_write(self.store.path, 'bad')
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_discovery_uses_one_client_and_closes(self):
        client = SimpleNamespace(resource_names=[SimpleNamespace(name='cam')], close=AsyncMock())
        factory = SimpleNamespace(Options=SimpleNamespace(with_api_key=lambda **kw: kw), at_address=AsyncMock(return_value=client))
        with patch.dict(sys.modules, {'viam':ModuleType('viam'), 'viam.robot':ModuleType('viam.robot'), 'viam.robot.client':SimpleNamespace(RobotClient=factory)}):
            self.assertEqual(asyncio.run(ViamAdapter.discover(Config(address='fake', key='fake', key_id='fake'))), ['cam'])
        factory.at_address.assert_awaited_once()
        client.close.assert_awaited_once()

    def test_discovery_closes_when_resource_read_fails(self):
        class BrokenClient:
            close = AsyncMock()
            @property
            def resource_names(self):
                raise RuntimeError('secret')
        client = BrokenClient()
        factory = SimpleNamespace(Options=SimpleNamespace(with_api_key=lambda **kw: kw), at_address=AsyncMock(return_value=client))
        with patch.dict(sys.modules, {'viam':ModuleType('viam'), 'viam.robot':ModuleType('viam.robot'), 'viam.robot.client':SimpleNamespace(RobotClient=factory)}):
            with self.assertRaises(RuntimeError):
                asyncio.run(ViamAdapter.discover(Config(address='fake', key='fake', key_id='fake')))
        client.close.assert_awaited_once()

    def test_live_action_always_disabled(self):
        with self.assertRaises(RuntimeError):
            ViamAdapter().execute()

    def test_cli_error_does_not_print_sdk_secrets(self):
        with patch.dict('os.environ', {'ALLOW_CLOUD':'true'}), patch.object(ViamAdapter, 'discover', side_effect=RuntimeError('secret-token')):
            code, output = self.cli(['viam-check', '--allow-cloud'])
        self.assertEqual(code, 1)
        self.assertNotIn('secret-token', output)

if __name__ == '__main__':
    unittest.main()
