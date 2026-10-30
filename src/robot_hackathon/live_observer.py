"""Read-only local run observer. No robot, model, database, or credential imports."""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False))
    tmp.replace(path)


class LiveRunLog:
    """Append-only observation/action evidence, separate from the legacy report."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frame = None
        self.lock = threading.Lock()
        self.closed = False
        self.finished = threading.Event()
        self.pulse = threading.Thread(target=self._heartbeat, daemon=True)
        self.pulse.start()

    def _heartbeat(self):
        while not self.finished.is_set():
            with self.lock:
                if not self.closed:
                    atomic_json(self.directory / 'observer-status.json', {'heartbeat': time.time(), 'status': 'running'})
            self.finished.wait(2)

    def emit(self, kind, **payload):
        with self.lock:
            if self.closed:
                return
            folder = payload.get('folder')
            if folder and kind in ('observation_received', 'sam3_capture'):
                self.frame = Path(folder).name
            record = {'time': time.time(), 'kind': kind, 'observation_frame': self.frame, **payload}
            with (self.directory / 'events.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
                f.flush()

    def close(self, status='ended'):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            atomic_json(self.directory / 'observer-status.json', {'heartbeat': time.time(), 'status': status})
        self.finished.set()


def read_json(path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def events_for(directory):
    journal = directory / 'events.jsonl'
    if journal.exists():
        records = []
        # A concurrent append can leave an incomplete final line; retain earlier records.
        for line in journal.read_text().splitlines():
            try:
                record = json.loads(line)
                if isinstance(record, dict) and isinstance(record.get('time'), (int, float)):
                    records.append(record)
            except ValueError:
                pass
        return records
    return read_json(directory / 'report.json', {}).get('events', [])


def safe_image(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() != '.png' or not path.is_file():
        raise ValueError('Image outside observation store')
    # Expose only images from captured frames or the saved final result.
    if path.name not in ('rgb.png', 'depth.png', 'detections.png', 'first-layer-result.png', 'plan.png'):
        raise ValueError('Not an observation image')
    return path


def snapshot(root):
    root = Path(root).resolve()
    runs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not re.fullmatch(r'\d{8}-\d{6}', directory.name):
            continue
        if not directory.resolve().is_relative_to(root):
            continue
        try:
            events = events_for(directory)
        except OSError:
            continue
        if not events:
            continue
        report = read_json(directory / 'report.json', {})
        status = read_json(directory / 'observer-status.json', {})
        heartbeat = status.get('heartbeat', 0)
        active = status.get('status') == 'running' and 0 <= time.time() - heartbeat < 8
        frames, normalized = {}, []
        last_frame = None
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            record = dict(event)
            folder = record.get('folder')
            if folder and record.get('kind') in ('observation_received', 'sam3_capture'):
                name = Path(folder).name
                if re.fullmatch(r'frame-\d+', name):
                    last_frame = name
                    frame = frames.setdefault(name, {'name': name})
                    frame.setdefault('time', record.get('time'))
                    frame['blocks'] = record.get('blocks', frame.get('blocks', []))
                    for label, file in [('rgb', 'rgb.png'), ('depth', 'depth.png')]:
                        relative = f'{directory.name}/{name}/{file}'
                        try:
                            safe_image(root, relative)
                            frame[label] = '/image/' + relative
                        except ValueError:
                            pass
            record['frame'] = record.get('observation_frame') or last_frame
            record['frame_link'] = 'recorded' if record.get('observation_frame') else 'inferred_from_event_order'
            record['id'] = f'{directory.name}:{index}'
            # Local paths are internal implementation details; frame URLs are separately scoped.
            record.pop('folder', None)
            normalized.append(record)
        runs.append({'id': directory.name, 'active': active, 'last_heartbeat': heartbeat or None,
                     'status': status.get('status') or report.get('status', 'recorded'),
                     'events': normalized, 'frames': frames})
    result = read_json(root / 'first-layer-result.json', None)
    return {'server_time': time.time(), 'runs': runs, 'result': result,
            'capture_mode': 'Workflow snapshots, not continuous video', 'can_control_robot': False}


def make_handler(root):
    root = Path(root).resolve()
    html = Path(__file__).parent / 'web' / 'observer.html'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = unquote(urlparse(self.path).path)
            try:
                if route == '/':
                    body, mime = html.read_bytes(), 'text/html; charset=utf-8'
                elif route == '/api/state':
                    body, mime = json.dumps(snapshot(root), allow_nan=False).encode(), 'application/json'
                elif route.startswith('/image/'):
                    body, mime = safe_image(root, route[len('/image/'):]).read_bytes(), 'image/png'
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    args.runs.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(args.runs))
    print(f'Observation viewer: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
