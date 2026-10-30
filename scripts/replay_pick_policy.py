"""Replay saved segmentation/embeddings through comparison and a proposal policy.

No camera or motion calls. Atlas access is read-only and explicitly enabled.
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robot_hackathon import rgbd
from robot_hackathon.candidate_store import CandidateLocalStore
from robot_hackathon.config import Config
from robot_hackathon.local_store import atomic_write
from robot_hackathon.pick_policy import POLICY_VERSION, propose
from robot_hackathon.pipeline import load_embedding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--observation", required=True)
    p.add_argument("--processing-version", required=True)
    p.add_argument("--mode", choices=("mock", "live"), default="live")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--comparisons", type=Path, help="Saved comparisons JSON; offline")
    group.add_argument("--atlas", action="store_true", help="Read-only vector comparisons")
    p.add_argument("--allow-cloud", action="store_true")
    p.add_argument("--geometry", type=Path, help="Optional verified geometry from the hardware owner")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    obs = rgbd.ObservationStore(args.data_dir, args.mode).load(args.observation, verify=True)
    candidates = [c for c in CandidateLocalStore(args.data_dir, args.mode).read()
                  if c["observation_id"] == args.observation and c["processing_version"] == args.processing_version]
    if not candidates:
        p.error("No saved candidates for this observation and processing version")
    geometry = json.loads(args.geometry.read_text()) if args.geometry else {}
    run_id = "replay_" + uuid.uuid4().hex[:12]
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    events = []
    def event(kind, payload):
        events.append({"event_id": f"{run_id}_{len(events):03d}", "episode_id": obs["episode_id"],
                       "observation_id": obs["observation_id"], "timestamp": rgbd.utc_now(),
                       "type": kind, "source": "saved_replay", "payload": payload})
        atomic_write(output / "events.jsonl", "".join(rgbd.canonical_json(e) + "\n" for e in events))
    started = time.monotonic()
    try:
        event("observation_loaded", {"artifact_hashes": [a["sha256"] for a in obs["artifacts"]]})
        event("segmentation_reused", {"processing_version": args.processing_version,
                                      "candidate_ids": [c["candidate_id"] for c in candidates],
                                      "segmentation_keys": sorted({c["segmentation"]["segmentation_key"] for c in candidates})})
        if args.atlas:
            config = Config.from_env()
            config.require_cloud(args.allow_cloud)
            from robot_hackathon.rgbd_mongo import RgbdMongoStore
            remote = RgbdMongoStore(config)
            try:
                comparisons = {}
                for c in candidates:
                    hits = remote.vector_search(load_embedding(args.data_dir, c),
                        {"mode": args.mode, "encoder_version": c["encoder_version"]}, limit=20)
                    records = []
                    for hit in hits:
                        if hit["observation_id"] == obs["observation_id"]:
                            continue  # exclude self and same-frame leakage from evidence
                        doc = remote.candidates.find_one({"candidate_id": hit["candidate_id"]},
                            {"_id": 0, "candidate_id": 1, "observation_id": 1, "mode": 1,
                             "encoder_version": 1, "pickability_label": 1})
                        if doc:
                            records.append({**doc, "score": hit["score"]})
                    comparisons[c["candidate_id"]] = records
            finally:
                remote.close()
        else:
            comparisons = json.loads(args.comparisons.read_text())
        atomic_write(output / "comparisons.json", rgbd.canonical_json(comparisons))
        event("comparison_completed", {"source": "atlas" if args.atlas else "saved",
                                       "matches": {cid: len(rows) for cid, rows in comparisons.items()}})
        proposal = propose(obs, candidates, comparisons, geometry)
        atomic_write(output / "proposal.json", rgbd.canonical_json(proposal))
        atomic_write(output / "inputs.json", rgbd.canonical_json({"observation": obs,
                     "candidates": candidates, "geometry": geometry, "policy_version": POLICY_VERSION}))
        event("decision_proposed", {"policy_version": POLICY_VERSION, "decision": proposal["decision"],
                                   "selected_candidate_id": proposal["selected_candidate_id"]})
        print(json.dumps({"run_id": run_id, "output": str(output.resolve()), "decision": proposal["decision"],
                          "selected_candidate_id": proposal["selected_candidate_id"],
                          "candidates": len(candidates), "seconds": round(time.monotonic() - started, 3)}))
        return 0
    except Exception as exc:
        event("replay_failed", {"error_type": type(exc).__name__})
        print(json.dumps({"error_type": type(exc).__name__, "output": str(output.resolve())}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
