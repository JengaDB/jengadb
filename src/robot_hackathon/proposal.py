"""Planning context, pick-and-place proposals, selection validator and the deterministic coordinator
(AGENT_HEURISTICS_PLAN steps 2, 6, 8 and 9).

A context ties one saved observation to SAM (or synthetic) masks, current geometry, retrieval evidence and the
open placement slot. A proposal is data with evidence IDs; it is never an executable command, carries
``executable: false`` and has no expiry policy yet (scene refresh is deferred by the plan). Retrieval and
geometry scores are kept separate so a reviewer can compare retrieval-assisted selection with geometry alone.
"""
import json
from pathlib import Path

from . import rgbd, retrieval
from . import grasp_geometry as geom
from .placement import TowerTarget, next_open_slot

CONTEXT_SCHEMA_VERSION = 1
PROPOSAL_SCHEMA_VERSION = 1
GRASP_PREFERENCE = ("middle", "side", "ends")
EXPIRY = {"policy": "undefined", "note": "Scene refresh and proposal expiry are deferred (plan step 10); a saved-scene "
                                         "proposal is not valid indefinitely for physical execution"}


class SelectionError(ValueError):
    pass


# ---------------------------------------------------------------- scenes

def synthetic_scene(data_dir, observation_id):
    """Scene from a saved synthetic observation (mock store) with ground-truth masks and synthetic calibration."""
    from . import synthetic_scene as synth
    from .pipeline import candidate_id
    store = rgbd.ObservationStore(data_dir, "mock")
    obs, rgb, depth = store.load_arrays(observation_id)
    doc, masks = synth.load_masks(data_dir, observation_id)
    geometry_doc = geom.GeometryCalibrationStore(data_dir, "mock").load(obs["setup_id"], synth.GEOMETRY_VERSION)
    for m in masks:
        m["candidate_id"] = candidate_id(observation_id, synth.PROVIDER, m["mask_index"])
        m["embedding"] = None
        m["candidate"] = None
    return {"source": "synthetic", "observation": obs, "rgb": rgb, "depth": depth, "masks": masks,
            "calibration": geom.bundle_from_records(obs, True, geometry_doc),
            "geometry_calibration_sha256": geom.GeometryCalibrationStore(data_dir, "mock").content_sha256(obs["setup_id"], synth.GEOMETRY_VERSION),
            "segmentation": {"provider": doc["provider"], "model_version": doc["model_version"], "segmentation_key": synth.PROVIDER},
            "processing_version": synth.PROVIDER}


def observation_scene(data_dir, mode, observation_id, processing_version=None, geometry_version=None):
    """Scene from a processed observation: candidates, masks and vectors from the SKY-66 pipeline outputs."""
    from .candidate_store import CandidateLocalStore, Review
    from .pipeline import load_candidate_file, load_embedding
    store = rgbd.ObservationStore(data_dir, mode)
    obs, rgb, depth = store.load_arrays(observation_id)
    rows = [c for c in CandidateLocalStore(data_dir, mode).read() if c["observation_id"] == observation_id]
    if not rows:
        raise FileNotFoundError(f"No processed candidates for {observation_id}; run rgbd-process first")
    versions = sorted({c["processing_version"] for c in rows}, key=lambda v: max(c["created_at"] for c in rows if c["processing_version"] == v))
    processing_version = processing_version or versions[-1]
    rows = [c for c in rows if c["processing_version"] == processing_version]
    if not rows:
        raise ValueError(f"No candidates for processing version {processing_version}")
    ignored = Review(data_dir, mode).ignored(observation_id)
    masks = []
    for c in sorted(rows, key=lambda c: c["mask_index"]):
        if c["candidate_id"] in ignored:
            continue
        array, kind = rgbd.decode_png(load_candidate_file(data_dir, c, "mask_png"))
        masks.append({"mask_index": c["mask_index"], "mask": array if kind == "mask" else array > 0, "bbox": c["bbox"],
                      "area": c["area_px"], "score": c["segmentation_score"], "candidate_id": c["candidate_id"],
                      "embedding": load_embedding(data_dir, c), "candidate": c})
    geometry_store = geom.GeometryCalibrationStore(data_dir, mode)
    geometry_version = geometry_version or geometry_store.current(obs["setup_id"])
    geometry_doc = geometry_store.load(obs["setup_id"], geometry_version) if geometry_version else None
    registration_verified = bool(rows[0].get("registration_verified", False))
    return {"source": "observation", "observation": obs, "rgb": rgb, "depth": depth, "masks": masks,
            "calibration": geom.bundle_from_records(obs, registration_verified, geometry_doc),
            "geometry_calibration_sha256": geometry_store.content_sha256(obs["setup_id"], geometry_version) if geometry_version else None,
            "segmentation": rows[0]["segmentation"], "processing_version": processing_version}


# ---------------------------------------------------------------- context

def _query_record(scene, m):
    cand = m.get("candidate") or {}
    return {"candidate_id": m["candidate_id"], "observation_id": scene["observation"]["observation_id"],
            "encoder_version": cand.get("encoder_version"), "embedding_dimension": cand.get("embedding_dimension"),
            "mode": scene["observation"]["mode"]}


def build_context(scene, policy=None, reference_search=None, tower_origin_xy_mm=None, filled_slots=(), tower_long_side_deg=0.0,
                  dry_run=False):
    """Deterministic planning context. Geometry and calibration are computed by code, never by the agent.

    dry_run=True keeps unverified-calibration reasons as flags instead of rejections so the geometry can be inspected;
    the decision then becomes a `geometry_preview`, never a proposal. Missing data is still missing."""
    import numpy as np
    policy = policy or retrieval.ReferencePolicy()
    obs, depth = scene["observation"], scene["depth"]
    calib = scene["calibration"]
    invalid = obs["depth"]["invalid_value"]
    # Known detections are kept out of the plane fit so block tops cannot pull the table towards the camera.
    exclude = np.zeros(depth.shape, dtype=bool)
    for m in scene["masks"]:
        exclude |= m["mask"]
    plane = geom.fit_table_plane(depth, invalid_value=invalid, exclude=geom.dilate(exclude, 8, step=4) if exclude.any() else None)
    table = None
    if plane is None:
        height_map = np.full(depth.shape, np.nan)
        table = {"status": "unavailable", "reason": "insufficient_valid_depth_for_plane_fit"}
    else:
        height_map = plane.height_map(depth, invalid)
        table = {"status": "fitted" if plane.reliable else "fitted_unreliable", **plane.to_dict()}
        if calib.intrinsics is not None and calib.extrinsics is not None:
            table["in_robot_base"] = geom.table_plane_in_base(plane, calib.intrinsics, calib.extrinsics, depth.shape[1], depth.shape[0])
            ss = calib.support_surface
            if ss.z_in_robot_base_mm is not None:
                table["support_surface_offset_mm"] = ss.z_in_robot_base_mm
                table["offset_minus_fitted_mean_mm"] = ss.z_in_robot_base_mm - table["in_robot_base"]["mean_z_mm"]
    candidates = []
    for m in scene["masks"]:
        est = geom.estimate_block(m["mask"], depth, height_map, calib, m["mask_index"], m["candidate_id"], invalid_value=invalid)
        if plane is None:
            est.rejection_reasons.append("table_plane_unavailable")
        elif not plane.reliable:
            est.rejection_reasons.append(f"table_plane_unreliable:inliers_{plane.inlier_fraction:.2f}")
        if dry_run:
            # Only *unverified* calibration is waived for the preview; missing intrinsics/extrinsics/offset still reject.
            waived = [r for r in est.rejection_reasons if r.startswith("calibration:") and r.endswith("_not_verified")]
            est.rejection_reasons = [r for r in est.rejection_reasons if r not in waived]
            est.flags.extend(f"preview_waived:{r}" for r in waived)
        grasps = geom.evaluate_grasps(est, height_map, calib)
        query = _query_record(scene, m)
        if m.get("embedding") is None:
            evidence = retrieval.unavailable_evidence(query, "no_query_embedding", policy, reference_search.identity() if reference_search else None)
        else:
            evidence = retrieval.retrieve(query, m["embedding"], reference_search, policy)
        reasons = list(est.rejection_reasons)
        if not reasons and not any(g["eligible"] for g in grasps):
            reasons.append("no_eligible_grasp")
        cand = m.get("candidate") or {}
        candidates.append({"candidate_id": m["candidate_id"], "mask_index": m["mask_index"], "bbox": m.get("bbox"),
                           "segmentation_score": m.get("score"), "exposure": cand.get("exposure"),
                           "geometry": est.to_dict(), "geometry_quality": geom.geometry_quality(est, calib.block),
                           "clearance_fraction": round(sum(g["eligible"] for g in grasps) / len(grasps), 4) if grasps else 0.0,
                           "grasps": grasps, "retrieval": evidence, "eligible": not reasons, "rejection_reasons": reasons})
    eligible = [c for c in candidates if c["eligible"]]
    # Retrieval evidence and geometric quality are both recorded; the combined ranking prefers curated matches, then
    # blocks with more clear grasp options, then dimensional/depth agreement. Geometry-only is the baseline.
    geometry_key = lambda c: (-c["clearance_fraction"], -c["geometry_quality"], c["candidate_id"])  # noqa: E731
    ranking = sorted(eligible, key=lambda c: (c["retrieval"]["status"] != "matched", -(c["retrieval"]["best_cosine_similarity"] or -2)) + geometry_key(c))
    geometry_only = sorted(eligible, key=geometry_key)
    placement = {"status": "unavailable", "reason": "tower_origin_not_configured", "next_slot": None}
    if tower_origin_xy_mm is not None and calib.support_surface.z_in_robot_base_mm is not None:
        target = TowerTarget(tuple(tower_origin_xy_mm), calib.support_surface, calib.block, calib.finger, tower_long_side_deg,
                             source="context_parameter", verified=False)
        slot = next_open_slot(target, filled_slots)
        placement = {"status": "available" if slot else "complete", "target": target.to_dict(filled_slots), "filled": sorted(filled_slots),
                     "next_slot": slot, "reason": None if slot else "all_slots_filled"}
        if slot is not None and not slot["neighbour_release_ok"]:
            placement.update(status="blocked", reason=f"slot_release_room_insufficient:{slot['slot_id']}:{slot['neighbour_release_room_mm']:.0f}mm")
    elif tower_origin_xy_mm is not None:
        placement = {"status": "unavailable", "reason": "support_surface_offset_missing", "next_slot": None}
    body = {"schema_version": CONTEXT_SCHEMA_VERSION, "source": scene["source"], "mode": obs["mode"], "dry_run": bool(dry_run),
            "episode_id": obs["episode_id"], "observation_id": obs["observation_id"], "setup_id": obs["setup_id"],
            "captured_at": obs["captured_at"], "processing_version": scene["processing_version"],
            "segmentation": scene["segmentation"], "calibration": calib.to_dict(),
            "geometry_calibration_sha256": scene.get("geometry_calibration_sha256"),
            "geometry_code_version": geom.GEOMETRY_CODE_VERSION, "retrieval_policy": policy.effective_settings(),
            "reference_index": reference_search.identity() if reference_search else {"source": "none"},
            "table_plane": table, "candidates": candidates,
            "eligible_candidate_ids": [c["candidate_id"] for c in eligible],
            "ranking": [{"candidate_id": c["candidate_id"], "retrieval_status": c["retrieval"]["status"],
                         "best_cosine_similarity": c["retrieval"]["best_cosine_similarity"], "geometry_quality": c["geometry_quality"],
                         "clearance_fraction": c["clearance_fraction"]} for c in ranking],
            "geometry_only_ranking": [c["candidate_id"] for c in geometry_only], "placement": placement}
    body["context_id"] = "ctx_" + rgbd.sha256_bytes(rgbd.canonical_json(body).encode())[:20]
    body["created_at"] = rgbd.utc_now()
    return body


def candidate_entry(context, candidate_id):
    for c in context["candidates"]:
        if c["candidate_id"] == candidate_id:
            return c
    raise SelectionError(f"unknown_candidate:{candidate_id}")


def validate_selection(context, candidate_id, grasp_style, shift_mm, slot_id):
    """Application-side check that a selection names an eligible candidate, an eligible grasp and the open slot."""
    entry = candidate_entry(context, candidate_id)
    if not entry["eligible"]:
        raise SelectionError(f"candidate_not_eligible:{candidate_id}:{','.join(entry['rejection_reasons'])}")
    grasp = next((g for g in entry["grasps"] if g["style"] == grasp_style and abs(g["shift_mm_along_long_side"] - float(shift_mm)) < 1e-6), None)
    if grasp is None:
        raise SelectionError(f"unknown_grasp:{grasp_style}@{shift_mm}")
    if not grasp["eligible"]:
        raise SelectionError(f"grasp_not_eligible:{grasp_style}@{shift_mm}:{','.join(grasp['reasons'])}")
    slot = context["placement"].get("next_slot")
    if slot is None or slot["slot_id"] != slot_id:
        raise SelectionError(f"slot_not_open:{slot_id}")
    if context["placement"].get("status") != "available":
        raise SelectionError(f"slot_blocked:{slot_id}:{context['placement'].get('reason')}")
    return entry, grasp, slot


def make_proposal(context, candidate_id, grasp_style, shift_mm, slot_id, rationale, decided_by):
    entry, grasp, slot = validate_selection(context, candidate_id, grasp_style, shift_mm, slot_id)
    g = entry["geometry"]
    ev = entry["retrieval"]
    preview = bool(context.get("dry_run"))
    body = {"schema_version": PROPOSAL_SCHEMA_VERSION, "kind": "geometry_preview" if preview else "pick_and_place_proposal",
            "context_id": context["context_id"],
            "episode_id": context["episode_id"], "observation_id": context["observation_id"], "mode": context["mode"],
            "candidate_id": candidate_id,
            "grasp": {k: grasp[k] for k in ("style", "claws_deg_to_long_side", "shift_mm_along_long_side", "tips_above_table_mm",
                                            "approach", "retreat", "lift_mm")} | {"claws_close_direction_deg_in_base": grasp.get("claws_close_direction_deg_in_base")},
            "source_pose": {"frame": "robot_base", "units": {"length": "mm", "angle": "deg"},
                            "block_top_center_mm": list(g["base_point_mm"]) if g["base_point_mm"] else None,
                            "long_side_deg": g["base_long_side_deg"], "height_above_table_mm": g["height_above_table_mm"],
                            "top_z_from_support_offset_mm": g["top_z_in_robot_base_mm"], "pixel_center": list(g["pixel_center"])},
            "destination": slot,
            "evidence": {"retrieval_status": ev["status"], "best_cosine_similarity": ev["best_cosine_similarity"],
                         "reference_candidate_ids": [m["reference_candidate_id"] for m in ev["matches"]],
                         "reference_index": ev["index"], "cosine_threshold": ev["policy"]["cosine_threshold"],
                         "geometry_quality": entry["geometry_quality"], "flags": g["flags"], "clearance": grasp["clearance"],
                         "calibration_version": context["calibration"]["calibration_version"],
                         "geometry_calibration_sha256": context["geometry_calibration_sha256"]},
            "rationale": rationale, "decided_by": decided_by,
            "validation_status": "unverified_calibration_preview" if preview else "validated_offline",
            "calibration_missing": context["calibration"]["missing"],
            "executable": False, "expiry": EXPIRY}
    body["proposal_id"] = ("preview_" if preview else "prop_") + rgbd.sha256_bytes(rgbd.canonical_json(body).encode())[:20]
    body["created_at"] = rgbd.utc_now()
    return body


def abstention(context, reason, decided_by, details=None):
    return {"schema_version": PROPOSAL_SCHEMA_VERSION, "kind": "abstention", "context_id": context["context_id"],
            "episode_id": context["episode_id"], "observation_id": context["observation_id"], "mode": context["mode"],
            "reason": reason, "details": details or {}, "decided_by": decided_by, "executable": False, "created_at": rgbd.utc_now()}


class DeterministicCoordinator:
    """Offline replacement for the model: top-ranked eligible candidate, preferred eligible grasp, open slot."""
    name = "deterministic-v1"

    def decide(self, context):
        if not context["ranking"]:
            return abstention(context, "no_eligible_candidate", self.name,
                              {"rejections": {c["candidate_id"]: c["rejection_reasons"] for c in context["candidates"]},
                               "calibration_missing": context["calibration"]["missing"], "table_plane": context["table_plane"].get("status"),
                               "placement": context["placement"].get("status")})
        if context["placement"].get("next_slot") is None or context["placement"].get("status") != "available":
            return abstention(context, context["placement"].get("reason") or "no_open_slot", self.name)
        top = candidate_entry(context, context["ranking"][0]["candidate_id"])
        grasp = next(g for style in GRASP_PREFERENCE for g in top["grasps"] if g["style"] == style and g["eligible"])
        ev = top["retrieval"]
        why = (f"best curated-reference match cosine {ev['best_cosine_similarity']:.3f}" if ev["status"] == "matched"
               else f"retrieval {ev['status']}; ranked by clear-grasp fraction {top['clearance_fraction']} and geometry quality {top['geometry_quality']}")
        return make_proposal(context, top["candidate_id"], grasp["style"], grasp["shift_mm_along_long_side"],
                             context["placement"]["next_slot"]["slot_id"],
                             f"{why}; {grasp['style']} grasp clear of obstructions", self.name)


def save_context(context, decision, directory):
    """Persist inputs for replay without network access. Returns the written paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ctx_path = directory / f"{context['context_id']}.json"
    dec_path = directory / f"{context['context_id']}-decision.json"
    ctx_path.write_text(json.dumps(context, indent=1, sort_keys=True))
    dec_path.write_text(json.dumps(decision, indent=1, sort_keys=True))
    return ctx_path, dec_path
