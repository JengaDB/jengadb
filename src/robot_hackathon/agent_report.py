"""Replayable review report for one planning context (AGENT_HEURISTICS_PLAN step 9).

Shows the source scene with mask overlays, depth and table-relative height, every candidate's crops, geometry,
grasp checks, curated-reference matches with both score types, and the proposal or abstention. The JSON
sidecars written next to it hold the full context and decision, so the page can be regenerated offline.
"""
import base64
import io
import json
import math
from html import escape
from pathlib import Path

from .local_store import atomic_write

PALETTE = [(240, 90, 70), (70, 170, 240), (90, 210, 120), (240, 200, 60), (200, 110, 240), (60, 220, 220), (250, 150, 50)]


def _png(array):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _img(png, cls=""):
    return f'<img class="{cls}" src="data:image/png;base64,{base64.b64encode(png).decode()}">' if png else ""


def _footprint(g):
    """Estimated block rectangle in pixels from centre, image angle and side lengths (Sam's save_target_picture)."""
    if not g["pixel_center"] or g["length_px"] is None:
        return None
    (u, v), a = g["pixel_center"], math.radians(g["image_angle_deg"])
    hl, hs = g["length_px"] / 2, g["width_px"] / 2
    return [(u + sl * hl * math.cos(a) - ss * hs * math.sin(a), v + sl * hl * math.sin(a) + ss * hs * math.cos(a))
            for sl, ss in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def scene_overlay(scene, context, decision):
    import numpy as np
    from PIL import Image, ImageDraw
    rgb = scene["rgb"].copy()
    for m in scene["masks"]:
        colour = np.array(PALETTE[m["mask_index"] % len(PALETTE)])
        rgb[m["mask"]] = (0.55 * rgb[m["mask"]] + 0.45 * colour).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    chosen = decision.get("candidate_id") if decision.get("kind") in ("pick_and_place_proposal", "geometry_preview") else None
    for c in context["candidates"]:
        colour = PALETTE[c["mask_index"] % len(PALETTE)]
        poly = _footprint(c["geometry"])
        if poly:
            draw.polygon(poly, outline=(255, 0, 0) if c["candidate_id"] == chosen else colour, width=3 if c["candidate_id"] == chosen else 1)
        if c["bbox"]:
            x0, y0, x1, y1 = c["bbox"]
            draw.rectangle([x0, y0, x1, y1], outline=colour, width=1)
            draw.text((x0 + 2, max(0, y0 - 11)), f"#{c['mask_index']} {'OK' if c['eligible'] else 'x'}", fill=colour)
    return _png(np.asarray(image))


def height_image(scene, context):
    import numpy as np
    from . import grasp_geometry as geom
    depth = scene["depth"]
    plane = context["table_plane"]
    if not str(plane.get("status", "")).startswith("fitted"):
        return None
    hm = geom.TablePlane(plane["c0"], plane["c1"], plane["c2"], 0, 0, 0, ()).height_map(depth, scene["observation"]["depth"]["invalid_value"])
    t = np.clip(np.nan_to_num(hm, nan=-1.0) / 45.0, 0, 1)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    out[..., 0] = (255 * t).astype(np.uint8)
    out[..., 1] = (255 * (1 - np.abs(2 * t - 1))).astype(np.uint8)
    out[..., 2] = (120 * (1 - t)).astype(np.uint8)
    out[np.isnan(hm)] = 0
    return _png(out)


def crops(scene, entry, margin=24):
    import numpy as np
    m = next((x for x in scene["masks"] if x["mask_index"] == entry["mask_index"]), None)
    if m is None or not m.get("bbox"):
        return None, None
    x0, y0, x1, y1 = m["bbox"]
    h, w = scene["rgb"].shape[:2]
    masked = np.zeros_like(scene["rgb"])
    masked[m["mask"]] = scene["rgb"][m["mask"]]
    cx0, cy0, cx1, cy1 = max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)
    return _png(masked[y0:y1, x0:x1]), _png(scene["rgb"][cy0:cy1, cx0:cx1])


def _fmt(v, nd=1):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else escape(str(v))


def candidate_card(scene, entry, chosen):
    g, ev = entry["geometry"], entry["retrieval"]
    cut, ctx = crops(scene, entry)
    grasps = "".join(f"<tr><td>{escape(x['style'])}</td><td>{_fmt(x['shift_mm_along_long_side'])}</td><td>{_fmt(x['tips_above_table_mm'])}</td>"
                     f"<td>{escape((x['clearance'] or {}).get('status', '—'))}</td><td>{'yes' if x['eligible'] else escape(', '.join(x['reasons']))}</td></tr>"
                     for x in entry["grasps"])
    matches = "".join(f"<tr><td>{escape(m['reference_candidate_id'])}</td><td>{escape(str(m['reference_observation_id']))}</td>"
                      f"<td>{_fmt(m['atlas_vector_search_score'], 4)}</td><td>{_fmt(m['cosine_similarity'], 4)}</td>"
                      f"<td>{escape(str((m['curation'].get('pickability_label') or {}).get('label_source')))}</td></tr>" for m in ev["matches"])
    excluded = {}
    for e in ev.get("excluded", []):
        for r in e["reasons"]:
            key = r.split(":")[0]
            excluded[key] = excluded.get(key, 0) + 1
    state = "chosen" if entry["candidate_id"] == chosen else ("eligible" if entry["eligible"] else "rejected")
    return f"""<div class="card {state}"><div class="imgs">{_img(ctx, 'ctx')}{_img(cut, 'cut')}</div>
<h3>#{entry['mask_index']} <code>{escape(entry['candidate_id'])}</code> <span class="state">{state}</span></h3>
<p>{escape(', '.join(entry['rejection_reasons'])) or 'eligible'}{' · flags: ' + escape(', '.join(g['flags'])) if g['flags'] else ''}</p>
<table><tr><th>height above table</th><td>{_fmt(g['height_above_table_mm'])} mm</td><th>length × width</th><td>{_fmt(g['length_mm'])} × {_fmt(g['width_mm'])} mm</td></tr>
<tr><th>valid depth</th><td>{_fmt(g['valid_fraction'], 2)}</td><th>geometry quality</th><td>{_fmt(entry['geometry_quality'], 3)}</td></tr>
<tr><th>top centre (robot_base)</th><td colspan="3">{escape(str([round(x, 1) for x in g['base_point_mm']]) if g['base_point_mm'] else '— (extrinsics missing)')} mm, long side {_fmt(g['base_long_side_deg'])}°</td></tr></table>
<h4>Grasps</h4><table class="grid"><tr><th>style</th><th>shift mm</th><th>tips above table mm</th><th>clearance</th><th>eligible</th></tr>{grasps}</table>
<h4>Curated references — {escape(ev['status'])}{(' (' + escape(str(ev.get('unavailable_reason'))) + ')') if ev['status'] == 'unavailable' else ''}</h4>
<p class="muted">threshold cosine ≥ {ev['policy']['cosine_threshold']} (Atlas score ≥ {ev['policy']['atlas_score_threshold']:.3f}); {ev['hits_considered']} hits considered; excluded: {escape(json.dumps(excluded)) if excluded else 'none'}</p>
{('<table class="grid"><tr><th>reference</th><th>observation</th><th>Atlas score</th><th>cosine</th><th>label source</th></tr>' + matches + '</table>') if matches else ''}
</div>"""


def decision_block(decision):
    if decision.get("kind") in ("pick_and_place_proposal", "geometry_preview"):
        sp, dst = decision["source_pose"], decision["destination"]
        preview = decision["kind"] == "geometry_preview"
        title = "GEOMETRY PREVIEW — calibration unverified, not a proposal" if preview else "Proposal"
        return f"""<div class="decision proposal"><h2>{title} <code>{escape(decision['proposal_id'])}</code></h2>
<p><strong>Not executable.</strong> validation {escape(decision['validation_status'])}; decided by {escape(decision['decided_by'])}; expiry policy {escape(decision['expiry']['policy'])}.
{('Waived for preview: ' + escape(', '.join(decision.get('calibration_missing', [])))) if preview else ''}</p>
<table><tr><th>pick</th><td>candidate <code>{escape(decision['candidate_id'])}</code>, {escape(decision['grasp']['style'])} grasp, shift {_fmt(decision['grasp']['shift_mm_along_long_side'])} mm, tips {_fmt(decision['grasp']['tips_above_table_mm'])} mm above table, {escape(decision['grasp']['approach'])} then {escape(decision['grasp']['retreat'])} {_fmt(decision['grasp']['lift_mm'])} mm</td></tr>
<tr><th>source pose</th><td>robot_base top centre {escape(str([round(x, 1) for x in sp['block_top_center_mm']]) if sp['block_top_center_mm'] else '—')} mm, long side {_fmt(sp['long_side_deg'])}°, height above table {_fmt(sp['height_above_table_mm'])} mm</td></tr>
<tr><th>destination</th><td>slot {escape(dst['slot_id'])} layer {dst['layer']}: block centre {escape(str([round(x, 1) for x in dst['block_center_mm']]))} mm, long side {_fmt(dst['long_side_deg'])}°, release z {_fmt(dst['release_z_mm'])} mm · {escape(dst['validation_status'])}</td></tr>
<tr><th>evidence</th><td>retrieval {escape(decision['evidence']['retrieval_status'])}, best cosine {_fmt(decision['evidence']['best_cosine_similarity'], 4)}, references {escape(', '.join(decision['evidence']['reference_candidate_ids'])) or 'none'}; geometry quality {_fmt(decision['evidence']['geometry_quality'], 3)}</td></tr>
<tr><th>rationale</th><td>{escape(decision['rationale'])}</td></tr></table></div>"""
    return f"""<div class="decision abstention"><h2>Abstention</h2><p>reason <strong>{escape(decision.get('reason', ''))}</strong>; decided by {escape(decision.get('decided_by', ''))}</p>
<pre>{escape(json.dumps(decision.get('details', {}), indent=1, default=str))}</pre></div>"""


def render_report(scene, context, decision, out_path):
    mode = context["mode"]
    label = "LIVE OBSERVATION" if mode == "live" else ("SYNTHETIC SCENE" if context["source"] == "synthetic" else "MOCK DATA")
    cal = context["calibration"]
    table = context["table_plane"]
    chosen = decision.get("candidate_id") if decision.get("kind") in ("pick_and_place_proposal", "geometry_preview") else None
    cards = "".join(candidate_card(scene, c, chosen) for c in context["candidates"])
    table_html = (f"<strong>{escape(table['status'])}</strong>: inliers {table['inlier_fraction']:.2f} (min {table['min_inlier_fraction']}), rms {table['rms_mm']:.2f} mm, "
                  f"final tolerance {table['tolerances_mm'][-1]:.1f} mm at median depth {table['median_depth_mm']:.0f} mm, detections excluded {table['excluded_fraction']:.2f}"
                  + (f"; in robot_base mean z {table['in_robot_base']['mean_z_mm']:.1f} mm, spread {table['in_robot_base']['z_spread_mm']:.1f} mm, tilt {table['in_robot_base']['normal_tilt_deg']:.2f}°" if table.get("in_robot_base") else "")
                  + (f"; configured support offset {table['support_surface_offset_mm']} mm (offset − fitted = {table['offset_minus_fitted_mean_mm']:.1f} mm)" if "support_surface_offset_mm" in table else "")
                  ) if str(table.get("status", "")).startswith("fitted") else escape(str(table))
    slots = context["placement"].get("target", {}).get("slots", []) if context["placement"].get("target") else []
    slot_rows = "".join(f"<tr><td>{s['slot_id']}</td><td>{s['layer']}</td><td>{[round(x, 1) for x in s['block_center_mm']]}</td><td>{s['long_side_deg']:.0f}°</td><td>{s['validation_status']}</td></tr>" for s in slots)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Proposal review — {escape(context['observation_id'])}</title><style>
body{{font:15px system-ui;background:#101b25;color:#eaf3f5;margin:0;padding:3vw}}h1,h2,h3,h4{{margin:.4em 0}}code{{font-size:.85em;color:#b8c9d3}}
.tag{{color:#75e2c3;font-weight:700;letter-spacing:.1em}}.muted{{color:#b8c9d3}}img{{background:#000;image-rendering:pixelated;max-width:100%}}
.scene img{{max-width:32%;margin-right:1%}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}}
.card{{background:#1c2d3b;border-radius:12px;padding:14px;border-left:6px solid #888}}.card.eligible{{border-color:#75e2c3}}.card.rejected{{border-color:#f28b82}}.card.chosen{{border-color:#ffd166;box-shadow:0 0 0 2px #ffd166}}
.imgs img{{max-width:46%;margin-right:2%}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:3px 6px;border-bottom:1px solid #2c4256;vertical-align:top}}th{{color:#b8c9d3;font-weight:500}}
.decision{{background:#1c2d3b;border-radius:12px;padding:14px;margin:16px 0;border-left:6px solid #ffd166}}.decision.abstention{{border-color:#f28b82}}pre{{white-space:pre-wrap;color:#b8c9d3}}.state{{font-size:.7em;color:#b8c9d3}}</style>
<p class="tag">{label}</p><h1>Pick-and-place proposal review</h1>
<p class="muted">observation <code>{escape(context['observation_id'])}</code> · episode <code>{escape(context['episode_id'])}</code> · context <code>{escape(context['context_id'])}</code> · processing <code>{escape(str(context['processing_version']))}</code> · captured {escape(str(context['captured_at']))}</p>
<div class="scene">{_img(scene_overlay(scene, context, decision))}{_img(height_image(scene, context))}</div>
<p class="muted">Left: RGB with mask tints, estimated footprints and the chosen block in red. Right: height above the fitted table (0–45 mm), invalid depth black.</p>
{decision_block(decision)}
<h2>Calibration and table</h2>
<p>calibration version <code>{escape(str(cal['calibration_version']))}</code>, geometry source <code>{escape(str(cal['source']))}</code>, registration verified {cal['registration_verified']}.
Missing/unverified: <strong>{escape(', '.join(cal['missing'])) or 'none'}</strong>.</p>
<p>support surface z in robot_base: {escape(str(cal['support_surface']['z_in_robot_base_mm']))} mm ({'verified' if cal['support_surface']['verified'] else 'unverified'}, {escape(str(cal['support_surface']['source']))}); finger geometry {'verified' if cal['finger']['verified'] else 'unverified'}; block dimensions {'verified' if cal['block']['verified'] else 'unverified'}.</p>
<p>table plane: {table_html}</p>
<h2>Placement target — {escape(context['placement']['status'])}{(' (' + escape(str(context['placement'].get('reason'))) + ')') if context['placement'].get('reason') else ''}</h2>
{('<table class="grid"><tr><th>slot</th><th>layer</th><th>block centre mm</th><th>long side</th><th>status</th></tr>' + slot_rows + '</table>') if slot_rows else ''}
<h2>Ranking</h2><p class="muted">retrieval-assisted: {escape(', '.join(r['candidate_id'] for r in context['ranking'])) or 'no eligible candidates'}<br>geometry only: {escape(', '.join(context['geometry_only_ranking'])) or 'no eligible candidates'}</p>
<h2>{len(context['candidates'])} candidates</h2><div class="grid">{cards}</div>
<p class="muted">Retrieval similarity measures resemblance to curated references, not grasp success. Geometry checks do not establish motion-path safety. No robot movement is implied by this page.</p>
</html>"""
    atomic_write(out_path, page)
    return Path(out_path)
