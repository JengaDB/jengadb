"""Exposure (visibility) heuristic. Produces exposed / occluded / unknown; never pickability.

Gate: alignment verified AND calibration version matches the registry's current version for the setup
AND depth intrinsics present. Without the gate every mask is unknown with explicit reasons.
Footprint reasoning stays disabled unless block dimensions are verified and the setup records a top-down
viewing pose; this slice ships it disabled.
"""
from dataclasses import dataclass, asdict

PICKABILITY_UNKNOWN = {"status": "unknown", "reason": "grasp eligibility not evaluated (SKY-87)"}


@dataclass(frozen=True)
class ExposureParams:
    params_label: str = "exposure-v1"
    min_valid_fraction: float = 0.5
    contact_band_px: int = 2
    closer_margin_mm: float = 10.0
    min_contact_fraction: float = 0.03
    footprint_enabled: bool = False
    block_dimensions_mm: tuple = (75.0, 25.0, 15.0)
    block_dimensions_verified: bool = False

    def effective_settings(self):
        d = asdict(self)
        d["block_dimensions_mm"] = list(self.block_dimensions_mm)
        return d


def gate(obs, registry_current_version, calibration_entry):
    reasons = []
    if obs["alignment"].get("status") != "verified":
        reasons.append("alignment_not_verified")
    if registry_current_version is None or obs["calibration_version"] != registry_current_version:
        reasons.append("calibration_mismatch")
    if calibration_entry is None or calibration_entry.get("alignment", {}).get("status") != "verified":
        reasons.append("calibration_not_verified")
    if not (obs.get("intrinsics") or {}).get("depth"):
        reasons.append("intrinsics_missing")
    if not obs.get("dimensions_equal", False):
        reasons.append("dimension_mismatch")
    return not reasons, reasons


def _band(mask, radius):
    import numpy as np
    grown = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                grown |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return grown & ~mask


def depth_stats(mask, depth, invalid_value=0):
    import numpy as np
    values = depth[mask]
    valid = values[values != invalid_value]
    total = int(values.size)
    return {"valid_fraction": float(valid.size / total) if total else 0.0,
            "median_mm": float(np.median(valid)) if valid.size else None,
            "p10_mm": float(np.percentile(valid, 10)) if valid.size else None,
            "p90_mm": float(np.percentile(valid, 90)) if valid.size else None,
            "pixels": total}


def classify(masks, depth, params, gated_in, gate_reasons, invalid_value=0):
    """Return a list aligned with masks: {exposure, pickability, depth_stats}."""
    results = []
    stats = [depth_stats(m["mask"], depth, invalid_value) for m in masks]
    for i, m in enumerate(masks):
        base = {"params_label": params.params_label, "label_source": "heuristic"}
        if not gated_in:
            results.append({"exposure": {"status": "unknown", "score": None, "reasons": list(gate_reasons), **base},
                            "pickability": dict(PICKABILITY_UNKNOWN), "depth_stats": None})
            continue
        st = stats[i]
        if st["valid_fraction"] < params.min_valid_fraction or st["median_mm"] is None:
            results.append({"exposure": {"status": "unknown", "score": None, "reasons": ["insufficient_valid_depth"], **base},
                            "pickability": dict(PICKABILITY_UNKNOWN), "depth_stats": st})
            continue
        band = _band(m["mask"], params.contact_band_px)
        band_pixels = max(int(band.sum()), 1)
        contact = 0.0
        closer_neighbors = []
        for j, other in enumerate(masks):
            if j == i or stats[j]["median_mm"] is None:
                continue
            overlap = int((band & other["mask"]).sum())
            if overlap == 0:
                continue
            if stats[j]["median_mm"] < st["median_mm"] - params.closer_margin_mm:
                contact += overlap / band_pixels
                closer_neighbors.append(j)
        reasons = []
        if closer_neighbors and contact >= params.min_contact_fraction:
            status = "occluded"
            reasons.append(f"closer_neighbor_contact:{contact:.3f}")
        else:
            status = "exposed"
            reasons.append("no_closer_neighbor_contact")
        if params.footprint_enabled and not params.block_dimensions_verified:
            reasons.append("footprint_check_skipped:block_dimensions_unverified")
        results.append({"exposure": {"status": status, "score": round(contact, 4), "reasons": reasons,
                                     "closer_neighbor_mask_indices": closer_neighbors, **base},
                        "pickability": dict(PICKABILITY_UNKNOWN), "depth_stats": st})
    return results
