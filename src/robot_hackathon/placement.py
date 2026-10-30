"""Two-layer placement target (AGENT_HEURISTICS_PLAN step 7): six explicit slots, alternating orientation.

Slots are data in the robot_base frame. Layer k rests at support_surface_z + k * block_height. Sam's supplied
place_block routine only ever set a block down on the table, so second-layer slots carry
``validation_status = "unvalidated_second_layer"``; nothing here proves physical feasibility.
"""
import math
from dataclasses import dataclass, asdict

from .grasp_geometry import BlockDimensions, FingerGeometry, SupportSurface, wrap_90

TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TowerTarget:
    origin_xy_mm: tuple                      # centre of the tower footprint in robot_base
    support_surface: SupportSurface          # table top height (signed, robot_base)
    block: BlockDimensions
    finger: FingerGeometry
    layer0_long_side_deg: float = 0.0        # world direction of the long side for layer 0; layer 1 is +90
    layers: int = 2
    blocks_per_layer: int = 3
    gap_mm: float = 0.0                      # intended gap between neighbours in a layer (0 = touching)
    release_drop_mm: float = 1.0             # let go this far above where the block rests (Sam's PLACE_DROP_MM)
    source: str = "unspecified"
    verified: bool = False

    def effective_settings(self):
        d = asdict(self)
        d["origin_xy_mm"] = list(self.origin_xy_mm)
        return d

    def slots(self, filled=()):
        """Six slots. `neighbour_release_ok` is evaluated against the slots already filled: opening a claw beside an
        existing neighbour needs claw_thickness of room (Sam's PLACE_CLEARANCE idea); the first block of a layer, or a
        slot whose neighbours are empty, needs none."""
        if self.support_surface.z_in_robot_base_mm is None:
            raise ValueError("Tower target needs a support-surface offset")
        filled = set(filled)
        slots = []
        L, W, H = self.block.length_mm, self.block.width_mm, self.block.height_mm
        pitch = W + self.gap_mm
        for layer in range(self.layers):
            long_deg = wrap_90(self.layer0_long_side_deg + 90.0 * layer)
            across = math.radians(long_deg + 90.0)
            support_z = self.support_surface.z_in_robot_base_mm + layer * H
            below = [f"L{layer - 1}S{i}" for i in range(self.blocks_per_layer)] if layer else []
            for i in range(self.blocks_per_layer):
                offset = (i - (self.blocks_per_layer - 1) / 2) * pitch
                cx = self.origin_xy_mm[0] + offset * math.cos(across)
                cy = self.origin_xy_mm[1] + offset * math.sin(across)
                neighbours = [f"L{layer}S{j}" for j in (i - 1, i + 1) if 0 <= j < self.blocks_per_layer]
                filled_neighbours = [n for n in neighbours if n in filled]
                neighbour_room = self.gap_mm - self.finger.claw_thickness_mm
                slots.append({
                    "slot_id": f"L{layer}S{i}", "layer": layer, "index": i,
                    "frame": "robot_base", "units": {"length": "mm", "angle": "deg"},
                    "block_center_mm": [cx, cy, support_z + H / 2], "long_side_deg": long_deg,
                    "support_z_mm": support_z, "release_z_mm": support_z + H / 2 + self.release_drop_mm,
                    "block_dimensions_mm": [L, W, H],
                    "requires_filled": below, "neighbours": neighbours, "filled_neighbours": filled_neighbours,
                    "neighbour_release_room_mm": neighbour_room,
                    "neighbour_release_ok": not filled_neighbours or neighbour_room >= 0,
                    # Claws on the block's short ends never enter the gap, so a touching neighbour only rules out
                    # grasps that close across the block (middle/side).
                    "requires_ends_grasp": bool(filled_neighbours) and neighbour_room < 0,
                    "validation_status": "offline_geometry_only" if layer == 0 else "unvalidated_second_layer",
                })
        return slots

    def to_dict(self, filled=()):
        return {"schema_version": TARGET_SCHEMA_VERSION, "settings": self.effective_settings(), "slots": self.slots(filled),
                "notes": ["Sam's place_block only placed on the table; second-layer placement is unvalidated",
                          "Held-block offset must be measured after the grasp closes; slots give the block pose, not the gripper pose",
                          "With gap 0 a block beside an already placed neighbour has no claw room; Sam placed rows spaced out and pushed "
                          "them together afterwards, which is outside this first workflow"]}


def next_open_slot(target, filled_slot_ids):
    """First slot whose supporting layer is complete and that is not yet filled. None when the tower is complete.
    Fills a layer from one end so at most one filled neighbour is ever adjacent (Sam's row order)."""
    filled = set(filled_slot_ids)
    for slot in target.slots(filled):
        if slot["slot_id"] in filled:
            continue
        if all(req in filled for req in slot["requires_filled"]):
            return slot
    return None


def footprint_clearance_mm(target):
    """Radius nothing else should occupy around the tower centre while placing (open claws reach past the row)."""
    row = target.blocks_per_layer * target.block.width_mm + (target.blocks_per_layer - 1) * target.gap_mm
    return max(row, target.block.length_mm) / 2 + target.finger.open_gap_mm / 2 + target.finger.claw_thickness_mm + target.finger.margin_mm
