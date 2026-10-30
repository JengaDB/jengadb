# STACK-01: Supervised layer stacking (human builds a layer, robot stacks it)

Status and owner: see board.json (single source of truth)
Role: data/integration (Kazy); builds on AGENT-01 (Kris) and Sam's motion conventions
Branch: codex/layer-stack (from agent-heuristics a16a577)

## Goal and acceptance

A human packs three blocks side by side on the table. One `layer-stack` command measures that layer and the top
layer of the tower, grips the three as one bundle across their 75 mm width, rotates it 90 deg to the layer below,
places it centred, and verifies it from a fresh capture. Acceptance: three consecutive supervised runs with the
new layer centre within 3 mm of the base centre and within 3 deg of perpendicular, measured from the verification
capture and checked by eye. Command acceptance or `grab()` true is never the outcome.

## Handoff (30 Oct 2026)

Status: NOT accepted. No layer has been stacked yet. Run 4 lifted the bundle; the run then stopped at the after-lift check.

Changes: `src/robot_hackathon/layer_stack.py` (pure layer validation, scene split, plan, residual, per-setup residual
log with a guarded constant-offset fit; supervised async run). `cli.py` adds `layer-stack` with:
`--plan-only` (capture and plan, never moves; arm must already be at the view pose), `--view x,y,z` (overview flange
pose, default home), `--direct-local` (LAN gRPC dial to viam-server, sessions stay on), `--no-pause` (skip Enter
prompts; every check still stops the run), `--grip-lower-mm` (deepen pick and release tips together),
`--apply-offset`. Safety checks: blocks cut off by the image edge are refused; flat detections under half a block
(paper, tape) are ignored; after the lift a capture checks that no blocks remain at the pick spot; verification
records "no new layer on the base" as a failure and keeps it out of the residual log. Resource APIs are imported
before connecting (the SDK only builds clients for APIs registered before dialing; same as live_executor.run_flow).
`tests/test_layer_stack.py`: 15 offline tests.

Checks: `python3 scripts/team.py check` and, with the project venv, `RGBD_TESTS_REQUIRED=1` -> all tests OK (see the
commit for the count at push time).

Live evidence (29-30 Oct, operator Kazy at the e-stop; episodes and captures in the operator's ignored data dir
local/layer-stack-data/live/episodes/):
- Setup: geometry calibration `layer-stack-unverified-v1`, every value unverified: table z 0 mm from Sam's touch-off
  (flange 171.5 at contact, gripper frame 172 below flange), camera pose taken from each capture, Sam's block and
  claw sizes. Camera registration `robot-camera-unverified-v1`.
- Reach: a layer at about (656, 6) mm was refused by the planner ("zero IK solutions produced"). Layers near
  x 440-570 mm worked.
- View: the two layers ~210 mm apart did not fit one image at flange z 400; z 460 did.
- Run 2: stopped at the source close-up: a flat paper note touching the image edge was counted as a block (fixed).
- Run 3: plan, grip ("holding"), lift, move and release all reported success, but the verification capture showed
  both layers unchanged (within 0.5 mm). The claws closed too high (operator observation). Gripper "holding" did not
  mean lifted.
- Run 4 (`--grip-lower-mm 3`, pick tips 3.0 mm above z 0): the bundle was lifted (after-lift photo shows three blocks
  in the claws). The run stopped at the after-lift check: with the bundle in front of the lens the table plane was
  `fitted_unreliable`. The operator then pressed the e-stop with the bundle held; it was handled by hand.
- Precision hints: overview and close-up placed the same base 5.5 mm apart in one run and 0.5 mm in another;
  depth puts the table at +4.3 to +5.3 mm in world (four captures, tilt < 0.7 deg) while Sam's touch-off implies 0 mm.

- Run on Kris's laptop (30 Oct 16:38-16:39, operator Kris, episode robot-live-4, `--view 300,150,460 --direct-local`):
  three overview captures stopped before any pick. Two saw three loose blocks (x 456-492, y -3..112) and no base layer;
  the third saw the packed base on the blue square and no source layer: the hand-built layer was outside the image.
  From the recorded camera pose (world z 446) that view covers x 199-546, y -128..491 mm at table height, with the
  claws occluding the bottom strip (x below ~280). Change (30 Oct, Kris): `visible_table_extent()` projects the image
  corners onto the table from the capture's camera pose; the "expected 3 ... saw 0" stop now reports that area and the
  tower-zone radius; `--source x,y` lets the operator name where the hand-built layer lies when it is out of frame
  (the run hovers there and measures it close up). Clarified with Kris: the blue square ("1") is the staging spot for
  the hand-built layer and the tower ("2") is elsewhere, so `--tower-origin` must point at "2". When the tower origin
  lies outside the overview image and the zone is empty, the run now hovers over the tower origin and measures the base
  there (`NoBaseLayer`, event `base_out_of_view`); an empty zone that *is* in view still stops. 3 offline tests added.
  Default overview pose moved from live_executor.HOME (z 400) to `OVERVIEW_POSE` (300,150,460, the proven view); the
  overview prints and records the table area it covered (`table_seen_mm`). Computed coverage for wider views (camera
  offset from the 30 Oct capture, table z 3): flange z 560 -> x 161-587, y -199..561 (42 px per block width);
  (400,150,560) -> x 261-687; (350,150,650) -> x 177-674, y -262..624 (36 px). Reachability of these poses is untested;
  the planner refuses before any other motion if one is out of reach.
- Episode robot-live-5 (16:52, `--view 350,150,650 --tower-origin 650,0`): the arm reached the z-650 view; both
  captures (obs_b174ec2236ec429a, obs_4219b8796ec8456b) showed "1" with its layer, "2" with blocks cut off at the
  top edge (~x 655, y 0 from the camera pose), and 15% floor past the table's left edge at ~1400 mm, plus a laptop,
  phone and hand on the right. Both stopped at `overview: table plane fitted_unreliable` (5-9% inliers): the plane
  fit seeded from every sample was skewed by the floor. Fix (grasp_geometry `INITIAL_DEPTH_BAND`): the first
  iteration uses samples within 10% of the median depth; the two captures now fit at 75% inliers, rms 2.2 mm, and
  the z-460 captures are unchanged (95%). The run also now looks at the tower from above when the base is clipped by
  the overview edge, not only when the origin is out of frame. Test added. Open risk: Kazy's planner refusal at
  (656, 6) is the same spot as "2"; move "2" nearer the robot (x <= ~570) if the hover or place is refused.

Limitations: the after-lift check cannot measure while the bundle fills the image (see run 4), so a full stack cannot
complete until it is changed; table height and camera offset are unverified; `--grip-lower-mm 3` is a single-run fix,
not a calibration; source and base must be reachable (x <= ~600 mm worked) and both visible from the view pose.

Next actions, in order:
1. After-lift check: use depth only at the pick spot (blocks still at table height = left behind; blocks far above
   the table = held) instead of requiring a table-plane fit over the whole image. Test offline on run 4's capture.
2. Touch-off at the pick location to measure the real table z; record it as a verified geometry calibration.
3. Three supervised runs; read `residuals-<setup>.jsonl`; only then consider `--apply-offset` or Sam's camera XY
   correction (calibrate_camera.py) for the viewpoint disagreement.

A local vision-language decision model (LiquidAI d1-omni-600M) was tested offline on 23 saved captures for these
checks and was not usable (held-vs-not separation absent, "stacked" true on every photo, ~1 s/photo on an M2 GPU);
nothing from it is in this branch.
