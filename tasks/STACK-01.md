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
- Episode robot-live-6: hover over (650, 0) at flange z 650 refused, "zero IK solutions" (same as Kazy's (656, 6)).
- Episode robot-live-7 (17:00-17:01, "2" moved to ~(524, -14), `--view 350,150,560`): two full runs reached the pick.
  Both captured, planned (source (439,193) 4 deg; base (524,-15) -85 deg), gripped ("holding" true) and lifted 100 mm,
  then stopped at `after_lift: table plane unavailable`: at camera z 261 the D435 returned 0.6% valid depth (inside
  its 1280x720 minimum range). The after-lift photo shows all three blocks still on the blue square: grip failed
  again, claws closing too high, as in Kazy's run 3. `--grip-lower-mm 15` had no extra effect: the pick tips are
  clamped at `tips_min_mm` 1.0 above the calibrated table (z 3.0, unverified, from depth; touch-off says ~0), so
  anything past `--grip-lower-mm 5` is cut off. Worse, the release height used the *requested* grip depth (23 mm)
  while the achieved one was 13.2 mm: release tips 5.9 mm would have pushed the bundle ~8 mm into the base.
  Fixes (Kris): `plan_stack` records `grip_below_top_mm` (achieved) and `requested_grip_below_top_mm` and derives the
  release from the achieved depth; `--tips-floor-mm` (default 1.0, bounded to -6.0) lets the operator lower the
  floor explicitly; the plan printout states tips height, achieved vs requested grip and the clamp; the after-lift
  capture is taken after rising to travel height (depth valid, pick spot visible, bundle mostly out of frame). 2
  tests added. Still unverified: table z and the claw-tip offset; the right next step is a touch-off at "1".
- Episode robot-live-8, 17:07:53 (`--grip-lower-mm 20 --tips-floor-mm -6`, achieved grip 20.6 mm below the top):
  FIRST LAYER STACKED BY THE ROBOT. Lift check 0 left behind, released, retreated. Verification stopped at
  "between layer heights" (obs_528dabf666f44d9f): the lower layer's ends protrude beside the crossed top layer and
  read 18.4/20.4 mm (mixed depth). By eye and from the capture the layer is on, crossed, roughly centred
  (top-layer centre ~(522.7,-18.7) vs base (523.8,-14.6): ~4 mm; yaw ~1 deg); not counted as a verified run because
  the run did not complete its measurement. Three further runs stopped at the same check before the pick.
  Changes (Kris): `split_scene` tolerates between-level detections *below* the top layer (logged as
  `between_level_ignored`), still stops on one on top; `tower_layers()` = top height / block height, rounded;
  `--layers N` asserts the count before the pick; the episode state keeps `tower_layers` and `layers_stacked`,
  updated on each verified placement; the run prints the count at overview, close-up and after verification.
  Tips floor -6 with table z 3.0 means tips commanded ~3 mm below the touch-off table; the claws reached the blocks
  only at this setting, so the real claw-tip offset or table z is off by several mm: measure it. 2 tests added.
- `live-run` (heap -> packed layer, live_executor.run_flow) given the same knobs so it can build the layer on "1":
  `--grip-lower-mm` / `--tips-floor-mm` lower the pick tips (release follows, relative), `--keep-out x,y` skips
  candidates within the footprint clearance of a second tower, `--view` moves to a capture pose first and returns
  there instead of HOME, `--direct-local` dials the LAN. Untested live as of this note.
- 17:22 capture (obs_a3e022054049411c): tower "2" measures 74.8 mm = 5 layers; three loose blocks lay in the heap
  area ((420,67), (371,41), (317,48), angles -49/89/28 deg) and layer-stack rightly refused them as a source layer.
  New `palletize` command (`run_palletize`): per cycle, run_flow assembles the layer on `--source` (the pallet spot)
  from the loose blocks with `--tower-origin` as a keep-out, then run_layer_stack moves it onto the tower. Stops when
  the pallet ends with fewer than three blocks or the stack adds no verified layer; `--cycles N` repeats; `--layers`
  advances automatically after each verified stack. Untested live as of this note.
- 17:32 palletize, cycle 1 assembly stage: picked and placed two heap blocks (L0S0 middle grasp at (316,47) 27 deg;
  L0S1 ends grasp at (419,66) -49 deg) at tips -6, pushed the row tight (gaps 9.4 -> 1.3 -> 0.0 mm) and centred it
  (offset 18.7 -> 35.3 -> 2.4 mm), final row (441,180)/(437,205). The third heap block ((371,41), standing at 89 deg)
  had `no_eligible_grasp` in the single start capture because the (419,66) block lay in the claws' sweep; the flow
  never re-looked after that neighbour was gone, so L0S2 stopped with no_candidate_with_required_grasp and the
  pallet ended 2 of 3 (correctly refused for stacking). Fixes: run_flow rescans from the view pose when its queue
  has no candidate for the slot (up to MAX_RESCANS=2 consecutive rescans without a placement); palletize carries an
  incomplete row into the next run (`pallet_incomplete` state) instead of building over it; state.json of
  robot-live-8 marked accordingly by hand. The first two pushes show the row push works but over-corrects at
  first (18.7 mm off -> 35.3 mm the other way), converging on round 3.
- 17:37 rerun: the pallet had been cleared by hand; the carried row was declared and not seen -> stop. palletize
  now drops a carried-over row that is not on the pallet (`filled_slots_dropped`) and starts empty.
- Operator requests (Kris, 10:44): (a) consecutive pallet blocks are released `--place-spacing-mm` (default 10)
  further apart than one pitch so the release cannot nudge the neighbour; the push rounds measure the real gaps and
  close them as before. (b) Grip check from the gripper's own report: `Robot.grip_state()` reads
  `is_holding_something` (+meta), `get_current_inputs` (opening) or `do_command` position queries, whichever the
  module supports; `grip_failed()` = grab() false, holding false, or opening <= 0.5 mm. On failure the pick
  movement (hover, align, descend, grab) is repeated up to `--grip-retries` (default 1) times in live-run and in
  layer-stack. This is a deliberate, operator-requested, bounded exception to the "no automatic motion retry" rule;
  the gripper report fields the robot's gripper module actually returns are unknown until the first live run (the
  event `grab` records `gripper.source`).

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
