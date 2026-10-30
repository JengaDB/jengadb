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
- Operator requests (10:58-10:59), all in run_flow unless noted: release spacing 14 mm (`PLACE_SPACING_MM`); the row
  push is one direction only (from the open end toward the anchor block) and one stroke closes the gaps and centres
  the row (`row_push_plan`); the anchor is released `ROW_PRE_OFFSET_MM` = 8 past nominal toward the open end so the
  centring shift is in the pushable direction, an off-centre the other way is reported, not pushed; pushes run at
  `--push-speed` (30 deg/s) and the last `APPROACH_MM` = 40 of every descent at `--approach-speed` (40 deg/s), travel
  speed restored afterwards (`--arm-speed`, else 180 as used on 30 Oct, announced); grasp order for a slot is taken
  from a target with the release spacing as gap, so middle/side grasps remain allowed next to a placed neighbour
  (`spaced_target`); `--layers` mismatch is reported and the measurement used (layer_stack `layers_mismatch`);
  the pallet count is measured (`reconcile_pallet` at start and after each rescan, `pallet_count` from the final
  capture) and palletize gates on that count; `--references local|atlas` embeds the eligible heap candidates'
  context cutouts (hosted Voyage, image-only) and ranks curated look-alikes first via the SKY-66 retrieval
  (`retrieval` event). Untested live as of this note; speed keys are the xArm module's `set_speed`.
- 11:15: palletize's pallet target is now a one-layer target (`layers=1`): a full row means `layer_already_complete`
  and the assembly stage returns after its first capture (no push rounds, no extra captures) so the stacking stage
  runs at once; previously a full row would have opened slot L1S0 on top of the pallet.
- 11:19: every pick attempt opens the gripper first (after a push the claws are closed). Multi-height heap: the
  tips floor is now relative to the surface a block rests on (`resting_level_mm` = rounded whole levels under the
  top; `grip_tips_mm(..., block_thickness_mm)`), so the operator's deep grip (depth 28, floor -6) gives tips -6 for
  a table block and +9 for a block lying on another, never through the lower block; the claw-clearance strips are
  checked down to those tips, so a lower block under the claws makes that grasp ineligible (side/ends may still be
  clear). run_flow applies grip depth and floor through the calibration bundle's GraspParams, so eligibility and
  the commanded descent agree, and takes the heap from the top (higher resting level first). 1 test added.
- 11:24 palletize on a real heap (obs_5533b073025c47a2; tower "2" meanwhile reduced to 2 layers): 4 heap blocks
  detected (three on the table, one lying across two of them at 29 mm), every grasp `fingers_blocked`: the claw
  strips (47-65 mm from the centre) land on neighbours 13-14 mm high while the tips were to go to -6 / +9. Also
  Atlas retrieval ran (3 embedded per capture, all `no_match`: no curated pickable references in the index yet).
  Fix: `claw_obstruction` distinguishes a hard obstruction (higher than block top - MIN_GRIP_DEPTH_MM 8) from a low
  neighbour the claws may stop on: the latter keeps the grasp eligible with `tips_raised_over_neighbour` (tips =
  p90 of the obstruction + 2 mm). The bridge block becomes pickable at tips 16 (13 mm grip); a table block beside
  a 26 mm stack stays blocked. The release height now subtracts the block's resting level (a block picked from on
  top of another hangs tips - resting below the tips). 1 test added.
- 11:32 (photo: pallet row with ends staggered ~18 mm per block): end alignment added to the push stage.
  `row_align_plan` lists blocks whose centre along the row's long axis is more than ROW_ALIGN_TOLERANCE_MM (5) from
  the row median; `push_block_end` sets closed claws (closing along the length, faces square to the push) just beyond
  the protruding end and slides the block back by the offset minus 1 mm at push speed. One block per round after
  gaps/centre are satisfied, ROW_PUSH_ROUNDS = 5 in total. 1 test added.
- 11:36 palletize: three heap blocks picked and placed (one off another block at raised tips 17.1; gripper width via
  `is_holding_something.meta.position`, ~220 for a held block, ~730 for an ends grip). Then: (a) blocks released with
  an across grasp 39 mm from the neighbour were dragged 16-30 mm when the claws opened (claw sweep ~40 mm/side);
  (b) a 60 mm constrained push ended in "Lost connection" (SDK probe: ResourceNames with 1 s deadline, missed while
  the server planned). 11:43 rerun: three placed, then SESSION_EXPIRED at layer_complete. Fixes: across-grasp
  release spacing = (open_gap - width)/2 + claw + margin = 52.5 mm (`release_spacing_mm`), ends grasp keeps 14;
  pushes in <= 25 mm segments; a complete-but-loose row on rerun goes through the push rounds; `--no-sessions`
  disables Viam sessions and the 1 s connectivity probe (operator decision; a dropped connection then no longer
  stops an executing move); Voyage embeddings run off the event loop; `--align-speed` (45 deg/s) for the claw
  alignment rotations and the slow final approach also in layer-stack. 1 test added.
- 18:48-18:50 palletize, FIRST COMPLETE ASSEMBLY STAGE: three heap blocks picked (bridge block at raised tips 19.9,
  then ends grasps; one rescan), released 39/38 mm apart, one 40 mm push (segments) closed the row to gaps 0.0 and
  centre offset 2.1 mm, ends aligned, final count 3. Stacking stage then refused at the overview: tower top layer
  pitch 24.9 / 28.3 mm against the 22-28 bundle window. Fix: `base_layer()` measures the tower's top (and the
  stacked layer at verification) with BASE_SPACING_RANGE_MM 20-33 and BASE_MIN_SHARED_CONTACT_MM 40; the bundle on
  the pallet keeps the strict window. 1 test added.
- 18:53-18:56 palletize: assembly stage complete again (3 picked, one rescan, one 34 mm push -> gaps 1.7, centre
  4.5 mm off toward the anchor, count 3). Stacking stage: the source search took every block outside the tower zone,
  so the remaining heap counted as part of the source (overview "saw 6", fell back to --source; close-up refused on
  a clipped heap block). Fix: with a source spot given, only blocks within the footprint zone of it are the source
  layer (`at_source`), overview and close-up.
- 12:01 operator request: the pallet must always end with three. run_flow restructured into `pick_phase` and
  `push_phase` inside a fill loop (MAX_FILL_ROUNDS = 3): after the push rounds the measured count is checked; if it is
  short (a block dropped in transit, knocked out, failed grasp with the gripper report), the pallet is re-measured,
  the slots re-derived from what is there and another block fetched, within the placement budget (palletize allows
  BLOCKS_PER_LAYER + 2 placements). Hard stops (planner refusal, grab failure after retries, no candidate after
  rescans) still end the stage.

- 30 Oct 19:19-19:29 UTC, episode kazy-palletize-30oct (operator Kazy, Kazy's laptop, main @6aafee2, `--direct-local
  --no-sessions --grip-lower-mm 15 --tips-floor-mm -3`; this laptop's calibration has table z 0, so -3 equals Kris's
  -6 with z 3.0): FIRST VERIFIED PALLETIZE CYCLE. Assembly: three heap blocks picked (one off another block at raised
  tips 16.7, one side and one ends grasp; one rescan), six close-and-centre pushes. Stacking: lift check 0 left
  behind; verification centre error 1.4 mm, yaw error 1.24 deg -> success; tower 4 layers. 1 of the 3 required runs.
  Operator notes on the push stage: the close-and-centre push repeated five times on a 3-4 mm residual gap that came
  from a block skewed 13 mm along its length, sliding the row 31.9 mm off-centre; end alignment never ran; the claws
  pushed 2 mm too low.
- Changes on branch kazy/push-tuning (Kazy): one close-and-centre push per push phase (`CENTRE_PUSH_ATTEMPTS`), after
  which end alignment runs even with a few mm of gap left; `PUSH_TIPS_MM` 5 -> 7. Gravity reorientation
  (`--drop-spot x,y`, live-run and palletize): a pick whose top face tilts more than `REORIENT_TILT_DEG` 8 (height-map
  plane fit seeded from the top band, so side faces do not bias it) or that stands on its edge is released
  `DROP_HEIGHT_MM` 25 above the clear spot, re-captured and checked to lie flat (one block thick, full length, level)
  before it can be placed; at most `MAX_REORIENT_DROPS` 2 per run, then stop. The spot must be clear and away from
  the pallet and keep-out. Offline: tests/test_reorient.py (tilt within 1.5 deg at 0/5/12/25 deg with side faces);
  a scratch simulation of a block leaning 12 deg measured 11.3 deg, dropped it, found it flat and finished the
  pallet. Untested live.
- Pallet repair (same branch, operator choice "pick-and-drop" over sweeping the pallet): with `--drop-spot`, after the
  push rounds a fresh capture checks every pallet block; one not one block thick (on a neighbour, on its edge) or
  tilted over 8 deg is picked (grasps from a zone-free context of the same capture; tips raised over the lower
  neighbours) and laid flat on the drop spot, highest first, one per fill round; the top-up replaces it. Only
  positive evidence counts on the pallet (a trimmed mask or unmeasurable face does not). No clear grasp or an
  occupied drop spot stops the run for the operator. Pick and drop motions are shared helpers (`grab_block`,
  `drop_and_check`) for heap and pallet. Scratch simulation: a pallet block put down on its neighbour measured
  29.7 mm, was taken off at tips 21.7, dropped, found flat and replaced. The same simulation still ends 2 of 3: the
  third across-grasp block lands outside the 102.5 mm zone and the top-up re-picks it. Untested live.
- Row-lane fix ported from codex/layer-stack efd55a1 (operator approved; that commit's other changes are not here):
  three across-grasp blocks are released 77.5 mm apart, so the third lands ~138 mm from the pallet centre, outside
  the 102.5 mm footprint circle; it was counted missing and re-picked by the top-up. Blocks in the released row's lane
  (within ROW_LANE_HALF_WIDTH_MM 20 of the row axis, out to the full release spread) now count as pallet blocks, are
  never picked again and are included in the pallet check; a --drop-spot in line with the row is refused at start
  (a block dropped there would count as a pallet block). tests/test_flow_harness.py (copied from codex/layer-stack,
  end-to-end only push contact fixed so side-by-side blocks are not shoved along) covers: three across grasps counted
  with no re-pick; a block put down on its neighbour taken off, laid flat and replaced to a full pallet; drop spot
  too close / in the lane refused before picking. Untested live.

- 30 Oct 19:38-19:43 UTC, episode kazy-palletize-30oct, second cycle (operator Kazy, kazy/push-tuning @fd95a0a,
  `--drop-spot 300,330 --layers 4`, otherwise as the first cycle). Assembly: an on-edge block resting on another
  (top 31.8 mm) was dropped at the spot and then placed normally; one close-and-centre push (gaps 105.7 -> 3.0 mm)
  then two end-alignment pushes (ends aligned, gaps 0.3, centre 2.2 mm off): the operator's push changes worked live.
  The third block released 137 mm out was counted through the row lane. Pallet check: 3 blocks, tilt 1.2-3.3 deg,
  heights 13.0-14.2, none flagged. Stacking: grab ok, lift check 0 left behind, released at (537.2, -20.7).
  Verification refused (`verify_failed`, outcome unknown): a detection at (541.7, 14.1), 70.4 mm, is the base
  layer's ends protruding ~5 mm past the new layer (photo obs_87bc5ecf52004a3b; base spans y -58.3..16.7, new layer
  ~-61.7..11.9) read with mixed depth. From the detected centres the new layer sits ~4.0 mm from the base centre
  (mostly along y) and ~1.6 deg from perpendicular: likely stacked, just over the 3 mm tolerance. Not counted.
  Two bugs from this run fixed: the post-drop flatness check failed a flat block that read 10.4 mm (only too-high
  readings count now); a tilted block was skipped as "drop spot occupied" by a block already moved to the pallet
  since that capture (picked blocks are now ignored).
  Open: verification treats a lower layer's protruding ends as a block "between layer heights" on top; a
  between-level detection below the top layer's height and outside its footprint could be ignored, as split_scene
  already does for the base. The ~4 mm placement offset points at the unverified camera XY / table calibration.

- Operator notes after the second cycle (30 Oct): (a) the claws hit the table by ~3 mm when picking: run with
  `--tips-floor-mm 0` on this laptop's calibration (table z 0), not -3; (b) wrist rotation too slow:
  DEFAULT_ALIGN_SPEED_DEG_S and the `--align-speed` default 45 -> 90 deg/s; (c) end alignment in one move instead of
  one block per round, only if the blue box matters only at the start of the pallet arrangement. It does: the
  stacking stage grips the measured row centre (second cycle: (435.4, 190.6) with --source 440,195) and uses the spot
  only as a zone (`at_source`). `row_align_push`: one stroke from the side that leaves the row nearer the pallet
  origin along the long axis, aimed between the blocks that stick out (claws 45 mm wide), back to the line of the
  block furthest back; ALIGN_PUSH_ATTEMPTS 1 per push phase. On the second cycle's ends (x 436.3 / 430.4 / 441.8)
  that is one 15.4 mm stroke instead of two pushes. Harness: two blocks 8 and 12 mm out lined up within 2 mm in one
  stroke (the simulated claw face now pushes every block it covers). Untested live.

- Knock-over (operator request, 30 Oct; `--knock-over`, live-run and palletize): a block standing on its end (about
  one block length tall with only its 25 x 15 end face visible; `standing_on_end`) is rejected for picking, so when
  a capture has nothing pickable the flow pushes it over: closed claws at KNOCK_TIPS_MM 45 above the table, from the
  far side, KNOCK_PAST_MM 30 past contact. `knock_plan` prefers pushing across the thin 15 mm side (it tips onto its
  wide face and lies flat), then across the 25 mm side (lands on its edge; the drop step lays it flat later); the
  landing spot must have no other block centre within 55 mm and lie outside the tower/pallet/keep-out/drop spot
  and the pallet lane, and nothing may lie within 30 mm of where the claws come down; otherwise the block is left
  (`knock_not_possible`). A fresh capture checks it fell (`knock_check`); MAX_KNOCKS 2 per run, then stop. Tests:
  standing detection (a 59 mm tower top with a 73 mm top is not a standing block), knock_plan direction choice and
  refusals, and a harness run (standing block rendered separately, camera resolution doubled so the end face
  clears the 200 px mask minimum) where it is knocked onto its wide face (lands 14.9 mm high) and then picked to a
  full pallet. Untested live: the real topple, the knock height and whether a SAM mask of a standing block's end
  face gives a usable height on the D435.

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
