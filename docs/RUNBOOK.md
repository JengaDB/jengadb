# Onsite runbook

## Next 15 minutes

1. Open `reports/mock.html` and run the README demo commands. You already have a network-independent fallback.
2. With teammates, choose one available object and one observable scoring rule. Default proposal: a foam cube reaches a mentor-approved region. This is a proposal only; no physical placement is implemented.
3. Assign one hardware operator, one application owner, and (if three teammates) one data/demo owner.
4. Ask the mentor for your assigned machine, permitted access, and safety briefing. Fill the assignment sheet below without credentials.

## Assignment sheet — not yet confirmed

| Item | Value |
| --- | --- |
| Assigned machine alias | TBD with mentor |
| Arm / gripper / camera models | TBD |
| Actual component names | TBD |
| Approved task and success rule | TBD |
| Operator / mentor | TBD |
| Emergency stop location and procedure | Demonstrate onsite |
| Calibrated frames, workspace, speed, payload | Mentor confirmation required |
| Authorized database / collection | TBD; README defaults are placeholders |
| Allowed data and artifact retention | TBD |

## Discovery

The supplied morning briefing directs participants to app.viam.com → hackathons → Arm Farm → assigned machine. Confirm the machine label on the onsite computer. Expected names `arm`, `gripper`, `cam` are not verified assignments.

After authorized access and SDK setup, use the README's `viam-check`. It creates one client, lists resource names, and closes in finally. It does not import or issue arm/gripper commands. If local setup stalls, the supplied briefing identifies CONTROL → Sandbox as an SDK alternative; browser code can control real hardware, so use discovery-only code after approval. Avoid generated examples that move hardware. Keep credentials out of screenshots and chat.

Review the frame system with the mentor in 3D SCENE. Do not use approximate slide offsets as calibration. Pixels and raw depth are not executable targets. The supplied briefing warns that direct arm move APIs bypass configured obstacle planning. No physical motion implementation is included here.

## Physical work remains a separate phase

Before a future supervised action, the mentor must approve the actual machine, frames, workspace, payload, and plan. Learn the physical emergency stop and keep one operator in control. Software flags and planners are not safety systems. Do not retry movement automatically after a fault or timeout; follow the onsite stop procedure. This project cannot be enabled for motion by changing a flag.

## Optional MongoDB

Use only the organizer-designated test destination and approved records. The organizer/user handles scoped credentials and a narrow temporary IP allowlist. Do not change network access or roles to make a connection work. Run bounded ping first. Obtain specific approval before unique-index creation and synthetic writes. Then run index setup and flush, repeat flush, and verify the same 20 IDs remain. The adapter performs read-back before acknowledgment. An unavailable database leaves the local demo usable.

## Two-minute demo script

- “This is a synthetic fallback; no robot is moving.”
- Run mock-demo and open the report. Show 20 attempts, 12 successes, 4 failures, 4 unknown.
- “We record observed outcomes separately from command acceptance. Unknowns do not inflate the success rate.”
- Expand the attempts table and explain one result.
- Repeat the command: still 20 attempts, same IDs.
- “Our tests simulate remote failure and interrupted uploads; local records survive and repeat delivery avoids duplicate attempts.” Clearly label this as tested simulation, not a live Atlas demonstration.
- State which real integrations, if any, were subsequently verified onsite. Never present mock results as hardware performance.

## Event timing from the supplied guide

Thursday 29 October: lunch 1 PM, dinner 6 PM, hard close 9 PM. Friday 30 October: doors 9 AM; demos 3:30 PM; awards 5 PM; end 6 PM. All times America/Los_Angeles (PDT). Follow later onsite announcements; logistics were not independently reverified during this build. Freeze features around Friday 2:30 PM and save an approved demo recording beforehand. There is no overnight hardware access according to the guide.

Bring laptop charger and usual adapter. Keep organizer confirmation accessible. No hardware purchase is needed for this preparation.

## Fallback and teardown

If hardware, credentials, network, or SDK setup stalls, demonstrate the local HTML report and tests. Do not add computer vision, LLMs, or a dashboard dependency to the critical path. After the event, coordinate credential revocation and approved data retention with the organizer. Publish nothing containing private captures, machine settings, or secrets without permission.
