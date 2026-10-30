"""OpenAI function-calling coordinator (AGENT_HEURISTICS_PLAN step 8).

The model only sees IDs, scores, geometry, uncertainty and rejection reasons from the planning context; it never
receives images here and never computes geometry. Every tool argument is validated application-side and the final
``submit_decision`` goes through proposal.validate_selection, so a rejected or nonexistent candidate cannot become a
proposal. Timeouts, malformed output, exhausted turns and HTTP errors end in a reviewable abstention.

Transport is plain urllib against the Responses API (no SDK import); the request body builder and tool loop are
testable offline through an injected opener. Sending any scene data to OpenAI requires ALLOW_CLOUD plus
--allow-cloud and a key; the model name and request budget are parameters pending implementation review.

OpenAI tool-interface reference: https://developers.openai.com/api/docs/guides/function-calling
"""
import json
import urllib.error
import urllib.request

from .config import ConfigError
from . import proposal as prop

RESPONSES_PATH = "/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"   # placeholder pending review; set OPENAI_MODEL
INSTRUCTIONS = (
    "You coordinate one pick-and-place proposal for a Jenga-block stacking robot from a saved RGB-D scene. "
    "You may only choose among candidates and grasps the tools report as eligible, and only the open placement slot. "
    "Geometry, calibration and clearance are already computed; do not estimate them yourself. Retrieval cosine similarity "
    "to curated good-pick references is evidence of resemblance, not grasp success. If nothing is eligible, or evidence is "
    "insufficient, abstain with a reason. Finish by calling submit_decision exactly once with a short rationale."
)


def tool_definitions():
    def tool(name, description, properties, required):
        return {"type": "function", "name": name, "description": description, "strict": True,
                "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}
    cid = {"type": "string", "description": "candidate_id from inspect_scene"}
    return [
        tool("inspect_scene", "Summary of the scene: every candidate with eligibility, rejection reasons, flags, geometry and scores.", {}, []),
        tool("get_retrieval_evidence", "Curated-reference matches for one candidate: cosine similarities, reference IDs, threshold, exclusions.",
             {"candidate_id": cid}, ["candidate_id"]),
        tool("get_grasp_options", "Grasp styles for one candidate with eligibility, finger-clearance results and reasons.",
             {"candidate_id": cid}, ["candidate_id"]),
        tool("list_placement_slots", "The open destination slot and the full two-layer slot plan.", {}, []),
        tool("submit_decision", "Final answer. Either propose one eligible candidate/grasp into the open slot, or abstain.",
             {"action": {"type": "string", "enum": ["propose", "abstain"]},
              "candidate_id": {"type": ["string", "null"]},
              "grasp_style": {"type": ["string", "null"], "enum": ["middle", "side", "ends", None]},
              "shift_mm": {"type": ["number", "null"], "description": "shift_mm_along_long_side of the chosen grasp option"},
              "slot_id": {"type": ["string", "null"]},
              "rationale": {"type": "string"}},
             ["action", "candidate_id", "grasp_style", "shift_mm", "slot_id", "rationale"]),
    ]


class ToolError(ValueError):
    pass


class ContextTools:
    """Deterministic tool handlers over one planning context. Only data the plan allows the agent to see."""

    def __init__(self, context):
        self.context = context
        self.calls = []

    def inspect_scene(self):
        out = []
        for c in self.context["candidates"]:
            g = c["geometry"]
            out.append({"candidate_id": c["candidate_id"], "eligible": c["eligible"], "rejection_reasons": c["rejection_reasons"],
                        "flags": g["flags"], "height_above_table_mm": g["height_above_table_mm"], "length_mm": g["length_mm"],
                        "width_mm": g["width_mm"], "valid_depth_fraction": g["valid_fraction"], "geometry_quality": c["geometry_quality"],
                        "retrieval_status": c["retrieval"]["status"], "best_cosine_similarity": c["retrieval"]["best_cosine_similarity"],
                        "eligible_grasps": [(x["style"], x["shift_mm_along_long_side"]) for x in c["grasps"] if x["eligible"]]})
        return {"observation_id": self.context["observation_id"], "mode": self.context["mode"], "source": self.context["source"],
                "calibration_missing": self.context["calibration"]["missing"], "table_plane": self.context["table_plane"].get("status"),
                "candidates": out, "ranking": self.context["ranking"], "geometry_only_ranking": self.context["geometry_only_ranking"]}

    def get_retrieval_evidence(self, candidate_id):
        return prop.candidate_entry(self.context, candidate_id)["retrieval"]

    def get_grasp_options(self, candidate_id):
        return prop.candidate_entry(self.context, candidate_id)["grasps"]

    def list_placement_slots(self):
        return self.context["placement"]

    def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        if not isinstance(arguments, dict):
            raise ToolError("arguments_not_object")
        if name == "inspect_scene":
            return self.inspect_scene()
        if name == "list_placement_slots":
            return self.list_placement_slots()
        if name in ("get_retrieval_evidence", "get_grasp_options"):
            cid = arguments.get("candidate_id")
            if not isinstance(cid, str):
                raise ToolError("candidate_id_required")
            try:
                return getattr(self, name)(cid)
            except prop.SelectionError as exc:
                return {"error": str(exc)}
        raise ToolError(f"unknown_tool:{name}")


def apply_decision(context, arguments, decided_by):
    """Validate a submit_decision payload into a proposal or an abstention. Never raises for model mistakes."""
    if not isinstance(arguments, dict) or arguments.get("action") not in ("propose", "abstain"):
        return prop.abstention(context, "malformed_decision", decided_by, {"arguments": arguments})
    rationale = arguments.get("rationale") if isinstance(arguments.get("rationale"), str) else ""
    if arguments["action"] == "abstain":
        return prop.abstention(context, "agent_abstained", decided_by, {"rationale": rationale})
    try:
        return prop.make_proposal(context, arguments.get("candidate_id"), arguments.get("grasp_style"),
                                  arguments.get("shift_mm") if arguments.get("shift_mm") is not None else 0.0,
                                  arguments.get("slot_id"), rationale, decided_by)
    except (prop.SelectionError, TypeError, ValueError) as exc:
        return prop.abstention(context, "invalid_selection", decided_by, {"error": str(exc), "arguments": arguments})


class OpenAICoordinator:
    name_prefix = "openai"

    def __init__(self, config, allow_cloud=False, model=None, max_turns=8, timeout_s=60, opener=None):
        self.config, self.allow_cloud = config, allow_cloud
        self.model = model or getattr(config, "openai_model", None) or DEFAULT_MODEL
        self.max_turns, self.timeout_s = max_turns, timeout_s
        self.opener = opener or urllib.request.urlopen
        self.calls = 0

    @property
    def name(self):
        return f"{self.name_prefix}:{self.model}"

    @property
    def url(self):
        return self.config.openai_base_url.rstrip("/") + RESPONSES_PATH

    def _gate(self):
        if not self.allow_cloud:
            raise ConfigError("The OpenAI coordinator sends scene metadata to OpenAI; requires ALLOW_CLOUD=true and --allow-cloud")
        self.config.require_openai()

    def request_body(self, input_items):
        # store=False keeps nothing server-side, so the whole exchange (model output items included) is resent each turn
        # instead of referencing a previous response id.
        return {"model": self.model, "instructions": INSTRUCTIONS, "tools": tool_definitions(), "tool_choice": "required",
                "input": input_items, "store": False}

    def _post(self, body):
        self._gate()
        self.calls += 1
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.config.openai_key}", "Content-Type": "application/json"})
        with self.opener(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode())

    def decide(self, context):
        self._gate()
        tools = ContextTools(context)
        note = (" DRY RUN: unverified calibration has been waived so geometry can be inspected; your answer is recorded as a geometry "
                "preview, never as an executable proposal, so do not abstain only because calibration is unverified or retrieval is "
                "unavailable. Still refuse anything the tools mark ineligible." if context.get("dry_run") else "")
        items = [{"role": "user", "content": f"Plan one pick-and-place for observation {context['observation_id']} "
                                               f"(context {context['context_id']}). Start with inspect_scene.{note}"}]
        for _ in range(self.max_turns):
            try:
                response = self._post(self.request_body(items))
            except urllib.error.HTTPError as exc:
                return prop.abstention(context, "openai_http_error", self.name, {"status": exc.code})
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return prop.abstention(context, "openai_unreachable", self.name, {"error": type(exc).__name__})
            except (ValueError, json.JSONDecodeError):
                return prop.abstention(context, "openai_malformed_response", self.name)
            outputs = response.get("output")
            if not isinstance(outputs, list):
                return prop.abstention(context, "openai_malformed_response", self.name)
            calls = [o for o in outputs if o.get("type") == "function_call"]
            if not calls:
                return prop.abstention(context, "no_tool_call", self.name, {"output_types": [o.get("type") for o in outputs]})
            # Echo every output item (reasoning items included, which reasoning models require) before the tool results.
            items = items + [o for o in outputs if isinstance(o, dict)]
            for call in calls:
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = None
                if call.get("name") == "submit_decision":
                    return apply_decision(context, arguments, self.name)
                try:
                    result = tools.dispatch(call.get("name"), arguments)
                except ToolError as exc:
                    result = {"error": str(exc)}
                items.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": json.dumps(result, default=str)})
        return prop.abstention(context, "turn_budget_exhausted", self.name, {"max_turns": self.max_turns, "tool_calls": len(tools.calls)})
