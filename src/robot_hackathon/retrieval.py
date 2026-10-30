"""Curated-reference retrieval evidence (AGENT_HEURISTICS_PLAN step 5).

Retrieval shortlists current candidates by resemblance to curated good-pick references. It never decides
grasp eligibility and never supplies coordinates: historical positions are evidence, not targets.

Score names stay distinct. Atlas returns ``vectorSearchScore`` for a cosine index, which is
``(1 + cosine) / 2``; the application converts it with ``cosine = 2 * score - 1`` and applies the agreed
``cosine_similarity >= 0.5`` threshold (Atlas score 0.75). The threshold is never lowered automatically.

Curation is explicit: ``pickable=true`` alone is not proof, because ``pickability_label.label_source`` may be
``human_default``. The default policy requires ``label_source == "human"`` and ``status == "pickable"``;
when the team designates a reference set, ``reference_set_id`` additionally requires ``reference_set`` membership.
Filters that the Atlas index does not expose are applied application-side over an over-fetched hit list.

MongoDB score reference: https://www.mongodb.com/docs/vector-search/indexes/vector-search-type/
"""
import math
from dataclasses import dataclass, asdict

from . import rgbd

EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_COSINE_THRESHOLD = 0.5
SCORE_CONVERSION = "cosine_similarity = 2 * atlas_vector_search_score - 1"


def cosine_from_atlas_score(score):
    """Atlas cosine index score -> cosine similarity in [-1, 1]."""
    return 2.0 * float(score) - 1.0


def atlas_score_from_cosine(cosine):
    return (1.0 + float(cosine)) / 2.0


def cosine_similarity(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Vectors must be non-empty and of equal length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        raise ValueError("Zero vector has no cosine similarity")
    return max(-1.0, min(1.0, dot / (na * nb)))


@dataclass(frozen=True)
class ReferencePolicy:
    """What counts as a curated reference and how the shortlist is cut. Values are recorded in the evidence."""
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD
    top_k: int = 10
    overfetch_factor: int = 4            # hits requested from the index before application-side filtering
    required_label_status: str = "pickable"
    require_human_label: bool = True     # label_source must be "human"; "human_default" is not curation
    reference_set_id: str | None = None  # when set, documents need reference_set == this id
    exclude_same_observation: bool = True
    exclude_observation_ids: tuple = ()  # evaluation scenes kept out of the reference pool

    def effective_settings(self):
        d = asdict(self)
        d["exclude_observation_ids"] = sorted(self.exclude_observation_ids)
        d["atlas_score_threshold"] = atlas_score_from_cosine(self.cosine_threshold)
        return d


def curation_reasons(doc, policy):
    """Why a stored candidate document is not an acceptable curated reference (empty list = acceptable)."""
    reasons = []
    label = doc.get("pickability_label") or {}
    if label.get("status") != policy.required_label_status:
        reasons.append(f"label_status:{label.get('status') or 'missing'}")
    if policy.require_human_label and label.get("label_source") != "human":
        reasons.append(f"label_source:{label.get('label_source') or 'missing'}")
    if policy.reference_set_id is not None and doc.get("reference_set") != policy.reference_set_id:
        reasons.append("not_in_reference_set")
    review = doc.get("review") or {}
    if review and review.get("status") not in (None, "included"):
        reasons.append(f"review_status:{review.get('status')}")
    return reasons


def compatibility_reasons(query, doc):
    """Encoder, dimension, phase and mode must match; synthetic and live records never mix."""
    reasons = []
    for key in ("encoder_version", "embedding_dimension", "mode"):
        if doc.get(key) != query.get(key):
            reasons.append(f"{key}_mismatch")
    if doc.get("phase", "pick") != "pick":
        reasons.append("phase_not_pick")
    return reasons


def index_filters(query, policy):
    """Filters the Atlas index exposes (see rgbd_mongo.VECTOR_FILTERS). Curation provenance is applied after."""
    filters = {"phase": "pick", "mode": query["mode"], "encoder_version": query["encoder_version"],
               "pickability_label.status": policy.required_label_status}
    return filters


def shortlist(query, hits, policy, index_identity):
    """Turn raw index hits into a reviewable evidence record.

    query: candidate record with candidate_id, observation_id, encoder_version, embedding_dimension, mode.
    hits: dicts with candidate_id, observation_id, score (Atlas-style cosine index score) and the stored fields
          used for curation/compatibility checks.
    """
    matches, excluded = [], []
    for hit in hits:
        reasons = []
        if hit.get("candidate_id") == query["candidate_id"]:
            reasons.append("self_match")
        if policy.exclude_same_observation and hit.get("observation_id") == query["observation_id"]:
            reasons.append("same_observation")
        if hit.get("observation_id") in policy.exclude_observation_ids:
            reasons.append("held_out_observation")
        reasons += compatibility_reasons(query, hit)
        reasons += curation_reasons(hit, policy)
        score = hit.get("score")
        cosine = None if score is None else cosine_from_atlas_score(score)
        if cosine is None:
            reasons.append("score_missing")
        elif cosine < policy.cosine_threshold:
            reasons.append(f"below_threshold:{cosine:.4f}<{policy.cosine_threshold}")
        row = {"reference_candidate_id": hit.get("candidate_id"), "reference_observation_id": hit.get("observation_id"),
               "atlas_vector_search_score": None if score is None else float(score), "cosine_similarity": cosine,
               "setup_id": hit.get("setup_id"), "processing_version": hit.get("processing_version"),
               "encoder_version": hit.get("encoder_version"), "embedding_dimension": hit.get("embedding_dimension"),
               "curation": {"pickability_label": hit.get("pickability_label"), "reference_set": hit.get("reference_set"),
                            "review": hit.get("review")}}
        if reasons:
            excluded.append({**row, "reasons": reasons})
        else:
            matches.append(row)
    matches.sort(key=lambda m: (-m["cosine_similarity"], m["reference_candidate_id"]))
    matches = matches[:policy.top_k]
    return {"schema_version": EVIDENCE_SCHEMA_VERSION,
            "query_candidate_id": query["candidate_id"], "query_observation_id": query["observation_id"],
            "encoder_version": query["encoder_version"], "embedding_dimension": query["embedding_dimension"],
            "mode": query["mode"], "index": dict(index_identity), "score_conversion": SCORE_CONVERSION,
            "policy": policy.effective_settings(), "hits_considered": len(hits),
            "status": "matched" if matches else "no_match", "matches": matches, "excluded": excluded,
            "best_cosine_similarity": matches[0]["cosine_similarity"] if matches else None}


def unavailable_evidence(query, reason, policy, index_identity=None):
    """Evidence record when retrieval could not run (no vector, no index). Never silently treated as a match."""
    return {"schema_version": EVIDENCE_SCHEMA_VERSION, "query_candidate_id": query.get("candidate_id"),
            "query_observation_id": query.get("observation_id"), "encoder_version": query.get("encoder_version"),
            "embedding_dimension": query.get("embedding_dimension"), "mode": query.get("mode"),
            "index": dict(index_identity or {"source": "none"}), "score_conversion": SCORE_CONVERSION,
            "policy": policy.effective_settings(), "hits_considered": 0, "status": "unavailable",
            "unavailable_reason": reason, "matches": [], "excluded": [], "best_cosine_similarity": None}


# ---------------------------------------------------------------- reference search backends

class ReferenceSearch:
    """Interface: identity() describes the index; search(vector, filters, limit) -> hits (Atlas-shaped dicts)."""

    def identity(self):
        raise NotImplementedError

    def search(self, vector, filters, limit):
        raise NotImplementedError


def _get_path(doc, path):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class LocalReferenceSearch(ReferenceSearch):
    """Brute-force cosine over in-memory documents, shaped like the Atlas result (score = (1 + cosine) / 2).

    Used for offline replay and fixture tests; the Atlas index behaviour itself still needs the read-only check
    named in the plan.
    """

    def __init__(self, documents, label="local-bruteforce"):
        self.documents = []
        for doc in documents:
            if not isinstance(doc.get("embedding"), list) or not doc["embedding"]:
                raise ValueError(f"Reference document {doc.get('candidate_id')} has no embedding")
            self.documents.append(doc)
        self.label = label

    def identity(self):
        return {"source": "local_bruteforce", "name": self.label, "similarity": "cosine",
                "documents": len(self.documents)}

    def search(self, vector, filters, limit):
        hits = []
        for doc in self.documents:
            if filters and any(_get_path(doc, k) != v for k, v in filters.items()):
                continue
            if len(doc["embedding"]) != len(vector):
                continue
            cosine = cosine_similarity(vector, doc["embedding"])
            hit = {k: v for k, v in doc.items() if k != "embedding"}
            hit["score"] = atlas_score_from_cosine(cosine)
            hits.append(hit)
        hits.sort(key=lambda h: (-h["score"], h["candidate_id"]))
        return hits[:limit]


class AtlasReferenceSearch(ReferenceSearch):
    """Read-only adapter over RgbdMongoStore.vector_search. Construction requires an already-authorized store."""

    def __init__(self, store):
        self.store = store

    def identity(self):
        from .rgbd_mongo import VECTOR_INDEX
        return {"source": "atlas", "name": VECTOR_INDEX, "similarity": "cosine",
                "collection": self.store.config.candidates_collection, "database": self.store.config.database}

    def search(self, vector, filters, limit):
        return self.store.vector_search(vector, filters, limit=limit, num_candidates=max(100, limit * 10))


def local_reference_documents(data_dir, mode, observation_ids=None):
    """Assemble reference documents from the local candidate store, review labels and embedding checkpoints.

    Mirrors candidate_store.remote_candidate without image bytes. `reference_set` is read from the review
    document when present (reserved for a future explicit curation marker); no field is invented.
    """
    from .candidate_store import CandidateLocalStore, Review
    from .pipeline import load_embedding
    review = Review(data_dir, mode)
    docs, review_docs = [], {}
    for cand in CandidateLocalStore(data_dir, mode).read():
        if observation_ids is not None and cand["observation_id"] not in observation_ids:
            continue
        rev = review_docs.setdefault(cand["observation_id"], review.load(cand["observation_id"]))
        if cand["candidate_id"] in set(rev.get("ignored", [])):
            continue
        doc = {k: cand[k] for k in ("candidate_id", "observation_id", "episode_id", "mode", "setup_id", "phase",
                                     "encoder_version", "embedding_dimension", "processing_version",
                                     "calibration_version", "segmentation", "exposure")}
        doc["pickability_label"] = Review.pickability_label(rev, cand["candidate_id"])
        doc["pickable"] = doc["pickability_label"]["status"] == "pickable"
        doc["review"] = {"status": "included", "reviewed_by": rev.get("reviewed_by"), "reviewed_at": rev.get("updated_at")}
        if rev.get("reference_set"):
            doc["reference_set"] = rev["reference_set"]
        try:
            doc["embedding"] = load_embedding(data_dir, cand)
        except (OSError, ValueError, KeyError):
            continue
        docs.append(doc)
    return docs


def retrieve(query, vector, search, policy):
    """Run one retrieval and return the evidence record. Empty shortlist on no match; threshold never lowered."""
    if vector is None:
        return unavailable_evidence(query, "no_query_embedding", policy, search.identity() if search else None)
    if search is None:
        return unavailable_evidence(query, "no_reference_index", policy)
    limit = max(policy.top_k * policy.overfetch_factor, policy.top_k)
    hits = search.search(vector, index_filters(query, policy), limit)
    return shortlist(query, hits, policy, search.identity())


def evidence_digest(evidence):
    return rgbd.sha256_bytes(rgbd.canonical_json(evidence).encode())[:16]
