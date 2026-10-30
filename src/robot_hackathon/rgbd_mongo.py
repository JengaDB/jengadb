"""Atlas destination for RGB-D observations, candidates, artifact sync state and media.

Collections: <observations> and <candidates> are insert-only with read-back; artifact_sync is mutable with a
monotonic remote_status (pending -> synced); upload_attempts + media_manifest implement the media ownership
protocol in media.py over GridFS bucket rgbd_media. The candidate vector index carries setup/version filters.
"""
import hashlib
from datetime import timedelta, timezone, datetime
from urllib.parse import urlsplit
from .media import MediaUploader

VECTOR_INDEX = "candidate_context_v1"
VECTOR_FILTERS = ("phase", "exposure.status", "pickability_label.status", "pickable", "setup_id", "mode", "encoder_version",
                  "processing_version", "segmentation.model_version", "calibration_version")


def _target(uri, database, *collections):
    endpoint = urlsplit(uri).netloc.rsplit("@", 1)[-1].lower()
    return hashlib.sha256(f"{endpoint}/{database}/{'/'.join(collections)}".encode()).hexdigest()[:24]


class MongoMediaBackend:
    def __init__(self, db):
        from gridfs import GridFSBucket
        self.bucket = GridFSBucket(db, bucket_name="rgbd_media")
        self.files = db["rgbd_media.files"]
        self.attempts = db["upload_attempts"]
        self.manifest = db["media_manifest"]

    def now(self):
        return datetime.now(timezone.utc)

    def insert_attempt(self, sha256, source_alias, started_at):
        from bson import ObjectId
        attempt_id = ObjectId()
        self.attempts.insert_one({"_id": attempt_id, "sha256": sha256, "source_alias": source_alias,
                                  "started_at": started_at, "state": "writing"})
        return attempt_id

    def set_attempt_state(self, attempt_id, state):
        self.attempts.update_one({"_id": attempt_id}, {"$set": {"state": state, "updated_at": self.now()}})

    def get_attempt(self, attempt_id):
        doc = self.attempts.find_one({"_id": attempt_id})
        if doc and doc["started_at"].tzinfo is None:
            doc["started_at"] = doc["started_at"].replace(tzinfo=timezone.utc)
        return doc

    def write_file(self, attempt_id, data, metadata):
        self.bucket.upload_from_stream_with_id(attempt_id, metadata["sha256"], data, metadata=metadata)

    def file_length(self, file_id):
        doc = self.files.find_one({"_id": file_id}, {"length": 1})
        return None if doc is None else doc["length"]

    def download(self, file_id):
        return self.bucket.open_download_stream(file_id).read()

    def delete_file(self, file_id):
        from gridfs.errors import NoFile
        try:
            self.bucket.delete(file_id)
        except NoFile:
            pass

    def find_manifest(self, sha256):
        return self.manifest.find_one({"_id": sha256})

    def insert_manifest(self, doc):
        from pymongo.errors import DuplicateKeyError
        try:
            self.manifest.insert_one(doc)
            return True
        except DuplicateKeyError:
            return False

    def list_attempts(self, states, older_than):
        rows = list(self.attempts.find({"state": {"$in": list(states)}, "started_at": {"$lt": older_than}}))
        for row in rows:
            if row["started_at"].tzinfo is None:
                row["started_at"] = row["started_at"].replace(tzinfo=timezone.utc)
        return rows


class RgbdMongoStore:
    def __init__(self, config):
        config.require_mongo()
        from pymongo import MongoClient
        from pymongo.write_concern import WriteConcern
        self.config = config
        self.client = MongoClient(config.uri, tls=True, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000,
                                  socketTimeoutMS=20000, timeoutMS=30000, retryWrites=False)
        self.db = self.client[config.database]
        wc = WriteConcern(w="majority", wtimeout=5000)
        self.observations = self.db.get_collection(config.observations_collection, write_concern=wc)
        self.candidates = self.db.get_collection(config.candidates_collection, write_concern=wc)
        self.artifact_sync = self.db.get_collection("artifact_sync", write_concern=wc)
        self.target = _target(config.uri, config.database, config.observations_collection, config.candidates_collection)
        self.media = MediaUploader(MongoMediaBackend(self.db), timedelta(hours=config.media_attempt_max_age_hours))

    def ping(self):
        self.client.admin.command("ping")

    # -- indexes
    def ensure_indexes(self):
        self.observations.create_index("observation_id", unique=True, name="unique_observation_id")
        self.candidates.create_index("candidate_id", unique=True, name="unique_candidate_id")
        self.artifact_sync.create_index([("sha256", 1), ("target", 1)], unique=True, name="unique_sha_target")
        self.db["upload_attempts"].create_index([("sha256", 1), ("state", 1), ("started_at", 1)], name="attempt_lookup")

    def ensure_vector_index(self, dimension=1024):
        from pymongo.operations import SearchIndexModel
        existing = {i["name"]: i for i in self.candidates.list_search_indexes()}
        fields = [{"type": "vector", "path": "embedding", "numDimensions": dimension, "similarity": "cosine"}]
        fields += [{"type": "filter", "path": p} for p in VECTOR_FILTERS]
        if VECTOR_INDEX in existing:
            current = existing[VECTOR_INDEX].get("latestDefinition", {}).get("fields", [])
            have = {(f.get("type"), f.get("path")) for f in current}
            want = {(f["type"], f["path"]) for f in fields}
            if want - have:
                self.candidates.update_search_index(VECTOR_INDEX, {"fields": fields})
                return {"created": False, "updated": True, "added_filters": sorted(p for t, p in want - have), **self.index_status()}
            return {"created": False, "updated": False, **self.index_status()}
        self.candidates.create_search_index(SearchIndexModel(definition={"fields": fields}, name=VECTOR_INDEX, type="vectorSearch"))
        return {"created": True, **self.index_status()}

    def index_status(self):
        for index in self.candidates.list_search_indexes():
            if index["name"] == VECTOR_INDEX:
                return {"name": VECTOR_INDEX, "status": index.get("status"), "queryable": index.get("queryable"),
                        "dimensions": next((f.get("numDimensions") for f in index.get("latestDefinition", {}).get("fields", [])
                                            if f.get("type") == "vector"), None)}
        return {"name": VECTOR_INDEX, "status": "absent", "queryable": False}

    def require_indexes(self):
        def has_unique(collection, key):
            return any(i.get("unique") and i.get("key") == [(key, 1)] for i in collection.index_information().values())
        if not (has_unique(self.observations, "observation_id") and has_unique(self.candidates, "candidate_id")):
            raise RuntimeError("Unique observation_id/candidate_id indexes required; run rgbd-index")

    # -- immutable records
    def _insert_only(self, collection, key, doc):
        from bson.binary import Binary
        expected = {k: (bytes(v) if isinstance(v, (bytes, bytearray)) else v) for k, v in doc.items()}
        sent = {k: (Binary(v) if isinstance(v, (bytes, bytearray)) else v) for k, v in doc.items()}
        result = collection.update_one({key: sent[key]}, {"$setOnInsert": sent}, upsert=True)
        if not result.acknowledged:
            raise RuntimeError("Write not acknowledged")
        stored = collection.find_one({key: sent[key]}, {"_id": 0})
        # Subtype-0 Binary decodes as bytes; bson.Binary.__eq__ deliberately returns False against bytes.
        stored = {k: (bytes(v) if isinstance(v, (bytes, bytearray)) else v) for k, v in (stored or {}).items()}
        if stored != expected:
            raise RuntimeError("Read-back mismatch")

    def upsert_observation(self, obs):
        self._insert_only(self.observations, "observation_id", obs)

    def upsert_candidate(self, doc):
        self._insert_only(self.candidates, "candidate_id", doc)

    # -- mutable sync state; remote_status never moves backwards
    def set_artifact_sync(self, sha256, observation_id, remote_status, gridfs_id, size, local_check):
        now = datetime.now(timezone.utc)
        update = {"$addToSet": {"observation_ids": observation_id},
                  "$set": {"bytes": size, "updated_at": now, "last_local_check": {**local_check, "checked_at": now}},
                  "$setOnInsert": {"sha256": sha256, "target": self.target, "created_at": now}}
        if remote_status == "synced":
            update["$set"].update({"remote_status": "synced", "gridfs_id": gridfs_id})
        else:
            update["$setOnInsert"]["remote_status"] = "pending"
        self.artifact_sync.update_one({"sha256": sha256, "target": self.target}, update, upsert=True)

    def vector_search(self, vector, filters=None, limit=5, num_candidates=100):
        stage = {"index": VECTOR_INDEX, "path": "embedding", "queryVector": vector, "numCandidates": num_candidates, "limit": limit}
        if filters:
            stage["filter"] = filters
        # Curation provenance and compatibility fields travel with every hit so retrieval.shortlist can audit them;
        # vectors and image bytes are never projected.
        projection = {"_id": 0, "candidate_id": 1, "observation_id": 1, "episode_id": 1, "setup_id": 1, "mode": 1, "phase": 1,
                      "exposure": 1, "processing_version": 1, "calibration_version": 1, "encoder_version": 1,
                      "embedding_dimension": 1, "segmentation": 1, "pickability_label": 1, "pickable": 1, "review": 1,
                      "reference_set": 1, "score": {"$meta": "vectorSearchScore"}}
        return list(self.candidates.aggregate([{"$vectorSearch": stage}, {"$project": projection}]))

    def close(self):
        self.client.close()
