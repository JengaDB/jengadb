import hashlib
from urllib.parse import urlsplit

class MongoStore:
    def __init__(self, config):
        config.require_mongo()
        from pymongo import MongoClient
        from pymongo.write_concern import WriteConcern
        self.client = MongoClient(config.uri, tls=True, serverSelectionTimeoutMS=5000,
                                  connectTimeoutMS=5000, socketTimeoutMS=5000, timeoutMS=8000,
                                  retryWrites=False)
        self.collection = self.client[config.database].get_collection(config.collection,
                          write_concern=WriteConcern(w="majority", wtimeout=5000))
        endpoint = urlsplit(config.uri).netloc.rsplit("@", 1)[-1].lower()
        self.target = hashlib.sha256(f"{endpoint}/{config.database}/{config.collection}".encode()).hexdigest()[:24]

    def ping(self):
        self.client.admin.command("ping")

    def ensure_index(self):
        self.collection.create_index("attempt_id", unique=True, name="unique_attempt_id")

    def require_index(self):
        indexes = self.collection.index_information().values()
        if not any(i.get("unique") and i.get("key") == [("attempt_id", 1)] and not i.get("partialFilterExpression") for i in indexes):
            raise RuntimeError("Unique attempt_id index required")

    def upsert(self, attempt):
        # Insert-only makes repeat delivery safe, including timeout-after-commit.
        result = self.collection.update_one({"attempt_id": attempt["attempt_id"]},
                                            {"$setOnInsert": attempt}, upsert=True)
        if not result.acknowledged:
            raise RuntimeError("Write not acknowledged")
        stored = self.collection.find_one({"attempt_id": attempt["attempt_id"]}, {"_id": 0})
        if stored != attempt:
            raise RuntimeError("Read-back mismatch")

    def close(self):
        self.client.close()
