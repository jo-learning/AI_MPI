from pymongo import MongoClient
from pathlib import Path
from bson import json_util
from bson import ObjectId

client = MongoClient("mongodb+srv://joes:v7fmZoWdOMhaChf4@cluster0.hkbmsuh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# List all databases
print("Databases:")
for db_name in client.list_database_names():
    print(" -", db_name)

# Select a database
db = client["test"]

# List collections
print("\nCollections in", db.name + ":")
for collection_name in db.list_collection_names():
    print(" -", collection_name)

collection = db["users"]

docs = list(collection.find({"role": "coach"}))
output_path = Path(__file__).resolve().parent / "coaches.json"
with output_path.open("w", encoding="utf-8") as write_handle:
    write_handle.write(json_util.dumps(docs, indent=2))

# target_org_id = ObjectId("690cdba9809365c990983391")
# organization_doc = collection.find_one({"_id": target_org_id})
# if not organization_doc:
#     print(f"Organization {target_org_id} not found.")
# else:
#     courts_collection = db["courts"]
#     courts_docs = list(courts_collection.find({"organization": target_org_id}))
#     courts_output_path = output_path.with_name(f"courts_{target_org_id}.json")
#     with courts_output_path.open("w", encoding="utf-8") as write_handle:
#         write_handle.write(json_util.dumps(courts_docs, indent=2))

#     def normalize_ids(values):
#         normalized = []
#         for value in values:
#             if isinstance(value, ObjectId):
#                 normalized.append(value)
#             elif isinstance(value, str):
#                 normalized.append(ObjectId(value))
#             elif isinstance(value, dict) and "$oid" in value:
#                 normalized.append(ObjectId(value["$oid"]))
#         return normalized

#     player_ids = normalize_ids(organization_doc.get("players", []))
#     coach_ids = normalize_ids(organization_doc.get("coaches", []))

#     players_collection = db["users"]
#     coaches_collection = db["users"]

#     players_docs = list(players_collection.find({"_id": {"$in": player_ids}})) if player_ids else []
#     # players_docs = list(
#     #     players_collection.find(
#     #         {"_id": {"$in": player_ids}},
#     #         {"_id": 1, "firstName": 1, "lastName": 1, "dateOfBirth": 1, "gender": 1, "phoneNumber": 1, "address": 1},
#     #     )
#     # ) if player_ids else []
#     coaches_docs = list(coaches_collection.find({"_id": {"$in": coach_ids}})) if coach_ids else []

#     players_output_path = output_path.with_name(f"players_{target_org_id}.json")
#     with players_output_path.open("w", encoding="utf-8") as write_handle:
#         write_handle.write(json_util.dumps(players_docs, indent=2))

#     coaches_output_path = output_path.with_name(f"coaches_{target_org_id}.json")
#     with coaches_output_path.open("w", encoding="utf-8") as write_handle:
#         write_handle.write(json_util.dumps(coaches_docs, indent=2))

# for doc in docs:
#     print(doc)

# doc = collection.find_one()

# if doc:
#     print("\nSchema (fields) in collection:", collection.name)
#     print(list(doc.keys()))
# else:
#     print("Collection is empty.")