# app_grouping.py
# Reference uploaded file (available in environment):
# /mnt/data/Hand Operated Corn Sheller Detailed Drawing v2.pdf

from __future__ import annotations

import math
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from bson.decimal128 import Decimal128
from bson.errors import InvalidId
from bson.objectid import ObjectId
from flask import Flask, jsonify, make_response, request
from pymongo import MongoClient

# Optional: scikit-learn for KMeans
try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - sklearn may be missing
    KMeans = None

# Optional: Gemini (google generative ai) - only used if GEMINI_API_KEY is set and you enable below
# import google.generativeai as genai

app = Flask(__name__)

client = MongoClient(
    "mongodb+srv://joes:v7fmZoWdOMhaChf4@cluster0.hkbmsuh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    tz_aware=True,
)


def get_db():
    return client["test"]


def try_parse_object_id(value: Any) -> ObjectId | None:
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str):
        try:
            return ObjectId(value)
        except InvalidId:
            return None
    if isinstance(value, dict):
        for key in ("$oid", "oid", "_id", "id"):
            if key in value:
                return try_parse_object_id(value[key])
    return None


def extract_object_ids(values: Any) -> list[ObjectId]:
    if not isinstance(values, (list, tuple, set)):
        return []
    seen: set[ObjectId] = set()
    results: list[ObjectId] = []
    for value in values:
        oid = try_parse_object_id(value)
        if oid and oid not in seen:
            seen.add(oid)
            results.append(oid)
    return results


def clean_document(value: Any) -> Any:
    """
    Convert BSON / non-JSON friendly fields to JSON-serializable Python types.
    """
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: clean_document(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_document(item) for item in value]
    if isinstance(value, tuple):
        return [clean_document(item) for item in value]
    return value


def fetch_users_by_ids(user_ids: list[ObjectId]) -> list[dict[str, Any]]:
    if not user_ids:
        return []
    users_collection = get_db()["users"]
    return list(
        users_collection.find(
            {"_id": {"$in": user_ids}},
            {
                "_id": 1,
                "firstName": 1,
                "lastName": 1,
                "dateOfBirth": 1,
                "phoneNumber": 1,
                "address": 1,
                "gender": 1,
                "emailAddress": 1,
                "avatar": 1,
                "role": 1,
                # If you have skillLevel, attendance, include them:
                "skillLevel": 1,
                "attendanceRate": 1,
            },
        )
    )


def fetch_courts_for_organization(org_oid: ObjectId) -> list[dict[str, Any]]:
    courts_collection = get_db()["courts"]
    return list(courts_collection.find({"organization": org_oid}))


def build_response_payload(
    organization: dict[str, Any],
    *,
    players: list[dict[str, Any]],
    coaches: list[dict[str, Any]],
    courts: list[dict[str, Any]],
) -> dict[str, Any]:
    cleaned_org = clean_document(organization)
    cleaned_players = clean_document(players)
    cleaned_coaches = clean_document(coaches)
    cleaned_courts = clean_document(courts)

    payload = {
        "organization": cleaned_org,
        "players": cleaned_players,
        "coaches": cleaned_coaches,
        "courts": cleaned_courts,
        "counts": {
            "players": len(cleaned_players),
            "coaches": len(cleaned_coaches),
            "courts": len(cleaned_courts),
        },
    }
    return payload


@app.get("/api/organizations/<string:org_id>")
def get_organization(org_id: str):
    org_oid = try_parse_object_id(org_id)
    if org_oid is None:
        message = {"error": "invalid_id", "message": f"{org_id} is not a valid ObjectId."}
        return make_response(jsonify(message), 400)

    db = get_db()
    organization = db["organizations"].find_one(
        {"_id": org_oid},
        {
            "_id": 1,
            "name": 1,
            "type": 1,
            "timeZone": 1,
            "phoneNumber": 1,
            "players": 1,
            "coaches": 1,
        },
    )
    if organization is None:
        message = {"error": "not_found", "message": f"Organization {org_id} was not found."}
        return make_response(jsonify(message), 404)

    player_ids = extract_object_ids(organization.get("players", []))
    coach_ids = extract_object_ids(organization.get("coaches", []))

    players = fetch_users_by_ids(player_ids)
    coaches = fetch_users_by_ids(coach_ids)
    courts = fetch_courts_for_organization(org_oid)

    response_payload = build_response_payload(
        organization,
        players=players,
        coaches=coaches,
        courts=courts,
    )
    return jsonify(response_payload)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# Grouping logic - Rule-based
# -----------------------------
def calculate_age(dob_iso: str | None) -> int | None:
    if not dob_iso:
        return None
    try:
        # handle 'YYYY-MM-DD' or ISO datetime
        dob = datetime.fromisoformat(dob_iso.split("T")[0])
        today = datetime.now()
        return (today - dob).days // 365
    except Exception:
        return None


def group_rule_based(players: list[dict[str, Any]], coaches: list[dict[str, Any]], max_players_per_coach: int = 4) -> dict[str, Any]:
    """
    Rule-based grouping:
      - Preferred rule: up to `max_players_per_coach` players per coach (default 4).
      - If there are fewer coaches than required, distribute players evenly among available coaches
        (this relaxes the rule but keeps grouping feasible).
    """
    players_clean = [clean_document(p) for p in players]
    coaches_clean = [clean_document(c) for c in coaches]

    total_players = len(players_clean)
    total_coaches = len(coaches_clean)

    if total_players == 0:
        return {"groups": [], "message": "no_players"}

    # Minimum coaches needed to respect the rule
    min_coaches_needed = math.ceil(total_players / max_players_per_coach)
    # If we have fewer coaches than needed, we'll distribute players among existing coaches
    if total_coaches == 0:
        return {"error": "no_coaches", "message": "No coaches available to assign."}

    # Determine how many groups we'll create:
    groups_count = min(total_coaches, min_coaches_needed) if total_coaches >= 1 else min_coaches_needed

    # If we have MORE coaches than groups_count, we'll only use as many coaches as groups_count (extra coaches remain unused)
    selected_coaches = coaches_clean[:groups_count]

    # Distribute players across groups_count groups (round-robin or chunking)
    groups: list[dict[str, Any]] = []
    # Use chunking to build groups up to max_players_per_coach where possible.
    # If coaches < min_coaches_needed, groups_count == total_coaches and some groups may exceed max_players_per_coach.
    chunk_size = math.ceil(total_players / groups_count)
    # But prefer not to exceed max_players_per_coach if we have enough coaches
    if total_coaches >= min_coaches_needed:
        chunk_size = max_players_per_coach

    for i in range(groups_count):
        start = i * chunk_size
        end = start + chunk_size
        group_players = players_clean[start:end]
        coach = selected_coaches[i]
        groups.append({"coach": coach, "players": group_players})

    # If any players remain (due to chunking), append them to the last group
    assigned = sum(len(g["players"]) for g in groups)
    if assigned < total_players:
        remaining = players_clean[assigned:]
        groups[-1]["players"].extend(remaining)

    # Return metadata about whether the requested rule had to be relaxed
    rule_respected = total_coaches >= min_coaches_needed
    return {
        "groups": groups,
        "meta": {
            "total_players": total_players,
            "total_coaches": total_coaches,
            "max_players_per_coach": max_players_per_coach,
            "min_coaches_needed": min_coaches_needed,
            "rule_respected": rule_respected,
        },
    }


# -----------------------------
# Grouping logic - KMeans
# -----------------------------
def build_feature_vector_for_player(player: dict[str, Any]) -> list[float]:
    """
    Create a numeric feature vector for each player.
    Uses:
      - age (years)
      - skillLevel (if numeric)
      - attendanceRate (if numeric 0..100)
    Missing values are replaced with 0.
    """
    age = calculate_age(player.get("dateOfBirth")) or 0
    skill = 0.0
    attendance = 0.0
    # skillLevel may be numeric or string (beginner/intermediate/advanced)
    skl = player.get("skillLevel")
    if isinstance(skl, (int, float)):
        skill = float(skl)
    elif isinstance(skl, str):
        mapping = {"beginner": 1.0, "intermediate": 2.0, "advanced": 3.0}
        skill = mapping.get(skl.lower(), 0.0)

    ar = player.get("attendanceRate")
    try:
        if ar is not None:
            attendance = float(ar)
    except Exception:
        attendance = 0.0

    return [float(age), float(skill), float(attendance)]


def group_kmeans(players: list[dict[str, Any]], coaches: list[dict[str, Any]], k: int | None = None) -> dict[str, Any]:
    """
    KMeans grouping:
      - Build feature vectors for players
      - Choose k (number of clusters):
          * if k provided use it
          * else, default to min(number of coaches, ceil(players / 4)) if coaches exist, otherwise  max(1, ceil(sqrt(players)))
      - Run KMeans and assign players to clusters
      - Assign a coach to each cluster if coaches available (round-robin)
    """
    if KMeans is None:
        return {"error": "sklearn_missing", "message": "scikit-learn is required for k-means grouping. pip install scikit-learn"}

    players_clean = [clean_document(p) for p in players]
    coaches_clean = [clean_document(c) for c in coaches]

    n_players = len(players_clean)
    if n_players == 0:
        return {"clusters": [], "message": "no_players"}

    # Determine default k
    if k is None:
        if len(coaches_clean) > 0:
            default_k = min(len(coaches_clean), math.ceil(n_players / 4))
            k = max(1, default_k)
        else:
            k = max(1, int(math.ceil(math.sqrt(n_players))))

    # Do not request more clusters than players
    k = max(1, min(k, n_players))

    # Build X matrix
    X = [build_feature_vector_for_player(p) for p in players_clean]

    # Run kmeans
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)

    clusters_map: dict[int, list[dict[str, Any]]] = {}
    for idx, label in enumerate(labels):
        clusters_map.setdefault(int(label), []).append(players_clean[idx])

    clusters_result = []
    for i in range(k):
        cluster_players = clusters_map.get(i, [])
        assigned_coach = None
        if coaches_clean:
            assigned_coach = coaches_clean[i % len(coaches_clean)]
        clusters_result.append({
            "cluster": i,
            "coach": assigned_coach,
            "players": cluster_players,
            "size": len(cluster_players),
        })

    return {
        "clusters": clusters_result,
        "k": k,
        "n_players": n_players,
        "n_coaches": len(coaches_clean),
    }


# -----------------------------
# Endpoints: Rule-based & KMeans
# -----------------------------
@app.get("/api/organizations/<string:org_id>/group/rule-based")
def group_rule_based_endpoint(org_id: str):
    org_oid = try_parse_object_id(org_id)
    if org_oid is None:
        return make_response(jsonify({"error": "invalid_id", "message": f"{org_id} is not a valid ObjectId."}), 400)

    db = get_db()
    organization = db["organizations"].find_one({"_id": org_oid})
    if organization is None:
        return make_response(jsonify({"error": "not_found", "message": f"Organization {org_id} was not found."}), 404)

    player_ids = extract_object_ids(organization.get("players", []))
    coach_ids = extract_object_ids(organization.get("coaches", []))

    players = fetch_users_by_ids(player_ids)
    coaches = fetch_users_by_ids(coach_ids)

    # allow overriding max players per coach via query param
    try:
        max_ppc = int(request.args.get("max_players_per_coach", 4))
    except Exception:
        max_ppc = 4

    result = group_rule_based(players, coaches, max_players_per_coach=max_ppc)
    return jsonify(result)


@app.get("/api/organizations/<string:org_id>/group/kmeans")
def group_kmeans_endpoint(org_id: str):
    org_oid = try_parse_object_id(org_id)
    if org_oid is None:
        return make_response(jsonify({"error": "invalid_id", "message": f"{org_id} is not a valid ObjectId."}), 400)

    db = get_db()
    organization = db["organizations"].find_one({"_id": org_oid})
    if organization is None:
        return make_response(jsonify({"error": "not_found", "message": f"Organization {org_id} was not found."}), 404)

    player_ids = extract_object_ids(organization.get("players", []))
    coach_ids = extract_object_ids(organization.get("coaches", []))

    players = fetch_users_by_ids(player_ids)
    coaches = fetch_users_by_ids(coach_ids)

    # read k from query param
    k_param = request.args.get("k")
    try:
        k = int(k_param) if k_param is not None else None
    except Exception:
        k = None

    result = group_kmeans(players, coaches, k=k)
    return jsonify(result)


# -----------------------------
# Gemini Prompt Endpoint
# -----------------------------
# Behavior:
# - Accepts JSON: { "prompt": "...", "mode": "rule" | "kmeans", "params": {...} }
# - If GEMINI_API_KEY configured and the SDK is available, it will call Gemini to generate JSON rules.
# - Otherwise it returns a simulated rule suggestion based on the prompt.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


@app.post("/api/grouping/prompt")
def group_with_gemini_rules():
    body = request.json or {}
    user_prompt = body.get("prompt")
    mode = body.get("mode", "rule")  # "rule" or "kmeans"
    params = body.get("params", {})

    if not user_prompt:
        return make_response(jsonify({"error": "missing_prompt", "message": "Please provide a prompt in the request body."}), 400)

    # If GEMINI_API_KEY is provided AND SDK installed, call the LLM to produce JSON rules.
    if GEMINI_API_KEY:
        # Example: pseudo-code for calling google generative ai
        # Uncomment and configure if you have the SDK installed
        """
        genai.configure(api_key=GEMINI_API_KEY)
        model = "gemini-pro"  # adjust model name if needed

        system = (
            "You are a grouping-rule generator. Parse the user's request and produce a JSON object with fields:\n"
            "  - mode: 'rule' or 'kmeans'\n"
            "  - rules: { ... }\n"
            "  - params: suggested params for grouping\n            " 
        )

        prompt_full = f"{system}\nUser prompt: {user_prompt}\nReturn ONLY JSON."

        resp = genai.generate_text(model=model, prompt=prompt_full)
        # resp.text expected to be JSON
        try:
            rules_json = json.loads(resp.text)
        except Exception:
            rules_json = {"raw": resp.text}
        return jsonify({"from": "gemini", "rules": rules_json})
        """
        # If you want me to enable this call for you, tell me and provide the environment variable.
        # For now, if GEMINI_API_KEY is set but SDK not enabled, we still proceed to local simulation below.
        pass

    # Local/simulated fallback: create a simple JSON rules structure
    # Try to interpret user prompt heuristically (best-effort)
    suggested = {
        "mode": mode,
        "rules": {},
        "params": {},
    }

    # Basic heuristics:
    p = user_prompt.lower()
    if "age" in p:
        suggested["rules"]["group_by"] = "age"
        suggested["params"]["age_bins"] = params.get("age_bins", [6, 12, 18, 25, 40])
    if "skill" in p or "experience" in p:
        suggested["rules"]["group_by"] = suggested["rules"].get("group_by", "skill")
        suggested["params"]["skill_preference"] = params.get("skill_preference", "mix")  # mix | segregate
    if "balanced" in p or "even" in p:
        suggested["rules"]["balance"] = True
    if "coach" in p:
        suggested["params"]["max_players_per_coach"] = params.get("max_players_per_coach", 4)

    # default fallback
    if not suggested["rules"]:
        suggested["rules"]["group_by"] = "default"
        suggested["params"]["max_players_per_coach"] = params.get("max_players_per_coach", 4)

    return jsonify({"from": "simulated", "prompt": user_prompt, "suggested": suggested})


if __name__ == "__main__":
    app.run(debug=True)
