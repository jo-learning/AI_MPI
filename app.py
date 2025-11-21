from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
import requests
import google.generativeai as genai

try:  # Optional dependency for clustering endpoint
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - sklearn may be missing
    KMeans = None

app = Flask(__name__)
CORS(app)

DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------

# Data loading utilities
def _read_json(path: Path) -> Any:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return json.loads(raw)


def _transform_extended_json(value: Any) -> Any:
    """Convert MongoDB extended JSON structures into plain Python types."""
    if isinstance(value, dict):
        if "$oid" in value and len(value) == 1:
            return str(value["$oid"])
        if "$date" in value and len(value) == 1:
            raw = value["$date"]
            if isinstance(raw, dict) and "$numberLong" in raw:
                milliseconds = int(raw["$numberLong"])
                return datetime.utcfromtimestamp(milliseconds / 1000).isoformat() + "Z"
            return str(raw)
        return {key: _transform_extended_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_transform_extended_json(item) for item in value]
    return value


def _load_collection(filename: str) -> list[dict[str, Any]]:
    data = _read_json(DATA_DIR / filename)
    if not isinstance(data, list):
        return []
    transformed: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            transformed.append(_transform_extended_json(entry))
    return transformed


@lru_cache(maxsize=1)
def load_organizations() -> list[dict[str, Any]]:
    return _load_collection("organizations.json")


@lru_cache(maxsize=1)
def load_players() -> list[dict[str, Any]]:
    return _load_collection("players.json")


@lru_cache(maxsize=1)
def load_coaches() -> list[dict[str, Any]]:
    return _load_collection("coaches.json")


@lru_cache(maxsize=1)
def load_courts() -> list[dict[str, Any]]:
    return _load_collection("courts.json")


# ---------------------------------------------------------------------------
# ID helpers and serialization
def coerce_object_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        if "$oid" in value:
            raw = value["$oid"]
            return raw if isinstance(raw, str) else None
        if "_id" in value:
            return coerce_object_id(value["_id"])
        if "id" in value:
            return coerce_object_id(value["id"])
    return None


def extract_object_ids(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    seen: set[str] = set()
    ids: list[str] = []
    for value in values:
        oid = coerce_object_id(value)
        if oid and oid not in seen:
            seen.add(oid)
            ids.append(oid)
    return ids


@lru_cache(maxsize=1)
def players_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for player in load_players():
        player_id = coerce_object_id(player.get("_id"))
        if player_id:
            index[player_id] = player
    return index


@lru_cache(maxsize=1)
def coaches_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for coach in load_coaches():
        coach_id = coerce_object_id(coach.get("_id"))
        if coach_id:
            index[coach_id] = coach
    return index


@lru_cache(maxsize=1)
def courts_by_org_index() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for court in load_courts():
        org_ref = court.get("organization")
        org_id = coerce_object_id(org_ref)
        if org_id:
            grouped.setdefault(org_id, []).append(court)
    return grouped


def clean_document(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: clean_document(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_document(item) for item in value]
    if isinstance(value, tuple):
        return [clean_document(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Fetch helpers
def get_organization_by_id(org_id: str) -> dict[str, Any] | None:
    for org in load_organizations():
        candidate_id = coerce_object_id(org.get("_id"))
        if candidate_id == org_id:
            return org
    return None


def fetch_players_by_ids(user_ids: list[str]) -> list[dict[str, Any]]:
    index = players_index()
    return [deepcopy(index[user_id]) for user_id in user_ids if user_id in index]


def fetch_coaches_by_ids(user_ids: list[str]) -> list[dict[str, Any]]:
    index = coaches_index()
    return [deepcopy(index[user_id]) for user_id in user_ids if user_id in index]


def fetch_courts_for_organization(org_id: str) -> list[dict[str, Any]]:
    specific_path = DATA_DIR / f"courts_{org_id}.json"
    if specific_path.exists():
        return _load_collection(specific_path.name)
    grouped = courts_by_org_index()
    return deepcopy(grouped.get(org_id, []))


# ---------------------------------------------------------------------------
# Response payload builders
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


# ---------------------------------------------------------------------------
# API endpoints


@app.get("/api/organizations")
def list_organizations():
    expand_param = request.args.get("expand", "")
    expand_parts = {part.strip().lower() for part in expand_param.split(",") if part.strip()}

    organizations_payload: list[dict[str, Any]] = []
    for organization in load_organizations():
        org_id = coerce_object_id(organization.get("_id"))
        if not org_id:
            continue

        player_ids = extract_object_ids(organization.get("players", []))
        coach_ids = extract_object_ids(organization.get("coaches", []))

        entry: dict[str, Any] = {
            "organization": clean_document(organization),
            "counts": {
                "players": len(player_ids),
                "coaches": len(coach_ids),
                "courts": 0,
            },
        }

        courts: list[dict[str, Any]] | None = None
        if "courts" in expand_parts:
            courts = fetch_courts_for_organization(org_id)
            entry["courts"] = clean_document(courts)
        else:
            courts_indexed = courts_by_org_index()
            courts = courts_indexed.get(org_id, [])

        entry["counts"]["courts"] = len(courts or [])

        if "players" in expand_parts:
            entry["players"] = clean_document(fetch_players_by_ids(player_ids))
        if "coaches" in expand_parts:
            entry["coaches"] = clean_document(fetch_coaches_by_ids(coach_ids))

        organizations_payload.append(entry)

    response = {
        "items": organizations_payload,
        "count": len(organizations_payload),
        "expand": sorted(expand_parts),
    }
    return jsonify(response)


def _organization_detail_response(org_id: str):
    organization = get_organization_by_id(org_id)
    if organization is None:
        message = {"error": "not_found", "message": f"Organization {org_id} was not found."}
        return make_response(jsonify(message), 404)

    player_ids = extract_object_ids(organization.get("players", []))
    coach_ids = extract_object_ids(organization.get("coaches", []))

    players = fetch_players_by_ids(player_ids)
    coaches = fetch_coaches_by_ids(coach_ids)
    courts = fetch_courts_for_organization(org_id)

    response_payload = build_response_payload(
        organization,
        players=players,
        coaches=coaches,
        courts=courts,
    )
    return jsonify(response_payload)


@app.get("/api/organizations/<string:org_id>")
def get_organization(org_id: str):
    return _organization_detail_response(org_id)


@app.get("/api/organizations/<string:org_id>/details")
def get_organization_details(org_id: str):
    return _organization_detail_response(org_id)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Grouping logic - Rule-based


def calculate_age(dob_iso: str | None) -> int | None:
    if not dob_iso:
        return None
    try:
        cleaned = dob_iso.replace("Z", "")
        date_part = cleaned.split("T")[0]
        dob = datetime.fromisoformat(date_part)
        today = datetime.utcnow().date()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None


def group_rule_based(
    players: list[dict[str, Any]],
    coaches: list[dict[str, Any]],
    max_players_per_coach: int = 4,
) -> dict[str, Any]:
    players_clean = [clean_document(p) for p in players]
    coaches_clean = [clean_document(c) for c in coaches]

    total_players = len(players_clean)
    total_coaches = len(coaches_clean)

    if total_players == 0:
        return {"groups": [], "message": "no_players"}

    if total_coaches == 0:
        return {"error": "no_coaches", "message": "No coaches available to assign."}

    min_coaches_needed = math.ceil(total_players / max_players_per_coach)
    groups_count = min(total_coaches, max(1, min_coaches_needed))
    selected_coaches = coaches_clean[:groups_count]

    chunk_size = math.ceil(total_players / groups_count)
    if total_coaches >= min_coaches_needed:
        chunk_size = max_players_per_coach

    groups: list[dict[str, Any]] = []
    for i in range(groups_count):
        start = i * chunk_size
        end = start + chunk_size
        group_players = players_clean[start:end]
        coach = selected_coaches[i]
        groups.append({"coach": coach, "players": group_players})

    assigned = sum(len(group["players"]) for group in groups)
    if assigned < total_players:
        remaining = players_clean[assigned:]
        groups[-1]["players"].extend(remaining)

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


# ---------------------------------------------------------------------------
# Grouping logic - KMeans


def build_feature_vector_for_player(player: dict[str, Any]) -> list[float]:
    age = calculate_age(player.get("dateOfBirth")) or 0
    skill = 0.0
    attendance = 0.0

    skill_level = player.get("skillLevel")
    if isinstance(skill_level, (int, float)):
        skill = float(skill_level)
    elif isinstance(skill_level, str):
        mapping = {"beginner": 1.0, "intermediate": 2.0, "advanced": 3.0}
        skill = mapping.get(skill_level.lower(), 0.0)

    attendance_rate = player.get("attendanceRate")
    try:
        if attendance_rate is not None:
            attendance = float(attendance_rate)
    except Exception:
        attendance = 0.0

    return [float(age), float(skill), float(attendance)]


def group_kmeans(
    players: list[dict[str, Any]],
    coaches: list[dict[str, Any]],
    k: int | None = None,
) -> dict[str, Any]:
    if KMeans is None:
        return {
            "error": "sklearn_missing",
            "message": "scikit-learn is required for k-means grouping. Install it with 'pip install scikit-learn'",
        }

    players_clean = [clean_document(p) for p in players]
    coaches_clean = [clean_document(c) for c in coaches]

    n_players = len(players_clean)
    if n_players == 0:
        return {"clusters": [], "message": "no_players"}

    if k is None:
        if coaches_clean:
            default_k = min(len(coaches_clean), math.ceil(n_players / 4))
            k = max(1, default_k)
        else:
            k = max(1, int(math.ceil(math.sqrt(n_players))))

    k = max(1, min(k, n_players))

    X = [build_feature_vector_for_player(player) for player in players_clean]

    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)

    clusters_map: dict[int, list[dict[str, Any]]] = {}
    for idx, label in enumerate(labels):
        clusters_map.setdefault(int(label), []).append(players_clean[idx])

    clusters_result: list[dict[str, Any]] = []
    for i in range(k):
        cluster_players = clusters_map.get(i, [])
        assigned_coach = coaches_clean[i % len(coaches_clean)] if coaches_clean else None
        clusters_result.append(
            {
                "cluster": i,
                "coach": assigned_coach,
                "players": cluster_players,
                "size": len(cluster_players),
            }
        )

    return {
        "clusters": clusters_result,
        "k": k,
        "n_players": n_players,
        "n_coaches": len(coaches_clean),
    }


# ---------------------------------------------------------------------------
# Grouping endpoints


def _resolve_org_entities(org_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    organization = get_organization_by_id(org_id)
    if organization is None:
        return None, [], []
    player_ids = extract_object_ids(organization.get("players", []))
    coach_ids = extract_object_ids(organization.get("coaches", []))
    players = fetch_players_by_ids(player_ids)
    coaches = fetch_coaches_by_ids(coach_ids)
    return organization, players, coaches


@app.get("/api/organizations/<string:org_id>/group/rule-based")
def group_rule_based_endpoint(org_id: str):
    organization, players, coaches = _resolve_org_entities(org_id)
    if organization is None:
        return make_response(jsonify({"error": "not_found", "message": f"Organization {org_id} was not found."}), 404)

    try:
        max_ppc = int(request.args.get("max_players_per_coach", 4))
    except Exception:
        max_ppc = 4

    result = group_rule_based(players, coaches, max_players_per_coach=max_ppc)
    return jsonify(result)


@app.get("/api/organizations/<string:org_id>/group/kmeans")
def group_kmeans_endpoint(org_id: str):
    organization, players, coaches = _resolve_org_entities(org_id)
    if organization is None:
        return make_response(jsonify({"error": "not_found", "message": f"Organization {org_id} was not found."}), 404)

    k_param = request.args.get("k")
    try:
        k = int(k_param) if k_param is not None else None
    except Exception:
        k = None

    result = group_kmeans(players, coaches, k=k)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Gemini prompt endpoint (simulated fallback)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_gemini_extraction(user_prompt):
    """
    Asks Gemini to convert natural language into structured JSON params.
    """
    if not GEMINI_API_KEY:
        return None

    # SYSTEM INSTRUCTION: We tell Gemini strictly what to output
    system_instruction = """
    You are an API helper. Extract grouping parameters from the user's prompt.
    Output ONLY valid JSON. No markdown, no conversational text.
    
    Possible keys:
    - "group_by": "age", "skill", "random"
    - "mode": "rule", "kmeans"
    - "params": {
        "age_bins": list of integers,
        "max_players_per_coach": integer,
        "k": integer (for kmeans)
    }
    """
    
    full_prompt = f"{system_instruction}\n\nUser Prompt: {user_prompt}"

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"} # Forces JSON output
    }

    # url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    # url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    # CHANGE THIS LINE
    # url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(full_prompt)
        # print(response)
        # response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract text and parse JSON
        text_response = data['candidates'][0]['content']['parts'][0]['text']
        return json.loads(text_response)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

@app.post("/api/grouping/prompt")
def group_with_gemini_rules():
    body = request.json or {}
    user_prompt = body.get("prompt")
    organization_id = body.get("organizationId") or body.get("organization_id")

    if not user_prompt:
        return make_response(jsonify({"error": "missing_prompt", "message": "Missing prompt."}), 400)
    if not organization_id:
        return make_response(jsonify({"error": "missing_org", "message": "Missing organizationId."}), 400)

    # 1. Load Data
    organization, players, coaches = _resolve_org_entities(organization_id)
    if not organization:
        return make_response(jsonify({"error": "not_found"}), 404)
        
    courts = fetch_courts_for_organization(organization_id)

    # 2. Initialize Default/Simulated Logic
    suggested = {
        "source": "simulated",
        "mode": "rule",
        "rules": {"group_by": "default"},
        "params": {"max_players_per_coach": 4}
    }

    # 3. Try Gemini Extraction
    ai_extraction = get_gemini_extraction(user_prompt)

    if ai_extraction:
        # OVERWRITE simulated logic with AI logic
        suggested["source"] = "gemini"
        if "mode" in ai_extraction:
            suggested["mode"] = ai_extraction["mode"]
        if "group_by" in ai_extraction:
            suggested["rules"]["group_by"] = ai_extraction["group_by"]
        if "params" in ai_extraction:
            suggested["params"].update(ai_extraction["params"])
    else:
        # Fallback to simple string matching if AI fails or no key
        prompt_lower = user_prompt.lower()
        if "age" in prompt_lower:
            suggested["rules"]["group_by"] = "age"
        if "kmeans" in prompt_lower or "cluster" in prompt_lower:
            suggested["mode"] = "kmeans"
            suggested["params"]["k"] = 3 # Default k

    # 4. Merge Params (Body params override AI/Simulated params)
    body_params = body.get("params", {})
    final_params = {**suggested["params"], **body_params}
    
    mode = suggested["mode"]

    # 5. Execute Grouping
    if mode == "kmeans":
        k_val = int(final_params.get("k", 3))
        grouping_result = group_kmeans(players, coaches, k=k_val)
    else:
        max_ppc = int(final_params.get("max_players_per_coach", 4))
        grouping_result = group_rule_based(players, coaches, max_players_per_coach=max_ppc)

    return jsonify({
        "prompt": user_prompt,
        "mode": mode,
        "organization_id": organization_id,
        "suggested": suggested,
        "grouping": grouping_result,
        "context": {
            "player_count": len(players),
            "coach_count": len(coaches)
        }
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9000)
