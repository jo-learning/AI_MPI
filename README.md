# Organization Aggregation API

A lightweight Flask API that serves organization data merged with related players, coaches, and courts sourced from the exported MongoDB JSON files in this directory.

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

```bash
python app.py
```

This starts a development server on `http://127.0.0.1:5000`.

### Endpoints

- `GET /api/health` — Simple readiness probe.
- `GET /api/organizations/<organization_id>` — Returns the organization metadata plus related players, coaches, and courts. Replace `<organization_id>` with the ObjectId string (for example, `690cdba9809365c990983391`).
- `GET /api/organizations/<organization_id>/group/rule-based` — Deterministic player grouping capped by `max_players_per_coach` (query parameter, default `4`).
- `GET /api/organizations/<organization_id>/group/kmeans` — Clusters players via K-Means. Requires `scikit-learn`; responds with an error payload if the package is missing.
- `POST /api/grouping/prompt` — Accepts a free-form prompt and returns a simulated grouping configuration (Gemini stub).

## Response Shape

```json
{
  "organization": { "_id": "...", "name": "...", "createdAt": "2025-11-06T17:32:25.573000" },
  "players": [ { "_id": "...", "firstName": "..." } ],
  "coaches": [ { "_id": "...", "firstName": "..." } ],
  "courts": [ { "_id": "...", "name": "..." } ],
  "counts": { "players": 1, "coaches": 1, "courts": 3 }
}
```

Timestamps are ISO 8601 strings and MongoDB `ObjectId` values are returned as hex strings for JSON compatibility.

## Notes

- Organization records are read from `organizations.json`.
- Matching players and coaches are loaded from the shared `players.json` and `coaches.json` exports and matched by ObjectId; courts are pulled from `courts_<org_id>.json` when available, otherwise from `courts.json`.
- Data is loaded from disk on demand and cached in-memory for subsequent requests, so restarts are required if the JSON files change.
- K-Means grouping is optional; install `scikit-learn` if you plan to use the `/group/kmeans` endpoint.
# AI_MPI
