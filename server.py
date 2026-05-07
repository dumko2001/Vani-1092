"""FastAPI server — REST API + operator dashboard + LiveKit token endpoint.

Run alongside the LiveKit agent:
    python server.py

Provides:
- /                  → API info
- /client            → Citizen web client (LiveKit JS)
- /dashboard         → Operator console HTML
- /api/token         → LiveKit access token (for client auth)
- /api/calls         → Active calls JSON
- /api/calls/{id}/accept → Assign call to operator
- /health            → Health check
"""

import json
import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import vani1092.database as db

app = FastAPI(title="Vani-1092 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


def _has_livekit_creds() -> bool:
    return all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET])


def _generate_token(room: str, identity: str) -> str:
    """Generate a LiveKit access token."""
    try:
        from livekit import api
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(identity)
        token.with_name(identity)
        token.with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        return token.to_jwt()
    except ImportError:
        # Fallback: manual JWT generation
        import jwt
        now = datetime.now(timezone.utc)
        payload = {
            "iss": LIVEKIT_API_KEY,
            "sub": identity,
            "nbf": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "video": {
                "roomJoin": True,
                "room": room,
                "canPublish": True,
                "canSubscribe": True,
            },
        }
        return jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")


@app.get("/")
async def root():
    return {
        "service": "vani-1092-api",
        "livekit_configured": _has_livekit_creds(),
        "endpoints": ["/client", "/dashboard", "/api/token", "/api/calls", "/health"],
    }


@app.get("/client")
async def client_page():
    try:
        html = open("static/index.html").read()
        # Inject LiveKit URL into client
        html = html.replace("wss://your-project.livekit.cloud", LIVEKIT_URL)
    except FileNotFoundError:
        html = "<h1>Client not found</h1>"
    return HTMLResponse(content=html)


@app.get("/dashboard")
async def dashboard_page():
    try:
        html = open("static/dashboard.html").read()
    except FileNotFoundError:
        html = "<h1>Dashboard not found</h1>"
    return HTMLResponse(content=html)


@app.get("/api/token")
async def api_token(room: str = Query(...), identity: str = Query(...)):
    """Generate a LiveKit access token for the web client."""
    if not _has_livekit_creds():
        return {"error": "LiveKit credentials not configured"}, 500
    jwt_token = _generate_token(room, identity)
    return {"token": jwt_token, "url": LIVEKIT_URL}


@app.get("/api/calls")
async def api_calls():
    return db.list_active_calls()


@app.get("/api/calls/{call_id}")
async def api_call_detail(call_id: str):
    return db.get_call(call_id)


@app.post("/api/calls/{call_id}/accept")
async def api_accept_call(call_id: str, operator_id: str = "op1"):
    db.assign_call_to_operator(call_id, operator_id)
    db.update_call(call_id, status="with_operator")
    return {"status": "assigned", "call_id": call_id}


@app.post("/api/calls/{call_id}/close")
async def api_close_call(call_id: str):
    db.update_call(call_id, status="closed", ended_at=datetime.now(timezone.utc).isoformat())
    return {"status": "closed", "call_id": call_id}


@app.post("/api/calls/{call_id}/feedback")
async def api_feedback(
    call_id: str,
    operator_id: str = Query("op1"),
    ai_was_correct: bool = Query(...),
    corrected_intent: str = Query(""),
    notes: str = Query(""),
):
    db.add_feedback(
        call_id=call_id,
        operator_id=operator_id,
        ai_was_correct=ai_was_correct,
        corrected_intent=corrected_intent,
        notes=notes,
    )
    return {"status": "saved", "call_id": call_id}


@app.get("/api/operators")
async def api_operators():
    return db.list_operators()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "vani-1092-api"}


if __name__ == "__main__":
    db.init_db()
    db.create_operator("op1", "Priya Sharma", ["hi", "en", "kn"])
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
