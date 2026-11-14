"""Kiosk server: serves the kiosk web page and connects it to the decision pipeline.

Run:   python server.py        then open  http://localhost:8000

Flow:  browser camera -> AWS Rekognition Face Liveness (liveness + spoof detection)
       -> /api/decide: face authentication (Rekognition) + campus schedule + rules
       -> ALLOW / STEP-UP VERIFICATION / DENY, logged, explained by Bedrock.
Offline: if AWS is unreachable, the page switches to the safe offline process (offline.py).
"""
import os
import time

import boto3
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import offline
import official_liveness
import pipeline
import store

app = FastAPI(title="HID Liveness Kiosk")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiosk_web", "dist")


class DecideIn(BaseModel):
    session_id: str
    location: str | None = None


class OverrideIn(BaseModel):
    person_id: str | None = None
    verdict: str
    note: str = ""


class OfflineIn(BaseModel):
    person_id: str
    code: str


class SimulateIn(BaseModel):
    offline: bool


def _offline_error():
    offline.mark_offline()
    return JSONResponse(status_code=503, content={"detail": "Cloud unreachable: switched to offline mode.",
                                                  "offline": True})


@app.get("/api/status")
def status():
    online = offline.is_online()
    synced = offline.sync() if online and offline.pending() else 0
    return {"online": online, "simulated_offline": offline.is_simulated_offline(),
            "location": config.KIOSK_LOCATION, "pending_sync": len(offline.pending()),
            "just_synced": synced}


@app.post("/api/session")
def create_session():
    if not offline.is_online():
        return _offline_error()
    try:
        session_id = official_liveness.create_session()
    except pipeline.NETWORK_ERRORS:
        return _offline_error()
    creds = boto3.Session().get_credentials()
    if creds is None:
        raise HTTPException(500, "No AWS credentials in this terminal. Paste the sandbox $Env: lines.")
    frozen = creds.get_frozen_credentials()
    credentials = {"accessKeyId": frozen.access_key, "secretAccessKey": frozen.secret_key}
    if frozen.token:
        credentials["sessionToken"] = frozen.token
    return {"sessionId": session_id, "region": config.REGION, "credentials": credentials}


@app.post("/api/decide")
def decide(body: DecideIn):
    t0 = time.perf_counter()
    try:
        live_result, reference = official_liveness.get_result(body.session_id)
        fetch_ms = int((time.perf_counter() - t0) * 1000)
        if reference is None:
            result = {"verdict": "RETRY", "verification": None, "reasons": live_result["reasons"],
                      "person_id": None, "similarity": None, "liveness": live_result,
                      "schedule": None, "decision_ms": fetch_ms, "timing": {"liveness_result": fetch_ms},
                      "explanation": "The liveness check did not finish. Please try again.",
                      "mode": "online"}
            return result
        result = pipeline.check_official(live_result, reference, use_ai=True,
                                         location=body.location)
    except pipeline.OfflineError:
        return _offline_error()
    except pipeline.NETWORK_ERRORS:
        return _offline_error()
    result["timing"] = {"liveness_result": fetch_ms, **result["timing"]}
    result["decision_ms"] = fetch_ms + result["decision_ms"]   # post-capture decision time
    result["timing"]["total"] = result["decision_ms"]
    result.pop("challenge", None)
    return result


@app.post("/api/override")
def override(body: OverrideIn):
    return pipeline.record_override(body.person_id, body.verdict, by="guard", note=body.note)


@app.post("/api/offline/decide")
def offline_decide(body: OfflineIn):
    from datetime import datetime
    return offline.offline_decide(body.person_id, body.code, config.KIOSK_LOCATION,
                                  datetime.now(), faces_in_view=1)


@app.post("/api/offline/simulate")
def simulate(body: SimulateIn):
    offline.set_simulated_offline(body.offline)
    return status()


@app.post("/api/offline/sync")
def sync():
    return {"synced": offline.sync(), "pending": len(offline.pending())}


@app.get("/api/log")
def log(limit: int = 50):
    return store.recent_events(limit)


if os.path.isdir(WEB_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))


if __name__ == "__main__":
    print("\n  Kiosk running:  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
