"""Offline mode with safe access: what the kiosk does when the cloud (AWS) is unreachable.

Rules (fail SAFE, never fail open):
  - Never allow on the face alone (identity can't be checked without AWS).
  - Fallback verification: a 6-digit code from an authenticator app (TOTP, works offline)
    AND today's campus schedule must allow the person AND exactly one face must be in view.
  - Store only minimum temporary data: TOTP secrets + a queue of decisions
    (person, time, location, verdict, method). NO photos. The queue is deleted after sync.
  - When the connection returns, queued decisions are synced into the central access log.

CONTRACT:
    is_online(force=False) -> bool
    set_simulated_offline(flag: bool) -> None
    is_simulated_offline() -> bool
    mark_offline() -> None                       # call when an AWS call fails
    enroll_code(person_id) -> {"secret": str, "otpauth_uri": str}
    has_code(person_id) -> bool
    verify_code(person_id, code) -> bool
    offline_decide(person_id, code, location, now, faces_in_view) -> AccessResult-like dict
    offline_face_decide(face_check, code, location, now) -> AccessResult-like dict
        face_check = local_face.verify(...) : offline face match + random-action liveness
    queue_decision(event) -> dict
    pending() -> list[dict]
    sync() -> int                                # number of decisions synced
"""
import json
import os
import time
from datetime import datetime

import boto3
import pyotp
from botocore.config import Config

import config
import store

SECRETS_FILE = os.path.join(config.DATA_DIR, "offline_secrets.json")
QUEUE_FILE = os.path.join(config.DATA_DIR, "offline_queue.json")

_state = {"simulated": False, "online": None, "checked_at": 0.0}
_probe = None


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------ connection state

def _probe_client():
    global _probe
    if _probe is None:
        _probe = boto3.client("rekognition", region_name=config.REGION, config=Config(
            connect_timeout=2, read_timeout=2, retries={"max_attempts": 0}))
    return _probe


def set_simulated_offline(flag):
    _state["simulated"] = bool(flag)
    _state["checked_at"] = 0.0


def is_simulated_offline():
    return _state["simulated"]


def mark_offline():
    _state["online"] = False
    _state["checked_at"] = time.time()


def is_online(force=False):
    if _state["simulated"]:
        return False
    fresh = time.time() - _state["checked_at"] < config.OFFLINE_CHECK_SECONDS
    if _state["online"] is not None and fresh and not force:
        return _state["online"]
    try:
        _probe_client().list_collections(MaxResults=1)
        _state["online"] = True
    except Exception:
        _state["online"] = False
    _state["checked_at"] = time.time()
    return _state["online"]


# ------------------------------------------------------------ fallback verification (TOTP)

def enroll_code(person_id):
    secrets = _load(SECRETS_FILE, {})
    secret = secrets.get(person_id) or pyotp.random_base32()
    secrets[person_id] = secret
    _save(SECRETS_FILE, secrets)
    uri = pyotp.TOTP(secret).provisioning_uri(name=person_id, issuer_name="HID Kiosk")
    return {"secret": secret, "otpauth_uri": uri}


def has_code(person_id):
    return person_id in _load(SECRETS_FILE, {})


def verify_code(person_id, code):
    secret = _load(SECRETS_FILE, {}).get(person_id)
    code = (code or "").strip().replace(" ", "")
    if not secret or not code.isdigit() or len(code) != 6:
        return False
    return pyotp.TOTP(secret).verify(code, valid_window=1)


# ------------------------------------------------------------ safe local decision

def offline_decide(person_id, code, location, now, faces_in_view):
    start = time.perf_counter()
    reasons = ["OFFLINE MODE: cloud unreachable, using the safe local process."]
    method = "offline_totp+schedule"

    def done(verdict, why):
        result = {
            "verdict": verdict, "reasons": reasons + why, "person_id": person_id or None,
            "similarity": None, "mode": "offline", "method": method, "location": location,
            "liveness": {"live": None, "score": 0.0, "spoof_type": None, "signals": {},
                         "source": "offline", "reasons": []},
            "decision_ms": int((time.perf_counter() - start) * 1000),
            "challenge": None, "explanation": f"{verdict}: " + " ".join(reasons + why),
        }
        queue_decision(result)
        return result

    if faces_in_view != 1:
        return done("DENY", ["Exactly one person must be in front of the camera."])
    person = store.get_person(person_id) if person_id else None
    if not person:
        return done("DENY", ["Unknown ID. Please see the guard."])
    if person.get("status") == "suspended":
        return done("DENY", ["Access is suspended."])
    if not has_code(person_id):
        return done("DENY", ["No offline code set up for this person. Please see the guard."])
    if not verify_code(person_id, code):
        return done("DENY", ["Wrong or expired authenticator code."])
    schedule = store.check_schedule(person_id, location, now)
    if schedule["status"] != "in_schedule":
        return done("DENY", [schedule["reason"], "Offline, only scheduled access is allowed. "
                                                 "Please see the guard."])
    import presence
    inside = presence.current(person_id, now)
    if inside and inside["location"] != location:
        return done("DENY", [f"Checked in at {inside['location']} and not checked out."])
    presence.check_in(person_id, location, now, schedule)
    return done("ALLOW", ["Authenticator code verified.", schedule["reason"]])


# ------------------------------------------------------------ offline face + random action

def offline_face_decide(face_check, code, location, now):
    """Local OpenCV face match + random-action liveness. Fails safe:
    ALLOW only if live AND strong match AND active AND in schedule AND no impossible travel.
    Weak match or near-schedule -> REQUEST_VERIFICATION (6-digit code). Anything else -> DENY."""
    import local_face
    import security
    start = time.perf_counter()
    reasons = ["OFFLINE MODE: cloud unreachable, using the local face model + random-action liveness."]
    method = "offline_face+challenge"
    pid = face_check.get("person_id")
    cos = face_check.get("cosine")
    live = face_check.get("live")
    for a in face_check.get("actions", []):
        reasons.append(f"Action '{a['label']}': {'done' if a['done'] else 'NOT done'}.")
    liveness = {"live": live, "score": 1.0 if live else 0.0,
                "spoof_type": face_check.get("spoof_type") if live is False else None,
                "signals": {"actions": face_check.get("actions", [])},
                "source": "offline_challenge", "reasons": [face_check.get("reason", "")]}

    def done(verdict, why, verification=None, schedule=None, queue=True, flag=None):
        result = {
            "verdict": verdict, "verification": verification, "reasons": reasons + why,
            "person_id": pid,
            "similarity": round(cos * 100, 1) if cos is not None else None,
            "mode": "offline", "method": method, "location": location, "liveness": liveness,
            "schedule": schedule, "flag": flag,
            "decision_ms": int((time.perf_counter() - start) * 1000),
            "challenge": None, "explanation": f"{verdict}: " + " ".join(why),
        }
        if queue:
            queue_decision(result)
        if flag:   # alert locally even while offline
            security.raise_alert(flag, " ".join(result["reasons"]), result)
        return result

    if live is None:
        return done("RETRY", [face_check.get("reason", "Please try again.")], queue=False)
    if live is False:
        return done("DENY", [f"Spoof suspected: {face_check.get('reason', '')}"])
    if not pid or cos is None or cos < local_face.REVIEW_COSINE:
        pid = None
        return done("DENY", ["Face not recognised by the offline model. Please see the guard."])
    person = store.get_person(pid)
    if not person:
        return done("DENY", ["Unknown person. Please see the guard."])
    if person.get("status") == "suspended":
        return done("DENY", ["Access is suspended."])
    import presence
    inside = presence.current(pid, now)
    if inside and inside["location"] != location:
        return done("DENY", [f"{pid} is checked in at {inside['location']} since {inside['since'][11:16]} "
                             "and has not checked out: possible cloned identity."], flag="already_checked_in")
    last = security.last_allow(pid, now)
    if last and last["location"] != location and last["minutes_ago"] < config.IMPOSSIBLE_TRAVEL_MINUTES:
        what = "checked out of" if last.get("direction") == "out" else "allowed at"
        return done("DENY", [f"Impossible travel: {what} {last['location']} only "
                             f"{last['minutes_ago']:.1f} min ago (minimum {config.IMPOSSIBLE_TRAVEL_MINUTES} min "
                             "between labs)."], flag="impossible_travel")
    schedule = store.check_schedule(pid, location, now)
    if schedule["status"] == "out_of_schedule":
        return done("DENY", [schedule["reason"], "Offline, only scheduled access is allowed."],
                    schedule=schedule)
    weak = cos < local_face.MATCH_COSINE
    near = schedule["status"] != "in_schedule"
    if weak or near:
        why = (["Offline face match is not strong enough."] if weak else []) + \
              ([schedule["reason"]] if near else [])
        if code and verify_code(pid, code):
            method = "offline_face+challenge+totp"
            rec = presence.check_in(pid, location, now, schedule)
            return {**done("ALLOW", why + ["Authenticator code verified."], schedule=schedule),
                    "presence": rec, "direction": "in"}
        if code:
            return done("DENY", why + ["Wrong or expired authenticator code."], schedule=schedule)
        return done("REQUEST_VERIFICATION", why + ["Enter your 6-digit authenticator code."],
                    verification="code", schedule=schedule, queue=False)
    rec = presence.check_in(pid, location, now, schedule)
    return {**done("ALLOW", [f"Live face (random actions passed), offline match {pid} "
                             f"({cos * 100:.1f}%).", schedule["reason"]], schedule=schedule),
            "presence": rec, "direction": "in"}


# ------------------------------------------------------------ queue + sync

def queue_decision(event):
    item = {   # minimum temporary information only
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": event.get("person_id"),
        "location": event.get("location"),
        "verdict": event.get("verdict"),
        "method": event.get("method", "offline"),
        "reasons": event.get("reasons", []),
    }
    queue = _load(QUEUE_FILE, [])
    queue.append(item)
    _save(QUEUE_FILE, queue)
    return item


def pending():
    return _load(QUEUE_FILE, [])


def sync():
    if not is_online(force=True):
        return 0
    queue = _load(QUEUE_FILE, [])
    for item in queue:
        store.log_event({
            "verdict": item["verdict"], "reasons": item["reasons"], "person_id": item["person_id"],
            "similarity": None, "liveness": None, "decision_ms": None,
            "explanation": f"[synced offline decision from {item['time']}] {item['verdict']}",
            "mode": "offline", "offline_time": item["time"], "location": item["location"],
        })
    _save(QUEUE_FILE, [])
    return len(queue)
