"""Security layer: least-privilege kiosk credentials, single-use sessions, threat-adaptive mode, alerts.

CONTRACT:
    kiosk_credentials() -> (credentials_dict, mode)   mode: "least-privilege" | "dev-full"
    ensure_kiosk_role() -> role_arn                    creates the minimal IAM role if missing
    register_session(session_id) -> None
    claim_session(session_id) -> (ok: bool, reason: str)   single use, max SESSION_MAX_AGE_SECONDS old
    threat_status(now=None) -> {"active": bool, "spoof_attempts": int, "until": str | None,
                                "targeted_ids": list[str], "reason": str}
    on_decision(result) -> None                        call after every logged decision (raises alerts)
    alerts(limit=50) -> list[dict]
    reset_threat() -> None                             # a guard confirms the threat is handled
    last_allow(person_id, now=None) -> {"location": str, "minutes_ago": float} | None
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta

import boto3

import config
import store

ALERTS_FILE = os.path.join(config.DATA_DIR, "alerts.json")
RESET_FILE = os.path.join(config.DATA_DIR, "threat_reset.txt")
_sessions = {}
_lock = threading.Lock()

# Only permission the browser ever needs: stream the liveness video to Rekognition.
KIOSK_POLICY = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Action": "rekognition:StartFaceLivenessSession",
                   "Resource": "*"}],
}


# ------------------------------------------------------------ least-privilege credentials

def _role_arn():
    account = boto3.client("sts", region_name=config.REGION).get_caller_identity()["Account"]
    return f"arn:aws:iam::{account}:role/{config.KIOSK_ROLE_NAME}"


def ensure_kiosk_role():
    """Create the kiosk role (only StartFaceLivenessSession) if it does not exist."""
    iam = boto3.client("iam")
    account = boto3.client("sts", region_name=config.REGION).get_caller_identity()["Account"]
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"AWS": f"arn:aws:iam::{account}:root"},
        "Action": "sts:AssumeRole"}]}
    try:
        role = iam.get_role(RoleName=config.KIOSK_ROLE_NAME)["Role"]
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(RoleName=config.KIOSK_ROLE_NAME,
                               AssumeRolePolicyDocument=json.dumps(trust),
                               Description="HID kiosk browser: may only start a Face Liveness session",
                               MaxSessionDuration=3600)["Role"]
    iam.put_role_policy(RoleName=config.KIOSK_ROLE_NAME, PolicyName="FaceLivenessOnly",
                        PolicyDocument=json.dumps(KIOSK_POLICY))
    return role["Arn"]


def kiosk_credentials():
    """15-minute credentials that can ONLY start a liveness session. Falls back to dev mode."""
    try:
        resp = boto3.client("sts", region_name=config.REGION).assume_role(
            RoleArn=_role_arn(), RoleSessionName="hid-kiosk", DurationSeconds=900,
            Policy=json.dumps(KIOSK_POLICY))
        c = resp["Credentials"]
        return ({"accessKeyId": c["AccessKeyId"], "secretAccessKey": c["SecretAccessKey"],
                 "sessionToken": c["SessionToken"]}, "least-privilege")
    except Exception:
        frozen = boto3.Session().get_credentials().get_frozen_credentials()
        creds = {"accessKeyId": frozen.access_key, "secretAccessKey": frozen.secret_key}
        if frozen.token:
            creds["sessionToken"] = frozen.token
        return creds, "dev-full"


# ------------------------------------------------------------ single-use sessions

def register_session(session_id):
    with _lock:
        _sessions[session_id] = {"issued": time.time(), "used": False}


def claim_session(session_id):
    with _lock:
        s = _sessions.get(session_id)
        if s is None:
            return False, "Unknown liveness session (not issued by this kiosk)."
        if s["used"]:
            return False, "This liveness session was already used (replay blocked)."
        if time.time() - s["issued"] > config.SESSION_MAX_AGE_SECONDS:
            return False, "This liveness session expired. Please start again."
        s["used"] = True
        return True, "ok"


# ------------------------------------------------------------ threat-adaptive mode

def _is_spoof(e):
    return e.get("verdict") == "DENY" and (e.get("liveness") or {}).get("live") is False


def reset_threat():
    os.makedirs(os.path.dirname(RESET_FILE) or ".", exist_ok=True)
    with open(RESET_FILE, "w", encoding="utf-8") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def _reset_time():
    try:
        with open(RESET_FILE, encoding="utf-8") as f:
            return datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
    except (FileNotFoundError, ValueError):
        return datetime.min


def threat_status(now=None):
    now = now or datetime.now()
    since = now - timedelta(minutes=config.THREAT_WINDOW_MINUTES)
    reset_at = _reset_time()
    recent = []
    for e in store.recent_events(300):
        try:
            t = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            continue
        if t >= since and t > reset_at and _is_spoof(e):
            recent.append((t, e))
    if not recent:
        return {"active": False, "spoof_attempts": 0, "until": None, "targeted_ids": [],
                "reason": "No recent attacks."}
    last = max(t for t, _ in recent)
    targeted = sorted({e["person_id"] for _, e in recent if e.get("person_id")})
    until = (last + timedelta(minutes=config.THREAT_WINDOW_MINUTES)).strftime("%H:%M")
    return {"active": True, "spoof_attempts": len(recent), "until": until, "targeted_ids": targeted,
            "reason": f"{len(recent)} spoof attempt(s) in the last {config.THREAT_WINDOW_MINUTES} min: "
                      f"heightened security until {until}."}


# ------------------------------------------------------------ alerts

def _load_alerts():
    try:
        with open(ALERTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _raise_alert(kind, message, event):
    alert = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": kind,
             "message": message, "person_id": event.get("person_id"),
             "location": event.get("location")}
    items = _load_alerts()
    items.append(alert)
    os.makedirs(os.path.dirname(ALERTS_FILE) or ".", exist_ok=True)
    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(items[-500:], f, indent=2)
    if config.ALERT_SNS_TOPIC_ARN:   # production: push to the guards' phones / email
        try:
            boto3.client("sns", region_name=config.REGION).publish(
                TopicArn=config.ALERT_SNS_TOPIC_ARN, Subject=f"HID kiosk alert: {kind}",
                Message=message)
        except Exception:
            pass
    return alert


def last_allow(person_id, now=None):
    now = now or datetime.now()
    for e in store.recent_events(300):
        if e.get("person_id") != person_id or e.get("verdict") != "ALLOW" or not e.get("location"):
            continue
        try:
            t = datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, ValueError):
            continue
        if t > now:
            continue
        return {"location": e["location"], "minutes_ago": (now - t).total_seconds() / 60}
    return None


def raise_alert(kind, message, event):
    return _raise_alert(kind, message, event)


def on_decision(result):
    if result.get("flag") == "impossible_travel":
        _raise_alert("impossible_travel", " ".join(result.get("reasons", [])), result)
    if _is_spoof(result):
        kind = (result.get("liveness") or {}).get("spoof_type") or "presentation_attack"
        who = f" (looked like {result['person_id']})" if result.get("person_id") else ""
        _raise_alert("spoof_attempt", f"Spoof attempt ({kind}) at {result.get('location')}{who}.",
                     result)


def alerts(limit=50):
    return list(reversed(_load_alerts()[-limit:]))
