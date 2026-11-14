"""capture -> liveness -> face authentication -> campus schedule -> decision -> log.
The UI calls ONLY this file for access checks.

CONTRACT:
    check_access(image_bytes, use_ai=True, log=True, location=None, now=None, use_schedule=True)
    verify_challenge(challenge, image_bytes, use_ai=True, log=True, location=None, now=None) -> AccessResult
    check_official(liveness_result, reference_image_bytes, use_ai=True, log=True, location=None, now=None)
    record_override(person_id, verdict, by, note="") -> dict
    warm_up() -> None                   # call once at app start so the first decision is fast
    OfflineError                        # raised when AWS can't be reached -> use offline.py

    AccessResult = {
        "verdict": "ALLOW" | "REQUEST_VERIFICATION" | "DENY" | "RETRY",
        "verification": "challenge" | "guard" | None,
        "reasons": list[str], "person_id": str | None, "similarity": float,
        "liveness": LivenessResult, "schedule": dict | None, "inspector": dict | None,
        "decision_ms": int, "timing": {step: ms}, "challenge": dict | None,
        "explanation": str, "location": str, "mode": "online",
    }
"""
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime

from botocore.exceptions import (ConnectionClosedError, ConnectTimeoutError,
                                 EndpointConnectionError, ReadTimeoutError)

import config
import face_service
import genai
import liveness
import offline
import rules
import security
import store

_pool = ThreadPoolExecutor(max_workers=6)
NETWORK_ERRORS = (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError,
                  ConnectionClosedError)


class OfflineError(Exception):
    """AWS is unreachable: the kiosk must switch to offline mode."""


def _timed(fn, *args):
    t0 = time.perf_counter()
    value = fn(*args)
    return value, int((time.perf_counter() - t0) * 1000)


def warm_up():
    try:
        face_service.client()
        face_service.list_people()
    except Exception:
        pass


def _analyse(image_bytes, with_inspector):
    """Run the AWS calls IN PARALLEL to stay well under 2 seconds."""
    jobs = {
        "identity": _pool.submit(_timed, face_service.identify, image_bytes),
        "face_attributes": _pool.submit(_timed, face_service.detect_faces, image_bytes),
        "object_detection": _pool.submit(_timed, liveness.detect_labels, image_bytes),
    }
    inspector_job = (_pool.submit(_timed, genai.spoof_opinion, image_bytes)
                     if with_inspector and config.USE_GENAI_INSPECTOR else None)
    timing = {}
    try:
        (face_result, timing["identity"]) = jobs["identity"].result()
        (face_details, timing["face_attributes"]) = jobs["face_attributes"].result()
        (labels, timing["object_detection"]) = jobs["object_detection"].result()
    except NETWORK_ERRORS as e:
        offline.mark_offline()
        raise OfflineError(str(e)) from e
    return face_result, face_details, labels, inspector_job, timing


def _apply_inspector(live_result, inspector_job, start, timing):
    """GenAI second opinion. It can only make the decision STRICTER, never more lenient."""
    if inspector_job is None:
        return None
    remaining = config.INSPECTOR_WAIT_SECONDS - (time.perf_counter() - start)
    try:
        opinion, timing["genai_inspector"] = inspector_job.result(timeout=max(0.05, remaining))
    except FutureTimeout:
        timing["genai_inspector"] = "skipped (too slow)"
        return None
    if opinion["spoof"] is True and live_result["live"] is not False:
        if opinion["confidence"] >= config.INSPECTOR_SPOOF_CONFIDENCE:
            live_result["live"] = False
            live_result["score"] = min(live_result["score"], 0.2)
            live_result["spoof_type"] = opinion["spoof_type"] or "presentation_attack"
        elif live_result["live"] is True:
            live_result["live"] = None
        live_result["reasons"] = live_result["reasons"] + [f"GenAI inspector: {opinion['reason']}"]
    return opinion


def _finish(face_result, live_result, face_details, start, timing, context, use_ai, log,
            location, now, inspector=None, use_schedule=True, extra=None):
    location = location or config.KIOSK_LOCATION
    now = now or datetime.now()
    t0 = time.perf_counter()
    person = store.get_person(face_result["person_id"]) if face_result["person_id"] else None
    schedule = (store.check_schedule(face_result["person_id"], location, now)
                if face_result["person_id"] and use_schedule else None)
    threat = security.threat_status(now)
    last = security.last_allow(face_result["person_id"], now) if face_result["person_id"] else None
    decision = rules.decide(face_result, live_result, person,
                            {**context, "schedule": schedule, "threat": threat,
                             "last_allow": last, "location": location})
    timing["schedule_and_rules"] = int((time.perf_counter() - t0) * 1000)
    decision_ms = int((time.perf_counter() - start) * 1000)   # decision time, before GenAI text
    timing["total"] = decision_ms

    challenge = None
    if decision["verification"] == "challenge":
        challenge = liveness.new_challenge(face_details[0] if face_details else None)

    plain = f"{decision['verdict']}: " + " ".join(decision["reasons"])
    explanation = (genai.explain(decision, face_result, live_result, person)
                   if use_ai and decision["verdict"] != "RETRY" else plain)

    result = {
        "verdict": decision["verdict"], "verification": decision["verification"],
        "reasons": decision["reasons"], "person_id": face_result["person_id"],
        "similarity": face_result["similarity"], "liveness": live_result,
        "schedule": schedule, "inspector": inspector, "decision_ms": decision_ms,
        "timing": timing, "challenge": challenge, "explanation": explanation,
        "location": location, "mode": "online", "threat": threat if threat["active"] else None,
        "flag": decision.get("flag"),
        **(extra or {}),
    }
    if log and decision["verdict"] != "RETRY":
        logged = store.log_event({k: v for k, v in result.items() if k != "challenge"})
        result["time"] = logged["time"]
        security.on_decision(result)
    return result


def check_access(image_bytes, use_ai=True, log=True, location=None, now=None, use_schedule=True):
    start = time.perf_counter()
    face_result, face_details, labels, inspector_job, timing = _analyse(image_bytes, True)
    t0 = time.perf_counter()
    live_result = liveness.passive_check(image_bytes, face_details, labels)
    timing["liveness_analysis"] = int((time.perf_counter() - t0) * 1000)
    inspector = _apply_inspector(live_result, inspector_job, start, timing)
    return _finish(face_result, live_result, face_details, start, timing, {}, use_ai, log,
                   location, now, inspector, use_schedule)


def verify_challenge(challenge, image_bytes, use_ai=True, log=True, location=None, now=None):
    """Second photo after REQUEST_VERIFICATION: passive check AND the random action."""
    start = time.perf_counter()
    face_result, face_details, labels, _, timing = _analyse(image_bytes, False)
    t0 = time.perf_counter()
    passive = liveness.passive_check(image_bytes, face_details, labels)
    active = liveness.verify_challenge(challenge, face_details)
    timing["liveness_analysis"] = int((time.perf_counter() - t0) * 1000)
    if passive["live"] is False:
        live_result = passive
    else:
        live_result = active
        live_result["signals"] = {**passive["signals"], **active["signals"]}
    return _finish(face_result, live_result, face_details, start, timing,
                   {"after_challenge": True}, use_ai, log, location, now)


def check_official(liveness_result, reference_image_bytes, use_ai=True, log=True,
                   location=None, now=None, extra=None):
    """Decision after AWS Face Liveness has checked liveness (official_liveness.py)."""
    start = time.perf_counter()
    try:
        face_result, timing_identity = _timed(face_service.identify, reference_image_bytes)
    except NETWORK_ERRORS as e:
        offline.mark_offline()
        raise OfflineError(str(e)) from e
    return _finish(face_result, liveness_result, [], start, {"identity": timing_identity},
                   {}, use_ai, log, location, now, extra=extra)


def record_override(person_id, verdict, by, note=""):
    return store.log_event({
        "verdict": verdict,
        "reasons": [f"Manual override by {by}." + (f" {note}" if note else "")],
        "person_id": person_id, "similarity": None, "liveness": None, "decision_ms": None,
        "explanation": f"A human ({by}) set this decision to {verdict}.", "mode": "override",
    })
