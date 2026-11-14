"""Joins all the parts: face + liveness -> rules -> explanation -> log. The UI calls ONLY this file.

OWNER: Afrah (integration)

CONTRACT:
    check_access(image_bytes, use_ai=True, log=True) -> AccessResult
    verify_challenge(challenge, image_bytes, use_ai=True, log=True) -> AccessResult
    check_official(liveness_result, reference_image_bytes, use_ai=True, log=True) -> AccessResult
    record_override(person_id, verdict, by, note="") -> dict

    AccessResult = {
        "verdict": "ALLOW" | "REQUEST_VERIFICATION" | "DENY" | "RETRY",
        "reasons": list[str],
        "person_id": str | None,
        "similarity": float,
        "liveness": LivenessResult,           # see liveness.py
        "decision_ms": int,                   # time to decide (must be <= 2000)
        "challenge": dict | None,             # set when verdict == REQUEST_VERIFICATION
        "explanation": str,
        "time": str,                          # only when logged
    }
"""
import time
from concurrent.futures import ThreadPoolExecutor

import face_service
import genai
import liveness
import rules
import store

_pool = ThreadPoolExecutor(max_workers=4)


def _analyse(image_bytes):
    """Run the three AWS calls IN PARALLEL to stay well under 2 seconds."""
    f_identity = _pool.submit(face_service.identify, image_bytes)
    f_faces = _pool.submit(face_service.detect_faces, image_bytes)
    f_labels = _pool.submit(liveness.detect_labels, image_bytes)
    return f_identity.result(), f_faces.result(), f_labels.result()


def _finish(face_result, live_result, start, context, use_ai, log):
    person = store.get_person(face_result["person_id"]) if face_result["person_id"] else None
    decision = rules.decide(face_result, live_result, person, context)
    decision_ms = int((time.perf_counter() - start) * 1000)   # decision time, before GenAI text

    challenge = liveness.new_challenge() if decision["verdict"] == "REQUEST_VERIFICATION" else None
    plain = f"{decision['verdict']}: " + " ".join(decision["reasons"])
    if use_ai and decision["verdict"] != "RETRY":
        explanation = genai.explain(decision, face_result, live_result, person)
    else:
        explanation = plain

    result = {
        "verdict": decision["verdict"],
        "reasons": decision["reasons"],
        "person_id": face_result["person_id"],
        "similarity": face_result["similarity"],
        "liveness": live_result,
        "decision_ms": decision_ms,
        "challenge": challenge,
        "explanation": explanation,
    }
    if log and decision["verdict"] != "RETRY":
        logged = store.log_event({k: v for k, v in result.items() if k != "challenge"})
        result["time"] = logged["time"]
    return result


def check_access(image_bytes, use_ai=True, log=True):
    start = time.perf_counter()
    face_result, face_details, labels = _analyse(image_bytes)
    live_result = liveness.passive_check(image_bytes, face_details, labels)
    return _finish(face_result, live_result, start, {}, use_ai, log)


def verify_challenge(challenge, image_bytes, use_ai=True, log=True):
    """Second attempt after REQUEST_VERIFICATION: passive check AND the random action."""
    start = time.perf_counter()
    face_result, face_details, labels = _analyse(image_bytes)
    passive = liveness.passive_check(image_bytes, face_details, labels)
    active = liveness.verify_challenge(challenge, face_details)
    if passive["live"] is False:
        live_result = passive
    else:
        live_result = active
        live_result["signals"] = {**passive["signals"], **active["signals"]}
    return _finish(face_result, live_result, start, {"after_challenge": True}, use_ai, log)


def check_official(liveness_result, reference_image_bytes, use_ai=True, log=True):
    """Decision when AWS Face Liveness has already checked liveness (official_liveness.py)."""
    start = time.perf_counter()
    face_result = face_service.identify(reference_image_bytes)
    return _finish(face_result, liveness_result, start, {}, use_ai, log)


def record_override(person_id, verdict, by, note=""):
    return store.log_event({
        "verdict": verdict,
        "reasons": [f"Manual override by {by}." + (f" {note}" if note else "")],
        "person_id": person_id,
        "similarity": None,
        "liveness": None,
        "decision_ms": None,
        "explanation": f"A human ({by}) set this decision to {verdict}.",
    })
