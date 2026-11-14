"""Decision rules: liveness + identity + campus schedule -> ALLOW / REQUEST_VERIFICATION / DENY.

CONTRACT:
    decide(face_result, liveness_result, person, context=None)
        -> {"verdict": "ALLOW" | "REQUEST_VERIFICATION" | "DENY" | "RETRY",
            "reasons": list[str],
            "verification": "challenge" | "guard" | None,   # which extra step is needed
            "needs_verification": bool}

    context keys:
        "after_challenge": True  when deciding after the random-action step
        "schedule": result of store.check_schedule(...)  (None = no schedule check)
"""
import config


def _out(verdict, reasons, verification=None):
    return {"verdict": verdict, "reasons": reasons, "verification": verification,
            "needs_verification": verification is not None}


def decide(face_result, liveness_result, person, context=None):
    context = context or {}
    after_challenge = context.get("after_challenge", False)
    schedule = context.get("schedule")
    reasons = list(liveness_result.get("reasons", []))

    if face_result["status"] == "NO_FACE":
        return _out("RETRY", ["No face detected. Look straight at the camera."])

    # 1) LIVENESS: spoof -> DENY, whoever the face looks like.
    if liveness_result["live"] is False:
        kind = (liveness_result.get("spoof_type") or "spoof").replace("_", " ")
        return _out("DENY", [f"Spoof detected ({kind})."] + reasons)

    # 2) FACE AUTHENTICATION: unknown face -> DENY.
    if face_result["status"] == "NO_MATCH":
        return _out("DENY", ["Face does not match any enrolled person."] + reasons)

    identity = f"{face_result['person_id']} ({face_result['similarity']}%)"

    # 3) ACCESS STATUS.
    if person and person.get("status") == "suspended":
        return _out("DENY", [f"Identity {identity} confirmed but access is suspended."])

    # 4) CAMPUS SCHEDULE: recognised is not the same as authorised.
    if schedule and schedule["status"] == "out_of_schedule":
        return _out("DENY", [f"Identity {identity} confirmed. {schedule['reason']}"])

    # 5) Unsure liveness or unsure identity -> random-action challenge (once).
    unsure = liveness_result["live"] is None or face_result["status"] == "POSSIBLE"
    if unsure:
        if after_challenge:
            return _out("DENY", ["Verification did not raise confidence enough.",
                                 f"Identity: {identity}."] + reasons)
        why = ("liveness is uncertain" if liveness_result["live"] is None
               else "identity confidence is low")
        return _out("REQUEST_VERIFICATION", [f"Extra verification needed: {why}.",
                                             f"Identity: {identity}."] + reasons, "challenge")

    # 6) Security mode: a passive check alone is not enough proof of liveness.
    if config.REQUIRE_ACTIVE_CHECK and liveness_result.get("source") == "passive":
        return _out("REQUEST_VERIFICATION",
                    ["Passive check passed; confirming liveness with a random action.",
                     f"Identity: {identity}."], "challenge")

    # 7) Near the scheduled slot -> a guard must approve (step-up).
    if schedule and schedule["status"] == "near_schedule":
        return _out("REQUEST_VERIFICATION",
                    [f"Live person, identity {identity} confirmed. {schedule['reason']}",
                     "Guard approval required."], "guard")

    # 8) Live + confirmed identity + allowed now -> ALLOW.
    why = f" {schedule['reason']}" if schedule else ""
    return _out("ALLOW", [f"Live person detected and identity confirmed: {identity}.{why}"])
