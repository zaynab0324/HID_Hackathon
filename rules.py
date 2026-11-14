"""Access decision rules: liveness + identity -> ALLOW / REQUEST_VERIFICATION / DENY.

OWNER: Afrah

CONTRACT:
    decide(face_result: dict, liveness_result: dict, person: dict | None, context: dict | None = None)
        -> {"verdict": "ALLOW" | "REQUEST_VERIFICATION" | "DENY" | "RETRY",
            "reasons": list[str],
            "needs_verification": bool}

    face_result:     face_service.identify(...)
    liveness_result: liveness.passive_check(...) / liveness.verify_challenge(...) / official result
    person:          store.get_person(...) or None
    context:         {"after_challenge": True} when deciding after a verification step
"""


def _out(verdict, reasons, needs_verification=False):
    return {"verdict": verdict, "reasons": reasons, "needs_verification": needs_verification}


def decide(face_result, liveness_result, person, context=None):
    context = context or {}
    after_challenge = context.get("after_challenge", False)
    reasons = list(liveness_result.get("reasons", []))

    if face_result["status"] == "NO_FACE":
        return _out("RETRY", ["No face detected. Look straight at the camera."])

    # 1) Spoof detected -> DENY, whoever the face looks like.
    if liveness_result["live"] is False:
        kind = (liveness_result.get("spoof_type") or "spoof").replace("_", " ")
        return _out("DENY", [f"Spoof detected ({kind})."] + reasons)

    # 2) Unknown face -> DENY.
    if face_result["status"] == "NO_MATCH":
        return _out("DENY", ["Face does not match any enrolled person."] + reasons)

    identity = f"{face_result['person_id']} ({face_result['similarity']}%)"

    # 3) Record rules.
    if person and person.get("status") == "suspended":
        return _out("DENY", [f"Identity {identity} confirmed but access is suspended."])

    # 4) Unsure liveness or unsure identity -> ask for an extra step (once).
    unsure = liveness_result["live"] is None or face_result["status"] == "POSSIBLE"
    if unsure:
        if after_challenge:
            return _out("DENY", ["Verification did not raise confidence enough.",
                                 f"Identity: {identity}."] + reasons)
        why = ("liveness is uncertain" if liveness_result["live"] is None
               else "identity confidence is low")
        return _out("REQUEST_VERIFICATION", [f"Extra verification needed: {why}.",
                                             f"Identity: {identity}."] + reasons, True)

    # 5) Live person + confirmed identity -> ALLOW.
    return _out("ALLOW", [f"Live person detected and identity confirmed: {identity}."])
