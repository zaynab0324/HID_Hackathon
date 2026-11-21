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
        "threat": result of security.threat_status()       (None = normal mode)
        "last_allow": {"location", "minutes_ago"} of this person's last ALLOW (None = none)
        "location": the door being decided
        "direction": "in" (entry, default) | "out" (check-out at the exit)
        "presence": presence.current(person) -> {"location", "since", "expires"} | None
"""
import config


def _out(verdict, reasons, verification=None, flag=None):
    return {"verdict": verdict, "reasons": reasons, "verification": verification,
            "needs_verification": verification is not None, "flag": flag}


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
    presence = context.get("presence")
    here = context.get("location")

    # CHECK-OUT at the exit: needs the same liveness + identity proof (nobody can check someone
    # else out), but never traps a person inside: schedule / suspension don't block leaving.
    if context.get("direction") == "out":
        if liveness_result["live"] is None or face_result["status"] == "POSSIBLE":
            return _out("REQUEST_VERIFICATION", ["Check-out needs a clear live face match. Please re-capture.",
                                                 f"Identity: {identity}."], "challenge")
        if presence and here and presence["location"] != here:
            return _out("ALLOW", [f"{face_result['person_id']} checked out at {here}, but was checked in at "
                                  f"{presence['location']} since {presence['since'][11:16]}. Mismatch flagged."],
                        flag="checkout_mismatch")
        if presence:
            return _out("ALLOW", [f"Checked out of {presence['location']} (in since {presence['since'][11:16]}). "
                                  f"Identity {identity}."])
        return _out("ALLOW", [f"Exit recorded for {identity}. No open check-in was found."])

    # 3) ACCESS STATUS.
    if person and person.get("status") == "suspended":
        return _out("DENY", [f"Identity {identity} confirmed but access is suspended."])

    # 3b) ALREADY CHECKED IN ELSEWHERE: still inside another lab (no check-out) -> one person,
    #     two places: the second face is a clone / spoof / shared identity.
    if presence and here and presence["location"] != here:
        return _out("DENY", [f"{face_result['person_id']} is checked in at {presence['location']} since "
                             f"{presence['since'][11:16]} and has not checked out. One person cannot be in two "
                             "labs: possible cloned identity, security alerted. Check out there first."],
                    flag="already_checked_in")

    # 3c) IMPOSSIBLE TRAVEL: seen at another lab (entry or check-out) less than the minimum walking
    #     time ago -> nobody can get here that fast: cloned identity?
    last = context.get("last_allow")
    if (last and here and last["location"] != here
            and last["minutes_ago"] < config.IMPOSSIBLE_TRAVEL_MINUTES):
        what = "checked out of" if last.get("direction") == "out" else "allowed at"
        return _out("DENY", [f"Impossible travel: {face_result['person_id']} {what} {last['location']} only "
                             f"{last['minutes_ago']:.1f} min ago; at least {config.IMPOSSIBLE_TRAVEL_MINUTES} min "
                             f"are needed to reach {here}. Possible cloned identity: security alerted."],
                    flag="impossible_travel")

    # 4) CAMPUS SCHEDULE: recognised is not the same as authorised.
    if schedule and schedule["status"] == "out_of_schedule":
        return _out("DENY", [f"Identity {identity} confirmed. {schedule['reason']}"])

    # 4b) THREAT-ADAPTIVE: the kiosk was attacked recently -> stricter rules.
    threat = context.get("threat") or {}
    if threat.get("active"):
        conf = (liveness_result.get("signals") or {}).get("aws_confidence")
        if (liveness_result.get("source") == "aws_face_liveness" and conf is not None
                and conf < config.THREAT_LIVE_CONFIDENCE):
            return _out("REQUEST_VERIFICATION",
                        [f"Heightened security: liveness {conf}% is below the "
                         f"{config.THREAT_LIVE_CONFIDENCE}% bar used after recent attacks.",
                         threat["reason"], "Please re-capture."], "challenge")
        if face_result["person_id"] in threat.get("targeted_ids", []):
            return _out("REQUEST_VERIFICATION",
                        [f"Identity {identity} was targeted by a recent spoof attempt.",
                         "Guard approval required."], "guard")

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
