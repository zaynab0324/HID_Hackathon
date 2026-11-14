"""Generative AI features with Amazon Bedrock.

OWNER: Abhi

CONTRACT (all functions NEVER raise; on any Bedrock problem they return safe fallback text):
    explain(decision: dict, face_result: dict, liveness_result: dict, person: dict | None) -> str
    ask(question: str, events: list[dict]) -> str
    summarize_performance(report: dict) -> str
    incident_report(events: list[dict], alerts: list[dict]) -> str
    parse_policy(text: str, people: list[str], locations: list[str], now) -> dict
        Natural-language admin command -> a structured, validated access-policy change (never applied here).
    spoof_opinion(image_bytes: bytes) -> {"spoof": True | False | None, "spoof_type": str | None,
                                          "confidence": float, "reason": str}
        A vision model's second opinion. Used ONLY to make decisions stricter.
"""
import json

import boto3
from botocore.config import Config

import config

_client = None


def _bedrock():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=config.REGION, config=Config(
            connect_timeout=config.AWS_TIMEOUT_SECONDS, read_timeout=15,
            retries={"max_attempts": 1}, max_pool_connections=20))
    return _client


def _chat(system_prompt, user_text, max_tokens=300):
    kwargs = {
        "modelId": config.MODEL_ID,
        "system": [{"text": system_prompt}],
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.2},
    }
    if config.GUARDRAIL_ID:
        kwargs["guardrailConfig"] = {
            "guardrailIdentifier": config.GUARDRAIL_ID,
            "guardrailVersion": config.GUARDRAIL_VERSION,
        }
    response = _bedrock().converse(**kwargs)
    return response["output"]["message"]["content"][0]["text"].strip()


def explain(decision, face_result, liveness_result, person):
    fallback = f"{decision['verdict']}: " + " ".join(decision["reasons"])
    system = (
        "You explain face-access decisions at an HID entry kiosk to security staff. "
        "Use ONLY the facts given; never invent facts. "
        "Say whether a live person was detected and whether the identity matched. "
        "At most 2 short sentences, plain English, no markdown."
    )
    facts = {
        "verdict": decision["verdict"],
        "reasons": decision["reasons"],
        "identity_similarity_percent": face_result.get("similarity"),
        "liveness_score_0_to_1": liveness_result.get("score"),
        "spoof_type": liveness_result.get("spoof_type"),
        "liveness_method": liveness_result.get("source"),
        "person_status": (person or {}).get("status"),
    }
    try:
        return _chat(system, "Explain this decision:\n" + json.dumps(facts, default=str))
    except Exception:
        return fallback


def ask(question, events):
    system = (
        "You are an assistant for an access-control admin. Answer ONLY from the access log "
        "provided. If the log does not contain the answer, say so. Be brief."
    )
    log_text = json.dumps(events[:200], default=str)
    try:
        return _chat(system, f"Access log (newest first):\n{log_text}\n\nQuestion: {question}",
                     max_tokens=500)
    except Exception as e:
        return f"The AI assistant is unavailable right now ({type(e).__name__})."


def summarize_performance(report):
    system = (
        "You summarise anti-spoofing test results for hackathon judges. Use ONLY the numbers "
        "given. 4 short bullet points: overall result, best and weakest spoof type, speed "
        "versus the 2-second requirement, one honest next improvement."
    )
    try:
        return _chat(system, json.dumps(report, default=str), max_tokens=400)
    except Exception as e:
        return f"AI summary unavailable ({type(e).__name__})."


INSPECTOR_PROMPT = (
    "You are a presentation-attack detector for a face-recognition door kiosk. Look at this "
    "webcam frame. Decide if it shows a LIVE person standing in front of the camera, or a spoof: "
    "a printed photo, a photo or video shown on a phone/tablet/laptop screen, or a mask. "
    "Clues of a spoof: phone or screen edges/bezel around the face, paper edges, fingers holding "
    "something flat, screen glare or pixel/moire patterns, unnaturally flat face, mask edges or "
    "eye holes. Reply with ONLY a JSON object, no other text: "
    '{"spoof": true or false, "spoof_type": "printed_photo" | "screen_replay" | "video_replay" | '
    '"mask" | null, "confidence": number between 0 and 1, "reason": "one short sentence"}'
)


def _parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def spoof_opinion(image_bytes):
    unknown = {"spoof": None, "spoof_type": None, "confidence": 0.0, "reason": "Inspector unavailable."}
    try:
        response = _bedrock().converse(
            modelId=config.VISION_MODEL_ID,
            messages=[{"role": "user", "content": [
                {"image": {"format": "png" if image_bytes[:4] == b"\x89PNG" else "jpeg",
                           "source": {"bytes": image_bytes}}},
                {"text": INSPECTOR_PROMPT},
            ]}],
            inferenceConfig={"maxTokens": 150, "temperature": 0.0},
        )
        data = _parse_json(response["output"]["message"]["content"][0]["text"])
        spoof = data.get("spoof")
        return {
            "spoof": spoof if isinstance(spoof, bool) else None,
            "spoof_type": data.get("spoof_type") or None,
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception:
        return unknown


def incident_report(events, alerts):
    system = (
        "You are a security operations assistant for HID entry kiosks. Using ONLY the access log "
        "and alerts given, write a short incident report for the security team: 1) summary, "
        "2) spoof/attack attempts (time, type, which identity was targeted), 3) denied or "
        "step-up patterns worth checking, 4) recommended actions. Be factual and brief. "
        "If there were no incidents, say so."
    )
    payload = {"alerts": alerts[:100],
               "events": [{k: e.get(k) for k in ("time", "verdict", "person_id", "location",
                                                 "reasons", "mode", "test_label")} for e in events[:200]]}
    try:
        return _chat(system, json.dumps(payload, default=str), max_tokens=700)
    except Exception as e:
        return f"AI incident report unavailable ({type(e).__name__})."


POLICY_ACTIONS = {"add_slot", "temp_pass", "suspend", "activate", "unknown"}


def parse_policy(text, people, locations, now):
    """Turn an admin's plain-English command into ONE structured policy change. A human must confirm."""
    system = (
        "You convert an access-control admin's instruction into ONE JSON action. Allowed actions: "
        "add_slot (recurring weekly access), temp_pass (temporary access for some hours from now), "
        "suspend, activate, unknown. Use ONLY these people ids: " + ", ".join(people) + ". "
        "Match names loosely (e.g. 'riya' -> 'Riya'). Use ONLY these locations: " + ", ".join(locations) + ". "
        "Days are numbers Monday=0 ... Sunday=6. Times are 24h 'HH:MM'. "
        f"Today is {now.strftime('%A %Y-%m-%d %H:%M')}. "
        'Reply with ONLY JSON: {"action": ..., "person_id": ..., "location": ..., "days": [..], '
        '"start": "HH:MM", "end": "HH:MM", "hours": number, "summary": "one sentence describing the change"}. '
        "Omit fields an action does not need. If the request is unclear or unsafe, use action unknown "
        "and explain in summary."
    )
    try:
        data = _parse_json(_chat(system, text, max_tokens=250))
    except Exception as e:
        return {"action": "unknown", "summary": f"AI unavailable ({type(e).__name__})."}
    return validate_policy(data, people, locations)


def validate_policy(data, people, locations):
    """Never trust model output: check every field before a human can apply it."""
    action = data.get("action")
    if action not in POLICY_ACTIONS or action == "unknown":
        return {"action": "unknown", "summary": data.get("summary") or "Could not understand the request."}
    pid = data.get("person_id")
    match = {p.lower(): p for p in people}.get(str(pid).lower().replace(" ", "_")) if pid else None
    if not match:
        return {"action": "unknown", "summary": f"Unknown person '{pid}'. Enroll them first."}
    out = {"action": action, "person_id": match, "summary": data.get("summary", "")}
    if action in ("add_slot", "temp_pass"):
        loc = {l.lower(): l for l in locations}.get(str(data.get("location", "")).lower())
        if not loc:
            return {"action": "unknown", "summary": f"Unknown location '{data.get('location')}'."}
        out["location"] = loc
    if action == "add_slot":
        days = [int(d) for d in data.get("days", []) if str(d).isdigit() and 0 <= int(d) <= 6]
        start, end = str(data.get("start", "")), str(data.get("end", ""))
        ok_time = all(len(t) == 5 and t[2] == ":" and t[:2].isdigit() and t[3:].isdigit()
                      and int(t[:2]) < 24 and int(t[3:]) < 60 for t in (start, end))
        if not days or not ok_time or start >= end:
            return {"action": "unknown", "summary": "Days or times were unclear. Please rephrase."}
        out.update(days=sorted(set(days)), start=start, end=end)
    if action == "temp_pass":
        try:
            hours = float(data.get("hours", 0))
        except (TypeError, ValueError):
            hours = 0
        if not 0 < hours <= 24:
            return {"action": "unknown", "summary": "Temporary passes must be 1-24 hours."}
        out["hours"] = hours
    return out
