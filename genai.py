"""Generative AI features with Amazon Bedrock.

OWNER: Abhi

CONTRACT (all functions NEVER raise; on any Bedrock problem they return safe fallback text):
    explain(decision: dict, face_result: dict, liveness_result: dict, person: dict | None) -> str
    ask(question: str, events: list[dict]) -> str
    summarize_performance(report: dict) -> str
"""
import json

import boto3

import config

_client = None


def _bedrock():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=config.REGION)
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
