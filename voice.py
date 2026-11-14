"""Talking kiosk: speaks the decision with Amazon Polly (accessibility + clear feedback at the door).

CONTRACT:
    speech_text(result: dict) -> str          short sentence for the person at the door (no sensitive detail)
    synthesize(text: str) -> bytes | None     MP3 audio, or None if Polly is unavailable
"""
import boto3
from botocore.config import Config

import config

_client = None


def _polly():
    global _client
    if _client is None:
        _client = boto3.client("polly", region_name=config.REGION, config=Config(
            connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}))
    return _client


def speech_text(result):
    verdict = result.get("verdict")
    name = (result.get("person_id") or "").split("_")[0]
    live = (result.get("liveness") or {}).get("live")
    schedule = (result.get("schedule") or {}).get("status")
    if verdict == "ALLOW":
        return f"Welcome{', ' + name if name else ''}. Access granted."
    if verdict == "DENY":
        if live is False:
            return "Access denied. A spoof attempt was detected. Security has been notified."
        if schedule == "out_of_schedule":
            return "Access denied. You are not scheduled for this room right now."
        if result.get("mode") == "offline":
            return "Access denied. Please see the security guard."
        return "Access denied."
    if verdict == "REQUEST_VERIFICATION":
        if result.get("verification") == "guard":
            return "Please wait. A security guard needs to approve your entry."
        return "Additional verification needed. Please try again, looking straight at the camera."
    return "Please look at the camera and try again."


def synthesize(text):
    for kwargs in ({"VoiceId": "Kajal", "Engine": "neural", "LanguageCode": "en-IN"},
                   {"VoiceId": "Joanna", "Engine": "standard"}):
        try:
            resp = _polly().synthesize_speech(Text=text, OutputFormat="mp3", **kwargs)
            return resp["AudioStream"].read()
        except Exception:
            continue
    return None
