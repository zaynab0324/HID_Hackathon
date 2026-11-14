"""Liveness detection: is a real person in front of the camera, or a spoof?

OWNER: Afrah

CONTRACT:
    passive_check(image_bytes: bytes, face_details: list[dict], labels: list[dict]) -> LivenessResult
        face_details = face_service.detect_faces(image_bytes)   (Rekognition FaceDetails)
        labels       = detect_labels(image_bytes)               (Rekognition Labels)
    detect_labels(image_bytes: bytes) -> list[dict]
    new_challenge(baseline_face: dict | None = None)
        -> {"id", "action", "instruction", "expires_at", "baseline"}   (action differs from the first photo)
    verify_challenge(challenge: dict, face_details: list[dict]) -> LivenessResult

    LivenessResult = {
        "live": True | False | None,     # None = unsure -> REQUEST VERIFICATION
        "score": float,                  # 0..1, 1 = surely live
        "spoof_type": str | None,        # "printed_photo" | "screen_replay" | "mask" | "challenge_failed" | None
        "signals": dict,                 # every measurement, for the log and the evaluation report
        "source": "passive" | "challenge" | "aws_face_liveness",
        "reasons": list[str],            # plain English
    }
"""
import random
import time
import uuid

import cv2
import numpy as np
from botocore.exceptions import ClientError

import config
import face_service


# ---------------------------------------------------------------- helpers

def detect_labels(image_bytes):
    try:
        response = face_service.client().detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=30,
            MinConfidence=config.LABEL_MIN_CONFIDENCE,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidParameterException":
            return []
        raise
    return response["Labels"]


def _result(live, score, spoof_type, signals, source, reasons):
    return {"live": live, "score": round(float(score), 2), "spoof_type": spoof_type,
            "signals": signals, "source": source, "reasons": reasons}


def _box_contains_point(box, x, y):
    return (box["Left"] <= x <= box["Left"] + box["Width"]
            and box["Top"] <= y <= box["Top"] + box["Height"])


def _texture_signals(image_bytes, face_box):
    """Measurements on the face area that differ between real skin, paper and screens."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {}
    h, w = img.shape[:2]
    x0 = max(int(face_box["Left"] * w), 0)
    y0 = max(int(face_box["Top"] * h), 0)
    x1 = min(int((face_box["Left"] + face_box["Width"]) * w), w)
    y1 = min(int((face_box["Top"] + face_box["Height"]) * h), h)
    face = img[y0:y1, x0:x1]
    if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
        return {}

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Sharpness: printed photos and re-captured screens are usually softer.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Glare: glossy paper and screens reflect light as bright white patches.
    glare_ratio = float((gray >= 245).mean())
    # Moire: screens create regular high-frequency patterns -> strong peaks in the spectrum.
    small = cv2.resize(gray, (128, 128)).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(small))))
    yy, xx = np.ogrid[-64:64, -64:64]
    high = spectrum[(xx ** 2 + yy ** 2) > 40 ** 2]
    moire_score = float(high.max() / (high.mean() + 1e-6))
    return {"sharpness": round(sharpness, 1), "glare_ratio": round(glare_ratio, 4),
            "moire_score": round(moire_score, 3)}


# ---------------------------------------------------------------- passive check

def passive_check(image_bytes, face_details, labels):
    if not face_details:
        return _result(None, 0.0, None, {"faces": 0}, "passive", ["No face detected."])
    if len(face_details) > 1:
        return _result(None, 0.5, None, {"faces": len(face_details)}, "passive",
                       ["More than one face in view."])

    face = face_details[0]
    box = face["BoundingBox"]
    cx, cy = box["Left"] + box["Width"] / 2, box["Top"] + box["Height"] / 2

    score = 0.9
    spoof_type = None
    reasons = []
    found = []

    # 1) A phone / screen / paper object around the face is the strongest spoof sign.
    for label in labels:
        name = label["Name"]
        kind = ("screen_replay" if name in config.SCREEN_LABELS else
                "printed_photo" if name in config.PAPER_LABELS else
                "mask" if name in config.MASK_LABELS else None)
        if kind is None:
            continue
        found.append(name)
        around_face = any(_box_contains_point(inst["BoundingBox"], cx, cy)
                          for inst in label.get("Instances", []))
        if around_face:
            score = min(score, 0.05)
            spoof_type = kind
            reasons.append(f"The face is inside a detected {name.lower()}.")
        elif name in config.ANYWHERE_LABELS:
            score -= 0.35
            spoof_type = spoof_type or kind
            reasons.append(f"A {name.lower()} is visible in the frame.")

    # 2) Face attributes.
    occluded = face.get("FaceOccluded", {})
    if occluded.get("Value") and occluded.get("Confidence", 0) > 80:
        score -= 0.1
        reasons.append("Face is partly covered (could be a hand holding a photo).")

    # 3) Image texture (only used once thresholds are set from evaluation data).
    texture = _texture_signals(image_bytes, box)
    if config.GLARE_MAX is not None and texture.get("glare_ratio", 0) > config.GLARE_MAX:
        score -= 0.25
        spoof_type = spoof_type or "screen_replay"
        reasons.append("Strong glare on the face, typical of screens or glossy photos.")
    if config.MOIRE_MAX is not None and texture.get("moire_score", 0) > config.MOIRE_MAX:
        score -= 0.3
        spoof_type = spoof_type or "screen_replay"
        reasons.append("Screen pixel pattern (moire) detected.")
    if config.SHARPNESS_MIN is not None and texture.get("sharpness", 1e9) < config.SHARPNESS_MIN:
        score -= 0.15
        spoof_type = spoof_type or "printed_photo"
        reasons.append("Face texture is too flat or soft for a live face.")

    score = max(0.0, min(1.0, score))
    signals = {"faces": 1, "spoof_labels": found, "face_occluded": bool(occluded.get("Value")),
               **texture}

    if score >= config.LIVE_THRESHOLD:
        return _result(True, score, None, signals, "passive",
                       reasons or ["No spoof signs found."])
    if score <= config.SPOOF_THRESHOLD:
        return _result(False, score, spoof_type, signals, "passive", reasons)
    return _result(None, score, spoof_type, signals, "passive",
                   reasons + ["Liveness is uncertain."])


# ---------------------------------------------------------------- active challenge
# Differential challenge: the action must CHANGE the face compared with the first photo.
# A printed photo or a screen cannot change its expression on command.

CHALLENGES = {
    "turn_left": "Turn your head to the LEFT",
    "turn_right": "Turn your head to the RIGHT",
    "smile": "Give a big SMILE",
    "close_eyes": "CLOSE your eyes",
    "open_mouth": "OPEN your mouth wide",
}


def _attribute(face, name):
    attr = face.get(name, {})
    return attr.get("Value") if attr.get("Confidence", 0) >= config.ATTRIBUTE_CONFIDENCE else None


def _state(face):
    """The facial state a challenge is compared against."""
    return {"smile": _attribute(face, "Smile"), "eyes_open": _attribute(face, "EyesOpen"),
            "mouth_open": _attribute(face, "MouthOpen"),
            "yaw": round(face.get("Pose", {}).get("Yaw", 0.0), 1)}


def new_challenge(baseline_face=None):
    """Pick a random action the person is NOT already doing in the first photo."""
    base = _state(baseline_face) if baseline_face else {}
    already = {"smile": base.get("smile") is True, "close_eyes": base.get("eyes_open") is False,
               "open_mouth": base.get("mouth_open") is True}
    options = [a for a in config.CHALLENGE_ACTIONS if not already.get(a)] or list(config.CHALLENGE_ACTIONS)
    action = random.choice(options)
    return {"id": uuid.uuid4().hex[:8], "action": action, "instruction": CHALLENGES[action],
            "expires_at": time.time() + config.CHALLENGE_SECONDS, "baseline": base}


def verify_challenge(challenge, face_details):
    if time.time() > challenge["expires_at"]:
        return _result(False, 0.0, "challenge_failed", {"action": challenge["action"]},
                       "challenge", ["The verification step timed out."])
    if len(face_details) != 1:
        return _result(False, 0.0, "challenge_failed", {"faces": len(face_details)},
                       "challenge", ["Exactly one face must be visible for verification."])

    now_state = _state(face_details[0])
    base = challenge.get("baseline") or {}
    action = challenge["action"]
    yaw = now_state["yaw"] * config.YAW_LEFT_SIGN
    base_yaw = (base.get("yaw") or 0.0) * config.YAW_LEFT_SIGN
    done = {
        "turn_left": yaw > config.TURN_YAW_DEGREES,
        "turn_right": yaw < -config.TURN_YAW_DEGREES,
        "smile": now_state["smile"] is True,
        "close_eyes": now_state["eyes_open"] is False,
        "open_mouth": now_state["mouth_open"] is True,
    }[action]
    changed = {
        "turn_left": yaw - base_yaw > config.TURN_YAW_DEGREES,
        "turn_right": base_yaw - yaw > config.TURN_YAW_DEGREES,
        "smile": base.get("smile") is not True,
        "close_eyes": base.get("eyes_open") is not False,
        "open_mouth": base.get("mouth_open") is not True,
    }[action]
    signals = {"action": action, "baseline": base, "now": now_state}
    if done and changed:
        return _result(True, 0.95, None, signals, "challenge",
                       [f"Completed the random action ({CHALLENGES[action].lower()}) "
                        "and the face changed from the first photo."])
    reason = (f"Did not complete the random action: {CHALLENGES[action].lower()}." if not done
              else "The face did not change from the first photo (possible photo or replay).")
    return _result(False, 0.1, "challenge_failed", signals, "challenge", [reason])
