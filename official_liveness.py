"""AWS Rekognition Face Liveness (official anti-spoofing) - backend part.

OWNER: Aman

How it works:
  1. create_session() -> session_id
  2. The browser runs AWS's FaceLivenessDetector (React, in official_liveness_web/) with that
     session_id. The user looks into an oval while the screen flashes colours.
  3. get_result(session_id) -> LivenessResult + the reference face image
  4. pipeline.check_official(result, reference_image) -> identity + ALLOW / DENY decision

CONTRACT:
    create_session() -> str
    get_result(session_id: str) -> (LivenessResult, reference_image_bytes | None)
        LivenessResult has the same shape as liveness.py, with source = "aws_face_liveness"
"""
import config
import face_service


def create_session():
    response = face_service.client().create_face_liveness_session(
        Settings={"AuditImagesLimit": 0})
    return response["SessionId"]


def get_result(session_id):
    response = face_service.client().get_face_liveness_session_results(SessionId=session_id)
    status = response["Status"]
    confidence = float(response.get("Confidence", 0.0))
    reference = (response.get("ReferenceImage") or {}).get("Bytes")
    signals = {"session_status": status, "aws_confidence": round(confidence, 1)}

    if status != "SUCCEEDED":
        result = {"live": None, "score": 0.0, "spoof_type": None, "signals": signals,
                  "source": "aws_face_liveness",
                  "reasons": [f"Liveness session not finished (status: {status})."]}
        return result, None

    live = confidence >= config.OFFICIAL_LIVE_CONFIDENCE
    result = {
        "live": True if live else (False if confidence < 50 else None),
        "score": round(confidence / 100.0, 2),
        "spoof_type": None if live else "presentation_attack",
        "signals": signals,
        "source": "aws_face_liveness",
        "reasons": [f"AWS Face Liveness confidence {confidence:.1f}%."],
    }
    return result, reference
