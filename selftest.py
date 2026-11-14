"""Offline self-test: checks every file still honours its contract. Needs NO AWS access.
Run before every git push:   python selftest.py
"""
import os
import shutil
import tempfile
import time

import boto3
import cv2
import numpy as np
from botocore.stub import Stubber

import config

TMP = tempfile.mkdtemp()
config.DATA_DIR = os.path.join(TMP, "data")
config.TEST_DATA_DIR = os.path.join(TMP, "test_data")

import evaluate  # noqa: E402
import face_service  # noqa: E402
import genai  # noqa: E402
import liveness  # noqa: E402
import official_liveness  # noqa: E402
import pipeline  # noqa: E402
import rules  # noqa: E402
import store  # noqa: E402

store.PEOPLE_FILE = os.path.join(config.DATA_DIR, "people.json")
store.LOG_FILE = os.path.join(config.DATA_DIR, "access_log.json")
evaluate.REPORT_FILE = os.path.join(config.DATA_DIR, "eval_report.json")
evaluate.DETAILS_FILE = os.path.join(config.DATA_DIR, "eval_details.csv")

# A real JPEG so the image-texture code runs for real.
_ok, _jpg = cv2.imencode(".jpg", (np.random.rand(240, 320, 3) * 255).astype(np.uint8))
IMG = _jpg.tobytes()
FACE_BOX = {"Left": 0.3, "Top": 0.2, "Width": 0.4, "Height": 0.6}

# ---- fake AWS answers (the pipeline calls these in parallel, so we patch the functions) ----
FAKE = {}


def fake_identify(_):
    return FAKE["identity"]


def fake_faces(_):
    return FAKE["faces"]


def fake_labels(_):
    return FAKE["labels"]


face_service.identify = fake_identify
face_service.detect_faces = fake_faces
liveness.detect_labels = fake_labels


def scene(similarity=99.4, pid="afrah", faces=1, labels=(), face_extra=None):
    if similarity is None:
        FAKE["identity"] = {"status": "NO_MATCH", "person_id": None, "similarity": 0.0, "message": ""}
    elif similarity == "noface":
        FAKE["identity"] = {"status": "NO_FACE", "person_id": None, "similarity": 0.0, "message": "No face"}
    else:
        status = "MATCH" if similarity >= config.MATCH_THRESHOLD else "POSSIBLE"
        FAKE["identity"] = {"status": status, "person_id": pid, "similarity": similarity, "message": ""}
    face = {"BoundingBox": FACE_BOX, "Pose": {"Yaw": 0.0}, **(face_extra or {})}
    FAKE["faces"] = [face] * faces
    FAKE["labels"] = list(labels)


AROUND_FACE = {"Left": 0.1, "Top": 0.05, "Width": 0.8, "Height": 0.9}

# ---- stubbed clients for single-call functions ----
rek = boto3.client("rekognition", region_name="us-east-1",
                   aws_access_key_id="x", aws_secret_access_key="x")
br = boto3.client("bedrock-runtime", region_name="us-east-1",
                  aws_access_key_id="x", aws_secret_access_key="x")
face_service._client = rek
genai._client = br
rek_stub, br_stub = Stubber(rek), Stubber(br)


def ai_says(text):
    br_stub.add_response("converse", {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    })


passed = 0


def check(name, condition):
    global passed
    assert condition, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


try:
    with rek_stub, br_stub:
        # ---------- face_service ----------
        check("clean_person_id handles spaces",
              face_service.clean_person_id(" Afrah Zainab ") == "Afrah_Zainab")
        check("clean_person_id rejects bad chars", face_service.clean_person_id("a/b") is None)
        rek_stub.add_response("index_faces", {"FaceRecords": [{"Face": {"FaceId": "f1"}}]})
        r = face_service.enroll(b"img", "Afrah Zainab")
        check("enroll ok", r["ok"] and r["person_id"] == "Afrah_Zainab")

        # ---------- capture -> liveness -> identity -> decision ----------
        scene()
        ai_says("A live person was detected and identified as afrah.")
        r = pipeline.check_access(IMG)
        check("live + known -> ALLOW", r["verdict"] == "ALLOW" and r["liveness"]["live"] is True)
        check("GenAI explanation used", r["explanation"].startswith("A live person"))
        check("decision time measured", isinstance(r["decision_ms"], int) and r["decision_ms"] < 2000)
        check("texture signals measured", "sharpness" in r["liveness"]["signals"])

        scene(labels=[{"Name": "Mobile Phone", "Instances": [{"BoundingBox": AROUND_FACE}]}])
        r = pipeline.check_access(IMG, use_ai=False)
        check("phone around face -> DENY screen replay",
              r["verdict"] == "DENY" and r["liveness"]["spoof_type"] == "screen_replay")

        scene(labels=[{"Name": "Paper", "Instances": [{"BoundingBox": AROUND_FACE}]}])
        r = pipeline.check_access(IMG, use_ai=False)
        check("paper around face -> DENY printed photo",
              r["verdict"] == "DENY" and r["liveness"]["spoof_type"] == "printed_photo")

        scene(labels=[{"Name": "Mobile Phone", "Instances": []}])
        r = pipeline.check_access(IMG, use_ai=False)
        check("phone visible -> REQUEST_VERIFICATION with challenge",
              r["verdict"] == "REQUEST_VERIFICATION" and r["challenge"] is not None)

        scene(labels=[{"Name": "Monitor", "Instances": []}])
        r = pipeline.check_access(IMG, use_ai=False)
        check("monitor in background does not block", r["verdict"] == "ALLOW")

        scene(similarity=None)
        r = pipeline.check_access(IMG, use_ai=False)
        check("live but unknown -> DENY", r["verdict"] == "DENY")

        scene(similarity=85.0)
        r = pipeline.check_access(IMG, use_ai=False)
        check("low identity confidence -> REQUEST_VERIFICATION", r["verdict"] == "REQUEST_VERIFICATION")

        store.upsert_person("afrah", status="suspended")
        scene()
        r = pipeline.check_access(IMG, use_ai=False)
        check("suspended -> DENY", r["verdict"] == "DENY")
        store.upsert_person("afrah", status="active")

        scene(similarity="noface", faces=0)
        r = pipeline.check_access(IMG)
        check("no face -> RETRY", r["verdict"] == "RETRY")

        # ---------- active challenge ----------
        challenge = {"id": "t1", "action": "smile", "instruction": "Smile",
                     "expires_at": time.time() + 30}
        scene(face_extra={"Smile": {"Value": True, "Confidence": 97}})
        r = pipeline.verify_challenge(challenge, IMG, use_ai=False)
        check("challenge done -> ALLOW", r["verdict"] == "ALLOW" and r["liveness"]["source"] == "challenge")

        scene(face_extra={"Smile": {"Value": False, "Confidence": 97}})
        r = pipeline.verify_challenge(challenge, IMG, use_ai=False)
        check("challenge not done -> DENY", r["verdict"] == "DENY")

        expired = {**challenge, "expires_at": time.time() - 1}
        scene(face_extra={"Smile": {"Value": True, "Confidence": 97}})
        r = pipeline.verify_challenge(expired, IMG, use_ai=False)
        check("expired challenge -> DENY", r["verdict"] == "DENY")

        turn = {"id": "t2", "action": "turn_left", "instruction": "Left", "expires_at": time.time() + 30}
        scene(face_extra={"Pose": {"Yaw": 30.0 * config.YAW_LEFT_SIGN}})
        check("turn left detected", pipeline.verify_challenge(turn, IMG, use_ai=False)["verdict"] == "ALLOW")

        scene(similarity=85.0, face_extra={"Smile": {"Value": True, "Confidence": 97}})
        r = pipeline.verify_challenge(challenge, IMG, use_ai=False)
        check("after challenge, identity still unsure -> DENY", r["verdict"] == "DENY")

        # ---------- GenAI never breaks the flow ----------
        scene()
        br_stub.add_client_error("converse", "AccessDeniedException")
        r = pipeline.check_access(IMG)
        check("Bedrock failure -> fallback text", r["verdict"] == "ALLOW" and r["explanation"].startswith("ALLOW:"))
        ai_says("Nobody was denied.")
        check("ask() returns text", genai.ask("Who was denied?", store.recent_events()) == "Nobody was denied.")
        br_stub.add_client_error("converse", "ThrottlingException")
        check("summarize_performance never raises", "unavailable" in genai.summarize_performance({}))

        # ---------- log ----------
        pipeline.record_override("bob", "ALLOW", by="guard")
        events = store.recent_events()
        check("log is newest first", events[0]["person_id"] == "bob")
        check("RETRY not logged", all(e["verdict"] != "RETRY" for e in events))

        # ---------- official AWS Face Liveness ----------
        rek_stub.add_response("create_face_liveness_session", {"SessionId": "a" * 36})
        check("official create_session", official_liveness.create_session() == "a" * 36)
        rek_stub.add_response("get_face_liveness_session_results", {
            "SessionId": "a" * 36, "Status": "SUCCEEDED", "Confidence": 97.5,
            "ReferenceImage": {"Bytes": IMG}})
        live_res, ref = official_liveness.get_result("a" * 36)
        check("official result live", live_res["live"] is True and ref == IMG)
        scene()
        r = pipeline.check_official(live_res, ref, use_ai=False)
        check("official + known -> ALLOW", r["verdict"] == "ALLOW"
              and r["liveness"]["source"] == "aws_face_liveness")
        rek_stub.add_response("get_face_liveness_session_results", {
            "SessionId": "a" * 36, "Status": "SUCCEEDED", "Confidence": 12.0})
        live_res, _ = official_liveness.get_result("a" * 36)
        r = pipeline.check_official(live_res, IMG, use_ai=False)
        check("official spoof -> DENY", r["verdict"] == "DENY")

        # ---------- evaluation ----------
        for folder in ("genuine", "printed_photo", "screen_photo"):
            os.makedirs(os.path.join(config.TEST_DATA_DIR, folder))
            for i in range(2):
                with open(os.path.join(config.TEST_DATA_DIR, folder, f"{i}.jpg"), "wb") as f:
                    f.write(IMG)
        real_check = pipeline.check_access

        def fake_check(image_bytes, use_ai=True, log=True):
            return real_check(image_bytes, use_ai=False, log=False)

        scene()  # every image looks live + known -> spoof images will be accepted
        pipeline.check_access = fake_check
        report = evaluate.run_evaluation()
        pipeline.check_access = real_check
        o = report["overall"]
        check("eval counts images", report["total_images"] == 6)
        check("eval false reject 0%", o["false_reject_rate_pct"] == 0.0)
        check("eval false accept 100% when spoofs pass", o["false_accept_rate_pct"] == 100.0)
        check("eval per-scenario table", set(report["scenarios"]) == {"genuine", "printed_photo", "screen_photo"})
        check("eval report saved", evaluate.load_report()["total_images"] == 6)

        rek_stub.assert_no_pending_responses()
        br_stub.assert_no_pending_responses()
    print(f"\nALL {passed} CHECKS PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
