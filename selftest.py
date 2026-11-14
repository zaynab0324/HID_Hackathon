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

import offline  # noqa: E402

config.REQUIRE_ACTIVE_CHECK = False   # most tests check the passive rules; security mode tested below
config.USE_GENAI_INSPECTOR = False    # tested separately below with a fake inspector
store.PEOPLE_FILE = os.path.join(config.DATA_DIR, "people.json")
store.SCHEDULE_FILE = os.path.join(config.DATA_DIR, "schedule.json")
store.LOG_FILE = os.path.join(config.DATA_DIR, "access_log.json")
offline.SECRETS_FILE = os.path.join(config.DATA_DIR, "offline_secrets.json")
offline.QUEUE_FILE = os.path.join(config.DATA_DIR, "offline_queue.json")
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
        store.upsert_person("afrah", status="active", role="staff")
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

        # ---------- security mode ----------
        config.REQUIRE_ACTIVE_CHECK = True
        scene()
        r = pipeline.check_access(IMG, use_ai=False)
        check("security mode: passive alone -> REQUEST_VERIFICATION",
              r["verdict"] == "REQUEST_VERIFICATION" and r["challenge"] is not None)
        scene(labels=[{"Name": "Mobile Phone", "Instances": [{"BoundingBox": AROUND_FACE}]}])
        check("security mode: obvious spoof still DENY",
              pipeline.check_access(IMG, use_ai=False)["verdict"] == "DENY")
        sm_challenge = {"id": "s1", "action": "smile", "instruction": "Smile",
                        "expires_at": time.time() + 30}
        scene(face_extra={"Smile": {"Value": True, "Confidence": 97}})
        check("security mode: challenge passed -> ALLOW",
              pipeline.verify_challenge(sm_challenge, IMG, use_ai=False)["verdict"] == "ALLOW")
        config.REQUIRE_ACTIVE_CHECK = False

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

        # ---------- campus schedule ----------
        from datetime import datetime, timedelta
        monday_10 = datetime(2026, 11, 16, 10, 0)            # a Monday
        store.upsert_person("riya", status="active", role="student")
        store.add_slot("riya", "Lab 3", [0], "09:00", "11:00")
        check("schedule in slot", store.check_schedule("riya", "Lab 3", monday_10)["status"] == "in_schedule")
        check("schedule near slot", store.check_schedule("riya", "Lab 3", monday_10.replace(hour=11, minute=10))["status"] == "near_schedule")
        check("schedule out of slot", store.check_schedule("riya", "Lab 3", monday_10.replace(hour=15))["status"] == "out_of_schedule")
        check("schedule wrong room", store.check_schedule("riya", "Lab 5", monday_10)["status"] == "out_of_schedule")
        store.upsert_person("riya", temp_pass={"location": "Lab 5", "until": (monday_10 + timedelta(hours=1)).isoformat()})
        check("temporary pass", store.check_schedule("riya", "Lab 5", monday_10)["status"] == "in_schedule")

        scene(pid="riya")
        r = pipeline.check_access(IMG, use_ai=False, location="Lab 3", now=monday_10.replace(hour=15))
        check("live + known + out of schedule -> DENY", r["verdict"] == "DENY" and r["schedule"]["status"] == "out_of_schedule")
        r = pipeline.check_access(IMG, use_ai=False, location="Lab 3", now=monday_10.replace(hour=11, minute=10))
        check("near schedule -> STEP-UP by guard", r["verdict"] == "REQUEST_VERIFICATION" and r["verification"] == "guard")
        r = pipeline.check_access(IMG, use_ai=False, location="Lab 3", now=monday_10)
        check("in schedule -> ALLOW", r["verdict"] == "ALLOW")
        r = pipeline.check_access(IMG, use_ai=False, location="Lab 3", now=monday_10.replace(hour=15), use_schedule=False)
        check("schedule can be skipped (evaluation)", r["verdict"] == "ALLOW")
        check("timing breakdown present", {"identity", "face_attributes", "object_detection", "total"} <= set(r["timing"]))

        # ---------- differential challenge ----------
        smiling = {"BoundingBox": FACE_BOX, "Smile": {"Value": True, "Confidence": 97}}
        for _ in range(20):
            check_c = liveness.new_challenge(smiling)
            assert check_c["action"] != "smile"
        check("challenge never asks for what you already do", True)
        static = {"id": "d1", "action": "smile", "instruction": "Smile", "expires_at": time.time() + 30,
                  "baseline": {"smile": True, "eyes_open": True, "mouth_open": False, "yaw": 0.0}}
        scene(face_extra={"Smile": {"Value": True, "Confidence": 97}})
        r = pipeline.verify_challenge(static, IMG, use_ai=False)
        check("no change from first photo -> DENY", r["verdict"] == "DENY")

        # ---------- GenAI inspector (can only make it stricter) ----------
        config.USE_GENAI_INSPECTOR = True
        real_opinion = genai.spoof_opinion
        genai.spoof_opinion = lambda img: {"spoof": True, "spoof_type": "screen_replay", "confidence": 0.95, "reason": "Phone bezel visible."}
        scene()
        r = pipeline.check_access(IMG, use_ai=False)
        check("inspector sure spoof -> DENY", r["verdict"] == "DENY" and r["liveness"]["spoof_type"] == "screen_replay")
        genai.spoof_opinion = lambda img: {"spoof": True, "spoof_type": None, "confidence": 0.6, "reason": "Maybe a screen."}
        r = pipeline.check_access(IMG, use_ai=False)
        check("inspector unsure -> STEP-UP", r["verdict"] == "REQUEST_VERIFICATION")
        genai.spoof_opinion = lambda img: {"spoof": False, "spoof_type": None, "confidence": 0.99, "reason": "Live."}
        scene(labels=[{"Name": "Mobile Phone", "Instances": [{"BoundingBox": AROUND_FACE}]}])
        r = pipeline.check_access(IMG, use_ai=False)
        check("inspector can never loosen a DENY", r["verdict"] == "DENY")
        genai.spoof_opinion = real_opinion
        config.USE_GENAI_INSPECTOR = False

        # ---------- offline mode ----------
        import pyotp
        offline.set_simulated_offline(True)
        check("simulated offline", offline.is_online() is False)
        secret = offline.enroll_code("riya")["secret"]
        good = pyotp.TOTP(secret).now()
        r = offline.offline_decide("riya", good, "Lab 3", monday_10, faces_in_view=1)
        check("offline: code + schedule -> ALLOW", r["verdict"] == "ALLOW" and r["mode"] == "offline")
        r = offline.offline_decide("riya", "000000" if good != "000000" else "111111", "Lab 3", monday_10, 1)
        check("offline: wrong code -> DENY", r["verdict"] == "DENY")
        r = offline.offline_decide("riya", pyotp.TOTP(secret).now(), "Lab 3", monday_10.replace(hour=15), 1)
        check("offline: out of schedule -> DENY", r["verdict"] == "DENY")
        r = offline.offline_decide("riya", pyotp.TOTP(secret).now(), "Lab 3", monday_10, faces_in_view=2)
        check("offline: two people -> DENY", r["verdict"] == "DENY")
        check("offline: never allows without a code", offline.offline_decide("afrah", "", "Lab 3", monday_10, 1)["verdict"] == "DENY")
        check("offline: decisions queued (no photos)", len(offline.pending()) == 5 and "image" not in offline.pending()[0])
        check("offline: no sync while offline", offline.sync() == 0)
        offline.set_simulated_offline(False)
        real_online = offline.is_online
        offline.is_online = lambda force=False: True
        before = len(store.recent_events(2000))
        check("sync when back online", offline.sync() == 5 and offline.pending() == [])
        check("synced into central log", len(store.recent_events(2000)) == before + 5)

        # ---------- kiosk server API ----------
        from fastapi.testclient import TestClient
        import server
        client = TestClient(server.app)
        check("api status", client.get("/api/status").json()["online"] is True)
        real_get, real_check_official = official_liveness.get_result, pipeline.check_official
        official_liveness.get_result = lambda sid: ({"live": True, "score": 0.98, "spoof_type": None,
                                                     "signals": {"aws_confidence": 98.0}, "source": "aws_face_liveness",
                                                     "reasons": ["AWS Face Liveness confidence 98.0%."]}, IMG)
        scene()
        res = client.post("/api/decide", json={"session_id": "a" * 36}).json()
        check("api decide -> ALLOW with timing", res["verdict"] == "ALLOW" and res["timing"]["total"] == res["decision_ms"])
        official_liveness.get_result = lambda sid: ({"live": False, "score": 0.1, "spoof_type": "presentation_attack",
                                                     "signals": {}, "source": "aws_face_liveness", "reasons": ["low"]}, IMG)
        check("api decide spoof -> DENY", client.post("/api/decide", json={"session_id": "a" * 36}).json()["verdict"] == "DENY")
        official_liveness.get_result = real_get
        offline.is_online = lambda force=False: False
        check("api session offline -> 503", client.post("/api/session", json={}).status_code == 503)
        r = client.post("/api/offline/decide", json={"person_id": "riya", "code": "12"}).json()
        check("api offline decide", r["verdict"] == "DENY" and r["mode"] == "offline")
        offline.is_online = real_online
        offline.QUEUE_FILE = os.path.join(config.DATA_DIR, "offline_queue.json")
        official_liveness.get_result = lambda sid: ({"live": True, "score": 0.97, "spoof_type": None,
                                                     "signals": {"aws_confidence": 97.0}, "source": "aws_face_liveness",
                                                     "reasons": []}, IMG)
        offline.is_online = lambda force=False: True
        scene()
        client.post("/api/decide", json={"session_id": "a" * 36, "test_label": "genuine"})
        official_liveness.get_result = lambda sid: ({"live": False, "score": 0.05, "spoof_type": "presentation_attack",
                                                     "signals": {"aws_confidence": 5.0}, "source": "aws_face_liveness",
                                                     "reasons": []}, IMG)
        client.post("/api/decide", json={"session_id": "a" * 36, "test_label": "screen_photo"})
        official_liveness.get_result = real_get
        offline.is_online = real_online
        live_rep = evaluate.live_kiosk_report(store.recent_events(2000))
        check("live kiosk report: labelled attempts", live_rep["attempts"] == 2)
        check("live kiosk report: metrics", live_rep["overall"]["liveness_accuracy_pct"] == 100.0
              and live_rep["overall"]["false_accept_rate_pct"] == 0.0)
        check("api override", client.post("/api/override", json={"person_id": "riya", "verdict": "ALLOW"}).json()["verdict"] == "ALLOW")

        # ---------- evaluation ----------
        empty = evaluate.run_evaluation(os.path.join(TMP, "no_such_folder"))
        check("evaluation with no test data does not crash", empty["total_images"] == 0)
        for folder in ("genuine", "printed_photo", "screen_photo"):
            os.makedirs(os.path.join(config.TEST_DATA_DIR, folder))
            for i in range(2):
                with open(os.path.join(config.TEST_DATA_DIR, folder, f"{i}.jpg"), "wb") as f:
                    f.write(IMG)
        real_check = pipeline.check_access

        def fake_check(image_bytes, use_ai=True, log=True, **kw):
            return real_check(image_bytes, use_ai=False, log=False, use_schedule=False)

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
        check("eval judges' metrics present", {"liveness_accuracy_pct", "face_auth_accuracy_pct", "apcer_pct",
                                               "bpcer_pct", "false_accept_rate_pct", "false_reject_rate_pct"} <= set(o))

        rek_stub.assert_no_pending_responses()
        br_stub.assert_no_pending_responses()
    print(f"\nALL {passed} CHECKS PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
