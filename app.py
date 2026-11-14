"""HID entry kiosk UI. Run:  streamlit run app.py

OWNER: Afrah (integration). Talks to other files only through their contracts.
"""
import hashlib
import time

import cv2
import streamlit as st

import config
import evaluate
import face_service
import genai
import pipeline
import store

st.set_page_config(page_title="HID Liveness Kiosk", page_icon="🛡️", layout="wide")

LABELS = {"ALLOW": "✅ ALLOW", "REQUEST_VERIFICATION": "⚠️ REQUEST VERIFICATION",
          "DENY": "⛔ DENY", "RETRY": "🔄 TRY AGAIN"}


@st.cache_resource
def get_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def result_text(r):
    live = r["liveness"]
    lines = [f"### {LABELS[r['verdict']]}", r["explanation"]]
    if r["verdict"] != "RETRY":
        speed = "✅" if r["decision_ms"] <= config.DECISION_BUDGET_MS else "❌"
        identity = f"{r['person_id']} ({r['similarity']}%)" if r["person_id"] else "no match"
        lines.append(f"⏱️ Decision time: **{r['decision_ms'] / 1000:.2f} s** {speed} · "
                     f"🧬 Liveness: **{live['score']}** ({live['source']}) · 🪪 Identity: **{identity}**")
        if live.get("spoof_type"):
            lines.append(f"Spoof type: **{live['spoof_type'].replace('_', ' ')}**")
    return "\n\n".join(lines)


def show_result(box, r):
    text = result_text(r)
    {"ALLOW": box.success, "DENY": box.error,
     "REQUEST_VERIFICATION": box.warning}.get(r["verdict"], box.info)(text)


def new_photo(key, photo):
    """True only the first time we see this exact photo (avoids repeat AWS calls on reruns)."""
    digest = hashlib.md5(photo.getvalue()).hexdigest()
    if st.session_state.get(key) == digest:
        return False
    st.session_state[key] = digest
    return True


st.title("🛡️ HID Liveness Kiosk")
st.caption("Checks that a REAL person is present (liveness) AND that it is the RIGHT person "
           "(identity), in under 2 seconds. AI makes the decision; uncertain cases get an extra "
           "verification step. Raw photos are not stored; face vectors can be deleted anytime.")

kiosk_tab, live_tab, enroll_tab, perf_tab, admin_tab = st.tabs(
    ["Kiosk (photo)", "Live Kiosk", "Enroll", "Performance", "Admin"])

# ======================================================== Kiosk (photo)
with kiosk_tab:
    use_ai = st.toggle("Explain decisions with GenAI", value=True, key="kiosk_ai")
    challenge = st.session_state.get("challenge")

    if challenge is None:
        photo = st.camera_input("Look at the camera", key="kiosk_cam")
        if photo and new_photo("kiosk_seen", photo):
            with st.spinner("Checking liveness and identity..."):
                st.session_state["kiosk_result"] = pipeline.check_access(photo.getvalue(), use_ai=use_ai)
            if st.session_state["kiosk_result"]["challenge"]:
                st.session_state["challenge"] = st.session_state["kiosk_result"]["challenge"]
                st.rerun()
        if st.session_state.get("kiosk_result"):
            show_result(st, st.session_state["kiosk_result"])
    else:
        st.warning(f"## Extra verification\n# 👉 {challenge['instruction']}\n"
                   f"Then take the photo (within {config.CHALLENGE_SECONDS} s).")
        photo = st.camera_input("Do the action, then take the photo", key="challenge_cam")
        if photo and new_photo("challenge_seen", photo):
            with st.spinner("Verifying..."):
                st.session_state["kiosk_result"] = pipeline.verify_challenge(
                    challenge, photo.getvalue(), use_ai=use_ai)
            st.session_state["challenge"] = None
            st.rerun()
        if st.button("Cancel verification"):
            st.session_state["challenge"] = None
            st.rerun()

    r = st.session_state.get("kiosk_result")
    if r and r["verdict"] in ("REQUEST_VERIFICATION", "DENY") and challenge is None:
        c1, c2 = st.columns(2)
        if c1.button("Guard override: allow"):
            pipeline.record_override(r["person_id"], "ALLOW", by="guard")
            st.success("Allowed by guard and logged.")
        if c2.button("Guard override: deny"):
            pipeline.record_override(r["person_id"], "DENY", by="guard")
            st.error("Denied by guard and logged.")

# ======================================================== Live Kiosk
with live_tab:
    st.write("Turn on and stand in front of the camera. It checks automatically.")
    st.caption("Turn it off before using other tabs: only one thing can use the webcam at a time.")
    live = st.toggle("Start live kiosk")
    frame_box, prompt_box, result_box = st.empty(), st.empty(), st.empty()

    if live:
        detector = get_face_detector()
        cam = (cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW")
               else cv2.VideoCapture(0))
        face_since, last_check, pending = None, 0.0, None
        try:
            while True:
                ok, frame = cam.read()
                if not ok:
                    result_box.error("Can't read the webcam. Close other apps or tabs using it.")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(80, 80))
                now = time.time()

                if pending and now > pending["expires_at"]:
                    prompt_box.empty()
                    result_box.error("### ⛔ DENY\nVerification step timed out.")
                    pending, last_check = None, now

                if len(faces) == 1:
                    face_since = face_since or now
                    steady = now - face_since >= config.STEADY_SECONDS
                    if pending and now - pending["issued"] >= 2.0:
                        _, jpg = cv2.imencode(".jpg", frame)
                        result = pipeline.verify_challenge(pending, jpg.tobytes(), use_ai=False)
                        if result["verdict"] == "ALLOW" or now - pending["issued"] > 6:
                            prompt_box.empty()
                            show_result(result_box, result)
                            pending, last_check = None, now
                    elif not pending and steady and now - last_check >= config.COOLDOWN_SECONDS:
                        last_check = now
                        _, jpg = cv2.imencode(".jpg", frame)
                        result = pipeline.check_access(jpg.tobytes(), use_ai=False)
                        show_result(result_box, result)
                        if result["challenge"]:
                            pending = {**result["challenge"], "issued": now}
                            prompt_box.warning(f"# 👉 {pending['instruction']}")
                else:
                    face_since = None
                    if len(faces) > 1:
                        result_box.warning("One person at a time, please.")
                    elif not pending and now - last_check >= config.COOLDOWN_SECONDS:
                        result_box.info("Waiting for a person...")

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame_box.image(frame, channels="BGR")
                time.sleep(0.03)
        finally:
            cam.release()

# ======================================================== Enroll
with enroll_tab:
    person_input = st.text_input("Person ID (e.g. Afrah Zainab)")
    status = st.selectbox("Access status", ["active", "suspended"])
    photo = st.camera_input("Enrollment photo (real person, good light)", key="enroll_cam")
    if st.button("Enroll"):
        if not person_input:
            st.warning("Type a Person ID first.")
        elif not photo:
            st.warning("Take a photo first.")
        else:
            with st.spinner("Enrolling..."):
                result = face_service.enroll(photo.getvalue(), person_input)
            if result["ok"]:
                store.upsert_person(result["person_id"], status=status)
                st.success(result["message"])
            else:
                st.error(result["message"])

# ======================================================== Performance
with perf_tab:
    st.write(f"Runs every image in `{config.TEST_DATA_DIR}/` through the real pipeline.")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating all test images..."):
            evaluate.run_evaluation()
    report = evaluate.load_report()
    if not report:
        st.info("No report yet. Add test images and click **Run evaluation** "
                "(or run `python evaluate.py`).")
    else:
        o = report["overall"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Spoof detection", f"{o['spoof_detection_pct']}%")
        c2.metric("False acceptance", f"{o['false_accept_rate_pct']}%")
        c3.metric("False rejection", f"{o['false_reject_rate_pct']}%")
        c4.metric("Avg decision", f"{o['avg_decision_ms']} ms")
        c5.metric("Within 2 s", f"{o['under_2s_pct']}%")
        st.caption(f"{report['total_images']} images · generated {report['generated']}")
        rows = [{"scenario": name, **vals} for name, vals in report["scenarios"].items()]
        st.dataframe(rows, width="stretch")
        if st.button("AI summary for judges"):
            with st.spinner("Summarising..."):
                st.write(genai.summarize_performance(report))

# ======================================================== Admin
with admin_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("People")
        faces = face_service.list_people()
        records = {p["person_id"]: p for p in store.list_persons()}
        st.dataframe([{"person_id": pid, "face_samples": n,
                       "status": records.get(pid, {}).get("status", "active")}
                      for pid, n in sorted(faces.items())], width="stretch")
        if faces:
            who = st.selectbox("Select person", sorted(faces))
            new_status = st.selectbox("Set status", ["active", "suspended"], key="admin_status")
            c1, c2 = st.columns(2)
            if c1.button("Update status"):
                store.upsert_person(who, status=new_status)
                st.success(f"{who} is now {new_status}.")
            if c2.button("Delete face data"):
                n = face_service.remove_person(who)
                store.delete_person(who)
                st.success(f"Deleted {n} face sample(s) for {who}.")
    with right:
        st.subheader("Ask the access log (GenAI)")
        question = st.text_input("e.g. Which spoof attempts happened today?")
        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                st.write(genai.ask(question, store.recent_events(200)))

    st.subheader("Access log")
    events = store.recent_events(100)
    st.dataframe([{
        "time": e.get("time"), "verdict": e.get("verdict"), "person": e.get("person_id"),
        "similarity": e.get("similarity"),
        "liveness": (e.get("liveness") or {}).get("score"),
        "spoof_type": (e.get("liveness") or {}).get("spoof_type"),
        "ms": e.get("decision_ms"), "explanation": e.get("explanation"),
    } for e in events], width="stretch")
