"""Staff admin console. Run:  streamlit run app.py
(The kiosk at the door is server.py -> http://localhost:8000)
"""
import io
import os
from datetime import datetime, timedelta

import streamlit as st

import config
import evaluate
import face_service
import genai
import offline
import pipeline
import store

st.set_page_config(page_title="HID Kiosk Admin", page_icon="🛡️", layout="wide")

st.title("🛡️ HID Kiosk: Admin Console")
st.caption("Checks that a REAL person is present (liveness) AND that it is the RIGHT person "
           "(identity), in under 2 seconds. AI makes the decision; uncertain cases get an extra "
           "verification step. Raw photos are not stored; face vectors can be deleted anytime.")

def aws_call(fn, *args, default=None):
    """Run an AWS call; show a clear message instead of a crash if credentials/network fail."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        st.error(f"AWS problem: {type(e).__name__}: {str(e)[:200]}  \n"
                 "Fix: in this terminal run `aws sts get-caller-identity`. If it fails, refresh the "
                 "sandbox credentials (aws configure set ...), then restart this app.")
        return default


pipeline.warm_up()
st.info("👉 The kiosk at the door (AWS Face Liveness) runs at **http://localhost:8000** "
        "(`python server.py`). This is the staff admin console.")

enroll_tab, sched_tab, offline_tab, perf_tab, admin_tab = st.tabs(
    ["Enroll", "Schedule", "Offline mode", "Performance", "Admin"])

# ======================================================== Enroll
with enroll_tab:
    person_input = st.text_input("Person ID (e.g. Afrah Zainab)")
    status = st.selectbox("Access status", ["active", "suspended"])
    role = st.selectbox("Role", ["student", "staff"], help="Staff can enter at any time.")
    photo = st.camera_input("Enrollment photo (real person, good light)", key="enroll_cam")
    if st.button("Enroll"):
        if not person_input:
            st.warning("Type a Person ID first.")
        elif not photo:
            st.warning("Take a photo first.")
        else:
            with st.spinner("Enrolling..."):
                result = aws_call(face_service.enroll, photo.getvalue(), person_input,
                                  default={"ok": False, "message": "Enrollment failed (see above)."})
            if result["ok"]:
                store.upsert_person(result["person_id"], status=status, role=role)
                st.success(result["message"] + ". Now add a schedule slot (Schedule tab) and an "
                           "offline code (Offline mode tab).")
            else:
                st.error(result["message"])

# ======================================================== Schedule
with sched_tab:
    st.write(f"This kiosk guards **{config.KIOSK_LOCATION}**. A recognised person is only allowed "
             "in when the schedule says so. Within "
             f"{config.NEAR_SCHEDULE_MINUTES} min of a slot -> step-up (guard approval).")
    people_ids = sorted(p["person_id"] for p in store.list_persons())
    if not people_ids:
        st.info("Enroll someone first.")
    else:
        c1, c2 = st.columns(2)
        who_s = c1.selectbox("Person", people_ids, key="sched_person")
        where = c2.selectbox("Location", config.LOCATIONS, key="sched_loc")
        days = st.multiselect("Days", list(range(7)), default=[datetime.now().weekday()],
                              format_func=lambda d: store.DAY_NAMES[d])
        c3, c4 = st.columns(2)
        hour_now = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_t = c3.time_input("From", value=hour_now.time(), step=900)
        end_t = c4.time_input("To", value=(hour_now + timedelta(hours=2)).time(), step=900)
        if st.button("Add slot"):
            store.add_slot(who_s, where, days, start_t.strftime("%H:%M"), end_t.strftime("%H:%M"))
            st.success("Slot added.")
        st.subheader("Temporary pass (e.g. lab timetable not updated)")
        c5, c6 = st.columns(2)
        pass_loc = c5.selectbox("Pass location", config.LOCATIONS, key="pass_loc")
        pass_hours = c6.number_input("Valid for (hours)", 1, 12, 2)
        if st.button("Give temporary pass"):
            until = (datetime.now() + timedelta(hours=int(pass_hours))).isoformat(timespec="minutes")
            store.upsert_person(who_s, temp_pass={"location": pass_loc, "until": until})
            st.success(f"{who_s} may enter {pass_loc} until {until[11:16]}.")
        st.subheader("Check now")
        res = store.check_schedule(who_s, config.KIOSK_LOCATION, datetime.now())
        st.write(f"**{who_s}** at **{config.KIOSK_LOCATION}** now: `{res['status']}`: {res['reason']}")
    slots = store.list_slots()
    if slots:
        st.dataframe([{**s_, "days": ", ".join(store.DAY_NAMES[d] for d in s_["days"])} for s_ in slots],
                     width="stretch")
        gone = st.selectbox("Delete slot", [""] + [s_["slot_id"] for s_ in slots])
        if gone and st.button("Delete"):
            store.delete_slot(gone)
            st.rerun()

# ======================================================== Offline mode
with offline_tab:
    online_now = offline.is_online(force=True)
    st.write(("🟢 **Online**" if online_now else "🔴 **Offline**")
             + f" · {len(offline.pending())} decision(s) waiting to sync")
    sim = st.toggle("Simulate network loss (demo)", value=offline.is_simulated_offline())
    if sim != offline.is_simulated_offline():
        offline.set_simulated_offline(sim)
        st.rerun()
    st.caption("Note: the kiosk server (python server.py) has its own switch: "
               "POST /api/offline/simulate, or stop the Wi-Fi to test for real.")
    if st.button("Sync now"):
        st.success(f"Synced {offline.sync()} decision(s) to the central log.")
    st.subheader("Set up a person's offline code")
    st.write("Scan the QR code with Google/Microsoft Authenticator. Offline, the person enters the "
             "6-digit code; it is accepted only within their schedule. No photos are stored offline.")
    ids = sorted(p["person_id"] for p in store.list_persons())
    if ids:
        who_o = st.selectbox("Person", ids, key="otp_person")
        if st.button("Create / show code"):
            info = offline.enroll_code(who_o)
            import qrcode
            buf = io.BytesIO()
            qrcode.make(info["otpauth_uri"]).save(buf, format="PNG")
            st.image(buf.getvalue(), width=220)
            st.code(info["secret"], language=None)
    if offline.pending():
        st.subheader("Waiting to sync (minimum data only)")
        st.dataframe(offline.pending(), width="stretch")

# ======================================================== Performance
with perf_tab:
    st.subheader("1. Live kiosk results (real AWS Face Liveness)")
    st.write("On the kiosk page (localhost:8000), pick a **🧪 Test mode** label at the bottom "
             "(genuine, printed photo, screen photo, video replay, mask, impostor), then do the "
             "attempt. Every labelled attempt is measured here.")
    live_rep = evaluate.live_kiosk_report(store.recent_events(2000))
    if not live_rep:
        st.info("No labelled kiosk attempts yet.")
    else:
        o = live_rep["overall"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Liveness detection accuracy", f"{o['liveness_accuracy_pct']}%")
        c2.metric("Face authentication accuracy", f"{o['face_auth_accuracy_pct']}%")
        c3.metric("False acceptance rate (FAR)", f"{o['false_accept_rate_pct']}%")
        c4.metric("False rejection rate (FRR)", f"{o['false_reject_rate_pct']}%")
        c5, c6 = st.columns(2)
        c5.metric("Avg decision time", f"{o['avg_decision_ms']} ms")
        c6.metric("Decisions within 2 s", f"{o['under_2s_pct']}%")
        st.caption(f"{live_rep['attempts']} labelled attempts. Percentages are 'None' until that "
                   "kind of attempt exists (e.g. FRR needs genuine attempts).")
        st.dataframe([{"scenario": k, **v} for k, v in live_rep["per_scenario"].items()],
                     width="stretch")
        if st.button("AI summary for judges (live kiosk)"):
            with st.spinner("Summarising..."):
                st.write(genai.summarize_performance(live_rep))

    st.divider()
    st.subheader("2. Image test set (backup photo checks + identity)")
    st.caption("Test images are personal data: consented team members only; delete the "
               "test_data folder after the event.")
    counts = evaluate.count_images()
    st.write("Images per scenario: " + " · ".join(f"**{k}** {v}" for k, v in counts.items()))
    c1, c2 = st.columns(2)
    scen = c1.selectbox("Scenario for the next photo", evaluate.SCENARIOS, key="eval_scen")
    pid = c2.text_input("Person ID shown (for genuine)", key="eval_pid")
    shot = st.camera_input("Take a test photo", key="eval_cam")
    if shot and st.button("Save photo to test set"):
        folder = os.path.join(config.TEST_DATA_DIR, scen)
        prefix = (pid.strip().replace(" ", "_") + "__") if (scen == "genuine" and pid.strip()) else ""
        name = f"{prefix}{scen}_{datetime.now().strftime('%H%M%S')}.jpg"
        with open(os.path.join(folder, name), "wb") as f:
            f.write(shot.getvalue())
        st.success(f"Saved {scen}/{name}. Click 'Clear photo' to take the next one.")
    if st.button("Run evaluation on the image test set"):
        if sum(counts.values()) == 0:
            st.warning("No test images yet. Save some photos above first.")
        else:
            with st.spinner("Evaluating all test images..."):
                aws_call(evaluate.run_evaluation)
    report = evaluate.load_report()
    if report and report.get("total_images"):
        o = report["overall"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Liveness accuracy (photo checks)", f"{o.get('liveness_accuracy_pct')}%")
        c2.metric("Face authentication accuracy", f"{o.get('face_auth_accuracy_pct')}%")
        c3.metric("APCER (attack passed as live)", f"{o.get('apcer_pct')}%")
        c4.metric("BPCER (real person flagged)", f"{o.get('bpcer_pct')}%")
        st.caption(f"{report['total_images']} images · generated {report['generated']}")
        st.dataframe([{"scenario": name, **vals} for name, vals in report["scenarios"].items()],
                     width="stretch")

# ======================================================== Admin
with admin_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("People")
        faces = aws_call(face_service.list_people, default={}) or {}
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
                n = aws_call(face_service.remove_person, who, default=0)
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
