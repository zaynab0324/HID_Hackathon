# HID Liveness Kiosk: "Can You Spot a Spoof?"

**Flow:** capture → liveness (AWS Rekognition Face Liveness) → face authentication (Rekognition face search)
→ campus schedule → **ALLOW / STEP-UP VERIFICATION / DENY**, decision time measured (target ≤ 2 s).

**Offline (Challenge 2):** when AWS is unreachable the kiosk switches automatically to an on-device
OpenCV face model (YuNet + SFace) with **random-action liveness** (2 random actions in random order, always
including a head turn: turn left / turn right / blink / move closer). Fails safe: weak match → 6-digit authenticator code, out of schedule → DENY.
No photos stored; decisions are queued and synced when the connection returns.

## Setup and run (Windows PowerShell, two terminals, venv active, AWS credentials configured)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python get_models.py          # REQUIRED once: downloads the offline face models (~39 MB) into models/
                              # (not included in this zip because of the upload size limit)

python selftest.py            # ALL CHECKS PASSED
python server.py              # terminal 1: kiosk at http://localhost:8000
streamlit run app.py          # terminal 2: admin console (enroll, schedule, offline, performance, security, log)
```
Without `python get_models.py` everything online still works, but the offline face check shows
"Offline face models are missing" and only the ID + 6-digit code fallback is available.

Model files downloaded by `get_models.py` (from the official OpenCV Model Zoo):
- `models/face_detection_yunet_2023mar.onnx`  face detection + 5 landmarks
- `models/face_recognition_sface_2021dec.onnx`  128-number face vector for recognition

## Demo of offline mode
1. Enroll a person from the admin console (Enroll tab). This also saves their offline face vector.
2. Tick **"Simulate AWS outage"** at the bottom of the kiosk page (or the switch in the admin Offline tab).
3. The kiosk switches to **"Offline mode: local face check"** → Start → look straight → do the 2 random actions.
4. Untick to go back online; queued offline decisions sync into the access log.

## Files
| File | What it does |
|---|---|
| `server.py` + `kiosk_web/dist/` | Kiosk web page (AWS Face Liveness component, pre-built) + API |
| `official_liveness.py` | Create liveness session, read result + reference face image |
| `pipeline.py` | liveness + identity + schedule → decision, timing breakdown, log |
| `rules.py` | ALLOW / STEP-UP (challenge or guard) / DENY |
| `face_service.py` | Rekognition enroll / identify |
| `security.py` | Least-privilege kiosk credentials, single-use sessions, threat mode, impossible travel, alerts, enrollment tokens |
| `presence.py` | Check-in / check-out: who is inside which lab; blocks entry at a 2nd lab while checked in elsewhere |
| `local_face.py` | **Offline** face detection + recognition (OpenCV YuNet + SFace) and random-action liveness |
| `get_models.py` | Downloads the two offline model files into `models/` |
| `offline.py` | Offline detection, offline face decision, TOTP fallback, minimal queue, sync |
| `liveness.py` | Backup passive checks + differential random-action challenge |
| `store.py` | People, campus schedule, temporary passes, tamper-evident access log |
| `genai.py` | Bedrock explanations, admin Q&A, report summary, incident report |
| `voice.py` | Spoken feedback (Amazon Polly) |
| `evaluate.py` | Liveness accuracy, face-auth accuracy, FAR/FRR, APCER/BPCER, decision time per attack |
| `app.py` | Admin console (Streamlit) |
| `selftest.py` | Offline self-test of all logic (no AWS needed) |
| `kiosk_web/src/` | Source of the kiosk page (only needed to rebuild: `npm install && npm run build`) |

Never commit keys, photos, `data/` or `test_data/`.
