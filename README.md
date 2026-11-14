# HID Liveness Kiosk: "Can You Spot a Spoof?"

**Flow:** capture → liveness (AWS Rekognition Face Liveness) → face authentication (Rekognition face search)
→ campus schedule → **ALLOW / STEP-UP VERIFICATION / DENY**, decision time measured (target ≤ 2 s).
**Offline:** safe local process (authenticator code + schedule, never face-only), minimum data, sync when back.

## Run (two terminals, both with the venv active and the sandbox `$Env:` lines pasted)
```
pip install -r requirements.txt
python selftest.py            # ALL CHECKS PASSED
python server.py              # terminal 1: kiosk at http://localhost:8000
streamlit run app.py          # terminal 2: admin console (enroll, schedule, offline codes, performance, log)
```

## Files
| File | What it does |
|---|---|
| `server.py` + `kiosk_web/dist/` | Kiosk web page (AWS Face Liveness component, pre-built) + API |
| `official_liveness.py` | Create liveness session, read result + reference face image |
| `pipeline.py` | liveness + identity + schedule → decision, timing breakdown, log |
| `rules.py` | ALLOW / STEP-UP (challenge or guard) / DENY |
| `face_service.py` | Rekognition enroll / identify |
| `liveness.py` | Backup passive checks + differential random-action challenge (photo kiosk) |
| `store.py` | People, campus schedule, temporary passes, access log |
| `offline.py` | Offline detection, TOTP fallback, minimal queue, sync |
| `genai.py` | Bedrock explanations, admin Q&A, report summary, vision spoof inspector (stricter only) |
| `evaluate.py` | Liveness accuracy, face-auth accuracy, FAR/FRR, APCER/BPCER, decision time per attack |
| `app.py` | Admin console + photo backup kiosk |
| `kiosk_web/src/` | Source of the kiosk page (only needed to rebuild: `npm install && npm run build`) |

Never commit keys, photos, `data/` or `test_data/`.
