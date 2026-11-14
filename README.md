# HID Liveness Kiosk: "Can You Spot a Spoof?"

**Flow:** capture → liveness detection → face authentication → decision (ALLOW / REQUEST VERIFICATION / DENY), in under 2 seconds.

**Adaptive trust:** every person gets a fast check (liveness clues + identity, run in parallel).
Only uncertain cases get an extra step: a random action (turn head, smile...) or official AWS Face Liveness.

## Run
```
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass      # Windows, once per terminal
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python selftest.py                              # must say ALL CHECKS PASSED
aws sts get-caller-identity                     # must show the team account
streamlit run app.py
```

## Files and owners (one owner per file = no merge conflicts)
| File | What it does | Owner |
|---|---|---|
| `config.py` | All settings and thresholds | Afrah |
| `liveness.py` | Passive spoof clues + random-action challenge | Afrah |
| `rules.py` | ALLOW / REQUEST_VERIFICATION / DENY logic | Afrah |
| `pipeline.py` | capture → liveness → identity → decision → log, with timing | Afrah |
| `app.py` | Kiosk UI | Afrah |
| `official_liveness.py` + `official_liveness_web/` | AWS Face Liveness (strong step) | Aman |
| `face_service.py` | Rekognition enroll / identify | Aman |
| `evaluate.py` | Performance report per spoof type | Abhi |
| `genai.py` | Bedrock explanations, admin Q&A, report summary | Abhi |
| `store.py` | People records + access log | Abhi |
| `selftest.py` | Offline contract tests (no AWS) | Everyone |

## Contracts (never change names, arguments or return keys without telling the team)
- `liveness.passive_check(img, face_details, labels)` / `liveness.verify_challenge(challenge, face_details)` → LivenessResult `{"live": True|False|None, "score", "spoof_type", "signals", "source", "reasons"}`
- `face_service.identify(img)` → `{"status": MATCH|POSSIBLE|NO_MATCH|NO_FACE, "person_id", "similarity", "message"}`
- `rules.decide(face_result, liveness_result, person, context)` → `{"verdict", "reasons", "needs_verification"}`
- `pipeline.check_access(img)` / `pipeline.verify_challenge(challenge, img)` / `pipeline.check_official(live_result, ref_img)` → AccessResult `{"verdict", "reasons", "person_id", "similarity", "liveness", "decision_ms", "challenge", "explanation"}`
- `official_liveness.create_session()` → `session_id`; `official_liveness.get_result(session_id)` → `(LivenessResult, reference_image_bytes)`
- `evaluate.run_evaluation()` → report; `genai.*` functions never raise

## Test data (never committed, faces are personal data)
`test_data/genuine/`, `test_data/printed_photo/`, `test_data/screen_photo/`, `test_data/video_replay/`, `test_data/mask/`

## Team rules
1. `git pull` before you start.
2. Edit only the files you own.
3. `python selftest.py` → ALL CHECKS PASSED before every push.
4. `git add .` → `git commit -m "..."` → `git push`.
5. Never commit keys (.csv), photos, `data/` or `test_data/`.
