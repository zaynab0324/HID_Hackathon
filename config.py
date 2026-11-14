"""Shared settings. Change values here (or via environment variables), never inside other files."""
import os

# ---------------- AWS ----------------
REGION = os.getenv("AWS_REGION", "us-east-1")
COLLECTION = os.getenv("FACE_COLLECTION", "hid-practice")
AWS_TIMEOUT_SECONDS = 3          # a call slower than this counts as "connection lost"

# ---------------- Kiosk ----------------
KIOSK_LOCATION = os.getenv("KIOSK_LOCATION", "Lab 3")   # the door this kiosk guards
LOCATIONS = ["Lab 3", "Lab 5", "Library", "Hostel A", "Exam Hall"]

# ---------------- Identity (face match %) ----------------
MATCH_THRESHOLD = 90.0    # >= this: identity confirmed
REVIEW_THRESHOLD = 80.0   # between this and MATCH_THRESHOLD: identity unsure -> step-up

# ---------------- Liveness (passive check) ----------------
# Liveness score is 0..1 (1 = surely a live person).
LIVE_THRESHOLD = 0.70     # score >= this: live
SPOOF_THRESHOLD = 0.30    # score <= this: spoof
# Labels Rekognition may return when a photo / phone / screen is held up to the camera.
SCREEN_LABELS = {"Mobile Phone", "Phone", "Cell Phone", "iPhone", "Screen", "Monitor",
                 "Display", "Tablet Computer", "Computer", "Electronics", "Laptop", "TV"}
PAPER_LABELS = {"Paper", "Photograph", "Poster", "Page", "Document", "Id Cards", "Business Card"}
MASK_LABELS = {"Mask"}
# Labels that count as suspicious even when not around the face (a phone or paper in view).
# Background things like monitors or laptops only count when they surround the face.
ANYWHERE_LABELS = {"Mobile Phone", "Phone", "Cell Phone", "iPhone", "Tablet Computer",
                   "Paper", "Poster", "Page", "Document", "Id Cards", "Business Card", "Mask"}
LABEL_MIN_CONFIDENCE = 70.0
# Image-texture thresholds: None = only measured, not used yet.
# Set them from the evaluation report (Performance tab) once you have test data.
GLARE_MAX = None          # spoof if glare_ratio above this
MOIRE_MAX = None          # spoof if moire_score above this
SHARPNESS_MIN = None      # spoof if sharpness below this

# ---------------- GenAI spoof inspector (can only make decisions STRICTER) ----------------
USE_GENAI_INSPECTOR = True
INSPECTOR_SPOOF_CONFIDENCE = 0.80   # inspector must be at least this sure to block
INSPECTOR_WAIT_SECONDS = 1.2        # never wait longer than this for it (2 s budget)

# ---------------- Security mode ----------------
# True = the passive check alone can NEVER give ALLOW: every person must also pass a random
# action (or AWS Face Liveness). Photos and replayed videos cannot follow a random instruction.
REQUIRE_ACTIVE_CHECK = True

# ---------------- Active challenge (REQUEST VERIFICATION) ----------------
CHALLENGE_ACTIONS = ["smile", "close_eyes", "open_mouth"]   # add "turn_left", "turn_right" if wanted
CHALLENGE_SECONDS = 15    # a challenge expires after this
ATTRIBUTE_CONFIDENCE = 80 # how sure Rekognition must be about smile / eyes / mouth
TURN_YAW_DEGREES = 20     # head turn must exceed this angle
YAW_LEFT_SIGN = 1         # flip to -1 if "turn left" is detected as right on your camera

# ---------------- Campus schedule ----------------
NEAR_SCHEDULE_MINUTES = 15   # this close to a slot (before/after) -> step-up instead of deny

# ---------------- Offline mode ----------------
OFFLINE_CHECK_SECONDS = 10   # re-check the connection at most this often

# ---------------- Speed ----------------
DECISION_BUDGET_MS = 2000

# ---------------- Live kiosk camera ----------------
STEADY_SECONDS = 1.0
COOLDOWN_SECONDS = 5

# ---------------- GenAI (Amazon Bedrock) ----------------
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
VISION_MODEL_ID = os.getenv("BEDROCK_VISION_MODEL_ID", "us.amazon.nova-lite-v1:0")
GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

# ---------------- AWS Face Liveness (official) ----------------
OFFICIAL_LIVE_CONFIDENCE = 90.0

# ---------------- Security ----------------
KIOSK_ROLE_NAME = "HIDKioskLivenessRole"       # browser credentials: StartFaceLivenessSession only
SESSION_MAX_AGE_SECONDS = 180                  # a liveness session must be decided within 3 min, once
THREAT_WINDOW_MINUTES = 15                     # heightened security lasts this long after a spoof
THREAT_LIVE_CONFIDENCE = 95.0                  # stricter liveness bar while under attack
ALERT_SNS_TOPIC_ARN = os.getenv("ALERT_SNS_TOPIC_ARN", "")   # optional: real-time alerts to guards
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "hid-admin-2026")
GUARD_PIN = os.getenv("GUARD_PIN", "2468")

ENROLL_LIVE_CONFIDENCE = 97.0                  # enrollment needs a very sure live face
IMPOSSIBLE_TRAVEL_MINUTES = 5                  # allowed at two different doors faster than this = clone

# ---------------- Server ----------------
SERVER_HOST = os.getenv("HOST", "127.0.0.1")   # 0.0.0.0 on EC2
SERVER_PORT = int(os.getenv("PORT", "8000"))

# ---------------- Local data (never committed) ----------------
DATA_DIR = os.getenv("DATA_DIR", "data")
TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", "test_data")
