
"""Shared settings. Change values here (or via environment variables), never inside other files."""
import os

# ---------------- AWS ----------------
REGION = os.getenv("AWS_REGION", "us-east-1")
COLLECTION = os.getenv("FACE_COLLECTION", "hid-practice")

# ---------------- Identity (face match %) ----------------
MATCH_THRESHOLD = 90.0    # >= this: identity confirmed
REVIEW_THRESHOLD = 80.0   # between this and MATCH_THRESHOLD: identity unsure -> REQUEST VERIFICATION

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
# Set them from the evaluation report (python evaluate.py) once you have test data.
GLARE_MAX = None          # spoof if glare_ratio above this
MOIRE_MAX = None          # spoof if moire_score above this
SHARPNESS_MIN = None      # spoof if sharpness below this

# ---------------- Security mode ----------------
# True = the passive check alone can NEVER give ALLOW: every person must also pass a random
# action (or AWS Face Liveness). Photos and replayed videos cannot follow a random instruction.
REQUIRE_ACTIVE_CHECK = True

# ---------------- Active challenge (REQUEST VERIFICATION) ----------------
CHALLENGE_SECONDS = 15    # a challenge expires after this
TURN_YAW_DEGREES = 20     # head turn must exceed this angle
YAW_LEFT_SIGN = 1         # flip to -1 if "turn left" is detected as right on your camera

# ---------------- Speed ----------------
DECISION_BUDGET_MS = 2000

# ---------------- Live door camera ----------------
STEADY_SECONDS = 1.0
COOLDOWN_SECONDS = 5

# ---------------- GenAI (Amazon Bedrock) ----------------
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
GUARDRAIL_ID = os.getenv("BEDROCK_GUARDRAIL_ID", "")
GUARDRAIL_VERSION = os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT")

# ---------------- AWS Face Liveness (official, stretch goal) ----------------
OFFICIAL_LIVE_CONFIDENCE = 90.0

# ---------------- Local data (never committed) ----------------
DATA_DIR = os.getenv("DATA_DIR", "data")
TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", "test_data")

