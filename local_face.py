"""Offline face engine: runs fully on this laptop with OpenCV (no AWS, no internet).

Models (in models/, downloaded once with  python get_models.py):
  - YuNet  (face_detection_yunet_2023mar.onnx)   finds faces + 5 landmarks (eyes, nose, mouth corners)
  - SFace  (face_recognition_sface_2021dec.onnx) turns a face into a 128-number vector

Offline liveness = challenge-response: the person does 2 RANDOM actions in a random order.
  turn_left / turn_right : the nose moves sideways relative to the eyes. Only a real 3D head does this;
                           a flat photo or screen, even tilted, keeps the same nose/eye ratio.
  blink                  : the eyes close then open again. An open eye has a dark pupil on white
                           (high contrast); a closed eye is flat skin (low contrast). A photo can't blink.
  move_closer            : the face gets bigger (never used alone: a photo can be moved closer).
Every frame must show exactly one face, and it must stay the SAME face for the whole challenge.

Privacy: only vectors are stored (data/offline_faces.json), saved at liveness-verified enrollment. No photos.

CONTRACT:
    available() -> bool
    save_vector(person_id, image_bytes) -> bool          # called by enrollment
    remove_vector(person_id) -> None
    has_vector(person_id) -> bool
    new_challenge() -> {"id", "actions": [a1, a2], "labels": [str, str], "expires": float}
    verify(challenge_id, frames) -> dict                  # frames: [{"phase": "baseline"|0|1, "image": bytes}]
    detect(image_bytes) -> {"faces": n, "box": [x, y, w, h] | None}   # live preview (0..1 coords)
    process_frame(challenge_id, image_bytes) -> live progress dict    # kiosk sends frames one by one
    finish(challenge_id) -> dict                          # same format as verify(); single use
"""
import json
import os
import random
import secrets
import threading
import time

import numpy as np

import config

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YUNET = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
VECTORS_FILE = os.path.join(config.DATA_DIR, "offline_faces.json")

ACTIONS = {
    "turn_left": "Turn your head to YOUR left",
    "turn_right": "Turn your head to YOUR right",
    "blink": "Blink slowly 2 times",
    "move_closer": "Move closer to the camera",
}
THREE_D_ACTIONS = ["turn_left", "turn_right"]   # at least one head turn every time (3D proof)

# thresholds (tuned for SFace / YuNet)
MATCH_COSINE = 0.45        # SFace: >= this is the same person
REVIEW_COSINE = 0.363      # OpenCV's published SFace threshold; between -> step-up (code)
SAME_FACE_COSINE = 0.30    # every frame must stay the same person as the baseline
TURN_DELTA = 0.12          # change in (nose - eye centre) / eye distance
BLINK_CLOSED_RATIO = 0.70  # eye contrast drops below 70 % of the open-eye baseline = closed
BLINK_OPEN_RATIO = 0.85    # and comes back above 85 % = opened again
CLOSER_RATIO = 1.20        # face width grows by >= 20 %
FRONTAL_MAX = 0.20         # baseline must be roughly frontal
MIN_FRAMES_PER_ACTION = 2  # action must be seen in at least 2 frames (noise filter)
CHALLENGE_SECONDS = 30
BASELINE_FRAMES = 3        # frontal frames needed before the actions start
BASELINE_SECONDS = 8       # time allowed to look straight at the camera
ACTION_SECONDS = 5         # time allowed for each random action

_lock = threading.Lock()
_models = {}
_challenges = {}


def available():
    return os.path.exists(YUNET) and os.path.exists(SFACE)


def _cv2():
    import cv2
    return cv2


def _detector(w, h):
    cv2 = _cv2()
    if "det" not in _models:
        _models["det"] = cv2.FaceDetectorYN.create(YUNET, "", (w, h), 0.8, 0.3, 5000)
    _models["det"].setInputSize((w, h))
    return _models["det"]


def _recognizer():
    if "rec" not in _models:
        _models["rec"] = _cv2().FaceRecognizerSF.create(SFACE, "")
    return _models["rec"]


def _decode(image_bytes):
    cv2 = _cv2()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > 640:                       # keep it fast
        s = 640 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def analyse(image_bytes, with_vector=True):
    """-> {"faces": n, "geometry": {...} | None, "vector": np.ndarray | None}"""
    img = _decode(image_bytes)
    if img is None:
        return {"faces": 0, "geometry": None, "vector": None, "box": None}
    h, w = img.shape[:2]
    with _lock:
        _, faces = _detector(w, h).detect(img)
        faces = [] if faces is None else faces
        if len(faces) != 1:
            return {"faces": len(faces), "geometry": None, "vector": None, "box": None}
        f = faces[0]
        vector = None
        if with_vector:
            rec = _recognizer()
            vector = rec.feature(rec.alignCrop(img, f)).flatten().astype(float)
    box = [float(f[0] / w), float(f[1] / h), float(f[2] / w), float(f[3] / h)]
    geometry = _geometry(f, w)
    geometry["eye_open"] = _eye_openness(img, f)
    return {"faces": 1, "geometry": geometry, "vector": vector, "box": box}


def _geometry(f, frame_w):
    # YuNet row: x, y, w, h, right_eye(x,y), left_eye(x,y), nose(x,y), mouth_right(x,y), mouth_left(x,y), score
    re, le, nose = f[4:6], f[6:8], f[8:10]
    mr, ml = f[10:12], f[12:14]
    eye_dist = float(np.linalg.norm(le - re)) or 1.0
    eye_mid_x = (re[0] + le[0]) / 2
    return {
        "yaw": float((nose[0] - eye_mid_x) / eye_dist),        # + = person turned to THEIR left
        "smile": float(np.linalg.norm(ml - mr) / eye_dist),
        "size": float(f[2] / frame_w),
    }


def _eye_openness(img, f):
    """Contrast inside small patches around both eyes (std / mean of gray levels).
    Open eye: dark pupil + white sclera -> high. Closed eye: eyelid skin -> low."""
    cv2 = _cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    re, le = f[4:6], f[6:8]
    d = float(np.linalg.norm(le - re)) or 1.0
    hw, hh = max(3, int(d * 0.22)), max(2, int(d * 0.12))
    vals = []
    for cx, cy in (re, le):
        x0, x1 = int(cx) - hw, int(cx) + hw
        y0, y1 = int(cy) - hh, int(cy) + hh
        if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
            continue
        patch = gray[y0:y1, x0:x1].astype(float)
        if patch.size:
            vals.append(patch.std() / (patch.mean() + 1.0))
    return float(np.mean(vals)) if vals else 0.0


def _cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


# ------------------------------------------------------------ vectors (saved at enrollment only)

def _load():
    try:
        with open(VECTORS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data):
    os.makedirs(os.path.dirname(VECTORS_FILE) or ".", exist_ok=True)
    tmp = VECTORS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, VECTORS_FILE)


def _padded(image_bytes):
    """AWS reference images can be a tight crop of the face; add a border so YuNet finds it."""
    cv2 = _cv2()
    img = _decode(image_bytes)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, h // 3, h // 3, w // 3, w // 3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.imencode(".jpg", img)[1].tobytes()


def save_vector(person_id, image_bytes):
    if not available():
        return False
    a = analyse(image_bytes)
    if a["faces"] == 0:
        a = analyse(_padded(image_bytes))
    if a["vector"] is None:
        return False
    data = _load()
    data[person_id] = [round(x, 6) for x in a["vector"]]
    _save(data)
    return True


def remove_vector(person_id):
    data = _load()
    if data.pop(person_id, None) is not None:
        _save(data)


def has_vector(person_id):
    return person_id in _load()


def best_match(vector):
    best, score = None, -1.0
    for pid, v in _load().items():
        s = _cosine(vector, v)
        if s > score:
            best, score = pid, s
    return best, score


# ------------------------------------------------------------ random challenge

def new_challenge():
    first = random.choice(THREE_D_ACTIONS)
    second = random.choice([a for a in ACTIONS if a != first])
    actions = [first, second]
    random.shuffle(actions)
    cid = secrets.token_urlsafe(12)
    ch = {"id": cid, "actions": actions, "labels": [ACTIONS[a] for a in actions],
          "expires": time.time() + CHALLENGE_SECONDS}
    with _lock:
        for k in [k for k, v in _challenges.items() if v["expires"] < time.time()]:
            _challenges.pop(k)
        _challenges[cid] = ch
    return ch


def _claim(challenge_id):
    with _lock:
        ch = _challenges.pop(challenge_id, None)     # single use
    if ch is None:
        return None, "Unknown or already used challenge (replay blocked)."
    if ch["expires"] < time.time():
        return None, "Challenge expired. Please start again."
    return ch, "ok"


def _action_done(action, base, frames):
    if action == "blink":       # needs order: eyes closed, then open again
        if not base.get("eye_open"):
            return False, 0
        closed = False
        for g in frames:
            e = g.get("eye_open", 0.0)
            if e <= base["eye_open"] * BLINK_CLOSED_RATIO:
                closed = True
            elif closed and e >= base["eye_open"] * BLINK_OPEN_RATIO:
                return True, MIN_FRAMES_PER_ACTION
        return False, 0
    hits = 0
    for g in frames:
        if action == "turn_left" and g["yaw"] - base["yaw"] >= TURN_DELTA:
            hits += 1
        elif action == "turn_right" and base["yaw"] - g["yaw"] >= TURN_DELTA:
            hits += 1
        elif action == "move_closer" and g["size"] >= base["size"] * CLOSER_RATIO:
            hits += 1
    return hits >= MIN_FRAMES_PER_ACTION, hits


def verify(challenge_id, frames):
    """Liveness + identity from the challenge frames. Never raises.
    -> {"ok", "live": True|False|None, "reason", "person_id", "cosine", "actions": [...], "frames": n}"""
    out = {"live": None, "person_id": None, "cosine": None, "actions": [], "frames": len(frames)}
    ch, why = _claim(challenge_id)
    if ch is None:
        return {**out, "live": False, "reason": why, "spoof_type": "replayed_challenge"}
    if not available():
        return {**out, "reason": "Offline face models are missing. Run python get_models.py."}

    analysed = {"baseline": [], 0: [], 1: []}
    for fr in frames:
        phase = fr.get("phase")
        if phase not in analysed:
            continue
        a = analyse(fr["image"], with_vector=True)
        if a["faces"] > 1:
            return {**out, "live": False, "reason": "More than one face in view.",
                    "spoof_type": "multiple_faces"}
        if a["faces"] == 1:
            analysed[phase].append(a)

    base_list = analysed["baseline"]
    if not base_list:
        return {**out, "reason": "No face found while looking straight. Please try again."}
    # baseline = most frontal frame
    base = min(base_list, key=lambda a: abs(a["geometry"]["yaw"]))
    if abs(base["geometry"]["yaw"]) > FRONTAL_MAX:
        return {**out, "reason": "Please look straight at the camera at the start."}

    # same person in every frame (blocks swapping faces / photos mid-challenge)
    for phase in ("baseline", 0, 1):
        for a in analysed[phase]:
            if _cosine(a["vector"], base["vector"]) < SAME_FACE_COSINE:
                return {**out, "live": False, "reason": "The face changed during the check.",
                        "spoof_type": "face_swap"}

    all_done = True
    for i, action in enumerate(ch["actions"]):
        done, hits = _action_done(action, base["geometry"], [a["geometry"] for a in analysed[i]])
        out["actions"].append({"action": action, "label": ACTIONS[action], "done": done, "frames": hits})
        all_done = all_done and done

    pid, cos = best_match(base["vector"])
    out["person_id"], out["cosine"] = pid, (round(cos, 3) if pid else None)
    if not all_done:
        missed = [a["label"] for a in out["actions"] if not a["done"]]
        return {**out, "live": False, "spoof_type": "failed_challenge",
                "reason": "Random action not performed: " + "; ".join(missed) + "."}
    return {**out, "live": True, "reason": "Both random actions performed by the same live face."}


# ------------------------------------------------------------ live mode (frame by frame, with feedback)

def detect(image_bytes):
    """Live preview before the check starts: is there exactly one face, and where."""
    if not available():
        return {"faces": 0, "box": None, "error": "Offline face models are missing. Run python get_models.py."}
    a = analyse(image_bytes, with_vector=False)
    return {"faces": a["faces"], "box": a.get("box")}


def _live_state(ch):
    if "live" not in ch:
        ch["live"] = {"phase": "baseline", "started": time.time(), "base_frames": [], "base": None,
                      "hits": [0] * len(ch["actions"]), "fail": None, "frames": 0}
    return ch["live"]


def _progress(ch, st, faces=1, box=None, message=None):
    phase = st["phase"]
    if phase == "baseline":
        instruction = "Look straight at the camera"
        left = BASELINE_SECONDS - (time.time() - st["started"])
    elif phase == "done":
        instruction = "Done"
        left = 0
    else:
        instruction = f"{phase + 1}/{len(ch['actions'])}: {ACTIONS[ch['actions'][phase]]}"
        left = ACTION_SECONDS - (time.time() - st["started"])
    return {"phase": phase, "instruction": instruction, "seconds_left": max(0, round(left, 1)),
            "faces": faces, "box": box, "message": message, "done": phase == "done",
            "steps": [{"label": ACTIONS[a], "done": st["hits"][i] >= MIN_FRAMES_PER_ACTION}
                      for i, a in enumerate(ch["actions"])],
            "failed": st["fail"][1] if st["fail"] else None}


def _fail(st, kind, why):
    st["fail"] = (kind, why)
    st["phase"] = "done"


def _single_hit(action, base, g):
    return _action_done(action, base, [g])[1] == 1


def process_frame(challenge_id, image_bytes):
    """Analyse ONE frame of the live challenge and move it forward. Never raises."""
    with _lock:
        ch = _challenges.get(challenge_id)
    if ch is None or ch["expires"] < time.time():
        return {"done": True, "phase": "done", "failed": "Challenge expired. Please start again.",
                "instruction": "Done", "steps": [], "faces": 0, "box": None, "message": None,
                "seconds_left": 0}
    st = _live_state(ch)
    if st["phase"] == "done":
        return _progress(ch, st)
    st["frames"] += 1
    t0 = time.perf_counter()
    a = analyse(image_bytes, with_vector=True)
    st["last_ms"] = int((time.perf_counter() - t0) * 1000)
    st["total_ms"] = st.get("total_ms", 0) + st["last_ms"]
    now = time.time()

    if a["faces"] > 1:
        _fail(st, "multiple_faces", "More than one face in view.")
        return _progress(ch, st, a["faces"])
    if a["faces"] == 0:
        if st["phase"] == "baseline" and now - st["started"] > BASELINE_SECONDS:
            _fail(st, "no_face", "No face found. Please stand in front of the camera.")
        elif st["phase"] != "baseline" and now - st["started"] > ACTION_SECONDS:
            _fail(st, "failed_challenge", "Random action not performed in time.")
        return _progress(ch, st, 0, None, "No face found: look at the camera")

    g = a["geometry"]
    if st["phase"] == "baseline":
        if abs(g["yaw"]) <= FRONTAL_MAX:
            st["base_frames"].append(a)
        if len(st["base_frames"]) >= BASELINE_FRAMES:
            st["base"] = min(st["base_frames"], key=lambda b: abs(b["geometry"]["yaw"]))
            for b in st["base_frames"]:
                if _cosine(b["vector"], st["base"]["vector"]) < SAME_FACE_COSINE:
                    _fail(st, "face_swap", "The face changed during the check.")
                    return _progress(ch, st, 1, a["box"])
            if "blink" in ch["actions"] and st["base"]["geometry"].get("eye_open", 0) < 0.02:
                # eyes not measurable (e.g. glare on glasses): swap blink for another action
                ch["actions"] = [a_ if a_ != "blink" else "move_closer" for a_ in ch["actions"]]
                if len(set(ch["actions"])) < len(ch["actions"]):
                    ch["actions"] = [ch["actions"][0], "turn_right" if ch["actions"][0] == "turn_left"
                                     else "turn_left"]
            st["phase"], st["started"] = 0, now
        elif now - st["started"] > BASELINE_SECONDS:
            _fail(st, "not_frontal", "Please look straight at the camera at the start.")
        return _progress(ch, st, 1, a["box"],
                         None if abs(g["yaw"]) <= FRONTAL_MAX else "Look straight at the camera")

    # action phases: the same person must stay in front of the camera
    if _cosine(a["vector"], st["base"]["vector"]) < SAME_FACE_COSINE:
        _fail(st, "face_swap", "The face changed during the check.")
        return _progress(ch, st, 1, a["box"])
    i = st["phase"]
    st.setdefault("seq", {}).setdefault(i, []).append(g)
    if ch["actions"][i] == "blink":
        if _action_done("blink", st["base"]["geometry"], st["seq"][i])[0]:
            st["hits"][i] = MIN_FRAMES_PER_ACTION
    elif _single_hit(ch["actions"][i], st["base"]["geometry"], g):
        st["hits"][i] += 1
    if st["hits"][i] >= MIN_FRAMES_PER_ACTION:
        st["phase"] = i + 1 if i + 1 < len(ch["actions"]) else "done"
        st["started"] = now
    elif now - st["started"] > ACTION_SECONDS:
        _fail(st, "failed_challenge", f"Action not performed: {ACTIONS[ch['actions'][i]]}.")
    return _progress(ch, st, 1, a["box"])


def finish(challenge_id):
    """Final liveness + identity result of a live challenge (same format as verify). Single use."""
    out = {"live": None, "person_id": None, "cosine": None, "actions": [], "frames": 0}
    ch, why = _claim(challenge_id)
    if ch is None:
        return {**out, "live": False, "reason": why, "spoof_type": "replayed_challenge"}
    st = _live_state(ch)
    out["frames"] = st["frames"]
    out["frame_ms"] = st.get("last_ms")
    out["avg_frame_ms"] = int(st.get("total_ms", 0) / st["frames"]) if st["frames"] else None
    out["actions"] = [{"action": a, "label": ACTIONS[a], "done": st["hits"][i] >= MIN_FRAMES_PER_ACTION,
                       "frames": st["hits"][i]} for i, a in enumerate(ch["actions"])]
    if st["fail"] and st["fail"][0] in ("no_face", "not_frontal"):
        return {**out, "reason": st["fail"][1]}
    if st["fail"] and st["base"] is None:          # e.g. two faces before the check started
        return {**out, "live": False, "spoof_type": st["fail"][0], "reason": st["fail"][1]}
    if st["base"] is None:
        return {**out, "reason": "No face found while looking straight. Please try again."}
    pid, cos = best_match(st["base"]["vector"])
    out["person_id"], out["cosine"] = pid, (round(cos, 3) if pid else None)
    if st["fail"]:
        return {**out, "live": False, "spoof_type": st["fail"][0], "reason": st["fail"][1]}
    if st["phase"] != "done" or not all(a["done"] for a in out["actions"]):
        missed = [a["label"] for a in out["actions"] if not a["done"]]
        return {**out, "live": False, "spoof_type": "failed_challenge",
                "reason": "Random action not performed: " + "; ".join(missed) + "."}
    return {**out, "live": True, "reason": "Both random actions performed by the same live face."}
