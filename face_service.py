"""Face recognition with Amazon Rekognition.

CONTRACT (other files rely on these exact names and return shapes - do not change them):
    clean_person_id(raw: str) -> str | None
    enroll(image_bytes: bytes, person_id: str) -> {"ok": bool, "person_id": str, "message": str}
    identify(image_bytes: bytes) -> {"status": "MATCH" | "POSSIBLE" | "NO_MATCH" | "NO_FACE",
                                     "person_id": str | None, "similarity": float, "message": str}
    list_people() -> dict[str, int]          # person_id -> number of stored face samples
    remove_person(person_id: str) -> int     # number of face samples deleted
    detect_faces(image_bytes: bytes) -> list[dict]   # Rekognition FaceDetails (all attributes)
    client()                                 # the shared Rekognition client
"""
import re

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import config

_ID_PATTERN = re.compile(r"[A-Za-z0-9_.\-:]+")
_client = None


def client():
    return _rek()


def _rek():
    """Create the Rekognition client and collection once, then reuse them."""
    global _client
    if _client is None:
        _client = boto3.client("rekognition", region_name=config.REGION, config=Config(
            connect_timeout=config.AWS_TIMEOUT_SECONDS, read_timeout=config.AWS_TIMEOUT_SECONDS,
            retries={"max_attempts": 1}, max_pool_connections=20))
        try:
            _client.create_collection(CollectionId=config.COLLECTION)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise
    return _client


def clean_person_id(raw):
    """'Afrah Zainab ' -> 'Afrah_Zainab'. Returns None if the ID can't be used."""
    cleaned = (raw or "").strip().replace(" ", "_")
    if cleaned and _ID_PATTERN.fullmatch(cleaned):
        return cleaned
    return None


def enroll(image_bytes, person_id):
    pid = clean_person_id(person_id)
    if pid is None:
        return {"ok": False, "person_id": person_id or "",
                "message": "Use only letters, numbers, spaces, _ - . or :"}
    try:
        response = _rek().index_faces(
            CollectionId=config.COLLECTION,
            Image={"Bytes": image_bytes},
            ExternalImageId=pid,
            MaxFaces=1,
            QualityFilter="AUTO",
        )
    except ClientError as e:
        return {"ok": False, "person_id": pid, "message": e.response["Error"]["Message"]}
    if response["FaceRecords"]:
        return {"ok": True, "person_id": pid, "message": f"Enrolled {pid}"}
    return {"ok": False, "person_id": pid,
            "message": "No clear face found. Try again with better light."}


def identify(image_bytes):
    try:
        response = _rek().search_faces_by_image(
            CollectionId=config.COLLECTION,
            Image={"Bytes": image_bytes},
            FaceMatchThreshold=config.REVIEW_THRESHOLD,
            MaxFaces=1,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidParameterException":
            return {"status": "NO_FACE", "person_id": None, "similarity": 0.0,
                    "message": "No face detected. Look straight at the camera."}
        raise

    matches = response["FaceMatches"]
    if not matches:
        return {"status": "NO_MATCH", "person_id": None, "similarity": 0.0,
                "message": "Face not recognised."}

    person_id = matches[0]["Face"]["ExternalImageId"]
    similarity = round(matches[0]["Similarity"], 1)
    if similarity >= config.MATCH_THRESHOLD:
        return {"status": "MATCH", "person_id": person_id, "similarity": similarity,
                "message": f"Matched {person_id} ({similarity}%)"}
    return {"status": "POSSIBLE", "person_id": person_id, "similarity": similarity,
            "message": f"Possible match {person_id} ({similarity}%)"}


def _all_faces():
    faces, token = [], None
    while True:
        kwargs = {"CollectionId": config.COLLECTION, "MaxResults": 1000}
        if token:
            kwargs["NextToken"] = token
        response = _rek().list_faces(**kwargs)
        faces += response["Faces"]
        token = response.get("NextToken")
        if not token:
            return faces


def list_people():
    counts = {}
    for face in _all_faces():
        pid = face.get("ExternalImageId", "?")
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def remove_person(person_id):
    ids = [f["FaceId"] for f in _all_faces() if f.get("ExternalImageId") == person_id]
    for i in range(0, len(ids), 4096):
        _rek().delete_faces(CollectionId=config.COLLECTION, FaceIds=ids[i:i + 4096])
    return len(ids)


def detect_faces(image_bytes):
    try:
        response = _rek().detect_faces(Image={"Bytes": image_bytes}, Attributes=["ALL"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidParameterException":
            return []
        raise
    return response["FaceDetails"]
