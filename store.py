"""People records and the access log, saved as JSON files in the data/ folder.

CONTRACT:
    get_person(person_id: str) -> dict | None
    upsert_person(person_id: str, **fields) -> dict         # create or update a record
    list_persons() -> list[dict]
    delete_person(person_id: str) -> bool
    log_event(event: dict) -> dict                          # adds "time" automatically
    recent_events(limit: int = 50) -> list[dict]            # newest first
"""
import json
import os
from datetime import datetime

import config

PEOPLE_FILE = os.path.join(config.DATA_DIR, "people.json")
LOG_FILE = os.path.join(config.DATA_DIR, "access_log.json")


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, data):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def get_person(person_id):
    return _load(PEOPLE_FILE, {}).get(person_id)


def upsert_person(person_id, **fields):
    people = _load(PEOPLE_FILE, {})
    record = people.get(person_id, {"person_id": person_id})
    record.update(fields)
    people[person_id] = record
    _save(PEOPLE_FILE, people)
    return record


def list_persons():
    return list(_load(PEOPLE_FILE, {}).values())


def delete_person(person_id):
    people = _load(PEOPLE_FILE, {})
    if person_id in people:
        del people[person_id]
        _save(PEOPLE_FILE, people)
        return True
    return False


def log_event(event):
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **event}
    log = _load(LOG_FILE, [])
    log.append(entry)
    _save(LOG_FILE, log[-1000:])
    return entry


def recent_events(limit=50):
    return list(reversed(_load(LOG_FILE, [])[-limit:]))
