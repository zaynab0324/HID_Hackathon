"""People records, campus schedule and the access log, saved as JSON files in data/.

CONTRACT:
    get_person(person_id) -> dict | None
    upsert_person(person_id, **fields) -> dict        # fields: status, role, temp_pass, ...
    list_persons() -> list[dict]
    delete_person(person_id) -> bool
    add_slot(person_id, location, days, start, end) -> dict   # days: list of 0-6 (Mon=0)
    list_slots() -> list[dict]
    delete_slot(slot_id) -> bool
    check_schedule(person_id, location, now: datetime) ->
        {"status": "in_schedule" | "near_schedule" | "out_of_schedule", "reason": str, "slot": dict | None}
    log_event(event: dict) -> dict                    # adds "time", "prev_hash", "hash" (tamper-evident)
    recent_events(limit=50) -> list[dict]             # newest first
    verify_log() -> {"ok": bool, "entries": int, "first_bad": int | None, "message": str}
"""
import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta

import config

PEOPLE_FILE = os.path.join(config.DATA_DIR, "people.json")
SCHEDULE_FILE = os.path.join(config.DATA_DIR, "schedule.json")
LOG_FILE = os.path.join(config.DATA_DIR, "access_log.json")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------ people

def get_person(person_id):
    return _load(PEOPLE_FILE, {}).get(person_id)


def upsert_person(person_id, **fields):
    people = _load(PEOPLE_FILE, {})
    record = people.get(person_id, {"person_id": person_id, "status": "active", "role": "student"})
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


# ------------------------------------------------------------ schedule

def add_slot(person_id, location, days, start, end):
    slots = _load(SCHEDULE_FILE, [])
    slot = {"slot_id": uuid.uuid4().hex[:6], "person_id": person_id, "location": location,
            "days": sorted(int(d) for d in days), "start": start, "end": end}
    slots.append(slot)
    _save(SCHEDULE_FILE, slots)
    return slot


def list_slots():
    return _load(SCHEDULE_FILE, [])


def delete_slot(slot_id):
    slots = _load(SCHEDULE_FILE, [])
    kept = [s for s in slots if s["slot_id"] != slot_id]
    _save(SCHEDULE_FILE, kept)
    return len(kept) != len(slots)


def _at(now, hhmm):
    h, m = (int(x) for x in hhmm.split(":"))
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


def check_schedule(person_id, location, now):
    person = get_person(person_id) or {}
    if person.get("role") == "staff":
        return {"status": "in_schedule", "reason": "Staff have access at all times.", "slot": None}

    temp = person.get("temp_pass")
    if temp and temp.get("location") == location:
        try:
            if now <= datetime.fromisoformat(temp["until"]):
                return {"status": "in_schedule",
                        "reason": f"Temporary pass for {location} until {temp['until'][11:16]}.",
                        "slot": None}
        except (KeyError, ValueError):
            pass

    near = timedelta(minutes=config.NEAR_SCHEDULE_MINUTES)
    best_near = None
    for slot in list_slots():
        if slot["person_id"] != person_id or slot["location"] != location:
            continue
        if now.weekday() not in slot["days"]:
            continue
        start, end = _at(now, slot["start"]), _at(now, slot["end"])
        if start <= now <= end:
            return {"status": "in_schedule",
                    "reason": f"Scheduled for {location} {slot['start']}-{slot['end']}.", "slot": slot}
        if start - near <= now < start or end < now <= end + near:
            best_near = slot
    if best_near:
        return {"status": "near_schedule",
                "reason": (f"Outside but within {config.NEAR_SCHEDULE_MINUTES} min of the slot "
                           f"{best_near['start']}-{best_near['end']} for {location}."),
                "slot": best_near}
    return {"status": "out_of_schedule",
            "reason": f"No schedule allows {person_id} in {location} at {now.strftime('%a %H:%M')}.",
            "slot": None}


# ------------------------------------------------------------ access log

def _entry_hash(entry):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def log_event(event):
    log = _load(LOG_FILE, [])
    entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **event,
             "prev_hash": log[-1].get("hash", "GENESIS") if log else "GENESIS"}
    entry["hash"] = _entry_hash(entry)
    log.append(entry)
    _save(LOG_FILE, log)
    return entry


def verify_log():
    """Every entry must hash correctly and point to the previous entry's hash."""
    log = _load(LOG_FILE, [])
    for i, entry in enumerate(log):
        expected_prev = log[i - 1].get("hash") if i else "GENESIS"
        if entry.get("hash") != _entry_hash(json.loads(json.dumps(entry, default=str))) \
                or entry.get("prev_hash") != expected_prev:
            return {"ok": False, "entries": len(log), "first_bad": i,
                    "message": f"Log tampered: entry {i} ({entry.get('time')}) was changed or "
                               "an entry before it was removed."}
    return {"ok": True, "entries": len(log), "first_bad": None,
            "message": f"All {len(log)} entries verified: no tampering."}


def recent_events(limit=50):
    return list(reversed(_load(LOG_FILE, [])[-limit:]))
