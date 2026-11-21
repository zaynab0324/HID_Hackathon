"""Presence: who is checked in where (check-in on entry ALLOW, check-out at the exit).

Rule (enforced in rules.py): while a person is checked in at one lab and has not checked out,
an entry attempt at a DIFFERENT lab is denied and flagged: one person cannot be in two labs, so the
second face is a clone, a photo/deepfake that fooled the camera, or a shared identity.

A forgotten check-out must not lock anyone out forever, so a check-in expires automatically at the
end of the person's schedule slot + PRESENCE_GRACE_MINUTES (or after PRESENCE_MAX_HOURS when there is
no slot, e.g. staff). Admins can also force a check-out.

Stored: person, lab, check-in time, expiry. No photos.

CONTRACT:
    current(person_id, now=None) -> {"location", "since", "expires"} | None
    check_in(person_id, location, now=None, schedule=None) -> record
    check_out(person_id, now=None) -> previous record | None
    force_checkout(person_id) -> bool
    present(now=None) -> list[{"person_id", "location", "since", "expires"}]
"""
import json
import os
from datetime import datetime, timedelta

import config

PRESENCE_FILE = os.path.join(config.DATA_DIR, "presence.json")
FMT = "%Y-%m-%d %H:%M:%S"


def _load():
    try:
        with open(PRESENCE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data):
    os.makedirs(os.path.dirname(PRESENCE_FILE) or ".", exist_ok=True)
    tmp = PRESENCE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, PRESENCE_FILE)


def _expired(rec, now):
    try:
        return datetime.strptime(rec["expires"], FMT) <= now
    except (KeyError, ValueError):
        return True


def current(person_id, now=None):
    if not person_id:
        return None
    now = now or datetime.now()
    rec = _load().get(person_id)
    if rec is None or _expired(rec, now):
        return None
    try:
        if datetime.strptime(rec["since"], FMT) > now:   # check-in in the "future" (tests / clock skew)
            return None
    except (KeyError, ValueError):
        return None
    return rec


def _expiry(now, schedule):
    slot = (schedule or {}).get("slot") or {}
    end = slot.get("end")
    if end:
        try:
            h, m = (int(x) for x in end.split(":"))
            end_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if end_dt > now:
                return end_dt + timedelta(minutes=config.PRESENCE_GRACE_MINUTES)
        except ValueError:
            pass
    return now + timedelta(hours=config.PRESENCE_MAX_HOURS)


def check_in(person_id, location, now=None, schedule=None):
    now = now or datetime.now()
    rec = {"location": location, "since": now.strftime(FMT), "expires": _expiry(now, schedule).strftime(FMT)}
    data = _load()
    data[person_id] = rec
    _save(data)
    return rec


def check_out(person_id, now=None):
    data = _load()
    rec = data.pop(person_id, None)
    _save(data)
    if rec and _expired(rec, now or datetime.now()):
        return None
    return rec


def force_checkout(person_id):
    data = _load()
    if data.pop(person_id, None) is None:
        return False
    _save(data)
    return True


def present(now=None):
    now = now or datetime.now()
    return [{"person_id": pid, **rec} for pid, rec in sorted(_load().items()) if not _expired(rec, now)]
