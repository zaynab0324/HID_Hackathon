"""Performance report: runs every test image through the real pipeline and measures results.

OWNER: Abhi

Folder layout (webcam photos taken with the team's laptop, with consent):
    test_data/genuine/        live faces of ENROLLED people   -> expected ALLOW
    test_data/printed_photo/  printed photo held to the camera -> expected NOT ALLOW
    test_data/screen_photo/   photo shown on a phone screen    -> expected NOT ALLOW
    test_data/video_replay/   video playing on a phone screen  -> expected NOT ALLOW
    test_data/mask/           (optional)                       -> expected NOT ALLOW
    test_data/impostor/       real faces of NON-enrolled people -> must not match anyone
Name genuine files "<Person_ID>__01.jpg" so face-authentication accuracy checks the right person.

Run:   python evaluate.py
Writes: data/eval_report.json (shown in the app's Performance tab) and data/eval_details.csv

CONTRACT:
    run_evaluation(test_dir=config.TEST_DATA_DIR) -> report dict (also saved to data/eval_report.json)
    load_report() -> report dict | None
"""
import csv
import json
import os
import statistics
from datetime import datetime

import config
import pipeline

REPORT_FILE = os.path.join(config.DATA_DIR, "eval_report.json")
DETAILS_FILE = os.path.join(config.DATA_DIR, "eval_details.csv")
IMAGE_EXT = (".jpg", ".jpeg", ".png")
GENUINE = "genuine"


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def run_evaluation(test_dir=None):
    test_dir = test_dir or config.TEST_DATA_DIR
    rows = []
    for scenario in sorted(os.listdir(test_dir)):
        folder = os.path.join(test_dir, scenario)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(IMAGE_EXT):
                continue
            with open(os.path.join(folder, name), "rb") as f:
                result = pipeline.check_access(f.read(), use_ai=False, log=False, use_schedule=False)
            live = result["liveness"]
            expected = name.split("__")[0] if "__" in name else None   # e.g. Afrah_Zainab__01.jpg
            rows.append({
                "scenario": scenario,
                "file": name,
                "expected_person": expected,
                "verdict": result["verdict"],
                "person_id": result["person_id"],
                "similarity": result["similarity"],
                "live": live["live"],
                "liveness_score": live["score"],
                "spoof_type": live["spoof_type"],
                "decision_ms": result["decision_ms"],
                **{f"sig_{k}": v for k, v in live["signals"].items()},
            })
            print(f"{scenario:15} {name:30} {result['verdict']:22} {result['decision_ms']:5} ms")

    scenarios = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        items = [r for r in rows if r["scenario"] == scenario]
        n = len(items)
        count = {v: sum(r["verdict"] == v for r in items)
                 for v in ("ALLOW", "REQUEST_VERIFICATION", "DENY", "RETRY")}
        times = [r["decision_ms"] for r in items]
        entry = {
            "n": n, **count,
            "avg_ms": int(statistics.mean(times)) if times else None,
            "max_ms": max(times) if times else None,
            "under_2s_pct": _pct(sum(t <= config.DECISION_BUDGET_MS for t in times), n),
        }
        live_vals = [r["live"] for r in items]
        entry["flagged_spoof_pct"] = _pct(sum(v is False for v in live_vals), n)
        entry["flagged_unsure_pct"] = _pct(sum(v is None for v in live_vals), n)
        entry["passed_as_live_pct"] = _pct(sum(v is True for v in live_vals), n)
        if scenario == GENUINE:
            entry["correct_allow_pct"] = _pct(count["ALLOW"], n)
            entry["false_reject_pct"] = _pct(count["DENY"], n)
            entry["extra_step_pct"] = _pct(count["REQUEST_VERIFICATION"], n)
        else:
            entry["detection_pct"] = _pct(n - count["ALLOW"], n)       # spoof not let in
            entry["denied_directly_pct"] = _pct(count["DENY"], n)
            entry["false_accept_pct"] = _pct(count["ALLOW"], n)
        # Average image signals per scenario: use these to set GLARE_MAX / MOIRE_MAX / SHARPNESS_MIN.
        for sig in ("sig_sharpness", "sig_glare_ratio", "sig_moire_score"):
            vals = [r[sig] for r in items if isinstance(r.get(sig), (int, float))]
            if vals:
                entry[f"avg_{sig[4:]}"] = round(statistics.mean(vals), 4)
        scenarios[scenario] = entry

    spoof_rows = [r for r in rows if r["scenario"] not in (GENUINE, "impostor")]
    genuine_rows = [r for r in rows if r["scenario"] == GENUINE]
    impostor_rows = [r for r in rows if r["scenario"] == "impostor"]

    def auth_ok(r):
        matched = r["person_id"] is not None and (r["similarity"] or 0) >= config.MATCH_THRESHOLD
        return matched and (r["expected_person"] in (None, r["person_id"]))

    live_correct = (sum(r["live"] is not False for r in genuine_rows)
                    + sum(r["live"] is not True for r in spoof_rows))
    all_times = [r["decision_ms"] for r in rows]
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(rows),
        "overall": {
            "liveness_accuracy_pct": _pct(live_correct, len(genuine_rows) + len(spoof_rows)),
            "face_auth_accuracy_pct": _pct(sum(auth_ok(r) for r in genuine_rows), len(genuine_rows)),
            "apcer_pct": _pct(sum(r["live"] is True for r in spoof_rows), len(spoof_rows)),
            "bpcer_pct": _pct(sum(r["live"] is False for r in genuine_rows), len(genuine_rows)),
            "identity_false_match_pct": _pct(sum(r["person_id"] is not None and (r["similarity"] or 0)
                                                 >= config.MATCH_THRESHOLD for r in impostor_rows),
                                             len(impostor_rows)),
            "spoof_detection_pct": _pct(sum(r["verdict"] != "ALLOW" for r in spoof_rows),
                                        len(spoof_rows)),
            "false_accept_rate_pct": _pct(sum(r["verdict"] == "ALLOW" for r in spoof_rows),
                                          len(spoof_rows)),
            "false_reject_rate_pct": _pct(sum(r["verdict"] == "DENY" for r in genuine_rows),
                                          len(genuine_rows)),
            "avg_decision_ms": int(statistics.mean(all_times)) if all_times else None,
            "p95_decision_ms": _p95(all_times),
            "under_2s_pct": _pct(sum(t <= config.DECISION_BUDGET_MS for t in all_times),
                                 len(all_times)),
        },
        "scenarios": scenarios,
    }

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if rows:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("sig_"), k))
        with open(DETAILS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return report


def load_report():
    try:
        with open(REPORT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    report = run_evaluation()
    print("\n" + json.dumps(report["overall"], indent=2))
    print(f"\nSaved {REPORT_FILE} and {DETAILS_FILE}")
