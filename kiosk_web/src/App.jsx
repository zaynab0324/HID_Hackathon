import React, { useCallback, useEffect, useRef, useState } from "react";
import { FaceLivenessDetectorCore } from "@aws-amplify/ui-react-liveness";

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.detail || data.error || `HTTP ${res.status}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

const LABEL = {
  ALLOW: "ALLOW",
  REQUEST_VERIFICATION: "STEP-UP VERIFICATION",
  DENY: "DENY",
  RETRY: "TRY AGAIN",
};
const ICON = { ALLOW: "✅", REQUEST_VERIFICATION: "⚠️", DENY: "⛔", RETRY: "🔄" };

function Result({ result, onGuard, onRetry, onNext, onResult, location }) {
  const v = result.verdict;
  const live = result.liveness || {};
  const t = result.timing || {};
  const fast = result.decision_ms != null && result.decision_ms <= 2000;
  return (
    <div className={`card result ${v}`}>
      <div className="dirtag">{result.direction === "out" ? "👋 CHECK-OUT" : "🚪 CHECK-IN"} · 📍 {result.location}</div>
      <div className="verdict">
        {ICON[v]} {LABEL[v] || v}
      </div>
      {v === "ALLOW" && result.direction !== "out" && result.presence && (
        <div className="presence in">
          ✅ <b>{(result.person_id || "").replaceAll("_", " ")}</b> is now <b>CHECKED IN at {result.presence.location}</b>{" "}
          (since {result.presence.since.slice(11, 16)}). Entry to any other lab is blocked until you check out here.
        </div>
      )}
      {v === "ALLOW" && result.direction === "out" && (
        <div className="presence out">
          👋 <b>{(result.person_id || "").replaceAll("_", " ")}</b> is now <b>CHECKED OUT</b> of {result.location}.
        </div>
      )}
      {result.flag === "already_checked_in" && (
        <div className="presence flag">🚨 <b>Same person, two places:</b> already checked in at another lab. Security alerted.</div>
      )}
      {result.flag === "impossible_travel" && (
        <div className="presence flag">🚨 <b>Impossible travel:</b> seen at another lab too recently to be here. Security alerted.</div>
      )}
      <p className="explain">{result.explanation}</p>
      <div className="facts">
        {result.decision_ms != null && (
          <span>
            ⏱️ Decision <b>{(result.decision_ms / 1000).toFixed(2)} s</b> {fast ? "✅" : "❌"}
          </span>
        )}
        {result.mode !== "offline" && (
          <span>
            🧬 Liveness <b>{live.signals?.aws_confidence ?? Math.round((live.score || 0) * 100)}%</b>
          </span>
        )}
        <span>
          🪪 Identity <b>{result.person_id ? `${result.person_id}${result.similarity ? ` (${result.similarity}%)` : ""}` : "no match"}</b>
        </span>
        {result.schedule && (
          <span>
            📅 <b>{result.schedule.status.replaceAll("_", " ")}</b>
          </span>
        )}
        {result.mode === "offline" && <span>📴 <b>offline decision</b></span>}

        {result.mode === "offline" && live.source === "offline_challenge" && (
          <span>
            🧬 Random actions <b>{live.live ? "passed" : live.live === false ? "FAILED" : "–"}</b>
          </span>
        )}
        {result.threat && <span>🚨 <b>heightened security</b></span>}
      </div>
      {Object.keys(t).length > 0 && (
        <div className="timing">
          {Object.entries(t).map(([k, ms]) => (
            <span key={k}>
              {k.replaceAll("_", " ")}: <b>{typeof ms === "number" ? `${ms} ms` : ms}</b>
            </span>
          ))}
        </div>
      )}
      <ul className="reasons">
        {(result.reasons || []).map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
      <div className="actions">
        {v === "REQUEST_VERIFICATION" && result.verification === "guard" && (
          <>
            <button className="ok" onClick={() => onGuard("ALLOW")}>Guard: approve</button>
            <button className="bad" onClick={() => onGuard("DENY")}>Guard: deny</button>
          </>
        )}
        {v === "REQUEST_VERIFICATION" && result.verification === "code" && result.stepup_ticket && (
          <CodeStepUp result={result} location={location} onResult={onResult} />
        )}
        {(v === "REQUEST_VERIFICATION" && result.verification !== "guard" && result.verification !== "code") || v === "RETRY" ? (
          <button className="ok" onClick={onRetry}>Re-capture (verify again)</button>
        ) : null}
        <button onClick={onNext}>Next person</button>
      </div>
    </div>
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const FRAME_GAP_MS = 120; // pause between frames (each frame is analysed on this laptop)

function cameraHelp(e) {
  const n = e?.name || "";
  if (n === "NotAllowedError") return "Camera permission blocked. Click the camera icon in the address bar → Allow, then reload.";
  if (n === "NotReadableError") return "The camera is being used by another app or tab (Zoom, Teams, the Streamlit page...). Close it and press Retry camera.";
  if (n === "NotFoundError") return "No camera found on this computer.";
  return `Camera error: ${e?.message || n || "unknown"}`;
}

function OfflinePanel({ status, location, testLabel, onResult }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const [camState, setCamState] = useState("starting"); // starting | on | error
  const [camError, setCamError] = useState("");
  const [preview, setPreview] = useState(null); // {faces, box}
  const [live, setLive] = useState(null); // progress from /api/offline/frame
  const [step, setStep] = useState("idle"); // idle | running | checking
  const [error, setError] = useState("");
  const [showCode, setShowCode] = useState(false);
  const [personId, setPersonId] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);

  const startCamera = useCallback(async () => {
    setCamState("starting");
    setCamError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot open the camera (use Chrome/Edge on localhost).");
      streamRef.current?.getTracks().forEach((t) => t.stop());
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      streamRef.current = stream;
      const v = videoRef.current;
      v.srcObject = stream;
      await v.play().catch(() => {});
      setCamState("on");
    } catch (e) {
      setCamState("error");
      setCamError(cameraHelp(e));
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => {
      runningRef.current = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [startCamera]);

  const grab = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;
    const c = document.createElement("canvas");
    const scale = Math.min(1, 480 / v.videoWidth);
    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height); // NOT mirrored: server sees the real image
    return c.toDataURL("image/jpeg", 0.8);
  };

  // live preview: face box + "face detected" before the check starts
  useEffect(() => {
    if (camState !== "on" || step !== "idle") return undefined;
    let stop = false;
    (async () => {
      while (!stop) {
        const img = grab();
        if (img) {
          try {
            const d = await api("/api/offline/detect", { image: img });
            if (!stop) setPreview(d);
          } catch {
            /* ignore preview errors */
          }
        }
        await sleep(400);
      }
    })();
    return () => {
      stop = true;
    };
  }, [camState, step]);

  const runCheck = async () => {
    setError("");
    setLive(null);
    setStep("running");
    runningRef.current = true;
    try {
      const ch = await api("/api/offline/challenge", {});
      const deadline = Date.now() + 25000;
      let p = null;
      while (runningRef.current && Date.now() < deadline) {
        const img = grab();
        if (!img) {
          await sleep(200);
          continue;
        }
        p = await api("/api/offline/frame", { challenge_id: ch.challenge_id, image: img });
        setLive(p);
        if (p.done) break;
        await sleep(FRAME_GAP_MS);
      }
      setStep("checking");
      onResult(await api("/api/offline/finish", { challenge_id: ch.challenge_id, location, test_label: testLabel || null }));
    } catch (e) {
      setError(e.message);
    } finally {
      runningRef.current = false;
      setStep("idle");
      setLive(null);
    }
  };

  const submitCode = async () => {
    setBusy(true);
    setError("");
    try {
      onResult(await api("/api/offline/decide", { person_id: personId.trim().replaceAll(" ", "_"), code }));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const box = (step === "running" ? live?.box : preview?.box) || null;
  const faces = step === "running" ? live?.faces : preview?.faces;
  const faceMsg =
    camState !== "on" ? "" : faces === 1 ? "✅ Face detected" : faces > 1 ? "⛔ More than one face" : "🔍 No face detected: look at the camera";

  return (
    <div className="card offline">
      <h2>📴 Offline mode: local face check</h2>
      {testLabel && <p className="testbadge">🧪 Test mode: this offline attempt is recorded as <b>{testLabel.replaceAll("_", " ")}</b></p>}
      <p>
        AWS is unreachable, so this kiosk uses its <b>on-device face model</b> (OpenCV YuNet + SFace) and a{" "}
        <b>random-action liveness check</b>. Photos, screens and recorded videos can't follow random instructions.
        Only scheduled access is allowed offline.
      </p>
      {status?.offline_face_ready === false && (
        <p className="error">Offline face models are missing on this kiosk: run <b>python get_models.py</b> and restart server.py.</p>
      )}
      <div className="offcam on">
        <video ref={videoRef} muted playsInline autoPlay />
        {box && (
          <div className={`facebox ${faces === 1 ? "good" : "bad"}`}
            style={{ left: `${(1 - box[0] - box[2]) * 100}%`, top: `${box[1] * 100}%`, width: `${box[2] * 100}%`, height: `${box[3] * 100}%` }} />
        )}
        {camState === "starting" && <div className="camnote">Starting camera... (allow camera access if the browser asks)</div>}
        {camState === "error" && <div className="camnote bad">{camError}</div>}
        {step === "running" && live && (
          <div className="instruction">
            {live.message || live.instruction} <span className="count">{Math.ceil(live.seconds_left)}s</span>
          </div>
        )}
        {step === "running" && !live && <div className="instruction">Getting your random actions...</div>}
      </div>
      <p className="facestate">{faceMsg}</p>
      {step === "running" && live?.steps && (
        <div className="steps">
          {live.steps.map((s, i) => (
            <span key={i} className={s.done ? "done" : ""}>{s.done ? "✅" : "⬜"} {s.label}</span>
          ))}
        </div>
      )}
      {step === "checking" && <p>Checking face and actions locally...</p>}
      {step === "idle" && camState === "on" && (
        <button className="big ok" onClick={runCheck}>Start offline face check</button>
      )}
      {camState === "error" && <button onClick={startCamera}>Retry camera</button>}
      {error && <p className="error">{error}</p>}
      <p className="muted">
        <button className="link" onClick={() => setShowCode(!showCode)}>
          {showCode ? "Hide" : "No camera / not recognised? Use ID + 6-digit code"}
        </button>
      </p>
      {showCode && (
        <div>
          <input placeholder="Person ID (e.g. Afrah Zainab)" value={personId} onChange={(e) => setPersonId(e.target.value)} />
          <input placeholder="6-digit code" value={code} maxLength={6} onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))} />
          <button className="ok" disabled={busy || !personId || code.length !== 6} onClick={submitCode}>
            {busy ? "Checking..." : "Verify"}
          </button>
        </div>
      )}
      <p className="muted">{status?.pending_sync || 0} decision(s) waiting to sync when the connection returns.</p>
    </div>
  );
}

function CodeStepUp({ result, location, onResult }) {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const go = async () => {
    setBusy(true);
    setErr("");
    try {
      onResult(await api("/api/offline/stepup", { ticket: result.stepup_ticket, code, location }));
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="stepup">
      <input placeholder="6-digit authenticator code" value={code} maxLength={6}
        onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))} />
      <button className="ok" disabled={busy || code.length !== 6} onClick={go}>{busy ? "Checking..." : "Verify code"}</button>
      {err && <p className="error">{err}</p>}
    </div>
  );
}

async function speak(text) {
  try {
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return;
    const url = URL.createObjectURL(await res.blob());
    await new Audio(url).play();
  } catch {
    /* voice is optional */
  }
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [phase, setPhase] = useState("idle"); // idle | starting | liveness | deciding | result
  const [session, setSession] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [testLabel, setTestLabel] = useState("");
  const [credMode, setCredMode] = useState("");
  const [door, setDoor] = useState("");
  const [direction, setDirection] = useState("in"); // in = check-in (entry) | out = check-out (exit)
  // Enrollment is started from the admin console, which opens this page with ?enroll=<one-time token>.
  const enrollToken = new URLSearchParams(window.location.search).get("enroll");
  const [mode] = useState(enrollToken ? "enroll" : "access"); // access | enroll
  const [enrollInfo, setEnrollInfo] = useState(null);
  const [enrollMsg, setEnrollMsg] = useState(null);
  const [voiceOn, setVoiceOn] = useState(true);
  const showResult = (r) => {
    setResult(r);
    setPhase("result");
    if (voiceOn && r?.speech) speak(r.speech);
  };

  const refresh = useCallback(async () => {
    try {
      setStatus(await api("/api/status"));
    } catch {
      setStatus((s) => ({ ...(s || {}), online: false }));
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    if (!enrollToken) return;
    api(`/api/enroll_token/${enrollToken}`)
      .then(setEnrollInfo)
      .catch((e) => setEnrollMsg({ ok: false, message: e.message }));
  }, [enrollToken]);

  const start = async (dir) => {
    if (dir === "in" || dir === "out") setDirection(dir);
    setError("");
    setResult(null);
    setPhase("starting");
    try {
      const s = await api("/api/session", {});
      setSession(s);
      setCredMode(s.credential_mode);
      setPhase("liveness");
    } catch (e) {
      if (e.status === 503) await refresh();
      setError(e.message);
      setPhase("idle");
    }
  };

  const onAnalysisComplete = async () => {
    setPhase("deciding");
    try {
      if (mode === "enroll") {
        setEnrollMsg(await api("/api/enroll", { session_id: session.sessionId, token: enrollToken }));
        setPhase("idle");
      } else {
        showResult(await api("/api/decide", {
          session_id: session.sessionId,
          test_label: testLabel || null,
          location: door || status?.location || null,
          direction,
        }));
      }
    } catch (e) {
      setError(e.message);
      setPhase("idle");
    }
    refresh();
  };

  const onGuard = async (verdict) => {
    const pin = window.prompt("Guard PIN");
    if (pin === null) return;
    try {
      await api("/api/override", {
        person_id: result.person_id, verdict, pin,
        location: door || status?.location || null, direction: result.direction || "in",
      });
      setResult({ ...result, verdict, verification: null, explanation: `Guard set the decision to ${verdict}.` });
      if (voiceOn) speak(verdict === "ALLOW" ? "Approved by security. Access granted." : "Access denied by security.");
    } catch (e) {
      setError(e.message);
    }
  };

  const reset = () => {
    setSession(null);
    setResult(null);
    setError("");
    setPhase("idle");
  };

  const offline = status && status.online === false;
  const m = status?.metrics?.overall;
  const pct = (v) => (v === null || v === undefined ? "–" : `${v}%`);

  return (
    <div className="page">
      <header>
        <div className="brand">🛡️ HID Liveness Kiosk</div>
        <div className="meta">
          📍{" "}
          <select className="door" value={door || status?.location || ""} onChange={(e) => setDoor(e.target.value)}>
            {(status?.locations || [status?.location || "..."]).map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>{" "}
          · {offline ? <span className="off">● Offline</span> : <span className="on">● Online</span>}
          {" · "}
          <button className="voice" onClick={() => setVoiceOn(!voiceOn)} title="Spoken feedback (accessibility)">
            {voiceOn ? "🔊 Voice on" : "🔇 Voice off"}
          </button>
        </div>
      </header>

      <div className="flow">
        <span className={phase === "liveness" ? "active" : ""}>1. Capture</span> →
        <span className={phase === "liveness" ? "active" : ""}>2. Liveness</span> →
        <span className={phase === "deciding" ? "active" : ""}>3. Face authentication</span> →
        <span className={phase === "deciding" ? "active" : ""}>4. Campus schedule</span> →
        <span className={phase === "result" ? "active" : ""}>5. Decision</span>
      </div>

      {status?.threat?.active && (
        <div className="card threat">🚨 <b>Heightened security.</b> {status.threat.reason}</div>
      )}

      {error && <div className="card error">⚠️ {error}</div>}

      {phase === "result" && result ? (
        <Result result={result} onGuard={onGuard} onRetry={result.mode === "offline" ? reset : start} onNext={reset}
          location={door || status?.location || null} onResult={(r) => { showResult(r); refresh(); }} />
      ) : offline ? (
        <OfflinePanel status={status} location={door || status?.location || null} testLabel={testLabel}
          onResult={(r) => { showResult(r); refresh(); }} />
      ) : phase === "liveness" && session ? (
        <div className="card detector">
          <FaceLivenessDetectorCore
            sessionId={session.sessionId}
            region={session.region}
            onAnalysisComplete={onAnalysisComplete}
            onUserCancel={reset}
            onError={(err) => {
              setError(`Liveness check error: ${err?.state || ""} ${err?.error?.message || ""}`);
              setPhase("idle");
            }}
            config={{ credentialProvider: async () => session.credentials }}
          />
        </div>
      ) : phase === "deciding" || phase === "starting" ? (
        <div className="card center">
          <div className="spinner" />
          <p>{phase === "starting" ? `Preparing the liveness check (${direction === "out" ? "check-out" : "check-in"})...` : "Checking liveness, identity and schedule..."}</p>
        </div>
      ) : mode === "enroll" ? (
        <div className="card">
          <h2>{enrollInfo?.kind === "offline_vector" ? "📴 Save offline face vector" : "👤 Liveness-verified enrollment"}</h2>
          {enrollInfo && !enrollMsg && enrollInfo.kind === "offline_vector" && (
            <p>Saving the offline face vector for <b>{enrollInfo.person_id.replaceAll("_", " ")}</b>. The person does the
              liveness check; the vector is stored only if the face is live AND it is the same face already enrolled.</p>
          )}
          {enrollInfo && !enrollMsg && enrollInfo.kind !== "offline_vector" && (
            <p>Enrolling <b>{enrollInfo.person_id.replaceAll("_", " ")}</b> ({enrollInfo.role}, {enrollInfo.status}).
              The person now does the liveness check. Only a LIVE face (≥ 97%) is accepted. Photos, screens and masks
              are rejected, and a face already enrolled under another name is refused.</p>
          )}
          {enrollMsg && <div className={`card ${enrollMsg.ok ? "okmsg" : "error"}`}>{enrollMsg.ok ? "✅" : "⛔"} {enrollMsg.message}</div>}
          {enrollInfo && !enrollMsg && (
            <button className="big" onClick={start}>Start liveness check</button>
          )}
          {enrollMsg && <p className="muted">You can close this tab and return to the admin console.</p>}
        </div>
      ) : (
        <div className="card center">
          <h1>Welcome</h1>
          {testLabel && <p className="testbadge">🧪 Test mode: this attempt is recorded as <b>{testLabel.replaceAll("_", " ")}</b></p>}
          <p>Look at the camera and follow the instructions. We check that you are a real person and that you are allowed in.</p>
          <div className="dirbuttons">
            <button className="big" onClick={() => start("in")}>🚪 Check in (enter)</button>
            <button className="big secondary" onClick={() => start("out")}>👋 Check out (leave)</button>
          </div>
          <div className="presentlist">
            <b>🚪 Checked in right now:</b>{" "}
            {status?.present?.length
              ? status.present.map((p) => `${p.person_id.replaceAll("_", " ")} @ ${p.location} (since ${p.since.slice(11, 16)})`).join(" · ")
              : "nobody"}
          </div>

          <p className="muted">
            AI is used to check liveness and identity. Your photo is not stored; only a face vector is kept,
            and it can be deleted on request. Uncertain cases go to a human.
          </p>
        </div>
      )}
      <div className="testmode">
        <label>
          🧪 Test mode (staff only, for the performance report):{" "}
          <select value={testLabel} onChange={(e) => setTestLabel(e.target.value)}>
            <option value="">off (normal use)</option>
            <option value="genuine">genuine (real enrolled person)</option>
            <option value="printed_photo">printed photo attack</option>
            <option value="screen_photo">photo on a phone/screen</option>
            <option value="video_replay">video replay attack</option>
            <option value="mask">mask attack</option>
            <option value="impostor">impostor (real, not enrolled)</option>
          </select>
        </label>
        {m && (
          <div className="metrics">
            📊 Online (AWS) test results ({status.metrics.attempts} labelled attempts) · Liveness detection accuracy <b>{pct(m.liveness_accuracy_pct)}</b> ·
            Face authentication accuracy <b>{pct(m.face_auth_accuracy_pct)}</b> · FAR <b>{pct(m.false_accept_rate_pct)}</b> ·
            FRR <b>{pct(m.false_reject_rate_pct)}</b> · Avg decision <b>{m.avg_decision_ms ?? "–"} ms</b>
          </div>
        )}
        <label className="simulate">
          <input type="checkbox" checked={!!status?.simulated_offline}
            onChange={async (e) => { try { setStatus(await api("/api/offline/simulate", { offline: e.target.checked })); } catch { /* ignore */ } }} />{" "}
          📴 Simulate AWS outage (demo of Challenge 2: kiosk switches to the offline face check)
        </label>
        {status?.offline_metrics?.overall && (
          <div className="metrics">
            📴 Offline test results ({status.offline_metrics.attempts} labelled attempts) · Liveness detection accuracy{" "}
            <b>{pct(status.offline_metrics.overall.liveness_accuracy_pct)}</b> · Face authentication accuracy{" "}
            <b>{pct(status.offline_metrics.overall.face_auth_accuracy_pct)}</b> · FAR <b>{pct(status.offline_metrics.overall.false_accept_rate_pct)}</b> ·
            FRR <b>{pct(status.offline_metrics.overall.false_reject_rate_pct)}</b>
          </div>
        )}
        {credMode && (
          <div>
            🔐 Browser credentials:{" "}
            <b>{credMode === "least-privilege" ? "least-privilege (liveness only, 15 min)" : "development mode (full)"}</b>
          </div>
        )}
      </div>
    </div>
  );
}
