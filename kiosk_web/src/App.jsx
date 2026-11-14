import React, { useCallback, useEffect, useState } from "react";
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

function Result({ result, onGuard, onRetry, onNext }) {
  const v = result.verdict;
  const live = result.liveness || {};
  const t = result.timing || {};
  const fast = result.decision_ms != null && result.decision_ms <= 2000;
  return (
    <div className={`card result ${v}`}>
      <div className="verdict">
        {ICON[v]} {LABEL[v] || v}
      </div>
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
        {(v === "REQUEST_VERIFICATION" && result.verification !== "guard") || v === "RETRY" ? (
          <button className="ok" onClick={onRetry}>Re-capture (verify again)</button>
        ) : null}
        <button onClick={onNext}>Next person</button>
      </div>
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
      </div>
    </div>
  );
}

function OfflinePanel({ status, onResult }) {
  const [personId, setPersonId] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const submit = async () => {
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
  return (
    <div className="card offline">
      <h2>📴 Offline mode: safe access</h2>
      <p>
        The kiosk cannot reach the central service. Face access is paused. Enter your ID and the
        6-digit code from your authenticator app. Only scheduled access is allowed offline.
      </p>
      <input placeholder="Person ID (e.g. Afrah Zainab)" value={personId} onChange={(e) => setPersonId(e.target.value)} />
      <input placeholder="6-digit code" value={code} maxLength={6} onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))} />
      <button className="ok" disabled={busy || !personId || code.length !== 6} onClick={submit}>
        {busy ? "Checking..." : "Verify"}
      </button>
      {error && <p className="error">{error}</p>}
      <p className="muted">{status?.pending_sync || 0} decision(s) waiting to sync when the connection returns.</p>
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
      </div>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [phase, setPhase] = useState("idle"); // idle | starting | liveness | deciding | result
  const [session, setSession] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [testLabel, setTestLabel] = useState("");

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

  const start = async () => {
    setError("");
    setResult(null);
    setPhase("starting");
    try {
      setSession(await api("/api/session", {}));
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
      setResult(await api("/api/decide", { session_id: session.sessionId, test_label: testLabel || null }));
      setPhase("result");
    } catch (e) {
      setError(e.message);
      setPhase("idle");
    }
    refresh();
  };

  const onGuard = async (verdict) => {
    await api("/api/override", { person_id: result.person_id, verdict });
    setResult({ ...result, verdict, verification: null, explanation: `Guard set the decision to ${verdict}.` });
  };

  const reset = () => {
    setSession(null);
    setResult(null);
    setError("");
    setPhase("idle");
  };

  const offline = status && status.online === false;

  return (
    <div className="page">
      <header>
        <div className="brand">🛡️ HID Liveness Kiosk</div>
        <div className="meta">
          📍 {status?.location || "..."} · {offline ? <span className="off">● Offline</span> : <span className="on">● Online</span>}
        </div>
      </header>

      <div className="flow">
        <span className={phase === "liveness" ? "active" : ""}>1. Capture</span> →
        <span className={phase === "liveness" ? "active" : ""}>2. Liveness</span> →
        <span className={phase === "deciding" ? "active" : ""}>3. Face authentication</span> →
        <span className={phase === "deciding" ? "active" : ""}>4. Campus schedule</span> →
        <span className={phase === "result" ? "active" : ""}>5. Decision</span>
      </div>

      {error && <div className="card error">⚠️ {error}</div>}

      {phase === "result" && result ? (
        <Result result={result} onGuard={onGuard} onRetry={start} onNext={reset} />
      ) : offline ? (
        <OfflinePanel status={status} onResult={(r) => { setResult(r); setPhase("result"); refresh(); }} />
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
          <p>{phase === "starting" ? "Preparing the liveness check..." : "Checking liveness, identity and schedule..."}</p>
        </div>
      ) : (
        <div className="card center">
          <h1>Welcome</h1>
          {testLabel && <p className="testbadge">🧪 Test mode: this attempt is recorded as <b>{testLabel.replaceAll("_", " ")}</b></p>}
          <p>Look at the camera and follow the instructions. We check that you are a real person and that you are allowed in.</p>
          <button className="big" onClick={start}>Start</button>
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
      </div>
    </div>
  );
}
