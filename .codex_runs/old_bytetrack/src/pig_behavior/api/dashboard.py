"""HTML dashboard for real-time pig behavior tracking."""


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pig Behavior Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d9dee7;
      --text: #17202a;
      --muted: #697586;
      --accent: #1f7a8c;
      --accent-strong: #155e6d;
      --danger: #b42318;
      --ok: #147d4f;
      --track: #334155;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 390px;
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 64px);
    }
    .video-pane, .side-pane, .section {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .video-pane {
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }
    .video-head, .section-head {
      min-height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    .video-wrap {
      min-height: 420px;
      display: grid;
      place-items: center;
      background: #111827;
      overflow: hidden;
    }
    #stream {
      width: 100%;
      height: auto;
      max-height: calc(100vh - 150px);
      object-fit: contain;
      display: block;
    }
    .side-pane {
      padding: 14px;
      overflow-y: auto;
      max-height: calc(100vh - 100px);
    }
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 14px;
    }
    button {
      min-height: 40px;
      border: 1px solid transparent;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
    }
    #startBtn { background: var(--accent); color: #fff; }
    #startBtn:hover { background: var(--accent-strong); }
    #stopBtn { background: #fff; color: var(--danger); border-color: #efb5ae; }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    input {
      width: 100%;
      height: 36px;
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 14px;
    }
    .field-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .section {
      margin-top: 14px;
      overflow: hidden;
    }
    .section-body { padding: 12px 14px; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
      background: #eef2f6;
      color: var(--track);
    }
    .status-pill.running {
      background: #dcfce7;
      color: var(--ok);
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      min-height: 70px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric strong {
      font-size: 22px;
      line-height: 1.1;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 96px 1fr 44px;
      gap: 8px;
      align-items: center;
      margin: 8px 0;
      font-size: 13px;
    }
    .bar-shell {
      height: 9px;
      border-radius: 999px;
      background: #edf1f5;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
    }
    th {
      color: var(--muted);
      font-weight: 700;
    }
    .path {
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 13px;
    }
    .error { color: var(--danger); font-size: 13px; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .side-pane { max-height: none; }
      .video-wrap { min-height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Pig Behavior Monitor</h1>
    <span id="state" class="status-pill">Idle</span>
  </header>
  <main>
    <section class="video-pane">
      <div class="video-head">
        <div>
          <strong>Detection & Tracking</strong>
          <div id="videoPath" class="path"></div>
          <div id="detectorPath" class="path"></div>
          <div id="behaviorPath" class="path"></div>
        </div>
        <div id="clock" class="path">0.00s</div>
      </div>
      <div class="video-wrap">
        <img id="stream" alt="Annotated video stream">
      </div>
    </section>
    <aside class="side-pane">
      <div class="field-grid">
        <div>
          <label for="confidence">Confidence</label>
          <input id="confidence" type="number" min="0.05" max="0.95"
            step="0.05" value="0.25">
        </div>
        <div>
          <label for="stride">Process stride</label>
          <input id="stride" type="number" min="1" max="30" step="1" value="1">
        </div>
        <div>
          <label for="behaviorStride">Behavior stride</label>
          <input id="behaviorStride" type="number" min="1" max="30"
            step="1" value="3">
        </div>
      </div>
      <div class="controls">
        <button id="startBtn">Start</button>
        <button id="stopBtn">Stop</button>
      </div>
      <div id="error" class="error"></div>

      <section class="section">
        <div class="section-head"><strong>Runtime</strong></div>
        <div class="section-body metric-grid">
          <div class="metric">
            <span>Tracked pigs</span><strong id="tracks">0</strong>
          </div>
          <div class="metric">
            <span>Processing FPS</span><strong id="pfps">0</strong>
          </div>
          <div class="metric"><span>Source FPS</span><strong id="sfps">0</strong></div>
          <div class="metric"><span>Frames</span><strong id="frames">0</strong></div>
        </div>
      </section>

      <section class="section">
        <div class="section-head"><strong>Current Behaviors</strong></div>
        <div id="currentBars" class="section-body"></div>
      </section>

      <section class="section">
        <div class="section-head"><strong>Behavior Time</strong></div>
        <div id="timeBars" class="section-body"></div>
      </section>

      <section class="section">
        <div class="section-head"><strong>Track Summary</strong></div>
        <div class="section-body">
          <table>
            <thead><tr><th>ID</th><th>Behavior</th><th>Obs.</th></tr></thead>
            <tbody id="trackRows"></tbody>
          </table>
        </div>
      </section>
    </aside>
  </main>
  <script>
    const stream = document.getElementById("stream");
    const state = document.getElementById("state");
    const error = document.getElementById("error");
    const confidence = document.getElementById("confidence");
    const stride = document.getElementById("stride");
    const behaviorStride = document.getElementById("behaviorStride");

    function cacheBustedStream() {
      stream.src = "/tracking/stream?ts=" + Date.now();
    }

    async function startTracking() {
      error.textContent = "";
      const response = await fetch("/tracking/start", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          confidence: Number(confidence.value),
          frame_stride: Number(stride.value),
          behavior_stride_frames: Number(behaviorStride.value),
          realtime: true
        })
      });
      const payload = await response.json();
      if (!response.ok) {
        error.textContent = payload.detail || "Could not start tracking.";
      }
      cacheBustedStream();
      await refreshStatus();
    }

    async function stopTracking() {
      await fetch("/tracking/stop", {method: "POST"});
      await refreshStatus();
    }

    function renderBars(targetId, values, suffix) {
      const target = document.getElementById(targetId);
      const entries = Object.entries(values || {}).sort((a, b) => b[1] - a[1]);
      if (!entries.length) {
        target.innerHTML = '<div class="path">No data yet</div>';
        return;
      }
      const maxValue = Math.max(...entries.map((entry) => entry[1]), 1);
      target.innerHTML = entries.map(([label, value]) => {
        const width = Math.max(4, (value / maxValue) * 100);
        const display = Number.isInteger(value) ? value : value.toFixed(1);
        return `<div class="bar-row">
          <div title="${label}">${label}</div>
          <div class="bar-shell">
            <div class="bar-fill" style="width:${width}%"></div>
          </div>
          <div>${display}${suffix}</div>
        </div>`;
      }).join("");
    }

    function renderTracks(rows) {
      const body = document.getElementById("trackRows");
      if (!rows || !rows.length) {
        body.innerHTML = '<tr><td colspan="3" class="path">No tracks yet</td></tr>';
        return;
      }
      body.innerHTML = rows.map((row) => `<tr>
        <td>${row.track_id}</td>
        <td>${row.dominant_behavior}</td>
        <td>${row.observations}</td>
      </tr>`).join("");
    }

    async function refreshStatus() {
      const response = await fetch("/tracking/status");
      const data = await response.json();
      state.textContent = data.running ? "Running" : "Idle";
      state.className = data.running ? "status-pill running" : "status-pill";
      error.textContent = data.error || "";
      document.getElementById("videoPath").textContent = data.video_path || "";
      document.getElementById("detectorPath").textContent =
        `Detector: ${data.detector_model_path || ""}`;
      document.getElementById("behaviorPath").textContent =
        `Behavior: ${data.behavior_model_path || ""}`;
      document.getElementById("clock").textContent = `${data.video_time_sec || 0}s`;
      document.getElementById("tracks").textContent = data.track_count || 0;
      document.getElementById("pfps").textContent = data.processing_fps || 0;
      document.getElementById("sfps").textContent = data.source_fps || 0;
      document.getElementById("frames").textContent = data.frames_processed || 0;
      renderBars("currentBars", data.current_counts, "");
      renderBars("timeBars", data.behavior_seconds, "s");
      renderTracks(data.top_tracks);
    }

    document.getElementById("startBtn").addEventListener("click", startTracking);
    document.getElementById("stopBtn").addEventListener("click", stopTracking);
    cacheBustedStream();
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
