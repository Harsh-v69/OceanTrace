/* All screens for the Operations Console. Each view renders into ctx.root and
   wires its own events. ctx = { user, root, go, toast }. */

import { api, fetchText } from "./api.js?v=ui5";
import {
  makeMap, anomalyMarker, vesselMarker, trackLine, polygon, fit, L,
  vesselTrackLayer, vesselPopupHtml, shorelineContact, mapLegend,
} from "./map.js?v=ui5";

/* -------------------------------------------------------------- helpers -- */
const h = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (v == null ? "-" : `${(v * 100).toFixed(0)}%`);
const num = (v, d = 2) => (v == null || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
const when = (s) => (s ? new Date(s).toLocaleString() : "-");
/* accept a bbox as [w,s,e,n] or {west,south,east,north} (or min_lon/… variants) */
function bboxStr(b) {
  if (Array.isArray(b) && b.length) return b.map((v) => num(v, 2)).join(", ");
  if (b && typeof b === "object") {
    const w = b.west ?? b.min_lon ?? b.w, s = b.south ?? b.min_lat ?? b.s;
    const e = b.east ?? b.max_lon ?? b.e, n = b.north ?? b.max_lat ?? b.n;
    if ([w, s, e, n].every((v) => v != null)) return [w, s, e, n].map((v) => num(v, 2)).join(", ");
  }
  return "-";
}
const $ = (sel, r = document) => r.querySelector(sel);
const $$ = (sel, r = document) => [...r.querySelectorAll(sel)];

function confBadge(c) {
  if (c == null) return `<span class="badge mut">n/a</span>`;
  const k = c >= 0.75 ? "crit" : c >= 0.5 ? "warn" : "info";
  return `<span class="badge ${k}">${pct(c)}</span>`;
}
function statusBadge(s) {
  const k = { RESOLVED: "ok", IN_PROGRESS: "info", OPEN: "warn", ARCHIVED: "mut" }[s] || "mut";
  return `<span class="badge ${k}">${h(s)}</span>`;
}
function alertBadge(s) {
  const k = { SENT: "ok", MOCKED: "info", PENDING: "warn", FAILED: "crit", SUPPRESSED: "mut" }[s] || "mut";
  return `<span class="badge ${k}">${h(s)}</span>`;
}
function page(title, sub, bodyHtml) {
  return `<div class="page-head"><div><h1>${h(title)}</h1>${sub ? `<p>${h(sub)}</p>` : ""}</div></div>${bodyHtml}`;
}
function kpi(v, l, tone) {
  return `<div class="kpi${tone ? " t-" + tone : ""}"><div class="v">${v}</div><div class="l">${h(l)}</div></div>`;
}
function kv(pairs) {
  return `<dl class="kv">${pairs.map(([k, v]) => `<dt>${h(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
}
function raw(obj) {
  return `<details class="raw"><summary class="muted">Raw payload</summary><pre>${h(JSON.stringify(obj, null, 2))}</pre></details>`;
}
async function loadInvestigations() {
  try { return await api.investigations(); } catch { return []; }
}

/* candidate component families for the bars */
const FAMILY = { physical: "physical", ais: "ais", behavioural: "behavioural" };

function candidateCard(c, truthMmsi) {
  const comps = c.components || {};
  const order = ["spatiotemporal", "axis_alignment", "proximity", "blackout", "ais_anomaly", "route_deviation", "vessel_prior"];
  const rows = order.filter((k) => comps[k]).map((k) => {
    const cc = comps[k];
    const fam = cc.family || "physical";
    const w = Math.max(0, Math.min(1, cc.value)) * 100;
    return `<div class="comprow ${fam}">
      <span title="${h(fam)} evidence">${h(k.replace(/_/g, " "))}</span>
      <span class="bar"><i style="width:${w.toFixed(0)}%"></i></span>
      <span class="mono">${(cc.points ?? 0).toFixed(1)}</span></div>`;
  }).join("");
  const isTruth = truthMmsi != null && c.identity.mmsi === truthMmsi;
  return `<div class="cand ${c.rank === 1 ? "top" : ""}">
    <div class="h">
      <b>#${c.rank ?? "-"} ${h(c.identity.name)}</b>
      <span class="score">${num(c.score, 1)}</span>
    </div>
    <div class="muted mono">MMSI ${h(c.identity.mmsi)} &middot; ${h(c.identity.vessel_type || "?")} &middot; ${h(c.identity.flag || "?")}
      ${isTruth ? ' &middot; <span class="badge ok">ground truth</span>' : ""}</div>
    <div class="muted">${h(c.assessment || "")}${c.margin_over_next != null ? ` &middot; margin +${num(c.margin_over_next, 1)}` : ""}</div>
    <div class="comps">${rows}</div>
    ${c.best_match_time_h != null ? `<div class="finding">Best space-time match at T${c.best_match_time_h >= 0 ? "+" : ""}${num(c.best_match_time_h, 1)} h &middot; CPA ${num(c.cpa_km, 1)} km</div>` : ""}
  </div>`;
}

/* ============================================================= MISSION == */
async function missionControl(ctx) {
  ctx.root.innerHTML = page(
    "Mission Control",
    "Live picture of the Indian coastline: active oil-like anomalies, their confidence, and the vessels under attribution.",
    `<div class="kpis" id="mc-kpis"></div>
     <p class="statline" id="mc-statline"></p>
     <div class="panel"><h2>Coastal picture <span class="h2-note" id="mc-map-note"></span></h2>
       <div id="mc-map" class="map"></div></div>
     <div class="grid cols-2">
       <div class="panel"><h2>Run a demo scenario</h2>
         <p class="muted">Each runs the full pipeline: SAR detection &rarr; look-alike filter &rarr; drift hindcast &rarr; AIS fusion &rarr; jurisdiction &rarr; SMS alert.</p>
         <div id="mc-scenarios" class="row"></div><p id="mc-run-msg" class="muted"></p></div>
       <div class="panel"><h2>Analyse an uploaded scene</h2>
         <form id="mc-upload" class="row" style="align-items:flex-end;gap:.6rem;flex-wrap:wrap">
           <label>Sentinel-1 GeoTIFF / PNG<input type="file" name="scene" accept=".tif,.tiff,.png,.jpg" required></label>
           <label>Ground-truth mask (optional &rarr; real IoU)<input type="file" name="ground_truth_mask" accept=".tif,.tiff,.png"></label>
           <label>bbox W,S,E,N<input name="bbox" placeholder="72.0,18.0,72.4,18.4" size="20"></label>
           <button class="btn btn-primary" type="submit">Analyse</button>
           <span id="mc-upload-msg" class="muted"></span>
         </form>
       </div>
       <div class="panel col-span-2"><h2>Recent investigations</h2><div id="mc-recent"></div></div>
     </div>`
  );

  const map = makeMap($("#mc-map"));
  const [invs, scen] = await Promise.all([loadInvestigations(), api.scenarios().catch(() => ({ scenarios: [] }))]);

  const markers = [];
  let mapped = 0, hasPrimeTrack = false;
  for (const inv of invs) {
    if (inv.centroid_lat == null) continue;
    mapped += 1;
    const sm = inv.summary_metrics || {};
    const c = sm.sar?.confidence;
    const cls = sm.sar?.scene_classification || "";
    const m = anomalyMarker(map, inv.centroid_lat, inv.centroid_lon, {
      confidence: c ?? 0, label: cls || "Anomaly", ref: inv.reference, classification: cls,
    });
    m.on("click", () => ctx.go(`#/workstation/${inv.id}`));
    m.on("mouseover", () => m.setSelected && m.setSelected(true));
    m.on("mouseout", () => m.setSelected && m.setSelected(false));
    markers.push(m);
    // draw the prime suspect's reconstructed track for this investigation
    const vts = sm.vessel_tracks || {};
    const prime = Object.values(vts).find((v) => v?.attribution?.is_prime);
    if (prime && prime.pings?.length) {
      const { group } = vesselTrackLayer(map, prime, {
        prime: true,
        onClick: () => ctx.go(`#/workstation/${inv.id}`),
      });
      markers.push(group); hasPrimeTrack = true;
    }
  }
  if (markers.length) fit(map, markers);
  mapLegend(map, ["anom-hi", "anom-mid", "anom-la", ...(hasPrimeTrack ? ["track-prime"] : [])]);
  $("#mc-map-note").textContent = mapped ? `${mapped} plotted` : "no located cases yet";

  let alertN = null;
  try {
    const al = await api.alerts();
    alertN = al.filter((a) => ["SENT", "MOCKED"].includes(a.status)).length;
  } catch { /* PILOT can't read alerts */ }

  // "what is happening right now" — derived only from real summary_metrics
  const clsOf = (i) => i.summary_metrics?.sar?.scene_classification;
  const confOf = (i) => i.summary_metrics?.sar?.confidence ?? 0;
  const hasPrime = (i) => {
    const vts = i.summary_metrics?.vessel_tracks || {};
    return Object.values(vts).some((v) => v?.attribution?.is_prime)
      || !!i.summary_metrics?.attribution?.summary?.prime_suspect;
  };
  const oilLike = invs.filter((i) => clsOf(i) === "Oil-like anomaly");
  const activeAnoms = oilLike.filter((i) => i.status === "OPEN" || i.status === "IN_PROGRESS");
  const highConf = invs.filter((i) => confOf(i) >= 0.75);
  const lookalikes = invs.filter((i) => clsOf(i) === "Likely look-alike");
  const primeCount = invs.filter(hasPrime).length;

  $("#mc-kpis").innerHTML =
    kpi(activeAnoms.length, "Active anomalies", "crit") +
    kpi(highConf.length, "High confidence", "hot") +
    kpi(primeCount || "-", "Prime suspects", "info") +
    kpi(lookalikes.length || "-", "Look-alikes filtered", "warn") +
    kpi(alertN == null ? "-" : alertN, "Alerts delivered", "ok");

  $("#mc-statline").innerHTML =
    `<b>${invs.length}</b> case(s) in your jurisdiction &middot; ` +
    `<b class="tone-crit">${activeAnoms.length}</b> active oil-like &middot; ` +
    `<b class="tone-warn">${lookalikes.length}</b> look-alike(s) filtered &middot; ` +
    `<b class="tone-info">${primeCount}</b> with a prime suspect`;

  $("#mc-scenarios").innerHTML = scen.scenarios.map((s) =>
    `<button class="btn" data-key="${h(s.key)}">${h(s.name)}</button>`).join("") || `<span class="muted">none</span>`;
  $$("#mc-scenarios .btn").forEach((b) => b.addEventListener("click", async () => {
    $$("#mc-scenarios .btn").forEach((x) => (x.disabled = true));
    $("#mc-run-msg").textContent = `Running "${b.dataset.key}" through the full pipeline...`;
    try {
      const r = await api.runScenario(b.dataset.key);
      ctx.toast(`${r.reference}: ${r.verdict || r.classification}`);
      ctx.go(`#/workstation/${r.investigation_id}`);
    } catch (e) {
      $("#mc-run-msg").textContent = `${e.status === 403 ? "Outside your jurisdiction. " : ""}${e.message}`;
      $$("#mc-scenarios .btn").forEach((x) => (x.disabled = false));
    }
  }));

  $("#mc-upload").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    if (!fd.get("ground_truth_mask")?.size) fd.delete("ground_truth_mask");
    if (!fd.get("bbox")) fd.delete("bbox");
    const btn = e.target.querySelector("button");
    btn.disabled = true; $("#mc-upload-msg").textContent = "Analysing the uploaded scene...";
    try {
      const inv = await api.uploadScene(fd);
      const iou = inv.summary_metrics?.iou;
      ctx.toast(`${inv.reference}: ${inv.summary_metrics?.sar?.scene_classification}`
        + (iou ? ` · IoU ${iou.value}` : ""));
      ctx.go(`#/workstation/${inv.id}`);
    } catch (err) {
      $("#mc-upload-msg").textContent = (err.status === 403 ? "Outside your jurisdiction. " : "")
        + (Array.isArray(err.detail) ? err.detail.map((d) => d.msg).join("; ") : (err.detail || err.message));
      btn.disabled = false;
    }
  });

  $("#mc-recent").innerHTML = invs.length ? `<table class="data"><thead><tr>
    <th>Ref</th><th>Title</th><th>Status</th><th>Conf.</th><th>Opened</th></tr></thead><tbody>${
    invs.slice(0, 12).map((i) => `<tr class="clickable" data-id="${i.id}">
      <td class="mono">${h(i.reference)}</td><td>${h(i.title)}</td>
      <td>${statusBadge(i.status)}</td><td>${confBadge(i.summary_metrics?.sar?.confidence)}</td>
      <td>${when(i.created_at)}</td></tr>`).join("")}</tbody></table>`
    : `<p class="empty">No investigations yet - run a scenario.</p>`;
  $$("#mc-recent tr.clickable").forEach((r) => r.addEventListener("click", () => ctx.go(`#/workstation/${r.dataset.id}`)));
}

/* ========================================================= MONITORING == */
const PIPE_STEPS = [
  ["detect", "SAR detection", "Refined-Lee speckle filter, adaptive dark-spot segmentation, RF/GB voting ensemble."],
  ["characterize", "Characterisation", "Area, centroid, perimeter, major/minor axis, orientation."],
  ["trace", "Drift hindcast + forecast", "Lagrangian RK4 backward to the release origin, then 48 h forward."],
  ["correlate", "AIS correlation", "Normalise vessel pings, remove impossible speeds, gate background traffic."],
  ["rank", "Attribution fusion", "Physical + AIS autoencoder + LSTM route-deviation, release-time feedback loop."],
  ["jurisdiction", "Jurisdiction mapping", "Point-in-polygon against the maritime-zone catalogue."],
  ["alert", "SMS alert", "Fingerprinted, deduplicated dispatch above the confidence threshold."],
];
async function monitoring(ctx) {
  const scen = await api.scenarios().catch(() => ({ scenarios: [] }));
  ctx.root.innerHTML = page(
    "Live Monitoring Stream",
    "Replay an incoming satellite observation and watch the unified pipeline execute stage by stage.",
    `<div class="panel"><div class="row">
       <label>Observation feed
         <select id="mon-key">${scen.scenarios.map((s) => `<option value="${h(s.key)}">${h(s.name)}</option>`).join("")}</select>
       </label>
       <button class="btn btn-primary" id="mon-run">Replay observation</button>
       <span class="spacer"></span><span id="mon-status" class="muted"></span>
     </div></div>
     <div class="panel"><h2>Pipeline</h2><div class="pipeline" id="mon-pipe">${
       PIPE_STEPS.map((s, i) => `<div class="pstep" data-step="${s[0]}">
         <span class="n">${i + 1}</span><div class="b"><div class="t">${h(s[1])}</div>
         <div class="d">${h(s[2])}</div><div class="s"></div></div></div>`).join("")
     }</div></div>
     <div class="panel" id="mon-result" hidden></div>`
  );

  $("#mon-run").addEventListener("click", async () => {
    const key = $("#mon-key").value;
    $("#mon-run").disabled = true;
    $("#mon-status").textContent = "Acquiring scene…";
    $$("#mon-pipe .pstep").forEach((p) => {
      p.classList.remove("done", "run", "skip", "warn");
      $(".s", p).textContent = "";
    });
    $("#mon-result").hidden = true;

    let r;
    try {
      r = await api.runScenario(key);
    } catch (e) {
      $("#mon-status").textContent = e.status === 403 ? "Outside your jurisdiction." : e.message;
      $("#mon-run").disabled = false;
      return;
    }
    const byStep = Object.fromEntries((r.pipeline || []).map((s) => [s.step, s]));
    for (const [name] of PIPE_STEPS) {
      const el = $(`#mon-pipe .pstep[data-step="${name}"]`);
      el.classList.add("run");
      await new Promise((res) => setTimeout(res, 420));
      const info = byStep[name];
      el.classList.remove("run");
      if (!info) { el.classList.add("done", "skip"); $(".s", el).textContent = "skipped"; continue; }
      el.classList.add("done");
      if (info.dispatched === false || (info.status && /FAIL|SUPPRESS/i.test(info.status))) el.classList.add("warn");
      $(".s", el).textContent =
        `${info.label} · ${num(info.seconds, 3)} s` +
        (info.classification ? ` · ${info.classification}` : "") +
        (info.dispatched === false ? " · no alert raised" : info.status ? ` · ${info.status}` : "");
    }
    $("#mon-status").textContent = "Pipeline complete.";
    $("#mon-run").disabled = false;

    const jname = (r.jurisdiction?.primary_name) || r.jurisdiction?.primary_code || "international waters";
    const tone = r.classification === "Oil-like anomaly" ? "crit"
      : r.classification === "Likely look-alike" ? "warn" : "info";
    const res = $("#mon-result");
    res.hidden = false;
    res.innerHTML = `
      <div class="outcome t-${tone}">
        <div class="outcome-main">
          <div class="outcome-verdict">${h(r.classification || "-")}</div>
          <div class="muted">${h(r.reference)} &middot; ${h(r.verdict || "")}</div>
        </div>
        <div class="outcome-conf">${confBadge(r.confidence)}</div>
      </div>
      ${kv([
        ["Jurisdiction", h(jname)],
        ["Prime suspect", r.prime_suspect ? h(r.prime_suspect.identity?.name || "-") : "&mdash;"],
        ["Candidates evaluated", r.n_candidates ?? "&mdash;"],
        ["Alert", r.alert_status ? alertBadge(r.alert_status) : "not raised"],
      ])}
      <div class="row" style="margin-top:.8rem"><button class="btn btn-primary" id="mon-open">Open investigation workstation</button></div>`;
    $("#mon-open").addEventListener("click", () => ctx.go(`#/workstation/${r.investigation_id}`));
  });
}

/* ====================================================== INVESTIGATIONS == */
async function investigations(ctx) {
  const invs = await loadInvestigations();
  ctx.root.innerHTML = page("Investigations", "Every case inside your jurisdiction closure.",
    invs.length ? `<div class="panel"><table class="data"><thead><tr>
      <th>Ref</th><th>Title</th><th>Status</th><th>Confidence</th><th>Prime suspect</th><th>Opened</th></tr></thead><tbody>${
      invs.map((i) => {
        const ps = i.summary_metrics?.attribution?.summary?.prime_suspect?.identity?.name;
        return `<tr class="clickable" data-id="${i.id}">
          <td class="mono">${h(i.reference)}</td><td>${h(i.title)}</td>
          <td>${statusBadge(i.status)}</td><td>${confBadge(i.summary_metrics?.sar?.confidence)}</td>
          <td>${h(ps || "-")}</td><td>${when(i.created_at)}</td></tr>`;
      }).join("")}</tbody></table></div>`
      : `<p class="empty">No investigations yet.</p>`);
  $$("#view tr.clickable").forEach((r) => r.addEventListener("click", () => ctx.go(`#/workstation/${r.dataset.id}`)));
}

/* ======================================================== WORKSTATION == */
async function workstation(ctx, params) {
  const id = params[0];
  if (!id) { return investigations(ctx); }
  let inv;
  try { inv = await api.investigation(id); }
  catch (e) {
    ctx.root.innerHTML = page("Investigation", null,
      `<p class="empty">${e.status === 403 ? "This case is outside your assigned jurisdiction." : h(e.message)}</p>`);
    return;
  }
  const m = inv.summary_metrics || {};
  const sar = m.sar || {};
  const hind = m.hindcast || {};
  const fore = m.forecast || {};
  const attr = m.attribution || {};
  const cands = attr.candidates || [];
  const truth = m.scenario?.truth_mmsi ?? null;
  const prim = sar.primary_detection?.characterization || {};

  ctx.root.innerHTML = page(`${h(inv.reference)} - ${h(inv.title)}`, m.verdict || inv.description,
    `<div class="row" style="margin-bottom:1rem">
      ${statusBadge(inv.status)} ${confBadge(sar.confidence)}
      <span class="badge mut">${h(sar.scene_classification || "-")}</span>
      ${m.iou ? `<span class="badge info" title="vs operator ground-truth mask">IoU ${num(m.iou.value, 3)}</span>` : ""}
      <span class="spacer"></span>
      <label class="btn" style="cursor:pointer">Ingest AIS CSV
        <input type="file" id="ws-ais" accept=".csv" hidden></label>
      <button class="btn" id="ws-evidence">Evidence dossier</button>
    </div>
    <p id="ws-ais-msg" class="muted" style="margin:-.6rem 0 1rem"></p>
    <div class="workstation">
      <div class="wcol">
        <div class="panel"><h2>Scene, drift &amp; vessel tracks</h2><div id="ws-map" class="map sm"></div>
          <div class="timeline"><div class="tl-label"><span>T-48h</span><span id="ws-tl-now">observation (T0)</span><span>T+48h</span></div>
          <input type="range" id="ws-tl" min="-48" max="48" value="0" step="1"></div>
          <p class="muted" style="font-size:.78rem">Red = prime suspect track · amber ring = loitering · dashed red = AIS blackout. Drag the slider to walk each vessel to its position at that time.</p>
          <div id="ws-vessel" class="finding" hidden></div>
        </div>
      </div>
      <div class="wcol">
        <div class="panel"><h2>Spill mask</h2>${kv([
          ["Classification", h(sar.scene_classification || "-")],
          ["Area", `${num(prim.area_km2, 2)} km&sup2;`],
          ["Centroid", `<span class="mono">${num(inv.centroid_lat, 3)}, ${num(inv.centroid_lon, 3)}</span>`],
          ["Major / minor axis", `${num(prim.major_axis_km, 2)} / ${num(prim.minor_axis_km, 2)} km`],
          ["Orientation", `${num(prim.orientation_deg, 0)}&deg;`],
          ["Look-alike filter", h(sar.primary_detection?.look_alike_filter?.verdict
            || (sar.scene_classification === "Oil-like anomaly" ? "passed - not a look-alike" : "-"))],
        ])}</div>
        <div class="panel"><h2>Reconstructed origin</h2>${kv([
          ["Best estimate", `<span class="mono">${(hind.best_estimate || []).map((v) => num(v, 3)).join(", ")}</span>`],
          ["Uncertainty radius", `${num(hind.uncertainty_radius_km, 1)} km`],
          ["Release window", (hind.release_window_h || []).map((v) => `T${v >= 0 ? "+" : ""}${num(v, 1)}h`).join(" .. ")],
          ["Age estimate", `${num(hind.age_point_estimate_h, 1)} h (${h(hind.age_source || "-")})`],
          ["Feedback loop", m.feedback_loop?.converged ? `converged in ${m.feedback_loop.n_iterations} iter` : `${m.feedback_loop?.n_iterations ?? "-"} iter`],
        ])}</div>
        <div class="panel"><h2>Forward forecast</h2>${
          (fore.horizons || []).length
            ? `<table class="data"><thead><tr><th>Horizon</th><th>Centroid</th><th>Radius</th><th>Beached</th></tr></thead><tbody>${
              fore.horizons.map((hz) => `<tr><td>+${h(hz.horizon_h ?? hz.t_h)}h</td>
                <td class="mono">${num(hz.centroid?.[0], 3)}, ${num(hz.centroid?.[1], 3)}</td>
                <td>${num(hz.mean_radius_km, 1)} km</td>
                <td>${hz.fraction_beached ? pct(hz.fraction_beached) : "-"}</td></tr>`).join("")}</tbody></table>`
            : `<p class="muted">Not computed (look-alike scene).</p>`
        }${(() => {
          const ci = fore.coastal_impact || {};
          if (!ci.will_beach) return `<p class="muted">${h(ci.note || "No shoreline contact modelled.")}</p>`;
          const fc = ci.first_contact_point || [];
          return `<h3>Shoreline contact</h3>${kv([
            ["First landfall ETA", `${num(ci.first_contact_eta_h ?? ci.eta_hours, 1)} h`],
            ["Contact point", `<span class="mono">${num(fc[0], 3)}, ${num(fc[1], 3)}</span>`],
            ["Oil ashore (48 h)", pct(ci.fraction_beached)],
          ])}`;
        })()}<p class="muted">${h(m.environmental_field?.label || "")}</p></div>
      </div>
      <div class="wcol">
        <div class="panel"><h2>Ranked candidate vessels</h2>
          <p class="muted">${h(attr.summary?.verdict || "No attribution for this scene.")}</p>
          <div id="ws-cands">${cands.map((c) => candidateCard(c, truth)).join("") || `<p class="empty">No vessels in the search window.</p>`}</div>
        </div>
      </div>
    </div>
    ${raw(m)}`);

  $("#ws-evidence").addEventListener("click", () => ctx.go(`#/evidence/${id}`));

  $("#ws-ais").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("investigation_id", id);
    fd.append("csv_file", file);
    fd.append("reattribute", "true");
    $("#ws-ais-msg").textContent = `Ingesting ${file.name} and re-running attribution...`;
    try {
      const r = await api.ingestAis(fd);
      ctx.toast(r.reattributed
        ? `${r.vessels} vessel(s) ingested; prime suspect: ${r.prime_suspect || "none"}`
        : `${r.vessels} vessel track(s) attached`);
      workstation(ctx, params);        // reload the view with the new attribution
    } catch (err) {
      $("#ws-ais-msg").textContent = (err.detail || err.message);
    }
  });

  // ---- map: anomaly, origin, release polygon, drift frames ----
  const map = makeMap($("#ws-map"), inv.centroid_lat ? { center: [inv.centroid_lat, inv.centroid_lon], zoom: 8 } : {});
  const layers = [];
  if (inv.centroid_lat != null)
    layers.push(anomalyMarker(map, inv.centroid_lat, inv.centroid_lon, { confidence: sar.confidence || 0, label: sar.scene_classification, ref: inv.reference }));
  if (hind.best_estimate)
    layers.push(L.circleMarker(hind.best_estimate, { radius: 6, color: "#35d07f", fillOpacity: 0.6 }).addTo(map).bindPopup("Reconstructed release origin"));
  if (hind.release_polygon?.length)
    layers.push(polygon(map, hind.release_polygon, { color: "#35d07f", lonlat: guessLonLat(hind.release_polygon) }));
  if (hind.confidence_ellipse?.length)
    polygon(map, hind.confidence_ellipse, { color: "#66c2ff", fillOpacity: 0.05, lonlat: guessLonLat(hind.confidence_ellipse) });
  const beachLayer = shorelineContact(map, fore.coastal_impact);
  if (beachLayer) layers.push(beachLayer);

  // vessels: reconstructed track per candidate (prime in red), clickable
  const vts = m.vessel_tracks || {};
  const vesselLayers = [];
  for (const c of cands) {
    const view = vts[String(c.identity.mmsi)];
    if (!view || !view.pings?.length) continue;
    const lyr = vesselTrackLayer(map, view, {
      prime: c.rank === 1,
      onClick: (v) => showVesselDetail(v),
    });
    vesselLayers.push(lyr);
    layers.push(lyr.group);
  }
  function showVesselDetail(view) {
    const box = $("#ws-vessel");
    if (!box) return;
    box.hidden = false;
    box.innerHTML = vesselPopupHtml(view);
  }
  if (cands[0]) {
    const pv = vts[String(cands[0].identity.mmsi)];
    if (pv) showVesselDetail(pv);
  }

  const frames = m.drift_frames || { hindcast: [], forecast: [] };
  let driftLayer = null;
  function showFrame(t) {
    if (driftLayer) { map.removeLayer(driftLayer); driftLayer = null; }
    for (const lyr of vesselLayers) lyr.positionAt(t);
    const pool = t <= 0 ? frames.hindcast : frames.forecast;
    if (pool && pool.length) {
      let best = pool[0];
      for (const f of pool) if (Math.abs(Math.abs(f.t_h) - Math.abs(t)) < Math.abs(Math.abs(best.t_h) - Math.abs(t))) best = f;
      const pts = (best.points || []).map((p) => [p[0], p[1]]);
      driftLayer = L.layerGroup(pts.map((p) => L.circleMarker(p, {
        radius: 2, color: t <= 0 ? "#f0b429" : "#2ea6ff", weight: 0, fillOpacity: 0.5,
      }))).addTo(map);
    }
    $("#ws-tl-now").textContent = t === 0 ? "observation (T0)" : `T${t > 0 ? "+" : ""}${t} h (${t < 0 ? "hindcast" : "forecast"})`;
  }
  $("#ws-tl").addEventListener("input", (e) => showFrame(Number(e.target.value)));
  if (layers.length) fit(map, layers);
  showFrame(0);
}
function guessLonLat(ring) {
  // GeoJSON rings are [lon,lat]; if the first coord's |x|>90 it's a longitude
  const a = ring?.[0]?.[0];
  return a != null && Math.abs(a) > 90;
}

/* ==================================================== VESSEL INTEL ===== */
async function vesselIntel(ctx) {
  const invs = await loadInvestigations();
  const withAttr = invs.filter((i) => i.summary_metrics?.attribution?.candidates?.length);
  ctx.root.innerHTML = page("Vessel Intelligence",
    "POSEatSea AIS autoencoder anomalies and LSTM trajectory deviations for every vessel under attribution.",
    `<div class="panel"><div class="row"><label>Investigation
      <select id="vi-inv">${withAttr.map((i) => `<option value="${i.id}">${h(i.reference)} - ${h(i.title)}</option>`).join("")}</select>
    </label></div></div><div id="vi-body"></div>`);
  if (!withAttr.length) { $("#vi-body").innerHTML = `<p class="empty">No attributed vessels yet.</p>`; return; }

  async function render(id) {
    const inv = await api.investigation(id);
    const sm = inv.summary_metrics || {};
    const cands = sm.attribution?.candidates || [];
    const vts = sm.vessel_tracks || {};
    $("#vi-body").innerHTML = cands.map((c) => {
      const cc = c.components || {};
      const ae = cc.ais_anomaly?.detail || {};
      const rd = cc.route_deviation?.detail || {};
      const bl = cc.blackout?.detail || {};
      const st = cc.spatiotemporal?.detail || {};
      const tv = vts[String(c.identity.mmsi)] || {};
      const tvm = tv.metrics || {};
      return `<div class="panel"><h2>#${c.rank} ${h(c.identity.name)} <span class="muted mono">MMSI ${h(c.identity.mmsi)}</span></h2>
        <p class="muted">Track: ${tvm.n_points ?? "-"} pings over ${num(tvm.duration_h, 1)} h ·
          ${num(tvm.sog_min_kn, 1)}–${num(tvm.sog_max_kn, 1)} kn ·
          heading ${tvm.mean_heading_deg != null ? num(tvm.mean_heading_deg, 0) + "°" : "-"} ·
          ${(tv.loiter || []).length} loiter span(s) · ${(tv.blackouts || []).length} AIS gap(s)</p>
        <div class="grid cols-3">
          <div><h3>AIS autoencoder</h3>${kv([
            ["Peak recon. error", num(ae.peak_reconstruction_error, 4)],
            ["Threshold", num(ae.threshold, 4)],
            ["Pings near slick", ae.pings_near_slick ?? "-"],
            ["Flagged pings", ae.flagged_pings ?? "-"],
            ["Score", num(cc.ais_anomaly?.value, 3)],
          ])}<p class="finding">${h(ae.finding || "")}</p></div>
          <div><h3>LSTM trajectory</h3>${kv([
            ["Usable", rd.usable ? "yes" : "no"],
            ["Score", num(cc.route_deviation?.value, 3)],
          ])}<p class="finding">${h(rd.finding || "")}</p></div>
          <div><h3>Behaviour &amp; space-time</h3>${kv([
            ["AIS gaps", bl.n_gaps ?? c.track?.n_gaps ?? 0],
            ["Longest blackout", `${num(bl.longest_blackout_over_window_min, 0)} min`],
            ["Track points", c.track?.n_points ?? "-"],
            ["First / last seen", `T${num(c.track?.first_seen_h, 0)} .. T${num(c.track?.last_seen_h, 0)} h`],
            ["Best match", st.best_match_time_h != null ? `T${st.best_match_time_h >= 0 ? "+" : ""}${num(st.best_match_time_h, 1)} h @ ${num(st.min_distance_km, 1)} km` : "-"],
          ])}<p class="finding">${h(bl.finding || "")}</p></div>
        </div>
        <div class="finding">${h(st.finding || "")}</div>
        <div class="finding">${h(c.assessment || "")} - fused score ${num(c.score, 1)}/100</div>
      </div>`;
    }).join("");
  }
  $("#vi-inv").addEventListener("change", (e) => render(e.target.value));
  render($("#vi-inv").value);
}

/* ==================================================== SPILL ANALYSIS === */
async function spillAnalysis(ctx, params) {
  const invs = await loadInvestigations();
  const id = params[0] || invs[0]?.id;
  ctx.root.innerHTML = page("Spill Analysis", "SAR detection detail: characterisation, backscatter and the look-alike ruling.",
    `<div class="panel"><div class="row"><label>Investigation
      <select id="sa-inv">${invs.map((i) => `<option value="${i.id}" ${i.id == id ? "selected" : ""}>${h(i.reference)} - ${h(i.title)}</option>`).join("")}</select>
    </label></div></div><div id="sa-body"></div>`);
  if (!invs.length) { $("#sa-body").innerHTML = `<p class="empty">No investigations.</p>`; return; }

  async function render(iid) {
   try {
    const inv = await api.investigation(iid);
    const sm = inv.summary_metrics || {};
    const sar = sm.sar || {};
    const dets = sar.detections || [];
    const wx = sm.weathering || {};
    const wxLast = (wx.series || []).slice(-1)[0] || {};
    const iou = sm.iou;
    $("#sa-body").innerHTML = `
      <div class="kpis">
        ${kpi(h(sar.scene_classification || "-"), "Scene verdict")}
        ${kpi(pct(sar.confidence), "Confidence")}
        ${kpi(sar.counts?.dark_spots ?? sar.counts?.candidates ?? "-", "Dark spots")}
        ${kpi(sar.counts?.oil_like ?? dets.length, "Oil-like")}
        ${iou ? kpi(num(iou.value, 3), "IoU vs ground truth") : ""}
        ${wxLast.evaporated_fraction != null ? kpi(pct(wxLast.evaporated_fraction), `Evaporated @ ${wxLast.t_h}h`) : ""}
      </div>
      ${wx.series ? `<div class="panel"><h2>Weathering &mdash; evaporation &amp; spreading</h2>
        <p class="muted">${h(wx.model || "")} &middot; oil class <b>${h(wx.oil_class)}</b>,
          water ${num(wx.water_temp_c, 0)}&deg;C, assumed initial film ${num(wx.assumed_initial_thickness_m * 1000, 2)} mm
          &rarr; initial volume ~${num(wx.initial_volume_m3, 0)} m&sup3;.</p>
        <table class="data"><thead><tr><th>t (h)</th><th>Evaporated</th><th>Volume left (m&sup3;)</th><th>Area (km&sup2;)</th><th>Mean thickness (mm)</th></tr></thead><tbody>${
          wx.series.map((s) => `<tr><td>${s.t_h}</td><td>${pct(s.evaporated_fraction)}</td>
            <td>${num(s.volume_remaining_m3, 0)}</td><td>${num(s.area_km2, 2)}</td>
            <td>${num(s.mean_thickness_mm, 3)}</td></tr>`).join("")
        }</tbody></table>
        <p class="muted">Order-of-magnitude: SAR gives area, not volume &mdash; the initial volume is an explicit assumption.</p>
      </div>` : ""}
      <div class="grid cols-2">
        <div class="panel"><h2>Pipeline stages</h2><ul class="clean">${
          ((Array.isArray(sar.pipeline) && sar.pipeline)
            || (Array.isArray(sar.pipeline?.timings) && sar.pipeline.timings)
            || []).map((p) => {
              if (typeof p === "string") return `<li>${h(p)}</li>`;
              const name = p.step || p.name;
              return `<li>${name ? h(name) + (p.seconds != null ? ` <span class="muted mono">${num(p.seconds, 3)} s</span>` : "") : h(JSON.stringify(p))}</li>`;
            }).join("") || `<li class="muted">n/a</li>`
        }</ul>
        <h3>Acquisition</h3>${kv([
          ["Acquired", when(sar.acquisition)],
          ["Bounding box", `<span class="mono">${bboxStr(sar.bounding_box || sar.bbox)}</span>`],
          ["Pixel size", `${num(sar.pixel_size_m, 1)} m`],
        ])}</div>
        <div class="panel"><h2>Detections</h2>${
          dets.map((d, i) => {
            const c = d.characterization || {};
            const la = d.look_alike_filter || {};
            return `<div class="cand"><div class="h"><b>Detection ${i + 1}</b><span class="score">${pct(d.confidence)}</span></div>
              <div class="muted">${h(d.classification || "")}</div>
              ${kv([
                ["Area", `${num(c.area_km2, 2)} km&sup2;`],
                ["Perimeter", `${num(c.perimeter_km, 2)} km`],
                ["Major / minor", `${num(c.major_axis_km, 2)} / ${num(c.minor_axis_km, 2)} km`],
                ["Orientation", `${num(c.orientation_deg, 0)}&deg;`],
                ["Look-alike ruling", `${h(la.verdict || "-")}${la.reasons?.length ? " - " + h(la.reasons.join("; ")) : ""}`],
              ])}</div>`;
          }).join("") || `<p class="muted">No discrete detections (scene rejected).</p>`
        }</div>
      </div>${raw(sar)}`;
   } catch (e) {
     $("#sa-body").innerHTML = `<div class="panel"><p class="empty">Could not render this scene's analysis (${h(e.message || e)}).</p></div>`;
   }
  }
  $("#sa-inv").addEventListener("change", (e) => render(e.target.value));
  render(id);
}

/* ==================================================== DRIFT & FORECAST = */
async function driftForecast(ctx, params) {
  const invs = (await loadInvestigations()).filter((i) => i.summary_metrics?.hindcast);
  const id = params[0] || invs[0]?.id;
  ctx.root.innerHTML = page("Drift & Forecast", "Lagrangian RK4 hindcast to the release origin and 48-hour forward projection.",
    `<div class="panel"><div class="row"><label>Investigation
      <select id="df-inv">${invs.map((i) => `<option value="${i.id}" ${i.id == id ? "selected" : ""}>${h(i.reference)} - ${h(i.title)}</option>`).join("")}</select>
    </label></div></div><div id="df-body"></div>`);
  if (!invs.length) { $("#df-body").innerHTML = `<p class="empty">No drift runs yet (look-alike scenes have none).</p>`; return; }

  async function render(iid) {
    const inv = await api.investigation(iid);
    const m = inv.summary_metrics || {};
    const hind = m.hindcast || {}, fore = m.forecast || {}, mo = m.met_ocean || {};
    $("#df-body").innerHTML = `
      <div class="panel"><h2>Trajectories</h2><div id="df-map" class="map"></div></div>
      <div class="grid cols-2">
        <div class="panel"><h2>Hindcast (origin reconstruction)</h2>${kv([
          ["Best estimate", `<span class="mono">${(hind.best_estimate || []).map((v) => num(v, 4)).join(", ")}</span>`],
          ["Uncertainty radius", `${num(hind.uncertainty_radius_km, 1)} km`],
          ["Release window", (hind.release_window_h || []).map((v) => `T${v >= 0 ? "+" : ""}${num(v, 1)}h`).join(" .. ")],
          ["Particles", hind.n_particles],
          ["Seed", h(hind.seed_kind || "-")],
        ])}</div>
        <div class="panel"><h2>Forecast horizons</h2>${
          (fore.horizons || []).length ? `<table class="data"><thead><tr><th>+h</th><th>Centroid</th><th>Mean radius</th><th>Beached</th></tr></thead><tbody>${
            fore.horizons.map((hz) => `<tr><td>${h(hz.horizon_h ?? hz.t_h)}</td>
              <td class="mono">${num(hz.centroid?.[0], 3)}, ${num(hz.centroid?.[1], 3)}</td>
              <td>${num(hz.mean_radius_km, 1)} km</td>
              <td>${hz.fraction_beached ? pct(hz.fraction_beached) : "-"}</td></tr>`).join("")}</tbody></table>`
            : `<p class="muted">n/a</p>`
        }<h3>Coastal impact</h3>${(() => {
          const ci = fore.coastal_impact || {};
          const rows = [["Will beach", ci.will_beach ? "yes" : "no"]];
          if (ci.will_beach) {
            const fc = ci.first_contact_point || [];
            rows.push(["First landfall ETA", `${num(ci.first_contact_eta_h ?? ci.eta_hours, 1)} h`]);
            rows.push(["Contact point", `<span class="mono">${num(fc[0], 3)}, ${num(fc[1], 3)}</span>`]);
            rows.push(["Oil ashore (48 h)", pct(ci.fraction_beached)]);
          } else {
            rows.push(["Note", h(ci.note || "-")]);
          }
          return kv(rows);
        })()}</div>
      </div>
      <div class="panel"><h2>Met-ocean field</h2><p class="muted">${h(m.environmental_field?.label || "")}</p>${kv([
        ["Mean current", `${num(mo.current_speed_ms ?? mo.current_ms, 2)} m/s @ ${num(mo.current_dir_deg, 0)}&deg;`],
        ["Mean wind", `${num(mo.wind_speed_ms ?? mo.wind_ms, 1)} m/s @ ${num(mo.wind_dir_deg, 0)}&deg;`],
        ["Wave height", `${num(mo.wave_height_m, 2)} m`],
      ])}</div>${raw({ hindcast: hind, forecast: fore })}`;

    const c = hind.best_estimate || [inv.centroid_lat, inv.centroid_lon];
    const map = makeMap($("#df-map"), { center: c, zoom: 8 });
    const fr = m.drift_frames || {};
    const L2 = window.L;
    const layers = [];
    if (inv.centroid_lat != null) layers.push(anomalyMarker(map, inv.centroid_lat, inv.centroid_lon, { confidence: m.sar?.confidence || 0, label: "Observed slick" }));
    if (hind.best_estimate) layers.push(L2.circleMarker(hind.best_estimate, { radius: 6, color: "#35d07f", fillOpacity: 0.7 }).addTo(map).bindPopup("Release origin"));
    for (const f of (fr.hindcast || [])) {
      const centroidPt = meanPoint(f.points);
      if (centroidPt) L2.circleMarker(centroidPt, { radius: 3, color: "#f0b429", weight: 0, fillOpacity: 0.7 }).addTo(map);
    }
    for (const f of (fr.forecast || [])) {
      const centroidPt = meanPoint(f.points);
      if (centroidPt) L2.circleMarker(centroidPt, { radius: 3, color: "#2ea6ff", weight: 0, fillOpacity: 0.7 }).addTo(map);
    }
    const hc = (fr.hindcast || []).map((f) => meanPoint(f.points)).filter(Boolean);
    const fc = (fr.forecast || []).map((f) => meanPoint(f.points)).filter(Boolean);
    if (hc.length > 1) trackLine(map, hc, { color: "#f0b429", dash: "5 4" });
    if (fc.length > 1) trackLine(map, fc, { color: "#2ea6ff" });
    const bl = shorelineContact(map, fore.coastal_impact);
    if (bl) layers.push(bl);
    if (layers.length) fit(map, layers);
  }
  $("#df-inv").addEventListener("change", (e) => render(e.target.value));
  render(id);
}
function meanPoint(pts) {
  if (!pts || !pts.length) return null;
  let a = 0, b = 0;
  for (const p of pts) { a += p[0]; b += p[1]; }
  return [a / pts.length, b / pts.length];
}

/* ========================================================== EVIDENCE == */
async function evidence(ctx, params) {
  const invs = await loadInvestigations();
  const id = params[0] || invs[0]?.id;
  ctx.root.innerHTML = page("Evidence Dossier", "Structured incident report - JSON is the system of record; the printable view produces a PDF.",
    `<div class="panel"><div class="row"><label>Investigation
      <select id="ev-inv">${invs.map((i) => `<option value="${i.id}" ${i.id == id ? "selected" : ""}>${h(i.reference)} - ${h(i.title)}</option>`).join("")}</select>
    </label>
    <button class="btn" id="ev-json">Download JSON</button>
    <button class="btn" id="ev-md">Download Markdown</button>
    <button class="btn btn-primary" id="ev-print">Open printable view</button>
    <span id="ev-msg" class="muted"></span></div></div>
    <div class="panel"><h2>Preview</h2><iframe id="ev-frame" class="dossier" title="dossier"></iframe></div>`);
  if (!invs.length) { $("#view .panel:last-child").innerHTML = `<p class="empty">No investigations.</p>`; return; }

  let current = id;
  async function loadFrame(iid) {
    current = iid;
    $("#ev-msg").textContent = "Rendering...";
    try {
      const html = await fetchText(`/investigations/${iid}/dossier.html`);
      const blob = new Blob([html], { type: "text/html" });
      $("#ev-frame").src = URL.createObjectURL(blob);
      $("#ev-msg").textContent = "";
    } catch (e) { $("#ev-msg").textContent = e.message; }
  }
  function download(name, text, type) {
    const blob = new Blob([text], { type });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }
  $("#ev-inv").addEventListener("change", (e) => loadFrame(e.target.value));
  $("#ev-json").addEventListener("click", async () => {
    try { download(`dossier-${current}.json`, JSON.stringify(await api.dossier(current), null, 2), "application/json"); }
    catch (e) { ctx.toast(e.message, true); }
  });
  $("#ev-md").addEventListener("click", async () => {
    try { download(`dossier-${current}.md`, await fetchText(`/investigations/${current}/dossier.md`), "text/markdown"); }
    catch (e) { ctx.toast(e.message, true); }
  });
  $("#ev-print").addEventListener("click", async () => {
    try {
      const html = await fetchText(`/investigations/${current}/dossier.html`);
      const w = window.open("", "_blank");
      w.document.write(html); w.document.close();
    } catch (e) { ctx.toast(e.message, true); }
  });
  loadFrame(id);
}

/* ============================================================ ALERTS == */
async function alerts(ctx) {
  ctx.root.innerHTML = page("Alerts", "Fingerprinted, deduplicated SMS dispatch. Mock provider is used locally; Twilio when configured.",
    `<div class="panel"><div class="row">
      <button class="btn btn-primary" id="al-test">Send test SMS to my number</button>
      <span id="al-msg" class="muted"></span>
    </div></div><div class="panel"><h2>Alert feed</h2><div id="al-body"></div></div>`);

  $("#al-test").addEventListener("click", async () => {
    $("#al-test").disabled = true;
    try {
      const a = await api.testSms("Operations Console test message.");
      $("#al-msg").textContent = `Test alert #${a.id}: ${a.status} via ${a.provider || "mock"}.`;
      load();
    } catch (e) { $("#al-msg").textContent = e.message; }
    $("#al-test").disabled = false;
  });

  async function load() {
    let rows;
    try { rows = await api.alerts(); }
    catch (e) { $("#al-body").innerHTML = `<p class="empty">${e.status === 403 ? "Alerts require REGIONAL or NATIONAL role." : h(e.message)}</p>`; return; }
    $("#al-body").innerHTML = rows.length ? `<table class="data"><thead><tr>
      <th>#</th><th>Status</th><th>Recipient</th><th>Message</th><th>Conf.</th><th>Provider</th><th>Att.</th><th>Sent</th><th></th></tr></thead><tbody>${
      rows.map((a) => `<tr>
        <td class="mono">${a.id}</td><td>${alertBadge(a.status)}</td><td class="mono">${h(a.recipient)}</td>
        <td>${h((a.message || "").slice(0, 90))}</td><td>${a.triggered_by_confidence != null ? pct(a.triggered_by_confidence) : "-"}</td>
        <td>${h(a.provider || "-")}</td><td>${a.attempts}</td><td>${when(a.sent_at)}</td>
        <td>${a.status === "FAILED" ? `<button class="btn sm" data-retry="${a.id}">Retry</button>` : ""}</td></tr>` +
        (a.last_error ? `<tr><td></td><td colspan="8" class="muted">${h(a.last_error)}</td></tr>` : "")).join("")
    }</tbody></table>` : `<p class="empty">No alerts raised yet.</p>`;
    $$("#al-body [data-retry]").forEach((b) => b.addEventListener("click", async () => {
      b.disabled = true;
      try { await api.retryAlert(b.dataset.retry); ctx.toast("Retry submitted."); load(); }
      catch (e) { ctx.toast(e.message, true); b.disabled = false; }
    }));
  }
  load();
}

/* ========================================================= ANALYTICS == */
async function analytics(ctx) {
  const invs = await loadInvestigations();
  let al = [];
  try { al = await api.alerts(); } catch {}
  const byStatus = tally(invs.map((i) => i.status));
  const byClass = tally(invs.map((i) => i.summary_metrics?.sar?.scene_classification || "n/a"));
  const gt = invs.map((i) => i.summary_metrics?.attribution?.summary?.ground_truth).filter(Boolean);
  const correct = gt.filter((g) => g.correctly_ranked_first).length;
  const delivered = al.filter((a) => ["SENT", "MOCKED"].includes(a.status)).length;

  ctx.root.innerHTML = page("Analytics", "Aggregate performance across every investigation in your scope.",
    `<div class="kpis">
      ${kpi(invs.length, "Investigations")}
      ${kpi(gt.length ? `${correct}/${gt.length}` : "-", "Top-1 attribution")}
      ${kpi(al.length ? `${delivered}/${al.length}` : "-", "Alerts delivered")}
      ${kpi(pctMean(invs.map((i) => i.summary_metrics?.sar?.confidence)), "Mean confidence")}
    </div>
    <div class="grid cols-2">
      <div class="panel"><h2>By status</h2>${barList(byStatus)}</div>
      <div class="panel"><h2>By scene verdict</h2>${barList(byClass)}</div>
    </div>
    <div class="panel"><h2>Attribution ground-truth checks</h2>${
      gt.length ? `<table class="data"><thead><tr><th>Culprit MMSI</th><th>Rank assigned</th><th>Top-1?</th></tr></thead><tbody>${
        gt.map((g) => `<tr><td class="mono">${g.culprit_mmsi}</td><td>${g.rank_assigned ?? "-"}</td>
          <td>${g.correctly_ranked_first ? '<span class="badge ok">yes</span>' : '<span class="badge crit">no</span>'}</td></tr>`).join("")
      }</tbody></table>` : `<p class="muted">Run scenarios with a known culprit to populate this.</p>`
    }</div>`);
}
function tally(arr) { const o = {}; for (const x of arr) o[x] = (o[x] || 0) + 1; return o; }
function barList(obj) {
  const max = Math.max(1, ...Object.values(obj));
  return Object.entries(obj).map(([k, v]) =>
    `<div class="comprow"><span>${h(k)}</span><span class="bar"><i style="width:${(v / max * 100).toFixed(0)}%"></i></span><span class="mono">${v}</span></div>`
  ).join("") || `<p class="muted">no data</p>`;
}
function pctMean(xs) { const v = xs.filter((x) => x != null); return v.length ? pct(v.reduce((a, b) => a + b, 0) / v.length) : "-"; }

/* ============================================================ SYSTEM == */
async function system(ctx) {
  const [health, info, models] = await Promise.all([
    api.health().catch((e) => ({ error: e.message })),
    api.info().catch((e) => ({ error: e.message })),
    api.models().catch((e) => ({ error: e.message, models: [] })),
  ]);
  const rows = (models.models || []);
  ctx.root.innerHTML = page("System", "Runtime status of the unified API.",
    `<div class="grid cols-2">
      <div class="panel"><h2>Health</h2>${kv(Object.entries(health).map(([k, v]) => [k, `<span class="mono">${h(typeof v === "object" ? JSON.stringify(v) : v)}</span>`]))}</div>
      <div class="panel"><h2>Capabilities</h2>${kv(Object.entries(info).map(([k, v]) => [k, `<span class="mono">${h(typeof v === "object" ? JSON.stringify(v) : v)}</span>`]))}</div>
    </div>
    <div class="panel"><h2>ML model registry
      <span class="badge ${models.lazy_load ? "ok" : "warn"}">lazy-load ${models.lazy_load ? "on" : "off"}</span>
      <span class="badge mut">${models.resident ?? rows.filter((m) => m.loaded).length} / ${rows.length} resident</span></h2>
      <p class="muted">Models load only when an investigation first needs them, then stay cached for the life of the process.</p>
      ${rows.length ? `<table class="data"><thead><tr><th>Model</th><th>State</th><th>First load</th><th>Parameters</th></tr></thead><tbody>${
        rows.map((m) => `<tr><td>${h(m.name)}</td>
          <td>${m.loaded ? '<span class="badge ok">resident</span>' : '<span class="badge mut">not loaded</span>'}${m.error ? ` <span class="badge crit">error</span>` : ""}</td>
          <td>${m.load_seconds != null ? num(m.load_seconds, 2) + " s" : "-"}</td>
          <td class="mono">${m.parameters != null ? m.parameters.toLocaleString() : "-"}</td></tr>`).join("")
      }</tbody></table>` : `<p class="empty">${h(models.error || "registry unavailable")}</p>`}
    </div>`);
}

/* =========================================================== PROFILE == */
async function profile(ctx) {
  const u = ctx.user;
  let zones = [];
  try { zones = await api.jurisdictions({ mine_only: true, with_geometry: false }); } catch {}
  ctx.root.innerHTML = page("Profile", null,
    `<div class="grid cols-2">
      <div class="panel"><h2>${h(u.name)}</h2>${kv([
        ["Email", h(u.email)],
        ["Role", `<span class="badge info">${h(u.role)}</span>`],
        ["Mobile", h(u.phone_number || "not set")],
        ["Account created", when(u.created_at)],
        ["Active", u.active ? "yes" : "no"],
      ])}</div>
      <div class="panel"><h2>Assigned jurisdictions</h2>${
        zones.length ? `<ul class="clean">${zones.map((z) => `<li><span class="mono">${h(z.code)}</span> - ${h(z.name)} <span class="muted">(${h(z.type)})</span></li>`).join("")}</ul>`
          : u.role === "NATIONAL" ? `<p class="muted">NATIONAL role - every maritime zone.</p>` : `<p class="muted">None assigned.</p>`
      }</div>
    </div>
    <div class="panel"><h2>Test SMS delivery</h2><div class="row">
      <button class="btn btn-primary" id="pf-sms">Send a test SMS to ${h(u.phone_number || "(no number on file)")}</button>
      <span id="pf-msg" class="muted"></span></div></div>`);
  $("#pf-sms").addEventListener("click", async () => {
    $("#pf-sms").disabled = true;
    try { const a = await api.testSms(); $("#pf-msg").textContent = `#${a.id}: ${a.status} via ${a.provider || "mock"}.`; }
    catch (e) { $("#pf-msg").textContent = e.message; }
    $("#pf-sms").disabled = false;
  });
}

/* ============================================================== AUTH == */
export function wireAuth(onAuthed) {
  const form = $("#auth-form");
  const tabs = $$("#auth-tabs button");
  const msg = $("#auth-msg");
  let mode = "login";

  function setMode(m) {
    mode = m;
    tabs.forEach((t) => t.classList.toggle("active", t.dataset.mode === m));
    $("#reg-extra").hidden = m !== "register";
    $("#name-label").hidden = m !== "register";
    $("#auth-submit").textContent = m === "login" ? "Sign in" : "Create account";
    form.password.autocomplete = m === "login" ? "current-password" : "new-password";
    msg.textContent = "";
  }
  tabs.forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
  setMode("login");

  // open self-registration is disabled by default; only expose the tab when the
  // server says it is on (dev / test).
  api.info().then((i) => {
    if (i && i.open_registration) $$('#auth-tabs [data-mode="register"]').forEach((b) => (b.hidden = false));
  }).catch(() => {});

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.textContent = "";
    const fd = new FormData(form);
    const email = fd.get("email").trim();
    const password = fd.get("password");
    try {
      if (mode === "register") {
        const codes = (fd.get("jurisdiction_codes") || "").split(",").map((s) => s.trim()).filter(Boolean);
        await api.register({
          name: fd.get("name").trim() || email.split("@")[0],
          email, password,
          phone_number: (fd.get("phone_number") || "").trim() || null,
          role: fd.get("role"),
          jurisdiction_codes: codes,
        });
      }
      const tok = await api.login(email, password);
      onAuthed(tok.user);
    } catch (err) {
      msg.textContent = Array.isArray(err.detail)
        ? err.detail.map((d) => d.msg).join("; ")
        : (err.detail || err.message);
    }
  });
}

/* ==================================================== USER MANAGEMENT == */
async function userManagement(ctx) {
  ctx.root.innerHTML = page("User Management",
    "Create and maintain operator accounts. NATIONAL manages every account; REGIONAL manages PILOT accounts inside its region.",
    `<div id="um-body"><p class="empty">Loading&hellip;</p></div>`);

  let scope, users, zones;
  try {
    [scope, users, zones] = await Promise.all([
      api.userScope(), api.users(),
      api.jurisdictions({ with_geometry: false }).catch(() => []),
    ]);
  } catch (e) {
    $("#um-body").innerHTML = `<p class="empty">${e.status === 403 ? "User management requires REGIONAL or NATIONAL role." : h(e.message)}</p>`;
    return;
  }
  const zoneById = Object.fromEntries(zones.map((z) => [z.id, z.code]));
  const closure = scope.jurisdiction_closure_ids; // null = all
  const pickZones = (closure == null ? zones : zones.filter((z) => closure.includes(z.id)))
    .filter((z) => z.type === "COASTAL_STATE" || z.type === "MARITIME_REGION");

  $("#um-body").innerHTML = `
    <div class="panel"><h2>Create account</h2>
      <form id="um-form" class="row" style="align-items:flex-end;gap:.7rem;flex-wrap:wrap">
        <label>Name<input name="name" required></label>
        <label>Email<input name="email" type="email" required></label>
        <label>Temp. password<input name="password" type="password" minlength="8" required></label>
        <label>Role<select name="role">${scope.can_create_roles.map((r) => `<option>${r}</option>`).join("")}</select></label>
        <label>Mobile<input name="phone_number" placeholder="+15005550006"></label>
        <label>Zones<select name="zones" multiple size="4">${
          pickZones.map((z) => `<option value="${h(z.code)}">${h(z.code)} - ${h(z.name)}</option>`).join("")
        }</select></label>
        <button class="btn btn-primary" type="submit">Create</button>
        <span id="um-msg" class="muted"></span>
      </form>
      <p class="muted">PILOT/REGIONAL accounts need at least one zone. The user changes the temporary password after first sign-in via Profile.</p>
    </div>
    <div class="panel"><h2>Accounts (${users.length})</h2>
      <table class="data"><thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Zones</th><th>Status</th><th></th></tr></thead>
      <tbody id="um-rows">${users.map((u) => `<tr data-id="${u.id}">
        <td>${h(u.name)}</td><td class="mono">${h(u.email)}</td>
        <td><span class="badge ${u.role === "NATIONAL" ? "crit" : u.role === "REGIONAL" ? "warn" : "info"}">${h(u.role)}</span></td>
        <td class="mono">${(u.jurisdiction_ids || []).map((i) => zoneById[i] || i).join(", ") || "-"}</td>
        <td>${u.active ? '<span class="badge ok">active</span>' : '<span class="badge mut">disabled</span>'}</td>
        <td><button class="btn sm" data-toggle="${u.id}" data-on="${u.active ? 0 : 1}">${u.active ? "Disable" : "Enable"}</button></td>
      </tr>`).join("") || `<tr><td colspan="6" class="empty">No manageable accounts.</td></tr>`}</tbody></table>
    </div>`;

  $("#um-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const codes = [...e.target.zones.selectedOptions].map((o) => o.value);
    const btn = e.target.querySelector("button[type=submit]");
    btn.disabled = true; $("#um-msg").textContent = "Creating...";
    try {
      await api.createUser({
        name: fd.get("name").trim(), email: fd.get("email").trim(),
        password: fd.get("password"), role: fd.get("role"),
        phone_number: (fd.get("phone_number") || "").trim() || null,
        jurisdiction_codes: codes,
      });
      ctx.toast("Account created.");
      userManagement(ctx);
    } catch (err) {
      $("#um-msg").textContent = Array.isArray(err.detail)
        ? err.detail.map((d) => d.msg).join("; ") : (err.detail || err.message);
      btn.disabled = false;
    }
  });
  $$("#um-rows [data-toggle]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    try { await api.setUserActive(b.dataset.toggle, b.dataset.on === "1"); userManagement(ctx); }
    catch (e) { ctx.toast(e.detail || e.message, true); b.disabled = false; }
  }));
}

/* ============================================================ EXPORT == */
export const views = {
  "mission-control": missionControl,
  "monitoring": monitoring,
  "investigations": investigations,
  "workstation": workstation,
  "vessels": vesselIntel,
  "spill": spillAnalysis,
  "drift": driftForecast,
  "evidence": evidence,
  "alerts": alerts,
  "analytics": analytics,
  "users": userManagement,
  "system": system,
  "profile": profile,
};

export const NAV_GROUPS = ["Operations", "Intelligence", "System"];

export const NAV = [
  { id: "mission-control", label: "Mission Control", min: "PILOT", group: "Operations", icon: "radar" },
  { id: "monitoring", label: "Live Monitoring", min: "PILOT", group: "Operations", icon: "activity" },
  { id: "investigations", label: "Investigations", min: "PILOT", group: "Operations", icon: "folder" },
  { id: "workstation", label: "Investigation Workstation", min: "PILOT", hideInNav: true },
  { id: "vessels", label: "Vessel Intelligence", min: "PILOT", group: "Intelligence", icon: "ship" },
  { id: "spill", label: "Spill Analysis", min: "PILOT", group: "Intelligence", icon: "droplet" },
  { id: "drift", label: "Drift & Forecast", min: "PILOT", group: "Intelligence", icon: "wind" },
  { id: "evidence", label: "Evidence", min: "PILOT", group: "Intelligence", icon: "file" },
  { id: "alerts", label: "Alerts", min: "REGIONAL", group: "System", icon: "bell" },
  { id: "analytics", label: "Analytics", min: "REGIONAL", group: "System", icon: "chart" },
  { id: "users", label: "User Management", min: "REGIONAL", group: "System", icon: "users" },
  { id: "system", label: "System", min: "PILOT", group: "System", icon: "server" },
  { id: "profile", label: "Profile", min: "PILOT", group: "System", icon: "user" },
];
