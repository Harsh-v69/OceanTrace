/* Lightweight vanilla-JS product tour + a "restart" hook.
   No dependencies, no build. State (completed) lives in localStorage. */

import { api } from "./api.js?v=ui22";

export const TOUR_KEY = "oceantrace.tour.done";
export function tourCompleted() {
  try { return localStorage.getItem(TOUR_KEY) === "1"; } catch { return false; }
}
function markDone() { try { localStorage.setItem(TOUR_KEY, "1"); } catch {} }
export function resetTour() { try { localStorage.removeItem(TOUR_KEY); } catch {} }

/* Steps: {title, body, view?, target?, needsInv?} — target is a CSS selector
   (first match wins). needsInv means the step opens the workstation. */
const STEPS = [
  { title: "Mission Control",
    body: "Your live coastal picture. The KPI row summarises active anomalies, high-confidence detections and prime suspects; the map plots them; below, you can run a deterministic demo scenario or upload a real SAR scene.",
    view: "#/mission-control", target: "#mc-kpis" },
  { title: "Live Monitoring",
    body: "Replay an incoming satellite observation and watch the unified pipeline run stage by stage — SAR detection → look-alike filter → drift hindcast → AIS fusion → jurisdiction → alert.",
    view: "#/monitoring", target: "#mon-run" },
  { title: "Select an anomaly",
    body: "Clicking an anomaly on Mission Control opens its Investigation Workstation. The left column is the anomaly itself: the canonical classification (Oil-like anomaly vs Likely look-alike), confidence and slick geometry.",
    needsInv: true, target: ".ws-left .panel" },
  { title: "Spill Analysis",
    body: "The SAR detection detail — Refined-Lee speckle filtering, adaptive dark-spot segmentation, the RF+GB ensemble verdict, the look-alike ruling, and the weathering (evaporation / spreading) estimate.",
    view: "#/spill", target: "#sa-body" },
  { title: "Drift & Origin",
    body: "The Lagrangian drift engine runs backward (hindcast) to reconstruct where and when the oil was released, then forward 6/12/24/48 h to project where it travels and whether it reaches shore.",
    view: "#/drift", target: "#df-body" },
  { title: "Vessel attribution",
    body: "Ranked candidate vessels. Each is scored 0–100 by fusing 8 evidence components — space-time match, CPA, dwell, AIS blackout, route deviation, AIS-anomaly and more. The top card is the prime suspect.",
    needsInv: true, target: "#ws-cands" },
  { title: "Route evidence",
    body: "Select a vessel to inspect its routes on the map: Observed AIS (solid line), Predicted (dashed — the LSTM's next-position estimate) and Reconstructed drift (dotted — the physics reverse-drift path). The timeline maps every event from historical AIS through the release window to the forecasts.",
    needsInv: true, target: "#ws-vessel, .ws-map-panel" },
  { title: "Evidence dossier",
    body: "The structured incident report — the JSON is the system of record; export Markdown, or open the printable / PDF view. Every panel is tagged with where its data came from (SYNTHETIC DEMO, RECONSTRUCTED, MODEL PREDICTION …).",
    view: "#/evidence", target: ".report-frame" },
];

/* --------------------------------------------------------------------- */
let _root = null;
let _i = 0;
let _invId = null;
let _nav = (h) => { location.hash = h; };
let _onKey = null;
let _onRelayout = null;

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

async function waitFor(selector, timeout = 4000) {
  const t0 = performance.now();
  return new Promise((resolve) => {
    const tick = () => {
      const found = selector && document.querySelector(selector);
      if (found) return resolve(found);
      if (performance.now() - t0 > timeout) return resolve(null);
      setTimeout(tick, 100);
    };
    tick();
  });
}

function teardown() {
  if (_onKey) { window.removeEventListener("keydown", _onKey); _onKey = null; }
  if (_onRelayout) {
    window.removeEventListener("resize", _onRelayout);
    window.removeEventListener("scroll", _onRelayout, true);
    document.removeEventListener("visibilitychange", _onRelayout);
    _onRelayout = null;
  }
  if (_root) { _root.remove(); _root = null; }
}

function finish(completed) {
  teardown();
  if (completed) markDone();
}

let _target = null;   // current step's spotlight element (or null)

/* Position the spotlight + card for the current step. Safe to call repeatedly
   (on resize / visibilitychange) — it re-measures every time. */
function layout() {
  if (!_root) return;
  const spot = _root.querySelector(".tour-spot");
  const card = _root.querySelector(".tour-card");
  const vw = window.innerWidth || 1280, vh = window.innerHeight || 800;
  const cw = card.offsetWidth || 340, chh = card.offsetHeight || 210;
  const r = _target && _target.getBoundingClientRect();
  const hasT = !!(r && r.width > 0 && r.height > 0);

  const pad = 8;
  const sx = hasT ? r.left - pad : -40;
  const sy = hasT ? r.top - pad : -40;
  const sw = hasT ? r.width + pad * 2 : 0;
  const sh = hasT ? r.height + pad * 2 : 0;
  spot.dataset.on = hasT ? "1" : "";
  spot.style.left = `${sx}px`;
  spot.style.top = `${sy}px`;
  spot.style.width = `${sw}px`;
  spot.style.height = `${sh}px`;

  let left, top;
  if (hasT) {
    if (r.right + 16 + cw < vw - 12) { left = r.right + 16; top = r.top; }
    else if (r.bottom + 14 + chh < vh - 12) { left = r.left; top = r.bottom + 14; }
    else { left = r.left; top = r.top - chh - 14; }
  } else {
    left = (vw - cw) / 2; top = (vh - chh) / 2;
  }
  card.style.left = `${Math.max(12, Math.min(left, vw - cw - 12))}px`;
  card.style.top = `${Math.max(12, Math.min(top, vh - chh - 12))}px`;
  card.style.opacity = "1";
}

async function show() {
  if (!_root) return;
  const step = STEPS[_i];
  const total = STEPS.length;

  if (step.needsInv && _invId != null) {
    if (!location.hash.startsWith(`#/workstation/${_invId}`)) _nav(`#/workstation/${_invId}`);
  } else if (step.view && location.hash !== step.view) {
    _nav(step.view);
  }

  const card = _root.querySelector(".tour-card");
  card.style.opacity = "0";
  _target = null;

  if (step.target) _target = await waitFor(step.target, (step.view || step.needsInv) ? 5000 : 1500);
  if (!_root) return;

  if (_target) {
    try { _target.scrollIntoView({ block: "center", behavior: "smooth" }); } catch {}
    await new Promise((res) => setTimeout(res, 240));
    if (!_root) return;
  }

  card.querySelector(".tour-step").textContent = `Step ${_i + 1} of ${total}`;
  card.querySelector(".tour-title").textContent = step.title;
  card.querySelector(".tour-body").textContent = step.body;
  card.querySelector('[data-act="back"]').disabled = _i === 0;
  card.querySelector('[data-act="next"]').textContent = _i === total - 1 ? "Finish" : "Next";

  layout();
  requestAnimationFrame(layout);
  // one more pass after the view/map has certainly settled
  setTimeout(() => { if (_root) layout(); }, 500);
}

function go(delta) {
  const n = _i + delta;
  if (n < 0) return;
  if (n >= STEPS.length) return finish(true);
  _i = n;
  show();
}

export async function startTour(opts = {}) {
  if (_root || document.querySelector(".tour-overlay")) return;   // already running (sync guard)
  _nav = opts.nav || _nav;
  _i = 0;

  // build the overlay synchronously so a concurrent startTour() is blocked
  _root = el("div", "tour-overlay");
  _root.innerHTML = `
    <div class="tour-spot"></div>
    <div class="tour-card" role="dialog" aria-modal="true">
      <div class="tour-step"></div>
      <h3 class="tour-title"></h3>
      <p class="tour-body"></p>
      <div class="tour-actions">
        <button class="btn btn-sm btn-ghost" data-act="skip">Skip tour</button>
        <span class="spacer"></span>
        <button class="btn btn-sm" data-act="back">Back</button>
        <button class="btn btn-sm btn-primary" data-act="next">Next</button>
      </div>
    </div>`;
  document.body.appendChild(_root);

  try {
    const invs = await api.investigations();
    _invId = invs && invs.length ? invs[invs.length - 1].id : null;
  } catch { _invId = null; }
  if (!_root) return;                   // skipped/torn down during the fetch

  _root.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "next") go(1);
    else if (act === "back") go(-1);
    else if (act === "skip") finish(true);
  });
  _onKey = (e) => {
    if (e.key === "Escape") finish(true);
    else if (e.key === "ArrowRight") go(1);
    else if (e.key === "ArrowLeft") go(-1);
  };
  window.addEventListener("keydown", _onKey);
  _onRelayout = () => { if (_root) layout(); };
  window.addEventListener("resize", _onRelayout, { passive: true });
  window.addEventListener("scroll", _onRelayout, { passive: true, capture: true });
  document.addEventListener("visibilitychange", _onRelayout);

  show();
}

/* If the app never onboarded this browser, offer the tour once. */
let _autoArmed = false;
export function maybeAutoStart(opts) {
  if (_autoArmed || tourCompleted()) return;
  _autoArmed = true;
  setTimeout(() => { if (!tourCompleted()) startTour(opts); }, 1100);
}
