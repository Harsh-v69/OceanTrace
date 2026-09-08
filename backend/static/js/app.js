/* Bootstrap: auth gate, role-filtered grouped nav, hash router, theme toggle,
   live API-status indicator. */

import { api, getToken, getStoredUser } from "./api.js?v=ui14";
import { views, NAV, NAV_GROUPS, wireAuth } from "./views.js?v=ui14";

const ROLE_RANK = { PILOT: 1, REGIONAL: 2, NATIONAL: 3 };
const THEME_KEY = "sn.theme";
const DEFAULT_VIEW = "mission-control";

const appEl = document.getElementById("app");
const authEl = document.getElementById("auth");
const navEl = document.getElementById("nav");
const viewEl = document.getElementById("view");
const toastEl = document.getElementById("toast");
const titleEl = document.getElementById("view-title");

let state = { user: null };

/* --------------------------------------------------------------- icons --- */
/* Inline 24-box stroke icons — no icon font, no CDN. */
const ICONS = {
  radar: '<path d="M12 12 4.5 5.5M12 3a9 9 0 1 0 9 9"/><circle cx="12" cy="12" r="3.2"/>',
  activity: '<path d="M3 12h4l3 8 4-16 3 8h4"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  ship: '<path d="M3 14l1.6 5.2a2 2 0 0 0 1.9 1.4h11a2 2 0 0 0 1.9-1.4L23 14M5 14V8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v6M12 3v3M8 14V9h8v5"/>',
  droplet: '<path d="M12 3s6 6.4 6 10.5A6 6 0 0 1 6 13.5C6 9.4 12 3 12 3z"/>',
  wind: '<path d="M3 9h11a3 3 0 1 0-3-4M3 15h15a3 3 0 1 1-3 4M3 12h8"/>',
  file: '<path d="M6 3h8l5 5v13H6z"/><path d="M14 3v5h5M9 13h7M9 17h7"/>',
  bell: '<path d="M6 9a6 6 0 1 1 12 0c0 5 2 6 2 6H4s2-1 2-6"/><path d="M10 20a2 2 0 0 0 4 0"/>',
  chart: '<path d="M4 20V4M4 20h16M9 16v-5M14 16V8M19 16v-9"/>',
  users: '<circle cx="9" cy="8" r="3.2"/><path d="M3.5 20a5.5 5.5 0 0 1 11 0M16 6.5a3 3 0 0 1 0 6M20.5 20a5.5 5.5 0 0 0-4-5.3"/>',
  server: '<rect x="3" y="4" width="18" height="7" rx="1.5"/><rect x="3" y="13" width="18" height="7" rx="1.5"/><path d="M7 7.5h.01M7 16.5h.01"/>',
  user: '<circle cx="12" cy="8" r="3.5"/><path d="M5 20a7 7 0 0 1 14 0"/>',
  dot: '<circle cx="12" cy="12" r="3"/>',
  alert: '<path d="M12 3 2 20h20z"/><path d="M12 9v5M12 17h.01"/>',
};
function icon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ICONS.dot}</svg>`;
}

/* ---------------------------------------------------------------- theme -- */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem(THEME_KEY, t); } catch {}
}
(function initTheme() {
  let t;
  try { t = localStorage.getItem(THEME_KEY); } catch {}
  applyTheme(t || "dark");
})();
document.getElementById("theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  applyTheme(cur === "dark" ? "light" : "dark");
  route();   // re-render so the map picks up the new palette
});

/* ---------------------------------------------------------------- toast -- */
let toastTimer = null;
function toast(msg, isErr = false) {
  toastEl.textContent = msg;
  toastEl.classList.toggle("err", isErr);
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toastEl.hidden = true), 4200);
}

/* ----------------------------------------------------------------- nav --- */
function renderNav() {
  const rank = ROLE_RANK[state.user.role] || 1;
  const cur = (location.hash.replace(/^#\//, "").split("/")[0]) || DEFAULT_VIEW;
  const visible = NAV.filter((n) => !n.hideInNav && rank >= (ROLE_RANK[n.min] || 1));
  navEl.innerHTML = NAV_GROUPS.map((grp) => {
    const items = visible.filter((n) => (n.group || "System") === grp);
    if (!items.length) return "";
    return `<div class="nav-group">
      <div class="nav-group-label">${grp}</div>
      ${items.map((n) => `<a href="#/${n.id}" class="nav-link${n.id === cur ? " active" : ""}" title="${n.label}">
        <span class="nav-ico">${icon(n.icon)}</span><span class="nav-label">${n.label}</span></a>`).join("")}
    </div>`;
  }).join("");
}

/* --------------------------------------------------------------- router -- */
let _routeGen = 0;
const ctx = {
  get user() { return state.user; },
  root: viewEl,
  go: (hash) => { location.hash = hash; },
  toast,
  stale: () => false,   // replaced per-navigation in route()
};

function stateBlock(iconName, title, msg) {
  return `<div class="state"><div class="state-icon">${icon(iconName)}</div>
    <h3>${title}</h3><p>${msg}</p></div>`;
}

async function route() {
  if (!state.user) return;
  const parts = location.hash.replace(/^#\//, "").split("/").filter(Boolean);
  const name = parts[0] || DEFAULT_VIEW;
  const params = parts.slice(1);

  const entry = NAV.find((n) => n.id === name);
  const rank = ROLE_RANK[state.user.role] || 1;
  if (titleEl) titleEl.textContent = entry?.label || "OceanTrace";

  const gen = ++_routeGen;
  ctx.stale = () => _routeGen !== gen;   // async view code can bail if the user moved on

  if (!views[name] || (entry && rank < (ROLE_RANK[entry.min] || 1))) {
    renderNav();
    viewEl.innerHTML = stateBlock("alert", "Not available",
      "This screen doesn't exist or your role can't see it.");
    return;
  }
  renderNav();
  viewEl.innerHTML = `<div class="loading-row"><span class="spin"></span> Loading&hellip;</div>`;
  try {
    await views[name](ctx, params);
    if (ctx.stale()) return;
    viewEl.scrollTop = 0;
  } catch (e) {
    if (ctx.stale()) return;
    console.error(e);
    viewEl.innerHTML = stateBlock("alert", "Something went wrong",
      String((e && e.message) || e || "Unexpected error"));
  }
}
window.addEventListener("hashchange", route);

/* ------------------------------------------------------------ op-status -- */
function setOpStatus(kind, text) {
  const el = document.getElementById("op-status");
  if (!el) return;
  el.classList.toggle("ok", kind === "ok");
  el.classList.toggle("bad", kind === "bad");
  document.getElementById("op-status-text").textContent = text;
}
function pollHealth() {
  api.health()
    .then((hth) => {
      const ok = hth && (hth.status === "ok" || hth.database === "up");
      setOpStatus(ok ? "ok" : "bad", ok ? "Operational" : "Degraded");
    })
    .catch(() => setOpStatus("bad", "API offline"));
}

/* ------------------------------------------------------------- session -- */
function showApp(user) {
  state.user = user;
  authEl.hidden = true;
  appEl.hidden = false;
  document.getElementById("user-chip").innerHTML = `<b>${esc(user.name)}</b> &middot; ${esc(user.role)}`;
  api.info().then((i) => {
    document.getElementById("env-chip").textContent =
      `v${i.version || "?"} · ${i.modes?.sms_provider || "mock"} SMS`;
  }).catch(() => {});
  pollHealth();
  clearInterval(showApp._hp);
  showApp._hp = setInterval(pollHealth, 30000);
  if (!location.hash) location.hash = `#/${DEFAULT_VIEW}`;
  else route();
}
function showAuth() {
  state.user = null;
  clearInterval(showApp._hp);
  appEl.hidden = true;
  authEl.hidden = false;
}
function esc(s) { return String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

document.getElementById("logout-btn").addEventListener("click", () => {
  api.logout();
  location.hash = "";
  showAuth();
});

wireAuth((user) => showApp(user));

/* --------------------------------------------------------------- start -- */
(async function start() {
  if (!getToken()) return showAuth();
  const cached = getStoredUser();
  if (cached) showApp(cached);          // optimistic
  try {
    const fresh = await api.me();        // validate token
    showApp(fresh);
  } catch {
    api.logout();
    showAuth();
  }
})();
