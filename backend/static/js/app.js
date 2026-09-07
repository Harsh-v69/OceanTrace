/* Bootstrap: auth gate, role-filtered nav, hash router, theme toggle. */

import { api, getToken, getStoredUser } from "./api.js?v=epic2";
import { views, NAV, wireAuth } from "./views.js?v=epic2";

const ROLE_RANK = { PILOT: 1, REGIONAL: 2, NATIONAL: 3 };
const THEME_KEY = "sn.theme";
const DEFAULT_VIEW = "mission-control";

const appEl = document.getElementById("app");
const authEl = document.getElementById("auth");
const navEl = document.getElementById("nav");
const viewEl = document.getElementById("view");
const toastEl = document.getElementById("toast");

let state = { user: null };

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
  // re-render current view so the map picks up the new palette
  route();
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
  navEl.innerHTML = NAV
    .filter((n) => !n.hideInNav && rank >= (ROLE_RANK[n.min] || 1))
    .map((n) => `<a href="#/${n.id}" class="${n.id === cur ? "active" : ""}"><span class="dot"></span>${n.label}</a>`)
    .join("");
}

/* --------------------------------------------------------------- router -- */
const ctx = { get user() { return state.user; }, root: viewEl, go: (hash) => { location.hash = hash; }, toast };

async function route() {
  if (!state.user) return;
  const parts = location.hash.replace(/^#\//, "").split("/").filter(Boolean);
  const name = parts[0] || DEFAULT_VIEW;
  const params = parts.slice(1);

  const entry = NAV.find((n) => n.id === name);
  const rank = ROLE_RANK[state.user.role] || 1;
  if (!views[name] || (entry && rank < (ROLE_RANK[entry.min] || 1))) {
    viewEl.innerHTML = `<div class="page-head"><h1>Not available</h1></div>
      <p class="empty">This screen doesn't exist or your role can't see it.</p>`;
    renderNav();
    return;
  }
  renderNav();
  viewEl.innerHTML = `<p class="empty">Loading&hellip;</p>`;
  try {
    await views[name](ctx, params);
  } catch (e) {
    console.error(e);
    viewEl.innerHTML = `<div class="page-head"><h1>Something went wrong</h1></div>
      <div class="panel"><p>${String(e && e.message || e)}</p></div>`;
  }
}
window.addEventListener("hashchange", route);

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
  if (!location.hash) location.hash = `#/${DEFAULT_VIEW}`;
  else route();
}
function showAuth() {
  state.user = null;
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
