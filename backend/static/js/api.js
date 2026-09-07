/* The single network seam. Every view talks to the backend through this module. */

const BASE = "/api/v1";
const TOKEN_KEY = "sn.token";
const USER_KEY = "sn.user";

export function getToken() {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function setToken(t) {
  try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch {}
}
export function getStoredUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); } catch { return null; }
}
export function setStoredUser(u) {
  try { u ? localStorage.setItem(USER_KEY, JSON.stringify(u)) : localStorage.removeItem(USER_KEY); } catch {}
}

export class ApiError extends Error {
  constructor(status, detail, body) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

async function request(path, { method = "GET", body, form, auth = true, raw = false } = {}) {
  const headers = {};
  const opts = { method, headers };

  if (auth) {
    const tok = getToken();
    if (tok) headers["Authorization"] = `Bearer ${tok}`;
  }
  if (form instanceof FormData) {
    opts.body = form;               // multipart; browser sets the boundary
  } else if (form) {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(form)) p.append(k, v);
    headers["Content-Type"] = "application/x-www-form-urlencoded";
    opts.body = p.toString();
  } else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }

  let res;
  try {
    res = await fetch(BASE + path, opts);
  } catch (e) {
    throw new ApiError(0, "Network error - is the API running?", null);
  }

  if (raw) return res;

  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = text; } }

  if (!res.ok) {
    const detail = data && typeof data === "object" && "detail" in data ? data.detail : (data || res.statusText);
    throw new ApiError(res.status, detail, data);
  }
  return data;
}

export const api = {
  // ---- auth ----
  async login(email, password) {
    const tok = await request("/auth/login", {
      method: "POST", auth: false, form: { username: email, password },
    });
    setToken(tok.access_token);
    setStoredUser(tok.user);
    return tok;
  },
  async register(payload) {
    return request("/auth/register", { method: "POST", auth: false, body: payload });
  },
  me: () => request("/auth/me"),
  logout() { setToken(null); setStoredUser(null); },

  // ---- user management (REGIONAL / NATIONAL) ----
  users: () => request("/users"),
  userScope: () => request("/users/me/scope"),
  createUser: (payload) => request("/users", { method: "POST", body: payload }),
  updateUser: (id, payload) => request(`/users/${id}`, { method: "PATCH", body: payload }),
  setUserActive: (id, on) => request(`/users/${id}/${on ? "enable" : "disable"}`, { method: "POST" }),

  // ---- system ----
  health: () => request("/system/health", { auth: false }),
  info: () => request("/system/info", { auth: false }),
  models: () => request("/system/models"),

  // ---- jurisdictions ----
  jurisdictions: (opts = {}) => {
    const q = new URLSearchParams(opts).toString();
    return request(`/jurisdictions${q ? "?" + q : ""}`);
  },
  resolve: (lat, lon) => request(`/jurisdictions/resolve?lat=${lat}&lon=${lon}`),

  // ---- investigations + evidence ----
  investigations: () => request("/investigations"),
  investigation: (id) => request(`/investigations/${id}`),
  dossier: (id) => request(`/investigations/${id}/dossier`),
  dossierMd: (id) => request(`/investigations/${id}/dossier.md`, { raw: false }),
  dossierHtmlUrl: (id) => `${BASE}/investigations/${id}/dossier.html`,

  // ---- anomalies / vessels ----
  anomalies: (investigationId) =>
    request(`/anomalies${investigationId ? `?investigation_id=${investigationId}` : ""}`),
  vessels: () => request("/vessels"),
  vessel: (mmsi) => request(`/vessels/${mmsi}`),

  // ---- alerts ----
  alerts: () => request("/alerts"),
  dispatchAlert: (payload) => request("/alerts/dispatch", { method: "POST", body: payload }),
  retryAlert: (id) => request(`/alerts/${id}/retry`, { method: "POST" }),
  testSms: (message) => request("/alerts/test-sms", { method: "POST", body: { message: message || null } }),

  // ---- scenarios ----
  scenarios: () => request("/scenarios"),
  runScenario: (key) => request(`/scenarios/${key}/run`, { method: "POST" }),

  // ---- uploads (Epic 3.3) ----
  uploadScene: (formData) => request("/investigations/upload-scene", { method: "POST", form: formData }),
  ingestAis: (formData) => request("/vessels/ingest-ais", { method: "POST", form: formData }),
};

// Fetch dossier .md / .html as text (needs auth header, so not a plain <a href>).
export async function fetchText(path) {
  const res = await request(path, { raw: true });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.text();
}
