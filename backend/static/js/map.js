/* Leaflet helpers. Offline-first: if OSM tiles fail we fall back to a plain
   ocean canvas so the console still works with no internet. */

const L = window.L;

// point Leaflet's default marker at our vendored images
if (L && L.Icon && L.Icon.Default) {
  L.Icon.Default.prototype.options.imagePath = "vendor/images/";
  delete L.Icon.Default.prototype._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconUrl: "vendor/images/marker-icon.png",
    iconRetinaUrl: "vendor/images/marker-icon-2x.png",
    shadowUrl: "vendor/images/marker-shadow.png",
  });
}

const INDIA_VIEW = { center: [15.6, 76.5], zoom: 5 };

export function makeMap(el, opts = {}) {
  const map = L.map(el, {
    zoomControl: true, attributionControl: false,
    center: opts.center || INDIA_VIEW.center,
    zoom: opts.zoom || INDIA_VIEW.zoom,
    worldCopyJump: true,
  });

  const dark = document.documentElement.getAttribute("data-theme") !== "light";

  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 12, minZoom: 3, crossOrigin: true,
    className: dark ? "tiles-dark" : "",
  });

  let failed = 0, fellBack = false, anyLoaded = false;
  const goOffline = () => {
    if (fellBack || anyLoaded) return;
    fellBack = true;
    offlineBasemap(map, el, dark);
  };
  tiles.on("tileerror", () => { failed += 1; if (failed >= 3) goOffline(); });
  tiles.on("tileload", () => { anyLoaded = true; failed = 0; });
  tiles.addTo(map);

  // if not a single tile has loaded after a few seconds, assume we're offline
  setTimeout(goOffline, 6500);
  // let the container settle before Leaflet measures it
  setTimeout(() => map.invalidateSize(), 60);
  return map;
}

/* Epic 2.4: when OSM tiles cannot load, draw the bundled simplified India
   coastline as a vector layer so the map still shows a recognisable outline. */
let _coastlineCache = null;
async function offlineBasemap(map, el, dark) {
  if (el.classList.contains("no-tiles")) return;   // already fell back
  el.classList.add("no-tiles");
  map.eachLayer((lyr) => { if (lyr instanceof L.TileLayer) map.removeLayer(lyr); });
  L.rectangle([[-60, -200], [75, 200]], {
    stroke: false, fillColor: dark ? "#0b1a2b" : "#dbeafe", fillOpacity: 1, interactive: false,
  }).addTo(map);
  try {
    if (!_coastlineCache) {
      _coastlineCache = await (await fetch("data/coastline_in.geojson")).json();
    }
    L.geoJSON(_coastlineCache, {
      style: {
        color: dark ? "#3a5372" : "#7d9bbd", weight: 1,
        fillColor: dark ? "#152a40" : "#eaf1f8", fillOpacity: 1,
      },
      interactive: false,
    }).addTo(map);
  } catch { /* no coastline file - the plain ocean canvas is the fallback */ }

  // tell the operator the basemap is degraded, not broken
  const badge = L.control({ position: "bottomleft" });
  badge.onAdd = () => {
    const d = L.DomUtil.create("div", "map-badge");
    d.innerHTML = `<span class="map-badge-dot"></span>Offline map &middot; coastline only`;
    return d;
  };
  badge.addTo(map);
}

/* tone from the canonical scene classification first, confidence second */
function anomTone(confidence, classification) {
  if (classification === "Likely look-alike") return "warn";
  if (classification === "No significant anomaly") return "info";
  return confidence >= 0.75 ? "crit" : confidence >= 0.5 ? "hot" : "info";
}

export function anomalyMarker(map, lat, lon, opts = {}) {
  const { confidence = 0, label = "Oil-like anomaly", ref = "", classification = "" } = opts;
  const tone = anomTone(confidence, classification);
  const m = L.marker([lat, lon], {
    icon: L.divIcon({
      className: "anom-div",
      html: `<span class="anom t-${tone}${tone === "crit" ? " pulse" : ""}"></span>`,
      iconSize: [28, 28], iconAnchor: [14, 14],
    }),
    keyboard: false,
    riseOnHover: true,
  }).addTo(map);
  m.bindPopup(
    `<b>${label}</b>${ref ? `<br><span class="mono">${ref}</span>` : ""}` +
    `<br>confidence ${(confidence * 100).toFixed(0)}%` +
    `<br><span class="mono">${lat.toFixed(3)}, ${lon.toFixed(3)}</span>`
  );
  m.setSelected = (on) => {
    const icon = m.getElement && m.getElement();
    if (icon) icon.classList.toggle("sel", !!on);
  };
  return m;
}

export function vesselMarker(map, lat, lon, { name = "", mmsi = "", rank = null, suspect = false } = {}) {
  const color = suspect ? "#ff6b6b" : "#91a3bf";
  const m = L.marker([lat, lon], {
    icon: L.divIcon({
      className: "vessel-div",
      html: `<span style="--vc:${color}">${rank ? "#" + rank : "▲"}</span>`,
      iconSize: [22, 22], iconAnchor: [11, 11],
    }),
  }).addTo(map);
  if (name || mmsi) m.bindPopup(`<b>${name || "Vessel"}</b><br>MMSI ${mmsi}${rank ? `<br>candidate rank #${rank}` : ""}`);
  return m;
}

export function trackLine(map, points, opts = {}) {
  if (!points || points.length < 2) return null;
  return L.polyline(points, {
    color: opts.color || "#2ea6ff", weight: opts.weight || 2,
    opacity: opts.opacity ?? 0.8, dashArray: opts.dash || null,
  }).addTo(map);
}

export function shorelineContact(map, coastalImpact) {
  if (!coastalImpact || !coastalImpact.will_beach) return null;
  const g = L.layerGroup().addTo(map);
  for (const p of (coastalImpact.contact_points || [])) {
    L.circleMarker(p, { radius: 3, color: "#ff6b6b", weight: 0, fillOpacity: 0.6 }).addTo(g);
  }
  const fc = coastalImpact.first_contact_point;
  if (fc) {
    L.circleMarker(fc, { radius: 7, color: "#ff3b3b", weight: 2, fillOpacity: 0.25 })
      .addTo(g)
      .bindPopup(`<b>First shoreline contact</b><br>ETA ${coastalImpact.first_contact_eta_h ?? coastalImpact.eta_hours} h`
        + `<br><span class="mono">${fc[0].toFixed(3)}, ${fc[1].toFixed(3)}</span>`
        + `<br>${Math.round((coastalImpact.fraction_beached || 0) * 100)}% of the modelled oil strands`);
  }
  return g;
}

export function polygon(map, ring, opts = {}) {
  if (!ring || !ring.length) return null;
  // accept GeoJSON [[lon,lat],...] or [[lat,lon],...]
  const latlng = ring.map(([a, b]) => (opts.lonlat ? [b, a] : [a, b]));
  return L.polygon(latlng, {
    color: opts.color || "#35d07f", weight: 1.5, fillOpacity: opts.fillOpacity ?? 0.12,
    dashArray: opts.dash || "4 4",
  }).addTo(map);
}

/* ---- vessel track rendering (Epic 1.4) ---- */
function _interpPing(pings, tH) {
  if (!pings || !pings.length) return null;
  if (tH <= pings[0].t_h) return [pings[0].lat, pings[0].lon];
  if (tH >= pings[pings.length - 1].t_h) {
    const p = pings[pings.length - 1]; return [p.lat, p.lon];
  }
  for (let i = 1; i < pings.length; i++) {
    if (pings[i].t_h >= tH) {
      const a = pings[i - 1], b = pings[i];
      const f = (tH - a.t_h) / ((b.t_h - a.t_h) || 1);
      return [a.lat + f * (b.lat - a.lat), a.lon + f * (b.lon - a.lon)];
    }
  }
  return [pings[pings.length - 1].lat, pings[pings.length - 1].lon];
}

/**
 * Render one reconstructed vessel track. Returns
 * { group, endMarker, positionAt(tH) } — positionAt moves a dot for the timeline.
 * `view` is the object from summary_metrics.vessel_tracks[mmsi] or /vessels/{mmsi}/track.
 */
export function vesselTrackLayer(map, view, opts = {}) {
  const pings = view?.pings || [];
  const g = L.layerGroup().addTo(map);
  if (pings.length < 1) return { group: g, endMarker: null, positionAt: () => {} };

  const prime = opts.prime ?? view?.attribution?.is_prime;
  const color = opts.color || (prime ? "#ff6b6b" : "#8aa0bd");
  const latlngs = pings.map((p) => [p.lat, p.lon]);

  if (latlngs.length > 1) {
    L.polyline(latlngs, { color, weight: prime ? 3 : 2, opacity: 0.9 }).addTo(g);
  }
  // blackout gaps as dashed segments between the fixes bracketing each gap
  for (const b of (view.blackouts || [])) {
    const a = _interpPing(pings, b.start_h), c = _interpPing(pings, b.end_h);
    if (a && c) L.polyline([a, c], { color: "#ff6b6b", weight: 2, dashArray: "3 6", opacity: 0.9 })
      .addTo(g).bindPopup(`AIS dark ${Math.round(b.minutes)} min`);
  }
  // loiter spans as amber rings at their midpoint
  for (const s of (view.loiter || [])) {
    const mid = _interpPing(pings, (s.start_h + s.end_h) / 2);
    if (mid) L.circleMarker(mid, { radius: 7, color: "#f0b429", weight: 2, fillOpacity: 0.15 })
      .addTo(g).bindPopup(`Loitering ${Math.round(s.minutes)} min @ ${s.mean_sog_kn} kn`);
  }
  // start (hollow) + end (vessel glyph)
  L.circleMarker(latlngs[0], { radius: 4, color, weight: 2, fillOpacity: 0 }).addTo(g);
  const end = latlngs[latlngs.length - 1];
  const rank = view?.attribution?.rank;
  const endMarker = L.marker(end, {
    icon: L.divIcon({
      className: "vessel-div",
      html: `<span style="--vc:${color}">${rank ? "#" + rank : "▲"}</span>`,
      iconSize: [22, 22], iconAnchor: [11, 11],
    }),
  }).addTo(g);
  if (opts.onClick) endMarker.on("click", () => opts.onClick(view));
  else endMarker.bindPopup(vesselPopupHtml(view));

  let dot = null;
  function positionAt(tH) {
    const p = _interpPing(pings, tH);
    if (!p) return;
    if (!dot) dot = L.circleMarker(p, { radius: 5, color, weight: 2, fillColor: color, fillOpacity: 0.9 }).addTo(g);
    else dot.setLatLng(p);
  }
  return { group: g, endMarker, positionAt };
}

export function vesselPopupHtml(view) {
  const a = view?.attribution || {};
  const m = view?.metrics || {};
  const ae = a.ais_anomaly || {};
  const rd = a.route_deviation || {};
  const line = (k, v) => (v == null || v === "" ? "" : `<br><span class="muted">${k}:</span> ${v}`);
  return `<b>${view?.name || "Vessel"}</b> <span class="mono">${view?.mmsi || ""}</span>`
    + line("type", view?.vessel_type) + line("flag", view?.flag)
    + (a.rank ? `<br><b>candidate #${a.rank}</b> — score ${Number(a.score).toFixed(1)} (${a.assessment || ""})` : "")
    + line("seen", `T${m.first_seen_h} … T${m.last_seen_h} h · ${m.n_points} pings`)
    + line("speed", `${m.sog_min_kn}–${m.sog_max_kn} kn (mean ${m.sog_mean_kn})`)
    + line("heading", m.mean_heading_deg != null ? `${m.mean_heading_deg}°` : null)
    + line("AIS gaps", `${m.n_gaps}${m.longest_blackout_over_window_min ? ` · ${Math.round(m.longest_blackout_over_window_min)} min over window` : ""}`)
    + line("AE anomaly", ae.peak_reconstruction_error != null
        ? `peak err ${ae.peak_reconstruction_error} / thr ${ae.threshold} — ${ae.flagged_pings ? ae.flagged_pings + " flagged" : "nothing flagged"}` : null)
    + line("LSTM route dev.", rd.usable ? `${rd.max_deviation_km} km peak` : (rd.finding ? "not usable (AOI-gated)" : null));
}

export function heat(map, cells, bbox) {
  // cells: flat list of probabilities over a grid inside bbox [w,s,e,n]
  if (!cells || !cells.length || !bbox) return [];
  return []; // kept lightweight for CPU/offline; heatmap omitted in favour of the ellipse
}

/* A small on-map key. Pass the ids relevant to the current view. */
const LEGEND_ITEMS = {
  "anom-hi":     ["sw dot t-crit pulse", "High-confidence anomaly"],
  "anom-mid":    ["sw dot t-hot", "Medium-confidence anomaly"],
  "anom-la":     ["sw dot t-warn", "Likely look-alike"],
  "anom-lo":     ["sw dot t-info", "Low-confidence / cleared"],
  "track-prime": ["sw line t-crit", "Prime suspect track"],
  "track-other": ["sw line t-mut", "Other vessel track"],
  "loiter":      ["sw ring t-warn", "Loitering"],
  "blackout":    ["sw line dash t-crit", "AIS blackout"],
  "origin":      ["sw dot t-ok", "Reconstructed origin"],
  "beach":       ["sw dot t-crit", "Shoreline contact"],
};
export function mapLegend(map, keys, opts = {}) {
  const ids = (keys && keys.length ? keys : Object.keys(LEGEND_ITEMS)).filter((k) => LEGEND_ITEMS[k]);
  if (!ids.length) return null;
  const ctl = L.control({ position: opts.position || "bottomright" });
  ctl.onAdd = () => {
    const d = L.DomUtil.create("div", "map-legend");
    d.innerHTML = ids.map((k) => {
      const [cls, label] = LEGEND_ITEMS[k];
      return `<span class="lg-row"><span class="${cls}"></span>${label}</span>`;
    }).join("");
    L.DomEvent.disableClickPropagation(d);
    return d;
  };
  ctl.addTo(map);
  return ctl;
}

export function fit(map, layers) {
  const bounds = L.latLngBounds([]);
  let any = false;
  for (const lyr of layers) {
    if (!lyr) continue;
    try {
      if (lyr.getBounds) bounds.extend(lyr.getBounds());
      else if (lyr.getLatLng) bounds.extend(lyr.getLatLng());
      any = true;
    } catch {}
  }
  if (any && bounds.isValid()) map.fitBounds(bounds.pad(0.25));
}

export { L };
