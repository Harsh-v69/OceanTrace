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
  const url = dark
    ? "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png" // never used, placeholder
    : "";

  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 12, minZoom: 3, crossOrigin: true,
    className: dark ? "tiles-dark" : "",
  });

  let failed = 0;
  tiles.on("tileerror", () => {
    failed += 1;
    if (failed === 4) {
      map.removeLayer(tiles);
      el.classList.add("no-tiles");
      L.rectangle([[-60, -200], [75, 200]], {
        stroke: false, fillColor: dark ? "#0d2036" : "#dbeafe", fillOpacity: 1,
      }).addTo(map);
    }
  });
  tiles.addTo(map);

  // let the container settle before Leaflet measures it
  setTimeout(() => map.invalidateSize(), 60);
  return map;
}

const CONF_COLOR = (c) => (c >= 0.75 ? "#ff6b6b" : c >= 0.5 ? "#f0b429" : "#66c2ff");

export function anomalyMarker(map, lat, lon, { confidence = 0, label = "Oil-like anomaly", ref = "" } = {}) {
  const m = L.circleMarker([lat, lon], {
    radius: 9, color: CONF_COLOR(confidence), weight: 2,
    fillColor: CONF_COLOR(confidence), fillOpacity: 0.35,
  }).addTo(map);
  m.bindPopup(
    `<b>${label}</b><br>${ref ? ref + "<br>" : ""}confidence ${(confidence * 100).toFixed(0)}%` +
    `<br><span class="mono">${lat.toFixed(3)}, ${lon.toFixed(3)}</span>`
  );
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

export function polygon(map, ring, opts = {}) {
  if (!ring || !ring.length) return null;
  // accept GeoJSON [[lon,lat],...] or [[lat,lon],...]
  const latlng = ring.map(([a, b]) => (opts.lonlat ? [b, a] : [a, b]));
  return L.polygon(latlng, {
    color: opts.color || "#35d07f", weight: 1.5, fillOpacity: opts.fillOpacity ?? 0.12,
    dashArray: opts.dash || "4 4",
  }).addTo(map);
}

export function heat(map, cells, bbox) {
  // cells: flat list of probabilities over a grid inside bbox [w,s,e,n]
  if (!cells || !cells.length || !bbox) return [];
  return []; // kept lightweight for CPU/offline; heatmap omitted in favour of the ellipse
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
