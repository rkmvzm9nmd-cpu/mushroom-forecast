/* Mushroom Forecast - client. Data comes from ./data/ (built daily by GitHub Actions). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ storage (fail-soft)
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };

  const S = {
    index: null, meta: null, species: store.get("mf_species", "psilocybe"),
    day: 0, mode: "score", opacity: store.get("mf_opacity", 0.75), base: store.get("mf_base", "topo"),
    overlays: store.get("mf_overlays", { sightings: false, forest: false, parcels: false, trails: false, geology: false, places: true }),
    spots: store.get("mf_spots", []), sightings: null, locating: false,
  };

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => (v == null ? "–" : Math.round(v * 100) + "%");
  const sp = (id) => S.index.species.find((s) => s.id === (id || S.species));
  const DATA = "data/";
  let bust = "";

  function toast(msg) {
    const t = $("toast"); t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2200);
  }

  function pxToLatLng(x, y, z) {
    const n = 256 * Math.pow(2, z);
    const lng = (x / n) * 360 - 180;
    const lat = (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI;
    return [lat, lng];
  }
  function latLngToPx(lat, lng, z) {
    const n = 256 * Math.pow(2, z);
    const phi = (Math.max(-85.0511, Math.min(85.0511, lat)) * Math.PI) / 180;
    return [((lng + 180) / 360) * n, ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * n];
  }

  // ------------------------------------------------------------------ colour ramps
  const HEAT = [[0, [255, 255, 178]], [0.25, [254, 204, 92]], [0.45, [253, 141, 60]], [0.65, [240, 59, 32]], [0.82, [189, 0, 38]], [1, [122, 1, 119]]];
  const BLUE = [[0, [222, 235, 247]], [0.3, [158, 202, 225]], [0.6, [66, 146, 198]], [1, [8, 48, 107]]];
  function ramp(stops, t) {
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
        const f = (t - t0) / (t1 - t0 || 1);
        return c0.map((c, k) => Math.round(c + (c1[k] - c) * f));
      }
    }
    return stops[stops.length - 1][1];
  }
  function makeLut(kind) {
    const lut = new Uint8ClampedArray(256 * 4);
    for (let v = 0; v < 256; v++) {
      const t = v / 255;
      let rgb, a;
      if (kind === "rain") { rgb = ramp(BLUE, t); a = t < 0.03 ? 0 : Math.min(1, 0.25 + t * 1.2); }
      else {
        // rescale so 0.06..0.6 covers most of the ramp: scores rarely reach 1
        const u = Math.min(1, Math.max(0, (t - 0.06) / 0.6));
        rgb = ramp(HEAT, u); a = t < 0.06 ? 0 : Math.min(1, 0.35 + u * 0.9);
      }
      lut.set([rgb[0], rgb[1], rgb[2], Math.round(a * 255)], v * 4);
    }
    return lut;
  }
  const LUTS = { heat: makeLut("heat"), rain: makeLut("rain") };
  function legendCss(kind) {
    const stops = kind === "rain" ? BLUE : HEAT;
    return `linear-gradient(90deg, ${stops.map(([t, c]) => `rgb(${c.join(",")}) ${Math.round(t * 100)}%`).join(",")})`;
  }

  // ------------------------------------------------------------------ grey PNG loading + cache
  const cache = new Map();
  function loadGray(path) {
    if (cache.has(path)) { const v = cache.get(path); cache.delete(path); cache.set(path, v); return v; }
    const p = new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
        const ctx = c.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0);
        const rgba = ctx.getImageData(0, 0, c.width, c.height).data;
        const g = new Uint8Array(c.width * c.height);
        for (let i = 0; i < g.length; i++) g[i] = rgba[i * 4];
        resolve({ w: c.width, h: c.height, data: g });
      };
      img.onerror = () => reject(new Error("missing " + path));
      img.src = DATA + path + bust;
    });
    cache.set(path, p);
    while (cache.size > 40) cache.delete(cache.keys().next().value);
    p.catch(() => cache.delete(path));
    return p;
  }
  async function colourise(gray, lut) {
    const c = document.createElement("canvas"); c.width = gray.w; c.height = gray.h;
    const ctx = c.getContext("2d");
    const im = ctx.createImageData(gray.w, gray.h);
    const d = im.data, g = gray.data;
    for (let i = 0; i < g.length; i++) {
      const o = g[i] * 4, j = i * 4;
      d[j] = lut[o]; d[j + 1] = lut[o + 1]; d[j + 2] = lut[o + 2]; d[j + 3] = lut[o + 3];
    }
    ctx.putImageData(im, 0, 0);
    return new Promise((r) => c.toBlob((b) => r(URL.createObjectURL(b)), "image/png"));
  }

  // which file + geometry for a given mode/species/day
  function layerPath(mode, sid, day) {
    const r = S.meta.id;
    if (mode === "habitat") return `${r}/habitat/${sid}.png`;
    if (mode === "wx") return `${r}/wx/${sid}_${day}.png`;
    if (mode === "rain") return `${r}/rain/${day}.png`;
    if (mode === "soil") return `${r}/wx/soil.png`;
    return `${r}/score/${sid}_${day}.png`;
  }
  const isHalf = (mode) => mode === "wx" || mode === "rain" || mode === "soil";
  const isBlue = (mode) => mode === "rain" || mode === "soil";
  function overlayBounds(half) {
    const g = S.meta.grid, k = half ? S.meta.half : 1;
    const w = half ? Math.ceil(g.width / k) * k : g.width, h = half ? Math.ceil(g.height / k) * k : g.height;
    return [pxToLatLng(g.x0, g.y0 + h, g.zoom), pxToLatLng(g.x0 + w, g.y0, g.zoom)];
  }
  function sampleAt(gray, lat, lng, half) {
    const g = S.meta.grid, k = half ? S.meta.half : 1;
    const [x, y] = latLngToPx(lat, lng, g.zoom);
    const col = Math.floor((x - g.x0) / k), row = Math.floor((y - g.y0) / k);
    if (row < 0 || col < 0 || row >= gray.h || col >= gray.w) return null;
    let m = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const r = row + dy, c = col + dx;
      if (r >= 0 && c >= 0 && r < gray.h && c < gray.w) m = Math.max(m, gray.data[r * gray.w + c]);
    }
    return m / 255;
  }

  function sampleCell(gray, lat, lng) {   // the single pixel under the tap, no neighbours
    const g = S.meta.grid;
    const [x, y] = latLngToPx(lat, lng, g.zoom);
    const col = Math.floor(x - g.x0), row = Math.floor(y - g.y0);
    if (row < 0 || col < 0 || row >= gray.h || col >= gray.w) return null;
    return gray.data[row * gray.w + col] / 255;
  }

  // ------------------------------------------------------------------ map
  const map = L.map("map", { zoomControl: false, attributionControl: true, preferCanvas: true });
  L.control.zoom({ position: "topleft" }).addTo(map);
  L.control.scale({ position: "bottomleft", imperial: false }).addTo(map);
  const ign = (layer, fmt) => `https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=${layer}&STYLE=normal&TILEMATRIXSET=PM&FORMAT=${fmt}&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}`;
  const IGN_ATTR = '<a href="https://www.ign.fr">IGN</a>';
  const BASES = {
    topo: { name: "Topographic", layer: L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", { maxZoom: 17, attribution: '© OpenStreetMap, SRTM | © OpenTopoMap' }) },
    ign: { name: "IGN map (France)", layer: L.tileLayer(ign("GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2", "image/png"), { maxZoom: 19, attribution: IGN_ATTR }) },
    ortho: { name: "IGN aerial (France)", layer: L.tileLayer(ign("ORTHOIMAGERY.ORTHOPHOTOS", "image/jpeg"), { maxZoom: 19, attribution: IGN_ATTR }) },
    sat: { name: "Satellite (worldwide)", layer: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Esri, Maxar, Earthstar" }) },
  };
  const OVERLAYS = {
    forest: { name: "Tree species map (IGN BD Forêt)", layer: L.tileLayer(ign("LANDCOVER.FORESTINVENTORY.V2", "image/png"), { maxZoom: 19, opacity: 0.6, attribution: IGN_ATTR }) },
    parcels: { name: "Farm parcels (IGN RPG)", layer: L.tileLayer(ign("LANDUSE.AGRICULTURE.LATEST", "image/png"), { maxZoom: 19, opacity: 0.6, attribution: IGN_ATTR }) },
    trails: { name: "Hiking routes (Waymarked Trails / OSM)", layer: L.tileLayer("https://tile.waymarkedtrails.org/hiking/{z}/{x}/{y}.png", { maxZoom: 18, opacity: 0.9, attribution: '<a href="https://hiking.waymarkedtrails.org">Waymarked Trails</a>' }) },
    geology: { name: "Geology 1:50k (BRGM)", layer: L.tileLayer.wms("https://geoservices.brgm.fr/geologie", { layers: "SCAN_H_GEOL50", format: "image/png", transparent: true, opacity: 0.55, attribution: "BRGM" }) },
  };
  let baseLayer = null;
  function setBase(k) {
    if (!BASES[k]) k = "topo";
    if (baseLayer) map.removeLayer(baseLayer);
    baseLayer = BASES[k].layer.addTo(map); baseLayer.bringToBack();
    S.base = k; store.set("mf_base", k);
  }
  function syncOverlays() {
    for (const [k, o] of Object.entries(OVERLAYS)) {
      if (S.overlays[k]) o.layer.addTo(map); else map.removeLayer(o.layer);
    }
    placesLayer.clearLayers();
    if (S.overlays.places && S.meta) for (const p of S.meta.places || [])
      L.circleMarker([p.lat, p.lon], { radius: 4, color: "#14160f", weight: 1, fillColor: "#f2efe6", fillOpacity: 1 })
        .bindTooltip(p.name, { permanent: true, direction: "right", className: "" }).addTo(placesLayer);
    renderSightings();
    renderSatWet();
    store.set("mf_overlays", S.overlays);
  }
  const placesLayer = L.layerGroup().addTo(map);
  const sightLayer = L.layerGroup().addTo(map);
  const spotsLayer = L.layerGroup().addTo(map);
  const hotLayer = L.layerGroup().addTo(map);
  let scoreOverlay = null, drawToken = 0;

  async function draw() {
    if (!S.meta) return;
    const token = ++drawToken;
    const mode = S.mode, half = isHalf(mode);
    const path = layerPath(mode, S.species, S.day);
    let url;
    try {
      const gray = await loadGray(path);
      url = await colourise(gray, isBlue(mode) ? LUTS.rain : LUTS.heat);
    } catch (e) {
      if (token === drawToken) { if (scoreOverlay) scoreOverlay.setOpacity(0); toast("No data for this view yet"); }
      return;
    }
    if (token !== drawToken) { URL.revokeObjectURL(url); return; }
    const bounds = overlayBounds(half);
    if (!scoreOverlay) {
      scoreOverlay = L.imageOverlay(url, bounds, { opacity: S.opacity, interactive: false, className: "score-ov" }).addTo(map);
    } else {
      const old = scoreOverlay._url;
      scoreOverlay.setUrl(url); scoreOverlay.setBounds(L.latLngBounds(bounds)); scoreOverlay.setOpacity(S.opacity);
      if (old && old.startsWith("blob:")) setTimeout(() => URL.revokeObjectURL(old), 3000);
    }
    $("legendBar").style.background = legendCss(isBlue(mode) ? "rain" : "heat");
    $("legendLo").textContent = mode === "rain" ? "0 mm" : mode === "soil" ? "dry" : "low";
    $("legendHi").textContent = mode === "rain" ? "100+ mm" : mode === "soil" ? `wet (${S.meta.soil_date || ""})` : "high";
    if (hotLayer.getLayers().length) showHotMarkers();
  }

  // ------------------------------------------------------------------ UI: chips, days, modes
  function renderChips() {
    const el = $("chips"); el.innerHTML = "";
    for (const s of S.index.species) {
      const b = document.createElement("button");
      b.className = "chip" + (s.id === S.species ? " on" : "") + (s.group !== "Edible" ? " look" : "");
      b.innerHTML = `<span class="dot"></span>${esc(s.name)}`;
      b.onclick = () => { S.species = s.id; store.set("mf_species", s.id); renderChips(); renderDays(); draw(); renderSightings(); refreshSheet(); };
      el.appendChild(b);
    }
  }
  function renderDays() {
    const el = $("days"); el.innerHTML = "";
    const dates = S.meta.dates.length ? S.meta.dates : [null];
    const st = S.meta.stats[S.species];
    const maxKm = st ? Math.max(1, ...st.good_km2) : 1;
    dates.forEach((d, i) => {
      const b = document.createElement("button");
      b.className = "day" + (i === S.day ? " on" : "");
      const dt = d ? new Date(d + "T12:00:00") : null;
      const top = i === 0 ? "Today" : dt.toLocaleDateString(undefined, { weekday: "short" });
      const num = dt ? dt.getDate() : "–";
      const fill = st ? Math.round((st.good_km2[i] / maxKm) * 100) : 0;
      b.innerHTML = `${esc(top)}<b>${num}</b><span class="meter"><i style="width:${fill}%"></i></span>`;
      b.title = st ? `${st.good_km2[i]} km² rated good` : "";
      b.onclick = () => { S.day = i; renderDays(); draw(); refreshSheet(); };
      el.appendChild(b);
    });
    const hab = S.mode === "habitat" || S.mode === "soil";
    el.style.opacity = hab ? 0.4 : 1;
  }
  document.querySelectorAll(".modes button").forEach((b) => {
    b.onclick = () => {
      if (!S.meta.weather_ok && b.dataset.mode !== "habitat") { toast("Weather data missing today"); return; }
      if (b.dataset.mode === "soil" && !S.meta.soil_date) { toast("Satellite soil moisture not available today"); return; }
      S.mode = b.dataset.mode;
      document.querySelectorAll(".modes button").forEach((x) => x.classList.toggle("on", x === b));
      renderDays(); draw();
    };
  });

  // ------------------------------------------------------------------ sheet
  let sheetKind = null;
  function openSheet(kind, title, html) {
    sheetKind = kind; $("sheetTitle").textContent = title; $("sheetBody").innerHTML = html; $("sheet").hidden = false;
    document.querySelectorAll(".fab").forEach((f) => f.classList.toggle("on", f.dataset.kind === kind));
  }
  function closeSheet() {
    $("sheet").hidden = true; sheetKind = null; hotLayer.clearLayers();
    document.querySelectorAll(".fab").forEach((f) => f.dataset.kind && f.classList.remove("on"));
  }
  $("sheetClose").onclick = closeSheet;
  function refreshSheet() {
    if (sheetKind === "hot") showHot();
    else if (sheetKind === "spots") showSpots();
    else if (sheetKind === "info") showInfo();
  }

  // --- species info
  function showInfo() {
    const s = sp(), v = (S.meta.validation || {})[s.id] || {};
    const auc = v.auc == null ? "not enough public records yet" :
      `${Math.round(v.auc * 100)}% (${v.n} public finds; 50% = no better than chance)`;
    openSheet("info", s.name, `
      <p><i>${esc(s.latin)}</i> · ${esc(s.group)}</p>
      ${s.status ? `<p class="warn">${esc(s.status)}</p>` : ""}
      <h3>Where and when</h3><p>${esc(s.notes)}</p>
      <h3>Look-alikes</h3><p>${esc(s.lookalikes)}</p>
      <h3>Model check</h3><p>Habitat map ranks real recorded finds above random ground: <b>${auc}</b>.</p>
      ${v.cal_auc_rules != null ? `<p class="note">Fair test against where mushroom recorders actually go: rules ${Math.round(v.cal_auc_rules * 100)}%${v.cal_auc_learned != null ? `, learned model ${Math.round(v.cal_auc_learned * 100)}%` : ""}.</p>` : ""}
      <p class="note">${[v.learned_habitat ? "Habitat weights <b>learned</b> from local finds" : "Habitat from expert rules",
        v.learned_timing ? "timing after rain <b>learned</b> from dated finds" : "timing from expert rules",
        v.learned_season ? "season <b>learned</b> from local records" : "season from expert rules"].join(" · ")}.</p>
      ${v.learned_habitat && v.top_factors ? `<p class="note">Strongest factors: ${v.top_factors.slice(0, 4).map((t) => `${esc(t.feature.replace(/^(ft_|lc_|pa_|pl_)/, "").replace(/_/g, " "))} ${t.weight > 0 ? "+" : "−"}`).join(", ")}</p>` : ""}
      ${v.auc_with_satellite != null ? `<p class="note">With satellite wetness added: ${Math.round(v.auc_with_satellite * 100)}% — ${v.satellite_used ? "better, so it is switched on for this species." : "no clear gain, so it is not used for this species."}</p>` : ""}
      <h3>How to read the map</h3>
      <p class="note"><b>Likelihood</b> = habitat × recent weather. <b>Habitat</b> = tree species or pasture type (old pasture vs re-sown), soil acidity, altitude and damp ground.
      <b>Weather</b> = rain in the right window before the date, temperature, frost, and (for the next few days) satellite-measured soil wetness. Grassland also loses score where satellites saw recent ploughing or frequent silage cuts. Colours show odds, not certainty.</p>
      <p class="warn">Never eat a mushroom on the strength of this map. Get every find checked by an expert (French pharmacists and local mycological societies do this for free). Check local picking limits and ask before entering farmland.</p>
      <p class="note">Data: Open-Meteo, Copernicus (soil water index, grassland ploughing & mowing, Sentinel-2), IGN BD Forêt & RPG farm parcels, ForestPaths tree genera, ESA WorldCover, Copernicus DEM, INRAE/GIS Sol & ISRIC SoilGrids soil pH, GBIF. <a href="data/status.json" style="color:inherit">Build log</a>.</p>`);
  }
  $("infoBtn").onclick = showInfo;

  // --- best areas
  function showHotMarkers() {
    hotLayer.clearLayers();
    const list = ((S.meta.hotspots[S.species] || [])[S.mode === "habitat" ? 0 : S.day]) || [];
    list.forEach((h, i) => L.marker([h.lat, h.lon], { icon: L.divIcon({ className: "", html: `<div class="hot-pin">${i + 1}</div>`, iconSize: [22, 22] }) })
      .on("click", () => openPoint(L.latLng(h.lat, h.lon))).addTo(hotLayer));
  }
  function showHot() {
    const list = ((S.meta.hotspots[S.species] || [])[S.day]) || [];
    const day = S.meta.dates[S.day] ? new Date(S.meta.dates[S.day] + "T12:00:00").toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "short" }) : "";
    const rows = list.length ? list.map((h, i) => `
      <div class="row"><div class="hot-pin">${i + 1}</div>
        <div class="grow"><div class="name">${h.lat.toFixed(4)}, ${h.lon.toFixed(4)}</div><div class="sub">area average</div></div>
        <span class="score-pill">${pct(h.score)}</span>
        <button class="btn" data-fly="${h.lat},${h.lon}">Show</button></div>`).join("")
      : `<p class="note">Nothing scores well for ${esc(sp().name)} on this day. Try a later day or another species.</p>`;
    openSheet("hot", `Best areas · ${sp().name}`, `<p class="note">${esc(day)}. Peaks of the likelihood map, at least ~3 km apart.</p>${rows}`);
    $("sheetBody").querySelectorAll("[data-fly]").forEach((b) => (b.onclick = () => {
      const [la, lo] = b.dataset.fly.split(",").map(Number); map.flyTo([la, lo], 14); openPoint(L.latLng(la, lo));
    }));
    showHotMarkers();
  }
  $("hotBtn").dataset.kind = "hot";
  $("hotBtn").onclick = () => (sheetKind === "hot" ? closeSheet() : showHot());

  // --- saved spots
  function saveSpots() { store.set("mf_spots", S.spots); renderSpotPins(); }
  function renderSpotPins() {
    spotsLayer.clearLayers();
    for (const s of S.spots) L.marker([s.lat, s.lon], { icon: L.divIcon({ className: "", html: '<div class="spot-pin"></div>', iconSize: [14, 14] }) })
      .bindTooltip(esc(s.name)).on("click", () => openPoint(L.latLng(s.lat, s.lon))).addTo(spotsLayer);
  }
  async function outlook(lat, lng, sid) {
    const n = S.meta.dates.length;
    const vals = await Promise.all([...Array(n).keys()].map((d) => loadGray(layerPath("score", sid, d)).then((g) => sampleAt(g, lat, lng)).catch(() => null)));
    return vals;
  }
  function spark(vals) {
    return `<span class="spark">${vals.map((v, i) => `<i class="${i === S.day ? "today" : ""}" style="height:${Math.max(2, Math.round((v || 0) * 26 / 0.8))}px" title="${pct(v)}"></i>`).join("")}</span>`;
  }
  const navUrl = (la, lo) => `https://www.google.com/maps/dir/?api=1&destination=${la},${lo}`;
  async function showSpots() {
    const s = sp();
    const head = `<p class="note">Saved on this phone only. Bars: ${esc(s.name)} likelihood, next ${S.meta.dates.length} days.</p>`;
    if (!S.spots.length) {
      openSheet("spots", "Saved spots", head + `<p>Tap anywhere on the map, then <b>Save spot</b>.</p>` + alertHelp());
      bindAlertCopy(); return;
    }
    openSheet("spots", "Saved spots", head + `<div id="spotRows"></div>` + alertHelp());
    bindAlertCopy();
    const rowsEl = $("spotRows");
    for (const spot of S.spots) {
      const row = document.createElement("div"); row.className = "row";
      row.innerHTML = `<div class="grow"><div class="name">${esc(spot.name)}</div><div class="sub">${spot.lat.toFixed(4)}, ${spot.lon.toFixed(4)}</div></div><span class="spk">…</span>
        <button class="btn" data-a="go">Show</button><a class="btn" href="${navUrl(spot.lat, spot.lon)}" target="_blank" rel="noopener">Go</a><button class="btn danger" data-a="del" aria-label="Delete">✕</button>`;
      row.querySelector('[data-a="go"]').onclick = () => { map.flyTo([spot.lat, spot.lon], 15); openPoint(L.latLng(spot.lat, spot.lon)); };
      row.querySelector('[data-a="del"]').onclick = () => { S.spots = S.spots.filter((x) => x !== spot); saveSpots(); showSpots(); };
      rowsEl.appendChild(row);
      if (S.meta.weather_ok) outlook(spot.lat, spot.lon, S.species).then((v) => (row.querySelector(".spk").innerHTML = spark(v)));
      else row.querySelector(".spk").textContent = "";
    }
  }
  function alertHelp() {
    return `<h3>Push alerts</h3>
      <p class="note">To get a daily phone notification when a saved spot looks good: copy the text below and paste it into your GitHub repository under
      Settings → Secrets and variables → Actions → <b>SPOTS_JSON</b>. Your spots stay private (they are not published on this site).</p>
      <div class="btns"><button class="btn primary" id="copySpots">Copy spots for alerts</button></div>`;
  }
  function bindAlertCopy() {
    const b = $("copySpots"); if (!b) return;
    b.onclick = async () => {
      const txt = JSON.stringify(S.spots.map(({ name, lat, lon }) => ({ name, lat: +lat.toFixed(5), lon: +lon.toFixed(5) })));
      try { await navigator.clipboard.writeText(txt); toast("Copied"); }
      catch { $("sheetBody").insertAdjacentHTML("beforeend", `<textarea readonly>${esc(txt)}</textarea>`); }
    };
  }
  $("spotsBtn").dataset.kind = "spots";
  $("spotsBtn").onclick = () => (sheetKind === "spots" ? closeSheet() : showSpots());

  // --- layers
  function showLayers() {
    const bases = Object.entries(BASES).map(([k, b]) => `<label class="opt"><input type="radio" name="base" value="${k}" ${k === S.base ? "checked" : ""}>${esc(b.name)}</label>`).join("");
    const ovs = [["sightings", "Recorded finds (GBIF, this species)"], ["places", "Reference places"],
      ...(S.meta.satellite_wetness ? [["satwet", "Satellite wetness (Sentinel-2, blue = wetter)"]] : []),
      ...Object.entries(OVERLAYS).map(([k, o]) => [k, o.name])]
      .map(([k, n]) => `<label class="opt"><input type="checkbox" data-ov="${k}" ${S.overlays[k] ? "checked" : ""}>${esc(n)}</label>`).join("");
    const regions = S.index.regions.length > 1 ? `<h3>Region</h3>` + S.index.regions.map((r) =>
      `<label class="opt"><input type="radio" name="region" value="${r.id}" ${r.id === S.meta.id ? "checked" : ""}>${esc(r.name)}</label>`).join("") : "";
    openSheet("layers", "Map layers", `
      <h3>Forecast opacity</h3><input type="range" id="opac" min="0.2" max="1" step="0.05" value="${S.opacity}">
      <h3>Base map</h3>${bases}<h3>Overlays</h3>${ovs}${regions}
      <p class="note">IGN layers cover France only.</p>`);
    $("opac").oninput = (e) => { S.opacity = +e.target.value; store.set("mf_opacity", S.opacity); if (scoreOverlay) scoreOverlay.setOpacity(S.opacity); };
    $("sheetBody").querySelectorAll('input[name="base"]').forEach((i) => (i.onchange = () => setBase(i.value)));
    $("sheetBody").querySelectorAll("[data-ov]").forEach((i) => (i.onchange = () => { S.overlays[i.dataset.ov] = i.checked; syncOverlays(); }));
    $("sheetBody").querySelectorAll('input[name="region"]').forEach((i) => (i.onchange = () => loadRegion(i.value, true)));
  }
  $("layersBtn").dataset.kind = "layers";
  $("layersBtn").onclick = () => (sheetKind === "layers" ? closeSheet() : showLayers());

  // --- satellite wetness overlay
  let satOverlay = null;
  async function renderSatWet() {
    if (satOverlay) { map.removeLayer(satOverlay); satOverlay = null; }
    if (!S.overlays.satwet || !S.meta || !S.meta.satellite_wetness) return;
    try {
      const url = await colourise(await loadGray(`${S.meta.id}/wx/satwet.png`), LUTS.rain);
      satOverlay = L.imageOverlay(url, overlayBounds(false), { opacity: 0.6, interactive: false }).addTo(map);
    } catch { toast("Satellite wetness not available"); }
  }

  // --- sightings
  async function renderSightings() {
    sightLayer.clearLayers();
    if (!S.overlays.sightings || !S.meta) return;
    if (!S.sightings) {
      try { S.sightings = await (await fetch(`${DATA}${S.meta.id}/sightings.json${bust}`)).json(); }
      catch { S.sightings = {}; }
    }
    for (const r of S.sightings[S.species] || [])
      L.circleMarker([r.lat, r.lon], { radius: 5, color: "#14160f", weight: 1.5, fillColor: "#7fd3ff", fillOpacity: 0.9 })
        .bindPopup(`<div class="pop-title">Recorded find</div>${esc(r.date || "date unknown")}<br><a href="https://www.gbif.org/occurrence/${encodeURIComponent(r.id)}" target="_blank" rel="noopener" style="color:#e8b04a">View record</a>`)
        .addTo(sightLayer);
  }

  // ------------------------------------------------------------------ tap on map
  async function openPoint(latlng) {
    const { lat, lng } = latlng;
    const popup = L.popup({ maxWidth: 280, autoPanPaddingTopLeft: [10, 110], autoPanPaddingBottomRight: [64, 175] }).setLatLng(latlng).setContent("Loading…").openOn(map);
    const day = S.day, weather = S.meta.weather_ok;
    const rows = await Promise.all(S.index.species.map(async (s) => {
      const path = weather ? layerPath("score", s.id, day) : layerPath("habitat", s.id);
      try { return [s, sampleAt(await loadGray(path), lat, lng)]; } catch { return [s, null]; }
    }));
    let rain = null, hab = null;
    try { if (weather) rain = sampleAt(await loadGray(layerPath("rain", null, day)), lat, lng, true); } catch { /* */ }
    try { hab = sampleAt(await loadGray(layerPath("habitat", S.species)), lat, lng); } catch { /* */ }
    let satw = null;
    try { if (S.meta.satellite_wetness) satw = sampleAt(await loadGray(`${S.meta.id}/wx/satwet.png`), lat, lng); } catch { /* */ }
    let bare = null;
    try { if (S.meta.bare_soil && S.meta.bare_soil.used) bare = sampleCell(await loadGray(`${S.meta.id}/wx/bare.png`), lat, lng); } catch { /* */ }
    const ground = {};
    for (const k of (S.meta.ground_layers || [])) {
      try { ground[k] = sampleCell(await loadGray(`${S.meta.id}/wx/${k}.png`), lat, lng); } catch { /* */ }
    }
    const pc = (v) => `${Math.round(v * 100)}%`;
    const why = [];
    if (ground.register_pasture != null || ground.register_arable != null) {
      const parts = [];
      if (ground.register_pasture > 0.05) parts.push(`${pc(ground.register_pasture)} permanent/rough pasture`);
      if (ground.register_resown > 0.05) parts.push(`${pc(ground.register_resown)} re-sown grass`);
      if (ground.register_arable > 0.05) parts.push(`${pc(ground.register_arable)} arable`);  // declared crops / cultivated
      const src = S.meta.sources && S.meta.sources.habitat_map ? "habitat map" : "farm register";
      why.push(`${src}: ${parts.length ? parts.join(", ") : "not listed"}`);
    }
    if (ground.ploughed_map != null) {
      const yr = S.meta.sources && S.meta.sources.ploughing_year;
      why.push(`ploughing map (to ${yr || "latest"}): ${ground.ploughed_map > 0.05 ? `ploughed on ${pc(ground.ploughed_map)}` : "not ploughed"}`);
    }
    if (bare != null) why.push(`bare earth in ${pc(bare)} of clear spring/autumn satellite views since ${(S.meta.bare_soil.built || "").slice(0, 4) - 2}${bare >= 0.1 ? " (tilled recently: score cut)" : ""}`);
    let soil = null;
    try { if (S.meta.soil_date) soil = sampleAt(await loadGray(layerPath("soil")), lat, lng, true); } catch { /* */ }
    const outside = rows.every(([, v]) => v == null);
    if (outside) { popup.setContent("Outside the modelled area."); return; }
    rows.sort((a, b) => (b[1] || 0) - (a[1] || 0));
    const label = weather ? (S.meta.dates[day] === S.meta.dates[0] ? "Likelihood today" : `Likelihood ${S.meta.dates[day]}`) : "Habitat (no weather today)";
    const outlookHtml = weather ? `<div style="margin:6px 0 2px" class="note">${esc(sp().name)}, next days</div><div id="popSpark">…</div>` : "";
    popup.setContent(`
      <div class="pop-title">${label}</div>
      <div class="pop-grid">${rows.map(([s, v]) => `<span>${esc(s.name)}</span><span class="v">${pct(v)}</span>`).join("")}</div>
      <div class="note">Habitat for ${esc(sp().name)}: ${pct(hab)}${rain != null ? ` · rain last 14 days: ${Math.round(rain * 100)}${rain >= 1 ? "+" : ""} mm` : ""}${soil != null ? ` · soil wetness (satellite, ${esc(S.meta.soil_date)}): ${Math.round(soil * 100)}%` : ""}${satw ? ` · vegetation wetter than ${Math.round(satw * 100)}% of similar ground (Sentinel-2)` : ""}</div>
      ${why.length ? `<div class="note"><b>Ground:</b> ${esc(why.join(" · "))}</div>` : ""}
      ${outlookHtml}
      <div class="btns"><button class="btn primary" id="saveHere">Save spot</button><a class="btn" href="${navUrl(lat.toFixed(5), lng.toFixed(5))}" target="_blank" rel="noopener">Directions</a></div>`);
    const saveBtn = document.getElementById("saveHere");
    if (saveBtn) saveBtn.onclick = () => {
      const name = prompt("Name this spot", `Spot ${S.spots.length + 1}`);
      if (name === null) return;
      S.spots.push({ name: name.trim() || `Spot ${S.spots.length + 1}`, lat: +lat.toFixed(5), lon: +lng.toFixed(5), created: Date.now() });
      saveSpots(); toast("Spot saved"); map.closePopup();
      if (sheetKind === "spots") showSpots();
    };
    if (weather) outlook(lat, lng, S.species).then((v) => { const el = document.getElementById("popSpark"); if (el) el.innerHTML = spark(v); });
  }
  map.on("click", (e) => openPoint(e.latlng));

  // ------------------------------------------------------------------ location
  let youMarker = null, youCircle = null, firstFix = true;
  $("locBtn").onclick = () => {
    if (S.locating) { map.stopLocate(); S.locating = false; $("locBtn").classList.remove("on"); return; }
    S.locating = true; firstFix = true; $("locBtn").classList.add("on");
    map.locate({ watch: true, enableHighAccuracy: true, setView: false });
  };
  map.on("locationfound", (e) => {
    if (!youMarker) {
      youMarker = L.marker(e.latlng, { icon: L.divIcon({ className: "", html: '<div class="you"></div>', iconSize: [16, 16] }), interactive: false }).addTo(map);
      youCircle = L.circle(e.latlng, { radius: e.accuracy, color: "#3d8bfd", weight: 1, fillOpacity: 0.08, interactive: false }).addTo(map);
    } else { youMarker.setLatLng(e.latlng); youCircle.setLatLng(e.latlng).setRadius(e.accuracy); }
    if (firstFix) {
      firstFix = false;
      // jump to the region you're standing in, if it isn't the one on screen
      const here = S.index && S.index.regions.find((r) => e.latlng.lng >= r.bbox[0] && e.latlng.lat >= r.bbox[1] && e.latlng.lng <= r.bbox[2] && e.latlng.lat <= r.bbox[3]);
      const go = () => map.setView(e.latlng, Math.max(map.getZoom(), 14));
      if (here && S.meta && here.id !== S.meta.id) loadRegion(here.id, false).then(go); else go();
    }
  });
  map.on("locationerror", (e) => { toast("Location unavailable"); S.locating = false; $("locBtn").classList.remove("on"); });

  // ------------------------------------------------------------------ boot
  async function loadRegion(id, fit) {
    S.meta = await (await fetch(`${DATA}${id}/meta.json${bust}`)).json();
    S.sightings = null; S.day = Math.min(S.day, Math.max(0, S.meta.dates.length - 1));
    if (!S.meta.weather_ok) {
      S.mode = "habitat";
      document.querySelectorAll(".modes button").forEach((x) => x.classList.toggle("on", x.dataset.mode === "habitat"));
    }
    const when = new Date(S.meta.generated);
    $("updated").textContent = "Updated " + when.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
    const view = store.get("mf_view", null);
    const rb = L.latLngBounds(S.meta.grid.bounds);
    if (fit || !view || view.z < 7 || !rb.contains([view.lat, view.lng])) map.fitBounds(rb);
    else map.setView([view.lat, view.lng], view.z);
    store.set("mf_region", id);
    const sel = $("regionSel");
    if (S.index.regions.length > 1) {
      sel.innerHTML = S.index.regions.map((r) => `<option value="${esc(r.id)}" ${r.id === id ? "selected" : ""}>${esc(r.name)}</option>`).join("");
      sel.hidden = false;
      sel.onchange = () => loadRegion(sel.value, true);
    }
    renderChips(); renderDays(); syncOverlays(); draw();
  }
  map.on("moveend", () => { if (!S.meta) return; const c = map.getCenter(); store.set("mf_view", { lat: c.lat, lng: c.lng, z: map.getZoom() }); });

  (async () => {
    setBase(S.base);
    renderSpotPins();
    try {
      S.index = await (await fetch(`${DATA}index.json?t=${Date.now()}`)).json();
      if (!S.index.species.some((s) => s.id === S.species)) S.species = S.index.species[0].id;
      const saved = store.get("mf_region", null);
      const start = S.index.regions.some((r) => r.id === saved) ? saved : S.index.regions[0].id;
      const firstMeta = await (await fetch(`${DATA}${start}/meta.json?t=${Date.now()}`)).json();
      bust = "?v=" + encodeURIComponent(firstMeta.generated);
      await loadRegion(start, false);
    } catch (e) {
      map.setView([45, 4.5], 8);
      toast("Forecast data not built yet");
    }
  })();
})();
