/* ============================================================
   CITYNEXUS — canvas + chart rendering
   Smooth interpolated animation + weather effects + sparklines.
   ============================================================ */

(function (global) {
  'use strict';

  // ---------- palette (mirrors styles.css) ----------
  const COLORS = {
    bg:           '#07070f',
    grid:         '#0d0d1d',
    zone: {
      empty:        '#15152a',
      residential:  '#1e3a8a',
      commercial:   '#0f766e',
      hospital:     '#9d174d',
      industrial:   '#92400e',
      road:         '#3a3a52',
    },
    accident:    '#ff3b6b',
    roadblock:   '#ffb020',
    ambulance:   '#10b981',
    police:      '#3b82f6',
    van:         '#f59e0b',
    incident:    '#a78bfa',
    text:        '#e6e6f0',
    textDim:     '#7a7a92',
    accent:      '#7c5cff',
    accent2:     '#21d4fd',
    border:      '#22223a',
    good:        '#10b981',
    bad:         '#ff3b6b',
    warn:        '#ffb020',
    soft:        '#a78bfa',
  };

  const ROLE_COLORS = {
    delivery:  COLORS.van,
    traffic:   COLORS.police,
    emergency: COLORS.ambulance,
    police:    COLORS.soft,
    planner:   COLORS.accent2,
  };

  // ---------- single source of truth for grid geometry ----------
  // Cells fill the canvas exactly: cellW = w/gridW, cellH = h/gridH, NO offset.
  // No centering math = nothing to drift between pickCell and drawCity.
  // (Our CSS forces the canvas square, so cellW === cellH; the per-axis
  //  split is a defensive fallback.)
  function gridGeometry(canvas, gridW, gridH) {
    const rect = (typeof canvas.getBoundingClientRect === 'function')
      ? canvas.getBoundingClientRect() : null;
    const w = (rect && rect.width)  || canvas.clientWidth  || canvas.width;
    const h = (rect && rect.height) || canvas.clientHeight || canvas.height;
    const cellW = w / gridW;
    const cellH = h / gridH;
    return {
      w, h,
      cellW, cellH,
      cellSize: Math.min(cellW, cellH),  // for marker sizing only
      // Legacy fields (kept = 0) so old call sites that destructure ox/oy
      // continue to work without changes:
      ox: 0,
      oy: 0,
    };
  }

  // ---------- main city canvas ----------
  function drawCity(ctx, state, opts) {
    opts = opts || {};
    const t = clamp01(opts.t || 0);   // animation interpolation [0..1]
    const W = state.world.width, H = state.world.height;
    const g = gridGeometry(ctx.canvas, W, H);
    const w = g.w, h = g.h, cellW = g.cellW, cellH = g.cellH, cellSize = g.cellSize;

    // Per-cell helpers — single source of truth for every coordinate transform.
    const cellX  = (cx) => cx * cellW;          // top-left x of cell column cx
    const cellY  = (cy) => cy * cellH;          // top-left y of cell row    cy
    const cellCX = (cx) => (cx + 0.5) * cellW;  // center x
    const cellCY = (cy) => (cy + 0.5) * cellH;  // center y

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, w, h);

    // ---------- 1. zone cells ----------
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const z = state.world.cells[y][x];
        ctx.fillStyle = COLORS.zone[z] || COLORS.zone.empty;
        ctx.fillRect(cellX(x), cellY(y), cellW, cellH);
      }
    }

    // grid lines (subtle)
    ctx.strokeStyle = 'rgba(255,255,255,0.025)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= W; i++) {
      const x = cellX(i) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let i = 0; i <= H; i++) {
      const y = cellY(i) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    // ---------- 2. traffic overlay on roads ----------
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (state.world.cells[y][x] !== 'road') continue;
        const v = state.traffic[y][x];
        if (v <= 0.05) continue;
        const a = 0.05 + 0.75 * v;
        const r = 255;
        const gn = Math.max(20, Math.round(220 - 220 * v));
        const b = Math.max(20, Math.round(80 - 60 * v));
        ctx.fillStyle = `rgba(${r}, ${gn}, ${b}, ${a.toFixed(3)})`;
        ctx.fillRect(cellX(x), cellY(y), cellW, cellH);
      }
    }

    // ---------- 3. memory zones (faint hint) ----------
    if (state.useMemory && state.memory && state.memory.zones) {
      for (const z of state.memory.zones) {
        if (z.risk < 0.5) continue;
        const a = 0.10 * z.risk;
        ctx.fillStyle = `rgba(255, 32, 80, ${a.toFixed(3)})`;
        ctx.fillRect(cellX(z.x), cellY(z.y), cellW, cellH);
      }
    }

    // ---------- 4. roadblocks ----------
    for (const r of state.roadblocks) {
      const cx = cellCX(r.x);
      const cy = cellCY(r.y);
      const sz = cellSize * 0.55;
      ctx.fillStyle = COLORS.roadblock;
      ctx.fillRect(cx - sz / 2, cy - sz / 2, sz, sz);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.2;
      ctx.strokeRect(cx - sz / 2, cy - sz / 2, sz, sz);
    }

    // ---------- 5. accidents (pulsing X) ----------
    const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 200);
    for (const a of state.accidents) {
      const cx = cellCX(a.x);
      const cy = cellCY(a.y);
      const sz = cellSize * (0.55 + 0.05 * a.severity);
      ctx.shadowColor = COLORS.accident;
      ctx.shadowBlur = 8 + 6 * pulse;
      ctx.strokeStyle = COLORS.accident;
      ctx.lineWidth = 2 + a.severity * 0.4;
      ctx.beginPath();
      ctx.moveTo(cx - sz / 2, cy - sz / 2);
      ctx.lineTo(cx + sz / 2, cy + sz / 2);
      ctx.moveTo(cx + sz / 2, cy - sz / 2);
      ctx.lineTo(cx - sz / 2, cy + sz / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // ---------- 6. incidents ----------
    for (const inc of state.incidents) {
      const cx = cellCX(inc.pos[0]);
      const cy = cellCY(inc.pos[1]);
      const sz = cellSize * 0.45;
      ctx.fillStyle = inc.assigned ? 'rgba(167,139,250,0.5)' : COLORS.incident;
      ctx.shadowColor = COLORS.incident;
      ctx.shadowBlur = 6;
      drawHex(ctx, cx, cy, sz);
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 0.8;
      drawHex(ctx, cx, cy, sz, true);
    }

    // ---------- 7. unit trails (if enabled) ----------
    if (opts.trails) {
      for (const u of state.units) {
        if (u.status !== 'en_route' && u.status !== 'returning') continue;
        if (u.prevPos[0] === u.pos[0] && u.prevPos[1] === u.pos[1]) continue;
        const x1 = cellCX(u.prevPos[0]);
        const y1 = cellCY(u.prevPos[1]);
        const x2 = cellCX(u.pos[0]);
        const y2 = cellCY(u.pos[1]);
        const grad = ctx.createLinearGradient(x1, y1, x2, y2);
        const c = unitColor(u);
        grad.addColorStop(0, hexA(c, 0.0));
        grad.addColorStop(1, hexA(c, 0.5));
        ctx.strokeStyle = grad;
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }

    // ---------- 8. units (interpolated position) ----------
    for (const u of state.units) {
      const px = lerp(u.prevPos[0], u.pos[0], t);
      const py = lerp(u.prevPos[1], u.pos[1], t);
      const cx = (px + 0.5) * cellW;
      const cy = (py + 0.5) * cellH;
      const c = unitColor(u);
      const r = cellSize * 0.30;
      ctx.shadowColor = c;
      ctx.shadowBlur = 8;
      ctx.fillStyle = c;
      ctx.beginPath();
      if (u.kind === 'van') {
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx + r * 0.85, cy + r * 0.7);
        ctx.lineTo(cx - r * 0.85, cy + r * 0.7);
        ctx.closePath();
      } else if (u.kind === 'police') {
        ctx.moveTo(cx, cy - r);
        ctx.lineTo(cx + r, cy);
        ctx.lineTo(cx, cy + r);
        ctx.lineTo(cx - r, cy);
        ctx.closePath();
      } else {
        ctx.arc(cx, cy, r * 0.85, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // ---------- 9. weather effects ----------
    if (opts.weatherFx && state.weather !== 'clear') {
      drawWeatherFx(ctx, state.weather, w, h);
    }

    // ---------- 10. selection highlight ----------
    // Uses the same cellW/cellH as pickCell — guaranteed pixel-perfect alignment.
    if (opts.selectedCell) {
      const [sx, sy] = opts.selectedCell;
      ctx.strokeStyle = COLORS.accent2;
      ctx.lineWidth = 2;
      ctx.shadowColor = COLORS.accent2;
      ctx.shadowBlur = 8;
      ctx.strokeRect(cellX(sx) + 1, cellY(sy) + 1, cellW - 2, cellH - 2);
      ctx.shadowBlur = 0;
    }
  }

  function drawHex(ctx, cx, cy, r, stroke) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const ang = Math.PI / 3 * i - Math.PI / 2;
      const x = cx + r * Math.cos(ang);
      const y = cy + r * Math.sin(ang);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    if (stroke) ctx.stroke(); else ctx.fill();
  }

  function unitColor(u) {
    if (u.kind === 'ambulance') return COLORS.ambulance;
    if (u.kind === 'police')    return COLORS.police;
    return COLORS.van;
  }

  // hex (#RRGGBB) → rgba(... , a)
  function hexA(hex, a) {
    const n = parseInt(hex.replace('#', ''), 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }

  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp01(t) { return t < 0 ? 0 : (t > 1 ? 1 : t); }

  // ---------- weather FX ----------
  // Persistent particle pool keyed by canvas to keep state across calls.
  const _fxState = new WeakMap();
  function getFx(canvas) {
    let s = _fxState.get(canvas);
    if (!s) {
      s = {
        rain: [],
        flashTtl: 0,
        flashCool: 0,
      };
      _fxState.set(canvas, s);
    }
    return s;
  }

  function drawWeatherFx(ctx, weather, w, h) {
    const fx = getFx(ctx.canvas);
    const target = weather === 'storm' ? 220 : 80;
    while (fx.rain.length < target) {
      fx.rain.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vy: 6 + Math.random() * 6 + (weather === 'storm' ? 5 : 0),
        len: 6 + Math.random() * 12 + (weather === 'storm' ? 6 : 0),
      });
    }
    while (fx.rain.length > target) fx.rain.pop();

    if (weather === 'storm') {
      ctx.fillStyle = 'rgba(20, 5, 40, 0.18)';
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.fillStyle = 'rgba(20, 30, 60, 0.06)';
      ctx.fillRect(0, 0, w, h);
    }

    ctx.strokeStyle = weather === 'storm' ? 'rgba(160, 200, 255, 0.45)' : 'rgba(140, 180, 255, 0.30)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const p of fx.rain) {
      p.y += p.vy;
      p.x += p.vy * 0.10;
      if (p.y > h) { p.y = -10; p.x = Math.random() * w; }
      if (p.x > w) p.x -= w;
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - p.vy * 0.10, p.y - p.len);
    }
    ctx.stroke();

    if (weather === 'storm') {
      if (fx.flashCool > 0) fx.flashCool--;
      if (fx.flashTtl > 0) fx.flashTtl--;
      if (fx.flashCool === 0 && fx.flashTtl === 0 && Math.random() < 0.012) {
        fx.flashTtl = 4;
        fx.flashCool = 60;
      }
      if (fx.flashTtl > 0) {
        ctx.fillStyle = `rgba(255, 255, 255, ${0.10 + 0.05 * fx.flashTtl})`;
        ctx.fillRect(0, 0, w, h);
      }
    }
  }

  // ---------- HiDPI scaling ----------
  function setupHiDPI(canvas) {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    if (canvas.width !== Math.round(cssW * dpr)) canvas.width = Math.round(cssW * dpr);
    if (canvas.height !== Math.round(cssH * dpr)) canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // ---------- charts: cumulative reward (per-agent) ----------
  function drawRewardChart(canvas, history) {
    const ctx = setupHiDPI(canvas);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    drawAxes(ctx, w, h);

    if (history.length < 2) {
      drawEmpty(ctx, w, h, 'Run a few ticks to see rewards');
      return;
    }

    const roles = ['delivery', 'traffic', 'emergency', 'police', 'planner'];
    let minV = 0, maxV = 0;
    for (const h of history) {
      for (const r of roles) {
        const v = h.cumulative[r] || 0;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    const pad = 16;
    const innerW = w - pad * 2 - 70;
    const innerH = h - pad * 2 - 8;
    const x0 = pad + 60, y0 = pad;
    const span = Math.max(0.5, maxV - minV);

    function tx(i) { return x0 + (i / Math.max(1, history.length - 1)) * innerW; }
    function ty(v) { return y0 + (1 - (v - minV) / span) * innerH; }

    // gridlines
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = minV + (span * i) / 4;
      const y = ty(v);
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + innerW, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(1), x0 - 6, y + 3);
    }
    // zero line
    if (minV < 0 && maxV > 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      const y = ty(0);
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x0 + innerW, y);
      ctx.stroke();
    }

    // role lines
    for (const role of roles) {
      ctx.strokeStyle = ROLE_COLORS[role];
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const v = history[i].cumulative[role] || 0;
        const x = tx(i), y = ty(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // legend
    ctx.font = '10.5px Inter, sans-serif';
    ctx.textAlign = 'left';
    let lx = x0;
    for (const role of roles) {
      ctx.fillStyle = ROLE_COLORS[role];
      ctx.fillRect(lx, h - 12, 8, 8);
      ctx.fillStyle = COLORS.text;
      ctx.fillText(role, lx + 11, h - 4);
      lx += 60;
    }
  }

  // ---------- charts: city score ----------
  function drawCityChart(canvas, history) {
    const ctx = setupHiDPI(canvas);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    drawAxes(ctx, w, h);

    if (history.length < 2) {
      drawEmpty(ctx, w, h, 'Run a few ticks to see city score');
      return;
    }

    const pad = 16;
    const innerW = w - pad * 2 - 50;
    const innerH = h - pad * 2 - 8;
    const x0 = pad + 40, y0 = pad;

    function tx(i) { return x0 + (i / Math.max(1, history.length - 1)) * innerW; }
    function ty(v) { return y0 + (1 - clamp01(v)) * innerH; }

    // grid
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = COLORS.textDim;
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = i / 4;
      const y = ty(v);
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + innerW, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), x0 - 6, y + 3);
    }

    // total area
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const v = history[i].city.total;
      const x = tx(i), y = ty(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineTo(tx(history.length - 1), ty(0));
    ctx.lineTo(tx(0), ty(0));
    ctx.closePath();
    ctx.fillStyle = 'rgba(124, 92, 255, 0.12)';
    ctx.fill();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const v = history[i].city.total;
      const x = tx(i), y = ty(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // dotted components
    const comps = [
      ['deliveryHealth', COLORS.warn],
      ['safety',         COLORS.bad],
      ['mobility',       COLORS.accent2],
      ['coordination',   COLORS.soft],
    ];
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1.4;
    for (const [k, c] of comps) {
      ctx.strokeStyle = c;
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const v = history[i].city[k];
        const x = tx(i), y = ty(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // legend
    ctx.font = '10.5px Inter, sans-serif';
    ctx.textAlign = 'left';
    let lx = x0;
    const items = [['total', COLORS.accent], ['delivery', COLORS.warn], ['safety', COLORS.bad], ['mobility', COLORS.accent2], ['coord', COLORS.soft]];
    for (const [name, c] of items) {
      ctx.fillStyle = c;
      ctx.fillRect(lx, h - 12, 8, 8);
      ctx.fillStyle = COLORS.text;
      ctx.fillText(name, lx + 11, h - 4);
      lx += 65;
    }
  }

  // ---------- sparkline (per-agent reward over time) ----------
  function drawSparkline(canvas, history, key) {
    const ctx = setupHiDPI(canvas);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    if (history.length < 2) return;

    const vals = history.map(rec => rec.cumulative[key] || 0);
    let mn = Math.min(...vals, 0), mx = Math.max(...vals, 0.5);
    const span = mx - mn || 1;

    const color = ROLE_COLORS[key] || COLORS.accent;

    // baseline
    const yZero = h - ((0 - mn) / span) * h;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, yZero); ctx.lineTo(w, yZero); ctx.stroke();

    // gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, hexA(color, 0.40));
    grad.addColorStop(1, hexA(color, 0.0));
    ctx.fillStyle = grad;
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
      const x = (i / (vals.length - 1)) * w;
      const y = h - ((vals[i] - mn) / span) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fill();

    // line
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < vals.length; i++) {
      const x = (i / (vals.length - 1)) * w;
      const y = h - ((vals[i] - mn) / span) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // last point dot
    const lastX = w - 1.5;
    const lastY = h - ((vals[vals.length - 1] - mn) / span) * h;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawAxes(ctx, w, h) {
    ctx.fillStyle = '#0e0e1c';
    ctx.fillRect(0, 0, w, h);
  }
  function drawEmpty(ctx, w, h, msg) {
    ctx.fillStyle = COLORS.textDim;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(msg, w / 2, h / 2);
  }

  // ---------- pick cell at canvas coords ----------
  // Uses the SAME gridGeometry() as drawCity. Direct division — no offsets.
  function pickCell(canvas, evt, gridSize) {
    const rect = canvas.getBoundingClientRect();
    const g = gridGeometry(canvas, gridSize, gridSize);
    const x = Math.floor((evt.clientX - rect.left) / g.cellW);
    const y = Math.floor((evt.clientY - rect.top)  / g.cellH);
    if (x < 0 || y < 0 || x >= gridSize || y >= gridSize) return null;
    return [x, y];
  }

  // ---------- exports ----------
  global.CityNexusRender = {
    drawCity,
    drawRewardChart,
    drawCityChart,
    drawSparkline,
    setupHiDPI,
    pickCell,
    COLORS,
    ROLE_COLORS,
  };
})(typeof window !== 'undefined' ? window : globalThis);
