/* ============================================================
   CITYNEXUS — app glue
   Wires DOM controls to the simulation, runs the animation loop,
   keeps panels and logs in sync.
   ============================================================ */

(function () {
  'use strict';

  const sim = window.CityNexus;
  const R   = window.CityNexusRender;

  // ---------- DOM refs ----------
  const $ = (id) => document.getElementById(id);
  const cityCanvas    = $('city-canvas');
  const cityChartC    = $('city-chart');
  const rewardChartC  = $('reward-chart');
  const overlayEl     = $('canvas-overlay');
  const decisionsLog  = $('decisions-log');
  const messagesLog   = $('messages-log');
  const zonesLog      = $('zones-log');

  const els = {
    cityScore: $('city-score'),
    cityScoreSub: $('city-score-sub'),
    tick: $('tick-counter'),
    tickSub: $('tick-sub'),
    episode: $('episode-counter'),
    diffSub: $('diff-sub'),
    weather: $('weather-pill'),
    hour: $('hour-pill'),
    accPill: $('acc-pill'),
    incPill: $('inc-pill'),
    delivered: $('stat-delivered'),
    pending: $('stat-pending'),
    failed: $('stat-failed'),
    accidents: $('stat-accidents'),
    incidents: $('stat-incidents'),
    congestion: $('stat-congestion'),
    memory: $('stat-memory'),
    shocks: $('stat-shocks'),
    shocksTotal: $('stat-shocks-total'),
    reward: $('stat-reward'),
    fps: $('footer-fps'),
    storage: $('footer-storage'),
    speedValue: $('speed-value'),
    diffValue: $('difficulty-value'),
  };
  const rewardEls = {
    delivery:  $('reward-delivery'),
    traffic:   $('reward-traffic'),
    emergency: $('reward-emergency'),
    police:    $('reward-police'),
    planner:   $('reward-planner'),
  };
  const statEls = {
    delivery:  $('stats-delivery'),
    traffic:   $('stats-traffic'),
    emergency: $('stats-emergency'),
    police:    $('stats-police'),
    planner:   $('stats-planner'),
  };
  const sparklines = Array.from(document.querySelectorAll('.sparkline'));

  // ---------- runtime state ----------
  let state = null;
  let episodeIndex = 0;
  let playing = false;
  let tickIntervalMs = 400;
  let lastTickAt = 0;
  let selectedCell = null;
  let weatherFx = $('weather-fx-cb').checked;
  let trails = $('trails-cb').checked;
  let useMemory = $('memory-cb').checked;

  // FPS tracking
  let fpsTimer = 0;
  let fpsFrames = 0;
  let fpsValue = 0;

  // ---------- init ----------
  function init() {
    state = sim.newState({
      seed: getSeed(),
      difficulty: getDifficulty(),
      episodeLen: getLength(),
      useMemory,
    });
    selectedCell = null;
    refreshAll();
    els.shocksTotal.textContent = state.scenario.shocks.length;
    els.storage.textContent = (typeof localStorage !== 'undefined') ? 'storage: ready' : 'storage: unavailable';
    requestAnimationFrame(loop);
  }

  function reset() {
    episodeIndex++;
    playing = false;
    state = sim.newState({
      seed: getSeed(),
      difficulty: getDifficulty(),
      episodeLen: getLength(),
      useMemory,
    });
    selectedCell = null;
    els.shocksTotal.textContent = state.scenario.shocks.length;
    refreshAll();
  }

  function getSeed() {
    const v = parseInt($('seed-input').value, 10);
    return Number.isFinite(v) ? v : 42;
  }
  function getDifficulty() {
    const v = parseInt($('difficulty-slider').value, 10);
    return Number.isFinite(v) ? v / 100 : 0.30;
  }
  function getLength() {
    const v = parseInt($('length-input').value, 10);
    return Number.isFinite(v) ? v : 200;
  }

  // ---------- main loop (animation + tick scheduler) ----------
  function loop(now) {
    requestAnimationFrame(loop);

    // FPS measurement
    fpsFrames++;
    if (now - fpsTimer >= 1000) {
      fpsValue = Math.round((fpsFrames * 1000) / (now - fpsTimer));
      fpsFrames = 0;
      fpsTimer = now;
      els.fps.textContent = fpsValue + ' fps';
    }

    // Schedule next sim tick
    if (playing && state.tick < state.episodeLen) {
      if (now - lastTickAt >= tickIntervalMs) {
        sim.step(state);
        lastTickAt = now;
        refreshAll();
      }
    } else if (state.tick >= state.episodeLen) {
      playing = false;
    }

    // Animate canvas every frame; smooth interpolation between ticks
    let t = playing && tickIntervalMs > 0 ? (now - lastTickAt) / tickIntervalMs : 1;
    if (t > 1) t = 1;
    const ctx = R.setupHiDPI(cityCanvas);
    R.drawCity(ctx, state, { t, weatherFx, trails, selectedCell });
  }

  // ---------- update DOM panels ----------
  function refreshAll() {
    // hero / pills
    els.cityScore.textContent  = state.cityScore.total.toFixed(2);
    els.cityScoreSub.textContent =
      `safety ${state.cityScore.safety.toFixed(2)} · mobility ${state.cityScore.mobility.toFixed(2)}`;
    els.tick.textContent       = state.tick;
    els.tickSub.textContent    = `/ ${state.episodeLen}`;
    els.episode.textContent    = episodeIndex + 1;
    els.diffSub.textContent    = `d=${state.difficulty.toFixed(2)}`;

    // weather pill — target the dedicated text span explicitly so we never
    // accidentally overwrite the dot or leave a stray text node behind.
    const wDot  = els.weather.querySelector('.pill-dot');
    const wText = els.weather.querySelector('.pill-text');
    if (wText) wText.textContent = state.weather;
    if (wDot) {
      wDot.style.background =
        state.weather === 'storm' ? '#ff3b6b' :
        (state.weather === 'rain' ? '#3b82f6' : '#10b981');
    }
    els.weather.setAttribute('data-state', state.weather);

    // hour pill
    const hour = state.tick % 24;
    els.hour.textContent = `${String(hour).padStart(2, '0')}:00`;

    els.accPill.textContent = state.accidents.length;
    els.incPill.textContent = state.incidents.length;

    // delivery stats
    let delivered = 0, failed = 0, pending = 0;
    for (const d of state.deliveries) {
      if      (d.status === 'delivered') delivered++;
      else if (d.status === 'failed')    failed++;
      else                                pending++;
    }
    els.delivered.textContent = delivered;
    els.pending.textContent   = pending;
    els.failed.textContent    = failed;
    els.accidents.textContent = state.accidents.length;
    els.incidents.textContent = state.incidents.length;
    const cong = sim.computeCongestion(state.traffic, state.world);
    els.congestion.textContent = `${Math.round(cong * 100)}%`;
    els.memory.textContent     = state.memory.zones.length;
    els.shocks.textContent     = state.shocksFired;

    let totalReward = 0;
    for (const role in state.cumulative) totalReward += state.cumulative[role];
    els.reward.textContent = (totalReward >= 0 ? '+' : '') + totalReward.toFixed(1);
    els.reward.parentElement.classList.toggle('bad',  totalReward < 0);
    els.reward.parentElement.classList.toggle('good', totalReward >= 0);

    // per-agent cards
    for (const role in rewardEls) {
      const v = state.cumulative[role] || 0;
      rewardEls[role].textContent = (v >= 0 ? '+' : '') + v.toFixed(2);
      rewardEls[role].classList.toggle('neg', v < 0);
    }
    statEls.delivery.textContent  = `${pending} pending · ${delivered} delivered`;
    statEls.traffic.textContent   = `congestion ${Math.round(cong * 100)}% · roadblocks ${state.roadblocks.length}`;
    const idleAmb = state.units.filter(u => u.kind === 'ambulance' && u.status === 'idle').length;
    statEls.emergency.textContent = `${state.accidents.length} active · ${idleAmb}/${state.units.filter(u => u.kind === 'ambulance').length} idle`;
    const idlePol = state.units.filter(u => u.kind === 'police' && u.status === 'idle').length;
    statEls.police.textContent    = `${state.incidents.length} active · ${idlePol}/${state.units.filter(u => u.kind === 'police').length} idle`;
    const aboveBaseline = Object.values(state.priorities).filter(v => v > 1).length;
    statEls.planner.textContent   = `${aboveBaseline} priorities elevated`;

    // logs
    refreshDecisions();
    refreshMessages();
    refreshZones();

    // sparklines
    for (const s of sparklines) {
      R.drawSparkline(s, state.history, s.dataset.key);
    }

    // big charts
    R.drawCityChart(cityChartC, state.history);
    R.drawRewardChart(rewardChartC, state.history);
  }

  function refreshDecisions() {
    if (state.decisions.length === 0) {
      decisionsLog.innerHTML = '<div class="log-empty">No decisions yet — press Step or Play.</div>';
      return;
    }
    const recent = state.decisions.slice(-30).reverse();
    decisionsLog.innerHTML = recent.map(d => `
      <div class="log-row" data-role="${d.role}">
        <span class="tick">t${d.tick}</span>
        <span class="role">${d.role}</span>
        <span class="body"><span class="kind">${d.action}</span> ${escapeHtml(d.detail || '')}</span>
      </div>
    `).join('');
  }

  function refreshMessages() {
    if (state.messages.length === 0) {
      messagesLog.innerHTML = '<div class="log-empty">No inter-agent messages yet.</div>';
      return;
    }
    const recent = state.messages.slice(-30).reverse();
    messagesLog.innerHTML = recent.map(m => `
      <div class="log-row" data-role="${m.from}">
        <span class="tick">t${m.tick}</span>
        <span class="role">${m.from}</span>
        <span class="body"><span class="kind">${m.kind}</span> ${escapeHtml(m.body || '')}</span>
      </div>
    `).join('');
  }

  function refreshZones() {
    if (!state.memory || state.memory.zones.length === 0) {
      zonesLog.innerHTML = '<div class="log-empty">No high-risk zones yet — accidents at locations build up zone records.</div>';
      return;
    }
    const sorted = state.memory.zones.slice().sort((a, b) => b.risk - a.risk).slice(0, 12);
    zonesLog.innerHTML = sorted.map(z => `
      <div class="zone-row">
        <span class="zone-coord">(${z.x},${z.y})</span>
        <span class="zone-bar"><span class="zone-bar-fill" style="width:${(z.risk * 100).toFixed(0)}%"></span></span>
        <span class="zone-meta">${z.samples}× · ${z.factor || 'mixed'}</span>
      </div>
    `).join('');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }

  // ---------- canvas interactions ----------
  cityCanvas.addEventListener('click', (e) => {
    const cell = R.pickCell(cityCanvas, e, state.world.width);
    if (!cell) {
      selectedCell = null;
      hideOverlay();
      return;
    }
    selectedCell = cell;
    showCellInfo(cell);
  });
  cityCanvas.addEventListener('mousemove', (e) => {
    const cell = R.pickCell(cityCanvas, e, state.world.width);
    if (!cell) {
      hideOverlay();
      return;
    }
    showCellInfo(cell);
  });
  cityCanvas.addEventListener('mouseleave', () => {
    if (!selectedCell) hideOverlay();
  });

  function showCellInfo([x, y]) {
    const z = state.world.cells[y][x];
    const t = state.world.cells[y][x] === 'road' ? state.traffic[y][x] : null;
    const acc = state.accidents.find(a => a.x === x && a.y === y);
    const rb  = state.roadblocks.find(r => r.x === x && r.y === y);
    const inc = state.incidents.find(i => i.pos[0] === x && i.pos[1] === y);
    const memZ = state.memory.zones.find(zz => zz.x === x && zz.y === y);
    const unit = state.units.find(u => u.pos[0] === x && u.pos[1] === y);
    const lines = [
      `<strong>(${x}, ${y}) — ${z}</strong>`,
      t !== null ? `traffic: ${(t * 100).toFixed(0)}%` : null,
      acc ? `accident · severity ${acc.severity} · ttl ${acc.ttl}` : null,
      rb  ? `roadblock · ttl ${rb.ttl == null ? '∞' : rb.ttl}` : null,
      inc ? `incident · ${inc.kind} · severity ${inc.severity}` : null,
      unit ? `unit: ${unit.id} (${unit.kind}) · status ${unit.status}` : null,
      memZ ? `memory: risk ${memZ.risk.toFixed(2)} · ${memZ.samples} samples` : null,
    ].filter(Boolean);
    overlayEl.innerHTML = lines.join('<br>');
    overlayEl.classList.add('visible');
  }
  function hideOverlay() {
    overlayEl.classList.remove('visible');
  }

  // ---------- control wiring ----------
  $('reset-btn').addEventListener('click', reset);
  $('step-btn').addEventListener('click', () => {
    sim.step(state);
    refreshAll();
  });
  $('play-btn').addEventListener('click', () => {
    if (state.tick >= state.episodeLen) reset();
    playing = true;
    lastTickAt = performance.now();
  });
  $('pause-btn').addEventListener('click', () => { playing = false; });

  $('speed-slider').addEventListener('input', (e) => {
    tickIntervalMs = parseInt(e.target.value, 10) || 400;
    els.speedValue.textContent = `${tickIntervalMs} ms`;
  });
  $('difficulty-slider').addEventListener('input', (e) => {
    const v = (parseInt(e.target.value, 10) || 30) / 100;
    els.diffValue.textContent = v.toFixed(2);
  });
  $('memory-cb').addEventListener('change', (e) => {
    useMemory = e.target.checked;
    state.useMemory = useMemory;
    if (useMemory) state.memory = sim.loadMemory();
    refreshAll();
  });
  $('weather-fx-cb').addEventListener('change', (e) => { weatherFx = e.target.checked; });
  $('trails-cb').addEventListener('change', (e) => { trails = e.target.checked; });

  $('clear-memory-btn').addEventListener('click', () => {
    sim.clearMemory();
    state.memory = { zones: [], failureCounts: {} };
    refreshAll();
    els.storage.textContent = 'storage: cleared';
    setTimeout(() => { els.storage.textContent = 'storage: ready'; }, 1500);
  });

  // keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea')) return;
    if (e.code === 'Space') {
      e.preventDefault();
      if (playing) { playing = false; }
      else { if (state.tick >= state.episodeLen) reset(); playing = true; lastTickAt = performance.now(); }
    } else if (e.code === 'KeyS') {
      sim.step(state); refreshAll();
    } else if (e.code === 'KeyR') {
      reset();
    }
  });

  // ---------- public API for the playback module ----------
  window.CityNexusApp = {
    // Pause auto-play and reset the sim with a specific seed/length, return state.
    setupPlayback({ seed, length }) {
      playing = false;
      $('seed-input').value      = String(seed);
      $('length-input').value    = String(length);
      state = sim.newState({
        seed,
        difficulty: getDifficulty(),
        episodeLen: length,
        useMemory,
      });
      selectedCell = null;
      els.shocksTotal.textContent = state.scenario.shocks.length;
      refreshAll();
      return state;
    },
    // Force the next sim tick to use the supplied mode, then refresh DOM.
    stepWithMode(mode) {
      state.forcedMode = mode;
      sim.step(state);
      state.forcedMode = null;
      refreshAll();
      return state;
    },
    getState() { return state; },
    isPlaying()    { return playing; },
    setPlaying(v)  { playing = !!v; if (playing) lastTickAt = performance.now(); },
  };

  // ---------- go ----------
  init();
})();
