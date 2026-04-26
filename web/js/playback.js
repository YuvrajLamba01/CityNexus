/* ============================================================
   CITYNEXUS — recorded-policy playback
   Loads runs/llm_rollouts.json (mirrored at web/data/llm_rollouts.json)
   and lets a reviewer watch the trained Qwen-0.5B Planner's actual
   per-tick mode choices animate the live city, compared with the
   random baseline and the heuristic expert (the GRPO training target).
   ============================================================ */
(function () {
  'use strict';

  const App = window.CityNexusApp;
  const Sim = window.CityNexus;
  if (!App || !Sim) return;

  const $ = (id) => document.getElementById(id);
  const panel       = $('playback-panel');
  const trackSel    = $('pb-track');
  const trackHint   = $('pb-track-hint');
  const seedSel     = $('pb-seed');
  const playBtn     = $('pb-play');
  const stopBtn     = $('pb-stop');
  const statusPill  = $('pb-status');
  const modePill    = $('pb-mode');
  const tickPill    = $('pb-tick');
  const rewardPill  = $('pb-reward');
  const barFill     = $('pb-bar-fill');
  const summary     = $('pb-summary');

  // Tick interval during playback (ms). 200 ms makes 80 ticks finish in ~16 s.
  const TICK_MS = 220;

  let payload = null;
  let timer = null;
  let activeRun = null;

  function fmt(n) {
    return (n >= 0 ? '+' : '') + n.toFixed(2);
  }

  function setStatus(text, kind) {
    statusPill.textContent = text;
    panel.dataset.state = kind || 'idle';
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function setSummary(html) {
    summary.innerHTML = html;
  }

  function summaryFor(track) {
    if (!track) return '';
    const seedKeys = Object.keys(track.rollouts || {});
    if (seedKeys.length === 0) {
      return `<strong>${escapeHtml(track.label)}</strong> — ${escapeHtml(track.note || track.description || 'No rollouts available yet.')}`;
    }
    const cums = seedKeys.map((k) => track.rollouts[k].cumulative_reward);
    const mean = cums.reduce((a, b) => a + b, 0) / cums.length;
    const min = Math.min(...cums), max = Math.max(...cums);
    const summary = track.summary || {};
    const tail = summary.welch_t !== undefined
      ? ` · Welch's t=${summary.welch_t.toFixed(2)} · p=${summary.p_value.toFixed(4)} · Cohen's d=${summary.cohens_d.toFixed(2)}`
      : '';
    return `<strong>${escapeHtml(track.label)}</strong> — Python eval over ${seedKeys.length} seeds: mean=${fmt(mean)}, min=${fmt(min)}, max=${fmt(max)}${tail}<br>` +
           `<span class="pb-summary-hint">${escapeHtml(track.description || '')}</span>`;
  }

  function refreshTrackHint() {
    const trackId = trackSel.value;
    const track = payload && payload.tracks ? payload.tracks[trackId] : null;
    if (!track) return;
    trackHint.textContent = track.available
      ? 'recorded policy ready'
      : (track.note || 'rollouts not yet generated');
    setSummary(summaryFor(track));
    playBtn.disabled = !track.available;
  }

  function buildSelectors(p) {
    trackSel.innerHTML = '';
    for (const [tid, track] of Object.entries(p.tracks)) {
      const opt = document.createElement('option');
      opt.value = tid;
      opt.textContent = `${track.label}${track.available ? '' : ' (not yet recorded)'}`;
      opt.disabled = !track.available;
      trackSel.appendChild(opt);
    }
    // Default to trained_llm if available, else heuristic_expert, else first.
    const order = ['trained_llm', 'heuristic_expert', 'random_baseline'];
    for (const id of order) {
      if (p.tracks[id] && p.tracks[id].available) {
        trackSel.value = id;
        break;
      }
    }

    seedSel.innerHTML = '';
    for (const seed of p.seeds) {
      const opt = document.createElement('option');
      opt.value = String(seed);
      opt.textContent = `seed ${seed}`;
      seedSel.appendChild(opt);
    }
    seedSel.value = String(p.seeds[0]);
  }

  function stop() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    activeRun = null;
    setStatus('idle', 'idle');
    stopBtn.disabled = true;
    playBtn.disabled = !(payload && payload.tracks && payload.tracks[trackSel.value] && payload.tracks[trackSel.value].available);
    barFill.style.width = '0%';
    modePill.textContent = 'mode: —';
    rewardPill.textContent = 'env reward 0.00';
  }

  function playTick() {
    if (!activeRun) { stop(); return; }
    if (activeRun.tick >= activeRun.modes.length) {
      const finalRew = activeRun.cumulative;
      const recorded = activeRun.recordedReward;
      setStatus('finished', 'finished');
      modePill.textContent = `mode: done`;
      tickPill.textContent = `tick ${activeRun.modes.length}/${activeRun.modes.length}`;
      rewardPill.textContent = `env reward ${fmt(finalRew)}`;
      barFill.style.width = '100%';
      stopBtn.disabled = true;
      playBtn.disabled = false;
      const note = recorded !== undefined
        ? `<br><span class="pb-summary-hint">Python-side recorded reward for this seed: <strong>${fmt(recorded)}</strong>. The JS sim re-evaluates the same mode trace on its own physics, so the live number above is a separate measurement of the same policy.</span>`
        : '';
      setSummary(summaryFor(payload.tracks[activeRun.trackId]) + note);
      if (timer) { clearInterval(timer); timer = null; }
      return;
    }
    const mode = activeRun.modes[activeRun.tick];
    App.stepWithMode(mode);
    const stateNow = App.getState();
    let total = 0;
    for (const r in stateNow.cumulative) total += stateNow.cumulative[r];
    activeRun.cumulative = total;
    activeRun.tick++;
    modePill.textContent = `mode: ${mode}`;
    tickPill.textContent = `tick ${activeRun.tick}/${activeRun.modes.length}`;
    rewardPill.textContent = `env reward ${fmt(total)}`;
    barFill.style.width = `${(100 * activeRun.tick) / activeRun.modes.length}%`;
  }

  function play() {
    if (!payload) return;
    const trackId = trackSel.value;
    const seedKey = seedSel.value;
    const track = payload.tracks[trackId];
    if (!track || !track.available) return;
    const rec = track.rollouts[seedKey];
    if (!rec) return;

    if (timer) { clearInterval(timer); timer = null; }

    App.setPlaying(false); // disable normal autoplay
    App.setupPlayback({ seed: parseInt(seedKey, 10), length: rec.modes.length });

    activeRun = {
      trackId,
      seed: seedKey,
      modes: rec.modes.slice(),
      tick: 0,
      cumulative: 0,
      recordedReward: rec.cumulative_reward,
    };
    setStatus(`playing ${track.label}`, 'playing');
    modePill.textContent = 'mode: —';
    tickPill.textContent = `tick 0/${rec.modes.length}`;
    rewardPill.textContent = 'env reward 0.00';
    barFill.style.width = '0%';
    stopBtn.disabled = false;
    playBtn.disabled = true;

    timer = setInterval(playTick, TICK_MS);
  }

  function init() {
    fetch('data/llm_rollouts.json', { cache: 'no-cache' })
      .then((r) => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then((p) => {
        payload = p;
        buildSelectors(p);
        refreshTrackHint();
        setStatus('ready', 'ready');
        const tickPillTxt = `tick 0/${p.max_ticks}`;
        tickPill.textContent = tickPillTxt;
      })
      .catch((err) => {
        setStatus('rollouts unavailable', 'error');
        playBtn.disabled = true;
        setSummary(`Could not load <code>data/llm_rollouts.json</code> (${escapeHtml(err.message)}). Run <code>python gen_rollouts.py</code> from the repo root, or rerun the Colab notebook section 6c, then reload.`);
      });
  }

  trackSel.addEventListener('change', refreshTrackHint);
  playBtn.addEventListener('click', play);
  stopBtn.addEventListener('click', stop);

  init();
})();
