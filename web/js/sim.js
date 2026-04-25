/* ============================================================
   CITYNEXUS — simulation engine (port of the Python core to JS)
   Engine + 5 heuristic agents + adversarial scenarios +
   per-agent rewards + city score + persistent memory.
   ============================================================ */

(function (global) {
  'use strict';

  // ---------- utilities ----------
  function mulberry32(seed) {
    let t = seed | 0;
    return function () {
      t = (t + 0x6d2b79f5) | 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function manhattan(a, b) { return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]); }
  function shuffleInPlace(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  function neighbors4(x, y, w, h) {
    const out = [];
    if (x + 1 < w) out.push([x + 1, y]);
    if (x > 0)     out.push([x - 1, y]);
    if (y + 1 < h) out.push([x, y + 1]);
    if (y > 0)     out.push([x, y - 1]);
    return out;
  }

  // ---------- world generation ----------
  const ROAD_SPACING = 4;

  function generateWorld(seed, width, height) {
    const rng = mulberry32(seed);
    const cells = [];
    for (let y = 0; y < height; y++) {
      const row = [];
      for (let x = 0; x < width; x++) {
        if (x % ROAD_SPACING === 0 || y % ROAD_SPACING === 0) row.push('road');
        else row.push('empty');
      }
      cells.push(row);
    }
    const empties = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (cells[y][x] === 'empty') empties.push([x, y]);
      }
    }
    shuffleInPlace(empties, rng);
    if (empties.length > 0) {
      const [hx, hy] = empties[0];
      cells[hy][hx] = 'hospital';
    }
    for (let i = 1; i < empties.length; i++) {
      const r = rng();
      const [x, y] = empties[i];
      if (r < 0.55)      cells[y][x] = 'residential';
      else if (r < 0.85) cells[y][x] = 'commercial';
      else               cells[y][x] = 'industrial';
    }
    const traffic = [];
    for (let y = 0; y < height; y++) {
      const row = new Float32Array(width);
      for (let x = 0; x < width; x++) row[x] = (cells[y][x] === 'road') ? 0.05 : -1;
      traffic.push(row);
    }
    return { cells, traffic, width, height };
  }

  // ---------- weather ----------
  const WEATHER_TRANSITIONS = {
    clear: [['clear', 0.88], ['rain', 0.10], ['storm', 0.02]],
    rain:  [['rain', 0.60], ['clear', 0.30], ['storm', 0.10]],
    storm: [['storm', 0.50], ['rain', 0.45], ['clear', 0.05]],
  };
  const WEATHER_CAPACITY = { clear: 1.0, rain: 0.7, storm: 0.4 };
  const WEATHER_ACC_RATE = { clear: 0.003, rain: 0.012, storm: 0.035 };

  function stepWeather(curr, rng) {
    const trans = WEATHER_TRANSITIONS[curr] || WEATHER_TRANSITIONS.clear;
    let r = rng();
    for (const [next, p] of trans) {
      if (r < p) return next;
      r -= p;
    }
    return curr;
  }

  // ---------- traffic dynamics ----------
  function timeOfDayDemand(hour) {
    if (hour >= 7 && hour <= 9)   return [0.35, 0.05]; // morning rush
    if (hour >= 17 && hour <= 19) return [0.05, 0.35]; // evening rush
    if (hour >= 10 && hour <= 16) return [0.10, 0.15]; // daytime
    return [0.03, 0.02];                               // night
  }

  function stepTraffic(world, weather, accidents, roadblocks, hour) {
    const W = world.width, H = world.height;
    const decay = 0.85, diffusion = 0.15;
    const cap = WEATHER_CAPACITY[weather] || 1;
    const [resSrc, comSrc] = timeOfDayDemand(hour);

    // 1. decay
    const next = [];
    for (let y = 0; y < H; y++) {
      const row = new Float32Array(W);
      for (let x = 0; x < W; x++) {
        row[x] = (world.cells[y][x] === 'road') ? world.traffic[y][x] * decay : -1;
      }
      next.push(row);
    }
    // 2. source from adjacent zones
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (world.cells[y][x] !== 'road') continue;
        const ns = neighbors4(x, y, W, H);
        for (let i = 0; i < ns.length; i++) {
          const z = world.cells[ns[i][1]][ns[i][0]];
          let s = 0;
          if      (z === 'residential') s = resSrc;
          else if (z === 'commercial')  s = comSrc;
          else if (z === 'industrial')  s = 0.08;
          else if (z === 'hospital')    s = 0.05;
          if (s) next[y][x] += s * cap;
        }
      }
    }
    // 3. diffusion (4-neighbour averaging on roads only)
    const base = next.map(row => Float32Array.from(row));
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (world.cells[y][x] !== 'road') continue;
        const ns = neighbors4(x, y, W, H);
        let sum = 0, count = 0;
        for (let i = 0; i < ns.length; i++) {
          if (world.cells[ns[i][1]][ns[i][0]] === 'road') {
            sum += base[ns[i][1]][ns[i][0]];
            count++;
          }
        }
        if (count) {
          const avg = sum / count;
          next[y][x] += diffusion * (avg - base[y][x]);
        }
      }
    }
    // 4. block effects
    const blocked = new Set();
    for (let i = 0; i < accidents.length; i++)  blocked.add(accidents[i].x + ',' + accidents[i].y);
    for (let i = 0; i < roadblocks.length; i++) blocked.add(roadblocks[i].x + ',' + roadblocks[i].y);
    blocked.forEach(key => {
      const [bx, by] = key.split(',').map(Number);
      if (world.cells[by] && world.cells[by][bx] === 'road') {
        next[by][bx] = 0;
        const ns = neighbors4(bx, by, W, H);
        for (let i = 0; i < ns.length; i++) {
          const [nx, ny] = ns[i];
          if (world.cells[ny][nx] === 'road' && !blocked.has(nx + ',' + ny)) {
            next[ny][nx] = Math.min(1, next[ny][nx] + 0.2);
          }
        }
      }
    });
    // 5. clamp
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (world.cells[y][x] === 'road') {
          next[y][x] = clamp(next[y][x], 0, 1);
        }
      }
    }
    return next;
  }

  // ---------- accidents ----------
  function spawnAccidents(traffic, weather, world, tick, rng) {
    const rate = WEATHER_ACC_RATE[weather] || 0.003;
    const out = [];
    for (let y = 0; y < world.height; y++) {
      for (let x = 0; x < world.width; x++) {
        if (world.cells[y][x] !== 'road') continue;
        const d = traffic[y][x];
        if (d < 0) continue;
        const p = rate * (0.3 + 0.7 * d);
        if (rng() < p) {
          const sevR = rng();
          const severity = sevR < 0.6 ? 1 : (sevR < 0.9 ? 2 : 3);
          out.push({ x, y, severity, ttl: severity * 4, spawnTick: tick });
        }
      }
    }
    return out;
  }
  function decayAccidents(accidents) {
    const out = [];
    for (let i = 0; i < accidents.length; i++) {
      const a = accidents[i];
      if (a.ttl - 1 > 0) out.push({ ...a, ttl: a.ttl - 1 });
    }
    return out;
  }
  function decayRoadblocks(roadblocks) {
    const out = [];
    for (let i = 0; i < roadblocks.length; i++) {
      const r = roadblocks[i];
      if (r.ttl == null) out.push(r);
      else if (r.ttl - 1 > 0) out.push({ ...r, ttl: r.ttl - 1 });
    }
    return out;
  }

  // ---------- BFS routing ----------
  function bfsRoute(world, blocked, start, goal) {
    const W = world.width, H = world.height;
    if (!isRoadFree(world, blocked, start) || !isRoadFree(world, blocked, goal)) return null;
    const parents = new Map();
    const startKey = start[0] + ',' + start[1];
    const goalKey = goal[0] + ',' + goal[1];
    parents.set(startKey, null);
    const queue = [start];
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      const ck = cur[0] + ',' + cur[1];
      if (ck === goalKey) break;
      const ns = neighbors4(cur[0], cur[1], W, H);
      for (let i = 0; i < ns.length; i++) {
        const [nx, ny] = ns[i];
        const nk = nx + ',' + ny;
        if (parents.has(nk)) continue;
        if (world.cells[ny][nx] !== 'road') continue;
        if (blocked.has(nk)) continue;
        parents.set(nk, ck);
        queue.push([nx, ny]);
      }
    }
    if (!parents.has(goalKey)) return null;
    const path = [];
    let cur = goalKey;
    while (cur !== null) {
      const [x, y] = cur.split(',').map(Number);
      path.unshift([x, y]);
      cur = parents.get(cur);
    }
    return path;
  }
  function isRoadFree(world, blocked, p) {
    return world.cells[p[1]] && world.cells[p[1]][p[0]] === 'road' && !blocked.has(p[0] + ',' + p[1]);
  }
  // BFS pathfinder for MOBILE UNITS (ambulance/police/van).
  // Start and goal may be any cell type (units live at hospitals, get dispatched
  // to building incidents). Intermediate cells must be ROADS — units never
  // teleport through buildings. Returns full path including start and goal,
  // or null if no road path exists.
  function findUnitPath(world, start, goal) {
    const W = world.width, H = world.height;
    if (start[0] === goal[0] && start[1] === goal[1]) return [start];
    const startKey = start[0] + ',' + start[1];
    const goalKey  = goal[0]  + ',' + goal[1];
    const parents  = new Map();
    parents.set(startKey, null);
    const queue = [start];
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      const ns = neighbors4(cur[0], cur[1], W, H);
      for (let i = 0; i < ns.length; i++) {
        const nx = ns[i][0], ny = ns[i][1];
        const nk = nx + ',' + ny;
        if (parents.has(nk)) continue;
        const isGoal = (nk === goalKey);
        const isRoad = world.cells[ny][nx] === 'road';
        // Only traverse roads — except the goal cell itself, which can be
        // a building (incident, delivery dest) or hospital (returning home).
        if (!isGoal && !isRoad) continue;
        parents.set(nk, cur[0] + ',' + cur[1]);
        if (isGoal) {
          // reconstruct
          const path = [[nx, ny]];
          let pk = parents.get(nk);
          while (pk !== null) {
            const [px, py] = pk.split(',').map(Number);
            path.unshift([px, py]);
            pk = parents.get(pk);
          }
          return path;
        }
        queue.push([nx, ny]);
      }
    }
    return null;
  }

  // BFS from any cell to the nearest ROAD cell. Returns [x, y] or null.
  // Used both for spawning unit homes and for snapping building targets to roads.
  function nearestRoadCell(world, point) {
    const x0 = point[0], y0 = point[1];
    if (world.cells[y0] && world.cells[y0][x0] === 'road') return [x0, y0];
    const visited = new Set([x0 + ',' + y0]);
    const queue = [[x0, y0]];
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      const ns = neighbors4(cur[0], cur[1], world.width, world.height);
      for (let i = 0; i < ns.length; i++) {
        const nx = ns[i][0], ny = ns[i][1];
        const k = nx + ',' + ny;
        if (visited.has(k)) continue;
        visited.add(k);
        if (world.cells[ny] && world.cells[ny][nx] === 'road') return [nx, ny];
        queue.push([nx, ny]);
      }
      if (visited.size > 400) break;
    }
    return null;
  }

  function nearestRoad(world, blocked, target) {
    if (isRoadFree(world, blocked, target)) return target;
    let best = null, bestD = Infinity;
    for (let y = 0; y < world.height; y++) {
      for (let x = 0; x < world.width; x++) {
        if (world.cells[y][x] !== 'road') continue;
        if (blocked.has(x + ',' + y)) continue;
        const d = Math.abs(x - target[0]) + Math.abs(y - target[1]);
        if (d < bestD) { bestD = d; best = [x, y]; }
      }
    }
    return best;
  }

  // ---------- agents (heuristic) ----------
  function deliveryAgent(state) {
    const actions = [];
    const blocked = new Set();
    for (const a of state.accidents)  blocked.add(a.x + ',' + a.y);
    for (const r of state.roadblocks) blocked.add(r.x + ',' + r.y);
    if (state.useMemory) {
      for (const z of state.memory.zones) {
        if (z.risk >= 0.85) blocked.add(z.x + ',' + z.y);
      }
    }
    for (const d of state.deliveries) {
      if (d.status !== 'pending') continue;
      const start = nearestRoad(state.world, blocked, d.origin);
      const end   = nearestRoad(state.world, blocked, d.dest);
      if (!start || !end) continue;
      let path = bfsRoute(state.world, blocked, start, end);
      if (!path) {
        // memory fallback: try without memory cells
        const noMemBlocked = new Set();
        for (const a of state.accidents)  noMemBlocked.add(a.x + ',' + a.y);
        for (const r of state.roadblocks) noMemBlocked.add(r.x + ',' + r.y);
        path = bfsRoute(state.world, noMemBlocked, start, end);
      }
      if (path) actions.push({ kind: 'assign_route', deliveryId: d.id, path });
    }
    return actions;
  }

  function trafficAgent(state) {
    const actions = [];
    const W = state.world.width, H = state.world.height;
    const hot = [];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (state.world.cells[y][x] !== 'road') continue;
        let n = 0;
        const ns = neighbors4(x, y, W, H);
        for (let i = 0; i < ns.length; i++) {
          if (state.world.cells[ns[i][1]][ns[i][0]] === 'road') n++;
        }
        if (n >= 3) {
          const t = state.traffic[y][x];
          if (t > 0.6) hot.push({ x, y, t });
        }
      }
    }
    hot.sort((a, b) => b.t - a.t);
    if (hot.length > 0) {
      actions.push({ kind: 'advisory', cells: hot.slice(0, 5).map(h => [h.x, h.y]) });
    }
    return actions;
  }

  function emergencyAgent(state) {
    const actions = [];
    const accs = state.accidents.slice().sort((a, b) => b.severity - a.severity);
    const idle = state.units.filter(u => u.kind === 'ambulance' && u.status === 'idle');
    const targeted = new Set();
    for (const u of state.units) {
      if (u.kind === 'ambulance' && u.target) targeted.add(u.target[0] + ',' + u.target[1]);
    }
    let avail = idle.slice();
    for (const a of accs) {
      const key = a.x + ',' + a.y;
      if (targeted.has(key)) continue;
      if (avail.length === 0) break;
      let best = null, bestD = Infinity;
      for (const u of avail) {
        const d = Math.abs(u.pos[0] - a.x) + Math.abs(u.pos[1] - a.y);
        if (d < bestD) { bestD = d; best = u; }
      }
      if (best) {
        actions.push({ kind: 'dispatch_ambulance', unitId: best.id, target: [a.x, a.y] });
        avail = avail.filter(u => u.id !== best.id);
        targeted.add(key);
      }
    }
    return actions;
  }

  function policeAgent(state) {
    const actions = [];
    const incs = state.incidents.slice().sort((a, b) => b.severity - a.severity);
    const idle = state.units.filter(u => u.kind === 'police' && u.status === 'idle');
    let avail = idle.slice();
    for (const inc of incs) {
      if (inc.assigned) continue;
      if (avail.length === 0) break;
      let best = null, bestD = Infinity;
      for (const u of avail) {
        const d = Math.abs(u.pos[0] - inc.pos[0]) + Math.abs(u.pos[1] - inc.pos[1]);
        if (d < bestD) { bestD = d; best = u; }
      }
      if (best) {
        actions.push({ kind: 'dispatch_police', unitId: best.id, incidentId: inc.id, target: [inc.pos[0], inc.pos[1]] });
        inc.assigned = best.id;
        avail = avail.filter(u => u.id !== best.id);
      }
    }
    return actions;
  }

  function plannerAgent(state) {
    const actions = [];
    const target = { delivery: 1, traffic: 1, emergency: 1, police: 1, planner: 1 };
    if (state.accidents.length >= 3) target.emergency = 2.0;
    if (state.incidents.length >= 2) target.police = 1.5;
    const cong = computeCongestion(state.traffic, state.world);
    if (cong > 0.5) target.traffic = 1.5;
    const openD = state.deliveries.filter(d => d.status === 'pending' || d.status === 'en_route').length;
    if (openD >= 5) target.delivery = 1.5;

    // Memory anticipation: recurring failure modes lead to pre-emptive raises.
    if (state.useMemory && state.memory && state.memory.failureCounts) {
      if ((state.memory.failureCounts.emergency || 0) >= 3) target.emergency = Math.max(target.emergency, 1.5);
      if ((state.memory.failureCounts.traffic || 0) >= 3)   target.traffic   = Math.max(target.traffic, 1.5);
    }

    for (const role of ['delivery', 'traffic', 'emergency', 'police']) {
      if (Math.abs((state.priorities[role] || 1) - target[role]) > 1e-6) {
        actions.push({ kind: 'set_priority', role, value: target[role] });
      }
    }
    return actions;
  }

  function computeCongestion(traffic, world) {
    let n = 0, hot = 0;
    for (let y = 0; y < world.height; y++) {
      for (let x = 0; x < world.width; x++) {
        if (world.cells[y][x] === 'road') {
          n++;
          if (traffic[y][x] > 0.7) hot++;
        }
      }
    }
    return n > 0 ? hot / n : 0;
  }

  // ---------- adversarial scenarios ----------
  function generateScenario(seed, difficulty, episodeLen, gridSize) {
    const rng = mulberry32(seed);
    const nShocks = Math.round(2 + difficulty * 8);
    const shocks = [];
    const kinds = ['traffic_spike', 'emergency_cluster', 'blocked_routes', 'weather_storm', 'incident_surge'];
    const weights = {
      traffic_spike:     1.0 + 0.5 * difficulty,
      emergency_cluster: 0.7 + 1.0 * difficulty,
      blocked_routes:    1.0 + 0.4 * difficulty,
      weather_storm:     0.4 + 0.6 * difficulty,
      incident_surge:    0.6 + 0.8 * difficulty,
    };
    const weightTotal = Object.values(weights).reduce((s, v) => s + v, 0);
    for (let i = 0; i < nShocks; i++) {
      let r = rng() * weightTotal, kind = kinds[0];
      for (const k of kinds) {
        if (r < weights[k]) { kind = k; break; }
        r -= weights[k];
      }
      const tick = Math.max(5, Math.floor(rng() * (episodeLen * 0.85)));
      const center = [Math.floor(rng() * gridSize), Math.floor(rng() * gridSize)];
      if (kind === 'traffic_spike') {
        shocks.push({ tick, kind, center, radius: Math.floor(2 + difficulty * 3), magnitude: 0.4 + 0.5 * difficulty });
      } else if (kind === 'emergency_cluster') {
        shocks.push({ tick, kind, center, radius: Math.floor(2 + difficulty * 3),
                      count: Math.floor(2 + difficulty * 4), severity: 1 + Math.round(difficulty * 2) });
      } else if (kind === 'blocked_routes') {
        const coords = [];
        const n = Math.floor(2 + difficulty * 6);
        for (let j = 0; j < n; j++) coords.push([Math.floor(rng() * gridSize), Math.floor(rng() * gridSize)]);
        shocks.push({ tick, kind, coords, ttl: Math.floor(8 + difficulty * 12) });
      } else if (kind === 'weather_storm') {
        shocks.push({ tick, kind, target: difficulty > 0.5 ? 'storm' : 'rain', duration: Math.floor(5 + difficulty * 15) });
      } else if (kind === 'incident_surge') {
        shocks.push({ tick, kind, count: Math.floor(2 + difficulty * 4), severity: 1 + Math.round(difficulty * 2) });
      }
    }
    shocks.sort((a, b) => a.tick - b.tick);
    return { shocks, difficulty, seed };
  }

  function applyShock(state, shock) {
    if (shock.kind === 'traffic_spike') {
      let n = 0;
      for (let dy = -shock.radius; dy <= shock.radius; dy++) {
        for (let dx = -shock.radius; dx <= shock.radius; dx++) {
          if (Math.abs(dx) + Math.abs(dy) > shock.radius) continue;
          const x = shock.center[0] + dx, y = shock.center[1] + dy;
          if (state.world.cells[y] && state.world.cells[y][x] === 'road') {
            state.traffic[y][x] = Math.min(1, state.traffic[y][x] + shock.magnitude);
            n++;
          }
        }
      }
      return `traffic spike ×${n} cells around (${shock.center[0]},${shock.center[1]})`;
    }
    if (shock.kind === 'emergency_cluster') {
      const candidates = [];
      for (let dy = -shock.radius; dy <= shock.radius; dy++) {
        for (let dx = -shock.radius; dx <= shock.radius; dx++) {
          if (Math.abs(dx) + Math.abs(dy) > shock.radius) continue;
          const x = shock.center[0] + dx, y = shock.center[1] + dy;
          if (state.world.cells[y] && state.world.cells[y][x] === 'road') candidates.push([x, y]);
        }
      }
      shuffleInPlace(candidates, state.rng);
      let n = 0;
      for (const [x, y] of candidates.slice(0, shock.count)) {
        state.accidents.push({ x, y, severity: shock.severity, ttl: shock.severity * 4, spawnTick: state.tick, fromShock: true });
        n++;
      }
      return `${n} accidents spawned around (${shock.center[0]},${shock.center[1]}), sev=${shock.severity}`;
    }
    if (shock.kind === 'blocked_routes') {
      let n = 0;
      for (const [x, y] of shock.coords) {
        if (state.world.cells[y] && state.world.cells[y][x] === 'road') {
          state.roadblocks.push({ x, y, ttl: shock.ttl, reason: 'shock' });
          n++;
        }
      }
      return `${n} roadblocks placed (ttl=${shock.ttl})`;
    }
    if (shock.kind === 'weather_storm') {
      state.weather = shock.target;
      state.weatherLockUntil = state.tick + shock.duration;
      return `weather forced to ${shock.target} for ${shock.duration} ticks`;
    }
    if (shock.kind === 'incident_surge') {
      const pool = [];
      for (let y = 0; y < state.world.height; y++) {
        for (let x = 0; x < state.world.width; x++) {
          const z = state.world.cells[y][x];
          if (z === 'commercial' || z === 'residential') pool.push([x, y]);
        }
      }
      shuffleInPlace(pool, state.rng);
      const kinds = ['protest', 'theft', 'disturbance'];
      let n = 0;
      for (const [x, y] of pool.slice(0, shock.count)) {
        state.incidents.push({
          id: 'inc-shock-' + (state.nextIncidentId++),
          pos: [x, y],
          kind: kinds[Math.floor(state.rng() * kinds.length)],
          severity: shock.severity,
          ttl: 8 + Math.floor(state.rng() * 12),
          spawnTick: state.tick,
          assigned: null,
        });
        n++;
      }
      return `${n} incidents spawned`;
    }
    return shock.kind;
  }

  // ---------- memory (localStorage) ----------
  const MEMORY_KEY = 'citynexus.memory.v1';

  function loadMemory() {
    if (typeof localStorage === 'undefined') return emptyMemory();
    try {
      const raw = localStorage.getItem(MEMORY_KEY);
      if (!raw) return emptyMemory();
      const m = JSON.parse(raw);
      if (!m.zones) m.zones = [];
      if (!m.failureCounts) m.failureCounts = {};
      return m;
    } catch (e) {
      return emptyMemory();
    }
  }
  function emptyMemory() { return { zones: [], failureCounts: {} }; }
  function saveMemory(memory) {
    if (typeof localStorage === 'undefined') return;
    try {
      localStorage.setItem(MEMORY_KEY, JSON.stringify(memory));
    } catch (e) {}
  }
  function clearMemory() {
    if (typeof localStorage === 'undefined') return;
    try { localStorage.removeItem(MEMORY_KEY); } catch (e) {}
  }
  function noteAccidentMemory(memory, x, y, severity, tick) {
    let z = null;
    for (let i = 0; i < memory.zones.length; i++) {
      if (memory.zones[i].x === x && memory.zones[i].y === y) { z = memory.zones[i]; break; }
    }
    if (z) {
      z.samples += 1;
      z.risk = Math.min(1.0, z.risk + severity * 0.10);
      z.timestamp = tick;
      z.factor = 'accident';
    } else {
      memory.zones.push({ x, y, risk: Math.min(1.0, severity * 0.10), samples: 1, factor: 'accident', timestamp: tick });
    }
    if (memory.zones.length > 60) {
      memory.zones.sort((a, b) => b.risk - a.risk);
      memory.zones.length = 60;
    }
  }
  function noteFailureMemory(memory, role) {
    memory.failureCounts[role] = (memory.failureCounts[role] || 0) + 1;
  }

  // ---------- rewards ----------
  function computeRewards(state, prevTraffic, prevAccCount) {
    const rewards = { delivery: 0, traffic: 0, emergency: 0, police: 0, planner: 0 };
    const components = { delivery: {}, traffic: {}, emergency: {}, police: {}, planner: {} };

    // Delivery completions / failures this tick
    let completed = 0, failed = 0;
    for (const d of state.deliveries) {
      if (d.completedTick === state.tick) completed++;
      if (d.failedTick === state.tick)    failed++;
    }
    if (completed) {
      rewards.delivery += completed;
      components.delivery.completion = completed;
    }
    if (failed) {
      rewards.delivery += failed * -0.5;
      components.delivery.failure = failed * -0.5;
      if (state.useMemory) noteFailureMemory(state.memory, 'delivery');
    }

    // Accident clearances
    if (state.accidentsClearedThisTick > 0) {
      rewards.emergency += 0.5 * state.accidentsClearedThisTick;
      components.emergency.clearance = 0.5 * state.accidentsClearedThisTick;
    }

    // Idle penalty for emergency when accidents exist
    const idleAmb = state.units.filter(u => u.kind === 'ambulance' && u.status === 'idle').length;
    if (state.accidents.length > 0 && idleAmb > 0) {
      const v = -0.05 * idleAmb;
      rewards.emergency += v;
      components.emergency.idle = v;
    }

    // Congestion delta
    const prevCong = computeCongestion(prevTraffic, state.world);
    const currCong = computeCongestion(state.traffic, state.world);
    if (currCong < prevCong) {
      const v = 0.30 * (prevCong - currCong);
      rewards.traffic += v;
      components.traffic.drop = v;
    } else if (currCong > prevCong) {
      const v = -0.20 * (currCong - prevCong);
      rewards.traffic += v;
      components.traffic.rise = v;
      if (currCong > 0.5 && state.useMemory) noteFailureMemory(state.memory, 'traffic');
    }

    // Collision penalty: new accidents at high-traffic cells
    let colPen = 0;
    for (const a of state.accidents) {
      if (a.spawnTick === state.tick && !a.fromShock) {
        const d = prevTraffic[a.y] ? prevTraffic[a.y][a.x] : 0;
        if (d > 0.5) colPen += -0.30 * ((d - 0.5) / 0.5);
      }
    }
    if (colPen) {
      rewards.traffic += colPen;
      components.traffic.collision = colPen;
    }

    // Process: per-cell movement progress
    for (const u of state.units) {
      if (!u.target || !u.prevPos) continue;
      const prevDist = Math.abs(u.prevPos[0] - u.target[0]) + Math.abs(u.prevPos[1] - u.target[1]);
      const currDist = Math.abs(u.pos[0] - u.target[0]) + Math.abs(u.pos[1] - u.target[1]);
      const gain = prevDist - currDist;
      if (gain > 0) {
        const role = u.kind === 'ambulance' ? 'emergency' : (u.kind === 'police' ? 'police' : 'delivery');
        const v = 0.05 * gain;
        rewards[role] += v;
        components[role].progress = (components[role].progress || 0) + v;
      }
    }

    // Dispatch intent: emergency dispatched to high-severity accident
    if (state.dispatchIntent) {
      for (const sev of state.dispatchIntent.emergency || []) {
        const v = 0.10 * (sev / 3);
        rewards.emergency += v;
        components.emergency.intent = (components.emergency.intent || 0) + v;
      }
      for (const sev of state.dispatchIntent.police || []) {
        const v = 0.10 * (sev / 3);
        rewards.police += v;
        components.police.intent = (components.police.intent || 0) + v;
      }
    }

    // Planner anticipation
    let antic = 0;
    if ((state.priorities.emergency || 1) > 1 && state.accidents.length >= 2) antic += 0.10;
    if ((state.priorities.police    || 1) > 1 && state.incidents.length >= 1) antic += 0.10;
    if ((state.priorities.traffic   || 1) > 1 && currCong > 0.4)             antic += 0.10;
    if (antic) {
      rewards.planner += antic;
      components.planner.anticipation = antic;
    }

    // Planner system share
    let positive = 0;
    for (const k of ['delivery', 'traffic', 'emergency', 'police']) {
      if (rewards[k] > 0) positive += rewards[k];
    }
    if (positive > 0) {
      const v = 0.15 * positive;
      rewards.planner += v;
      components.planner.system_share = v;
    }

    return { rewards, components, prevCong, currCong };
  }

  function computeCityScore(state) {
    let completed = 0, failed = 0;
    for (const d of state.deliveries) {
      if (d.status === 'delivered') completed++;
      else if (d.status === 'failed') failed++;
    }
    const deliveryHealth = (completed + failed) === 0 ? 1.0 : completed / (completed + failed);
    const safety = Math.max(0, 1 - Math.min(1, state.accidents.length / 10));
    const cong = computeCongestion(state.traffic, state.world);
    const mobility = Math.max(0, 1 - cong);
    let aligned = 0, expected = 0;
    if (state.accidents.length >= 2) { expected++; if ((state.priorities.emergency || 1) > 1) aligned++; }
    if (state.incidents.length >= 1) { expected++; if ((state.priorities.police    || 1) > 1) aligned++; }
    if (cong > 0.4)                  { expected++; if ((state.priorities.traffic   || 1) > 1) aligned++; }
    const coordination = expected === 0 ? 1.0 : aligned / expected;
    const total = 0.30 * deliveryHealth + 0.30 * safety + 0.25 * mobility + 0.15 * coordination;
    return { total, deliveryHealth, safety, mobility, coordination, peakCong: cong };
  }

  // ---------- coordinator: build initial state + step ----------
  function newState(opts) {
    const seed = opts.seed | 0;
    const difficulty = clamp(opts.difficulty || 0.30, 0, 1);
    const episodeLen = Math.max(20, opts.episodeLen | 0 || 200);
    const useMemory = !!opts.useMemory;
    const gridSize = opts.gridSize || 20;

    const world = generateWorld(seed, gridSize, gridSize);
    const memory = useMemory ? loadMemory() : emptyMemory();

    const hospitals = [], commercials = [], industrials = [];
    for (let y = 0; y < world.height; y++) {
      for (let x = 0; x < world.width; x++) {
        const z = world.cells[y][x];
        if      (z === 'hospital')   hospitals.push([x, y]);
        else if (z === 'commercial') commercials.push([x, y]);
        else if (z === 'industrial') industrials.push([x, y]);
      }
    }
    const ambHomes = hospitals.length > 0 ? hospitals : (commercials.length > 0 ? [commercials[0]] : [[0, 0]]);
    const polHomes = commercials.length > 0 ? commercials : ambHomes;
    const vanHomes = industrials.length > 0 ? industrials : ambHomes;

    let nextUnitId = 1;
    const units = [];
    // Snap a building to its nearest road cell. Units physically live on roads
    // (the building is just their conceptual base — hospital, station, depot).
    // Falls back to the building cell only if no road is reachable.
    function snap(building) {
      return nearestRoadCell(world, building) || building;
    }
    for (let i = 0; i < 3; i++) {
      const h = snap(ambHomes[i % ambHomes.length]);
      units.push({ id: 'amb-' + (nextUnitId++), kind: 'ambulance', home: [h[0], h[1]], pos: [h[0], h[1]], prevPos: [h[0], h[1]], status: 'idle', target: null, assignedTo: null });
    }
    for (let i = 0; i < 3; i++) {
      const h = snap(polHomes[i % polHomes.length]);
      units.push({ id: 'pol-' + (nextUnitId++), kind: 'police',    home: [h[0], h[1]], pos: [h[0], h[1]], prevPos: [h[0], h[1]], status: 'idle', target: null, assignedTo: null });
    }
    for (let i = 0; i < 4; i++) {
      const h = snap(vanHomes[i % vanHomes.length]);
      units.push({ id: 'van-' + (nextUnitId++), kind: 'van',       home: [h[0], h[1]], pos: [h[0], h[1]], prevPos: [h[0], h[1]], status: 'idle', target: null, assignedTo: null });
    }

    return {
      // config
      seed, difficulty, episodeLen, useMemory, gridSize,
      // world + entities
      tick: 0,
      world,
      traffic: world.traffic,
      weather: 'clear',
      weatherLockUntil: null,
      accidents: [],
      roadblocks: [],
      deliveries: [],
      incidents: [],
      units,
      // metadata
      priorities: { delivery: 1, traffic: 1, emergency: 1, police: 1, planner: 1 },
      memory,
      scenario: generateScenario(seed, difficulty, episodeLen, gridSize),
      shocksFired: 0,
      // RNG
      rng: mulberry32(seed * 7 + 13),
      // counters
      nextDeliveryId: 1,
      nextIncidentId: 1,
      // tallies
      cumulative: { delivery: 0, traffic: 0, emergency: 0, police: 0, planner: 0 },
      cityScore: { total: 0, deliveryHealth: 1, safety: 1, mobility: 1, coordination: 1, peakCong: 0 },
      decisions: [],
      messages: [],
      history: [],   // [{tick, city, cumulative, weather, accidents, incidents, peakCong}]
      // per-tick scratch
      accidentsClearedThisTick: 0,
      dispatchIntent: { emergency: [], police: [] },
      // ui-only counters
      peakCongCum: 0,
    };
  }

  function step(state, opts) {
    if (state.tick >= state.episodeLen) return state;
    opts = opts || {};

    // snapshot prev
    const prevTraffic = state.traffic;
    const prevAccidentCount = state.accidents.length;

    // 1. fire scheduled shocks
    state.accidentsClearedThisTick = 0;
    state.dispatchIntent = { emergency: [], police: [] };
    for (const shock of state.scenario.shocks) {
      if (shock.tick === state.tick && !shock.fired) {
        shock.fired = true;
        const desc = applyShock(state, shock);
        state.shocksFired++;
        state.messages.push({ tick: state.tick, from: 'adversary', kind: shock.kind, body: desc });
      }
    }

    // 2. spawn deliveries / incidents (organic)
    if (state.rng() < 0.30) spawnDelivery(state);
    if (state.rng() < 0.10) spawnIncident(state);

    // 3. agent decisions
    const allActions = {
      delivery:  deliveryAgent(state),
      traffic:   trafficAgent(state),
      emergency: emergencyAgent(state),
      police:    policeAgent(state),
      planner:   plannerAgent(state),
    };

    // 4. apply actions
    for (const a of allActions.delivery)  applyDeliveryAction(state, a);
    for (const a of allActions.traffic)   applyTrafficAction(state, a);
    for (const a of allActions.emergency) applyEmergencyAction(state, a);
    for (const a of allActions.police)    applyPoliceAction(state, a);
    for (const a of allActions.planner)   applyPlannerAction(state, a);

    // 5. move units + resolve arrivals — units route THROUGH ROADS, never
    //    cut diagonally through buildings.
    for (const u of state.units) {
      u.prevPos = [u.pos[0], u.pos[1]];
      if (u.status !== 'en_route' && u.status !== 'returning') continue;
      if (!u.target) continue;

      // Arrived?
      if (u.pos[0] === u.target[0] && u.pos[1] === u.target[1]) {
        if (u.status === 'en_route') {
          resolveUnitArrival(state, u);
          u.status = 'returning';
          u.target = [u.home[0], u.home[1]];
          u.assignedTo = null;
        } else {
          u.status = 'idle';
          u.target = null;
        }
        u.pathCache = null;
        continue;
      }

      // Recompute path if we don't have one, or it's stale, or our position
      // has drifted off it. Cheap enough — BFS over a 20×20 grid.
      const tgtKey = u.target[0] + ',' + u.target[1];
      const head = u.pathCache && u.pathCache[0];
      const stillOnPath = head && head[0] === u.pos[0] && head[1] === u.pos[1] && u.pathCacheTarget === tgtKey;
      if (!stillOnPath) {
        u.pathCache = findUnitPath(state.world, u.pos, u.target);
        u.pathCacheTarget = tgtKey;
      }

      if (u.pathCache && u.pathCache.length > 1) {
        // Step to the next cell on the planned road path.
        u.pos = [u.pathCache[1][0], u.pathCache[1][1]];
        u.pathCache.shift();
      }
      // No path → unit waits in place this tick.
    }

    // 6. update deliveries (deadline)
    for (const d of state.deliveries) {
      if ((d.status === 'pending' || d.status === 'en_route') && state.tick > d.deadlineTick) {
        d.status = 'failed';
        d.failedTick = state.tick;
      }
    }
    // 7. drop resolved/expired incidents
    state.incidents = state.incidents.filter(i => {
      if (i.resolved) return false;
      if (state.tick > i.spawnTick + i.ttl) return false;
      return true;
    });

    // 8. step world dynamics
    if (state.weatherLockUntil != null && state.tick < state.weatherLockUntil) {
      // weather locked
    } else {
      state.weather = stepWeather(state.weather, state.rng);
      state.weatherLockUntil = null;
    }
    const spawned = spawnAccidents(state.traffic, state.weather, state.world, state.tick + 1, state.rng);
    if (state.useMemory) {
      for (const a of spawned) noteAccidentMemory(state.memory, a.x, a.y, a.severity, state.tick);
    }
    state.traffic = stepTraffic(state.world, state.weather, state.accidents, state.roadblocks, (state.tick) % 24);
    state.accidents = decayAccidents(state.accidents).concat(spawned);
    state.roadblocks = decayRoadblocks(state.roadblocks);

    // 9. rewards + city score
    const { rewards, components, currCong } = computeRewards(state, prevTraffic, prevAccidentCount);
    for (const r in rewards) state.cumulative[r] += rewards[r];
    state.cityScore = computeCityScore(state);
    if (currCong > state.peakCongCum) state.peakCongCum = currCong;

    // 10. record history
    state.history.push({
      tick: state.tick,
      city: state.cityScore,
      cumulative: { ...state.cumulative },
      weather: state.weather,
      accidents: state.accidents.length,
      incidents: state.incidents.length,
      tickRewards: rewards,
      tickComponents: components,
    });
    if (state.history.length > 250) state.history.shift();
    if (state.decisions.length > 100) state.decisions = state.decisions.slice(-100);
    if (state.messages.length > 100) state.messages = state.messages.slice(-100);

    state.tick++;

    // periodic memory persist
    if (state.useMemory && state.tick % 25 === 0) saveMemory(state.memory);

    return state;
  }

  function spawnDelivery(state) {
    const origins = [], dests = [];
    for (let y = 0; y < state.world.height; y++) {
      for (let x = 0; x < state.world.width; x++) {
        const z = state.world.cells[y][x];
        if (z === 'commercial' || z === 'industrial') origins.push([x, y]);
        if (z === 'residential' || z === 'commercial') dests.push([x, y]);
      }
    }
    if (origins.length === 0 || dests.length === 0) return;
    const o = origins[Math.floor(state.rng() * origins.length)];
    const d = dests[Math.floor(state.rng() * dests.length)];
    if (o[0] === d[0] && o[1] === d[1]) return;
    state.deliveries.push({
      id: 'd-' + (state.nextDeliveryId++),
      origin: [o[0], o[1]], dest: [d[0], d[1]],
      spawnTick: state.tick, deadlineTick: state.tick + 30,
      status: 'pending', priority: 1,
    });
  }

  function spawnIncident(state) {
    const pool = [];
    for (let y = 0; y < state.world.height; y++) {
      for (let x = 0; x < state.world.width; x++) {
        const z = state.world.cells[y][x];
        if (z === 'commercial' || z === 'residential') pool.push([x, y]);
      }
    }
    if (pool.length === 0) return;
    const p = pool[Math.floor(state.rng() * pool.length)];
    const kinds = ['protest', 'theft', 'disturbance'];
    state.incidents.push({
      id: 'i-' + (state.nextIncidentId++),
      pos: [p[0], p[1]],
      kind: kinds[Math.floor(state.rng() * kinds.length)],
      severity: 1 + Math.floor(state.rng() * 3),
      ttl: 8 + Math.floor(state.rng() * 12),
      spawnTick: state.tick,
      assigned: null,
    });
  }

  function applyDeliveryAction(state, a) {
    if (a.kind !== 'assign_route') return;
    const d = state.deliveries.find(x => x.id === a.deliveryId);
    if (!d || d.status !== 'pending') return;
    d.status = 'en_route';
    d.path = a.path;
    const van = state.units.find(u => u.kind === 'van' && u.status === 'idle');
    if (van) {
      // Snap delivery destination (a building) to its nearest road cell —
      // the van delivers from the curb, never enters the building.
      const tgt = nearestRoadCell(state.world, d.dest) || [d.dest[0], d.dest[1]];
      van.status = 'en_route';
      van.target = tgt;
      van.assignedTo = d.id;
    }
    state.decisions.push({ tick: state.tick, role: 'delivery', action: 'AssignRoute', detail: `${a.deliveryId} (${a.path.length} cells)` });
  }
  function applyTrafficAction(state, a) {
    if (a.kind !== 'advisory') return;
    state.decisions.push({ tick: state.tick, role: 'traffic', action: 'IssueAdvisory', detail: `${a.cells.length} hot cells` });
    state.messages.push({ tick: state.tick, from: 'traffic', kind: 'advisory', body: `${a.cells.length} hot intersections to avoid` });
  }
  function applyEmergencyAction(state, a) {
    if (a.kind !== 'dispatch_ambulance') return;
    const u = state.units.find(x => x.id === a.unitId);
    if (!u || u.status !== 'idle') return;
    u.status = 'en_route';
    u.target = [a.target[0], a.target[1]];
    u.assignedTo = `accident@${a.target[0]},${a.target[1]}`;
    // Track dispatch intent for reward
    const acc = state.accidents.find(x => x.x === a.target[0] && x.y === a.target[1]);
    if (acc) state.dispatchIntent.emergency.push(acc.severity);
    state.decisions.push({ tick: state.tick, role: 'emergency', action: 'DispatchUnit', detail: `${a.unitId} → (${a.target[0]},${a.target[1]})` });
    state.messages.push({ tick: state.tick, from: 'emergency', kind: 'dispatch_notice', body: `${a.unitId} dispatched to (${a.target[0]},${a.target[1]})` });
  }
  function applyPoliceAction(state, a) {
    if (a.kind !== 'dispatch_police') return;
    const u = state.units.find(x => x.id === a.unitId);
    if (!u || u.status !== 'idle') return;
    // Snap incident location (a building) to nearest road cell — police
    // arrives by road and resolves the incident from there.
    const tgt = nearestRoadCell(state.world, [a.target[0], a.target[1]]) || [a.target[0], a.target[1]];
    u.status = 'en_route';
    u.target = tgt;
    u.assignedTo = a.incidentId;
    const inc = state.incidents.find(x => x.id === a.incidentId);
    if (inc) state.dispatchIntent.police.push(inc.severity);
    state.decisions.push({ tick: state.tick, role: 'police', action: 'DispatchUnit', detail: `${a.unitId} → ${a.incidentId}` });
  }
  function applyPlannerAction(state, a) {
    if (a.kind !== 'set_priority') return;
    state.priorities[a.role] = a.value;
    state.decisions.push({ tick: state.tick, role: 'planner', action: 'SetPriority', detail: `${a.role} → ${a.value.toFixed(1)}` });
  }
  function resolveUnitArrival(state, u) {
    if (u.kind === 'ambulance') {
      const idx = state.accidents.findIndex(a => a.x === u.pos[0] && a.y === u.pos[1]);
      if (idx >= 0) {
        state.accidents.splice(idx, 1);
        state.accidentsClearedThisTick++;
      }
    } else if (u.kind === 'police' && u.assignedTo) {
      const inc = state.incidents.find(i => i.id === u.assignedTo);
      if (inc) inc.resolved = true;
    } else if (u.kind === 'van' && u.assignedTo) {
      const d = state.deliveries.find(x => x.id === u.assignedTo);
      if (d) {
        d.status = 'delivered';
        d.completedTick = state.tick;
      }
    }
  }

  // ---------- exports ----------
  global.CityNexus = {
    newState,
    step,
    saveMemory,
    loadMemory,
    clearMemory,
    computeCongestion,
    // utilities the renderer needs
    timeOfDayDemand,
    WEATHER_ACC_RATE,
    WEATHER_CAPACITY,
  };
})(typeof window !== 'undefined' ? window : globalThis);
