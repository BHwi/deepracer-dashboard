'use strict';
/* MiniRacer 대시보드. 외부 라이브러리 없이 동작한다 (인터넷이 없는 환경에서도 열리도록). */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}

const C = { ink: '#232F3E', line: '#FF9900', muted: '#5F6B78', grid: '#DDE3E9', good: '#0B8F6A', bad: '#C8352B', gray: '#8794A1', sky: '#3E86C4' };
const RUN_COLORS = ['#FF9900', '#232F3E', '#0B8F6A', '#C8352B', '#3E86C4', '#8A6D3B'];
const LEVEL_COLORS = ['#232F3E', '#FF9900', '#0B8F6A', '#C8352B', '#3E86C4', '#8A6D3B', '#8794A1'];
const FONT = getComputedStyle(document.body).fontFamily;
const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;

const STATE_KO = { running: '학습 중', finished: '완료', aborted: '비정상 종료', starting: '시작 중', failed: '시작 실패', unknown: '알 수 없음' };
const STOP_KO = { term_cond_max_episodes: '목표 에피소드 수에 도달', term_cond_avg_score: '목표 점수에 도달', stopped_by_user: '사용자가 중지',
  interrupted: '중단됨', max_minutes: '제한 시간 도달', reward_error: '보상함수 오류로 종료' };
const END_KO = { lap_complete: '완주', offtrack: '트랙 이탈', timeout: '시간 초과' };

async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  let r;
  try { r = await fetch(path, opt); } catch (e) { throw new Error('대시보드 서버에 연결할 수 없습니다. 터미널에서 python dashboard.py 가 실행 중인지 확인하세요.'); }
  let data = null;
  try { data = await r.json(); } catch (e) { /* 본문 없음 */ }
  if (!r.ok) throw new Error((data && data.error) || `요청이 실패했습니다 (${r.status})`);
  return data;
}

const num = (v, d = 1) => (v == null || !Number.isFinite(v) ? '–' : v.toFixed(d));
const secs = v => (v == null || !Number.isFinite(v) ? '–' : v.toFixed(2) + '초');
const dur = s => { s = Math.round(s || 0); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const lastOf = a => (a && a.length ? a[a.length - 1] : null);

/* ---------------------------------------------------------------- 차트 */
function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toPrecision(12));
  return t;
}
const tickFmt = v => (v === 0 ? '0' : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? String(+v.toFixed(2)) : String(+v.toPrecision(2)));

class Hub {                         // 같이 움직이는 차트 묶음 (커서와 선택한 iteration 을 공유)
  constructor() { this.charts = []; this.hover = null; this.sel = null; this.best = null; this.onPick = null; }
  add(c) { this.charts.push(c); c.hub = this; return c; }
  setHover(it) { if (this.hover !== it) { this.hover = it; this.draw(); } }
  setSel(it) { this.sel = it; this.draw(); }
  draw() { this.charts.forEach(c => c.draw()); }
}

class Chart {
  constructor(host, opt) {
    this.opt = Object.assign({ height: 96, xLabels: false, yMin: null, yMax: null, fmt: v => num(v, 2), title: '' }, opt);
    this.series = []; this.xMax = 2; this.last = null; this.hub = null;
    this.readEl = h('span', { class: 'pread' });
    this.cv = h('canvas');
    this.cv.style.height = this.opt.height + 'px';
    this.box = h('div', { class: 'panel' }, h('div', { class: 'phead' }, h('span', { class: 'ptitle' }, this.opt.title), this.readEl), this.cv);
    host.append(this.box);
    const pos = e => { const g = this._geom(); return this._it(e.offsetX, g); };
    this.cv.addEventListener('mousemove', e => this.hub && this.hub.setHover(this._clampIt(pos(e))));
    this.cv.addEventListener('mouseleave', () => this.hub && this.hub.setHover(null));
    this.cv.addEventListener('click', e => this.hub && this.hub.onPick && this.last && this.hub.onPick(this._clampIt(pos(e))));
  }
  set(series, xMax, last) { this.series = series; this.xMax = Math.max(2, xMax || 2); this.last = last; this.draw(); }
  _clampIt(it) { return Math.max(1, Math.min(this.last || 1, it)); }
  _geom() {
    const w = this.cv.clientWidth, hh = this.cv.clientHeight, pad = { l: 48, r: 10, t: 8, b: this.opt.xLabels ? 22 : 8 };
    return { w, h: hh, pad, pw: w - pad.l - pad.r, ph: hh - pad.t - pad.b };
  }
  _x(it, g) { return g.pad.l + (it - 1) / (this.xMax - 1) * g.pw; }
  _it(px, g) { return Math.round(1 + (px - g.pad.l) / g.pw * (this.xMax - 1)); }
  _yrange() {
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series) for (const v of s.y) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!Number.isFinite(lo)) return null;
    const { yMin, yMax } = this.opt;
    if (yMin != null) lo = yMin;
    if (yMax != null) hi = yMax;
    if (hi - lo < 1e-9) { const d = Math.abs(hi) * 0.1 || 1; lo -= d; hi += d; }
    else { const p = (hi - lo) * 0.08; if (yMin == null) lo -= p; if (yMax == null) hi += p; }
    return [lo, hi];
  }
  draw() {
    const cv = this.cv, dpr = window.devicePixelRatio || 1, g = this._geom();
    if (!g.w) return;
    if (cv.width !== Math.round(g.w * dpr) || cv.height !== Math.round(g.h * dpr)) { cv.width = Math.round(g.w * dpr); cv.height = Math.round(g.h * dpr); }
    const c = cv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, g.w, g.h);
    c.font = '11.5px ' + FONT;
    const yr = this._yrange();
    this._readout();
    if (!yr) { c.fillStyle = C.muted; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(this.opt.empty || '첫 iteration 이 끝나면 표시됩니다', g.pad.l, g.h / 2); return; }
    const [lo, hi] = yr, Y = v => g.pad.t + (1 - (v - lo) / (hi - lo)) * g.ph;
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const t of niceTicks(lo, hi, 3)) {
      c.strokeStyle = C.grid; c.lineWidth = 1; c.beginPath(); c.moveTo(g.pad.l, Math.round(Y(t)) + 0.5); c.lineTo(g.w - g.pad.r, Math.round(Y(t)) + 0.5); c.stroke();
      c.fillStyle = C.muted; c.fillText(tickFmt(t), g.pad.l - 7, Y(t));
    }
    if (this.opt.xLabels) {
      c.textAlign = 'center'; c.textBaseline = 'top'; c.fillStyle = C.muted;
      for (const t of niceTicks(1, this.xMax, 5)) c.fillText(String(t), this._x(t, g), g.h - g.pad.b + 6);
    }
    const hub = this.hub;
    if (hub && hub.best != null && hub.best <= this.xMax) {          // best iteration 표시 (아래쪽 주황 눈금)
      const x = this._x(hub.best, g); c.fillStyle = C.line; c.fillRect(x - 1.5, g.pad.t + g.ph - 6, 3, 6);
    }
    for (const s of this.series) {
      c.strokeStyle = s.color; c.fillStyle = s.color; c.lineWidth = s.w || 1.8; c.setLineDash(s.dash || []);
      c.beginPath(); let pen = false;
      for (let i = 0; i < s.x.length; i++) {
        if (!Number.isFinite(s.y[i])) { pen = false; continue; }
        const x = this._x(s.x[i], g), y = Y(s.y[i]);
        if (pen) c.lineTo(x, y); else { c.moveTo(x, y); pen = true; }
      }
      c.stroke(); c.setLineDash([]);
      if (s.dots || s.x.length < 3) for (let i = 0; i < s.x.length; i++) if (Number.isFinite(s.y[i])) { c.beginPath(); c.arc(this._x(s.x[i], g), Y(s.y[i]), 2.4, 0, 6.3); c.fill(); }
    }
    const mark = (it, dash, color) => { if (it == null || it < 1 || it > this.xMax) return; const x = Math.round(this._x(it, g)) + 0.5; c.strokeStyle = color; c.lineWidth = 1; c.setLineDash(dash); c.beginPath(); c.moveTo(x, g.pad.t); c.lineTo(x, g.pad.t + g.ph); c.stroke(); c.setLineDash([]); };
    if (hub) { mark(hub.sel, [], 'rgba(35,47,62,.7)'); mark(hub.hover, [3, 3], C.muted); }
  }
  _readout() {
    const hub = this.hub;
    const it = hub && hub.hover != null ? hub.hover : hub && hub.sel != null ? hub.sel : this.last;
    this.readEl.replaceChildren();
    if (it == null) return;
    for (const s of this.series) {
      let v = null, best = Infinity;
      for (let i = 0; i < s.x.length; i++) { const d = Math.abs(s.x[i] - it); if (d < best && Number.isFinite(s.y[i])) { best = d; v = s.y[i]; } }
      if (v == null || best > 2) continue;
      this.readEl.append(h('span', {}, h('i', { style: `background:${s.color}` }), s.label + ' ', h('b', {}, this.opt.fmt(v))));
    }
  }
}

/* ---------------------------------------------------------------- 트랙 그리기 / 지도 */
function transformFor(bounds, w, hh) {
  const [x0, x1, y0, y1] = bounds, s = Math.min(w / (x1 - x0), hh / (y1 - y0));
  return { s, ox: (w - (x1 - x0) * s) / 2 - x0 * s, oy: (hh - (y1 - y0) * s) / 2 + y1 * s };
}
function paintTrack(g, tr, T, w, hh) {
  g.fillStyle = '#00C389'; g.fillRect(0, 0, w, hh);
  const P = pts => { g.moveTo(T.ox + pts[0][0] * T.s, T.oy - pts[0][1] * T.s); for (const p of pts) g.lineTo(T.ox + p[0] * T.s, T.oy - p[1] * T.s); g.closePath(); };
  g.beginPath(); P(tr.outer); P(tr.inner); g.fillStyle = C.ink; g.fill('evenodd');
  g.strokeStyle = '#fff'; g.lineWidth = Math.max(1.5, T.s * 0.025);
  g.beginPath(); P(tr.outer); g.stroke(); g.beginPath(); P(tr.inner); g.stroke();
  g.strokeStyle = C.line; g.lineWidth = Math.max(1, T.s * 0.018); g.setLineDash([T.s * 0.1, T.s * 0.08]);
  g.beginPath(); P(tr.center); g.stroke(); g.setLineDash([]);
  g.strokeStyle = '#fff'; g.lineWidth = Math.max(2, T.s * 0.04);                 // 출발선
  g.beginPath(); g.moveTo(T.ox + tr.inner[0][0] * T.s, T.oy - tr.inner[0][1] * T.s); g.lineTo(T.ox + tr.outer[0][0] * T.s, T.oy - tr.outer[0][1] * T.s); g.stroke();
}
const lerp = (a, b, u) => a + (b - a) * u;
const ramp = u => { u = Math.max(0, Math.min(1, u)); return `rgb(${lerp(127, 255, u) | 0},${lerp(179, 176, u) | 0},${lerp(230, 0, u) | 0})`; };

class MapView {
  constructor(canvas) {
    this.cv = canvas; this.track = null; this.layer = 'replay'; this.eps = []; this.idx = 0; this.t = 0; this.hold = 0;
    this.playing = !REDUCE; this.mul = 2; this.heat = null; this.base = document.createElement('canvas'); this.onEpisode = null; this.vmin = 0; this.vmax = 1;
    this.lastTs = performance.now();
    window.addEventListener('resize', () => this.resize());
    const loop = now => { requestAnimationFrame(loop); const dt = Math.min(0.1, (now - this.lastTs) / 1000); this.lastTs = now; this._tick(dt); };
    requestAnimationFrame(loop);
  }
  setTrack(tr) {
    if (!tr || this.track === tr) return;
    this.track = tr;
    const [x0, x1, y0, y1] = tr.bounds;
    this.cv.style.aspectRatio = `${x1 - x0} / ${y1 - y0}`;
    this.resize();
  }
  resize() {
    if (!this.track) return;
    const dpr = window.devicePixelRatio || 1, w = this.cv.clientWidth, hh = this.cv.clientHeight;
    if (!w || !hh) return;
    this.cv.width = this.base.width = Math.round(w * dpr); this.cv.height = this.base.height = Math.round(hh * dpr);
    this.dpr = dpr; this.W = w; this.H = hh; this.T = transformFor(this.track.bounds, w, hh);
    const g = this.base.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); paintTrack(g, this.track, this.T, w, hh);
    this.draw();
  }
  setLayer(l) { this.layer = l; this.draw(); }
  setHeat(d) { this.heat = d; this.draw(); }
  setReplay(eps) {
    this.eps = eps || [];
    for (const ep of this.eps) {           // 진행 방향은 위치 변화에서 계산
      const hd = []; let prev = 0;
      for (let i = 0; i < ep.x.length; i++) {
        const j = Math.min(i + 1, ep.x.length - 1), k = j === i ? Math.max(0, i - 1) : i;
        const dx = ep.x[j] - ep.x[k], dy = ep.y[j] - ep.y[k];
        if (Math.hypot(dx, dy) > 1e-4) prev = Math.atan2(dy, dx);
        hd.push(prev);
      }
      ep.hd = hd;
    }
    let lo = Infinity, hi = -Infinity;
    for (const ep of this.eps) for (const v of ep.speed) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    this.vmin = 0; this.vmax = Number.isFinite(hi) && hi > 1 ? hi : 1;   // iteration 을 넘나들어도 같은 색이 같은 속도
    if (this.idx >= this.eps.length) this.idx = 0;
    this._restart();
  }
  pick(i) { this.idx = i; this._restart(); }
  _restart() {
    this.t = 0; this.hold = 0;
    if (REDUCE && this.eps[this.idx]) this.t = this.eps[this.idx].x.length - 1;
    if (this.onEpisode) this.onEpisode(this.idx);
    this.draw();
  }
  _tick(dt) {
    if (this.layer !== 'replay' || !this.eps.length || !this.playing || !this.cv.offsetParent) return;
    const n = this.eps[this.idx].x.length;
    if (this.t >= n - 1) { this.hold += dt; if (this.hold > 1.0) { this.t = 0; this.hold = 0; } }
    else this.t = Math.min(n - 1, this.t + dt * 15 * this.mul);
    this.draw();
  }
  draw() {
    if (!this.track || !this.W) return;
    const g = this.cv.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(this.base, 0, 0);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const { s, ox, oy } = this.T;
    if (this.layer === 'heat') {
      const d = this.heat; if (!d || !d.x.length) return;
      const sorted = [...d.r].sort((a, b) => a - b), lo = sorted[0], hi = sorted[sorted.length - 1] || 1;
      for (let i = 0; i < d.x.length; i++) { g.fillStyle = ramp((d.r[i] - lo) / (hi - lo || 1)); g.fillRect(ox + d.x[i] * s - 1.5, oy - d.y[i] * s - 1.5, 3, 3); }
      return;
    }
    const ep = this.eps[this.idx]; if (!ep) return;
    const k = Math.floor(this.t);
    g.lineWidth = Math.max(2.5, s * 0.05); g.lineCap = 'round';
    for (let i = 1; i <= k; i++) {
      g.strokeStyle = ramp((ep.speed[i] - this.vmin) / (this.vmax - this.vmin));
      g.beginPath(); g.moveTo(ox + ep.x[i - 1] * s, oy - ep.y[i - 1] * s); g.lineTo(ox + ep.x[i] * s, oy - ep.y[i] * s); g.stroke();
    }
    const f = this.t - k, i2 = Math.min(k + 1, ep.x.length - 1);
    const x = lerp(ep.x[k], ep.x[i2], f), y = lerp(ep.y[k], ep.y[i2], f);
    g.save(); g.translate(ox + x * s, oy - y * s); g.rotate(-ep.hd[k]);
    const L = 0.5 * s, Wd = 0.3 * s;
    g.fillStyle = C.line; g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.rect(-L / 2, -Wd / 2, L, Wd); g.fill(); g.stroke();
    g.fillStyle = C.ink; g.fillRect(L / 2 - 0.12 * s, -Wd / 2 + 1.5, 0.12 * s, Wd - 3);
    g.restore();
  }
}

/* ---------------------------------------------------------------- 상태 */
const S = {
  info: null, runs: [], tab: 'new', active: null, checked: new Set(), data: null, selIt: null, follow: false, layer: 'replay',
  product: 'mr', last: {}, geom: {}, usage: null, usageAt: 0, heatAt: 0, replayKey: '', pending: null, cmp: {}, drive: null,
};
const HUB = new Hub(), CMP = new Hub();
let MAP, DMAP, CH = {}, CMPCH = [];

async function trackGeom(name) {
  if (!S.geom[name]) S.geom[name] = await api('/api/track?name=' + encodeURIComponent(name));
  return S.geom[name];
}

/* ---------------------------------------------------------------- 탭 */
/* 사이드바 그룹(접고 펼치기)의 열림 상태는 브라우저에 기억한다 */
const GROUP_KEY = 'trainer.groups';
function groupState() { try { return Object.assign({ mr: true, dr: false }, JSON.parse(localStorage.getItem(GROUP_KEY) || '{}')); } catch (e) { return { mr: true, dr: false }; } }
function expandGroup(p, open, save = true) {
  const head = $(`.group[data-group="${p}"] .grouphead`), body = $('#grp-' + p);
  head.setAttribute('aria-expanded', String(open)); body.hidden = !open;
  if (save) { const st = groupState(); st[p] = open; try { localStorage.setItem(GROUP_KEY, JSON.stringify(st)); } catch (e) { /* 저장 불가 */ } }
}
function gotoProduct(p) {
  expandGroup(p, true); expandGroup(p === 'mr' ? 'dr' : 'mr', false);
  setTab(S.last[p] || (p === 'mr' ? (S.runs.length ? 'result' : 'new') : DRUI.defaultTab()));
}

function setTab(tab) {
  if (typeof DRUI !== 'undefined') DRUI.leave(tab);      // 영상 연결을 끊는다 (브라우저의 동시 연결 수를 아낀다)
  S.tab = tab;
  const product = tab.startsWith('dr-') ? 'dr' : 'mr';
  S.product = product; S.last[product] = tab;
  $$('.tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.product === product)));
  $$('.menu button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  if (!$('#grp-' + product).offsetParent) expandGroup(product, true);
  $$('.tab').forEach(s => { s.hidden = s.id !== 'tab-' + tab; });
  if (tab === 'result') { renderResult(); requestAnimationFrame(() => { HUB.draw(); MAP.resize(); }); }
  if (tab === 'drive') renderDriveOptions();
  if (tab === 'compare') refreshCompare();
  if (tab === 'clean') renderClean();
  if (tab === 'new') drawPreview();
  if (product === 'dr') DRUI.enter(tab);
}

/* ---------------------------------------------------------------- 실험 정리 (공통 선택 표, MiniRacer) */
const fmtBytes = n => (n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n > 0 ? Math.max(1, Math.round(n / 1024)) + ' KB' : '0');

/* cols=[{label, get(row)}], locked(row)=선택할 수 없는 이유(없으면 ''). 선택된 이름 목록을 돌려주는 함수를 반환한다. */
function selTable(table, cols, rows, locked, onChange) {
  const boxes = [];
  const all = h('input', { type: 'checkbox', 'aria-label': '전체 선택' });
  all.addEventListener('change', () => { boxes.forEach(([cb]) => { if (!cb.disabled) cb.checked = all.checked; }); onChange(); });
  const body = rows.map(r => {
    const why = locked(r);
    const cb = h('input', { type: 'checkbox', 'aria-label': r.name + ' 선택' });
    if (why) cb.disabled = true;
    cb.addEventListener('change', onChange);
    boxes.push([cb, r]);
    return h('tr', { class: why ? 'locked' : '', title: why || '' }, h('td', {}, cb), ...cols.map(c => h('td', {}, c.get(r))));
  });
  table.replaceChildren(h('thead', {}, h('tr', {}, h('th', {}, all), ...cols.map(c => h('th', {}, c.label)))), h('tbody', {}, ...body));
  return () => boxes.filter(([cb]) => cb.checked).map(([, r]) => r.name);
}

const CLEAN = { sel: () => [], upd: () => {} };
async function renderClean() {
  const msg = $('#cl-msg'); msg.className = 'msg'; msg.textContent = '';
  let rows;
  try { rows = await api('/api/runs/usage'); } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; return; }
  $('#cl-empty').textContent = '삭제할 실험 기록이 없습니다.';
  $('#cl-empty').hidden = rows.length > 0; $('#cl-body').hidden = rows.length === 0;
  if (!rows.length) return;
  CLEAN.upd = () => {
    const n = CLEAN.sel().length, ok = n > 0 && $('#cl-confirm').value.trim() === '삭제';
    $('#cl-go').disabled = !ok; $('#cl-go').textContent = n ? `선택한 ${n}개 삭제` : '선택한 실험 삭제';
  };
  CLEAN.sel = selTable($('#cl-table'), [
    { label: '이름', get: r => r.name }, { label: '트랙', get: r => r.track || '–' }, { label: '상태', get: r => STATE_KO[r.state] || r.state || '–' },
    { label: '최고 랩', get: r => (r.best_lap != null ? secs(r.best_lap) : '–') }, { label: '용량', get: r => fmtBytes(r.bytes) },
  ], rows, r => (r.state === 'running' || r.state === 'starting' ? '학습 중이라 삭제할 수 없습니다' : ''), () => CLEAN.upd());
  $('#cl-confirm').value = ''; CLEAN.upd();
}
function initClean() {
  $('#cl-confirm').addEventListener('input', () => CLEAN.upd());
  $('#cl-go').addEventListener('click', async () => {
    const names = CLEAN.sel(), msg = $('#cl-msg'); msg.className = 'msg'; msg.textContent = '';
    if (!names.length) return;
    $('#cl-go').disabled = true;
    try {
      await api('/api/runs/delete', { names, confirm: true });
      names.forEach(n => S.checked.delete(n));
      await refreshRuns();
      if (S.active && names.includes(S.active)) { S.active = S.runs.length ? S.runs[0].name : null; S.data = null; }
      await renderClean(); const m = $('#cl-msg'); m.className = 'msg ok'; m.textContent = `${names.length}개를 삭제했습니다.`;
    } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; CLEAN.upd(); }
  });
}

/* ---------------------------------------------------------------- 사이드: 실험 기록 */
async function refreshRuns() {
  try { S.runs = await api('/api/runs'); } catch (e) { $('#railnote').textContent = e.message; return; }
  const gone = [...S.checked].filter(n => !S.runs.some(r => r.name === n)); gone.forEach(n => S.checked.delete(n));
  renderRail(); renderLive();
  if (S.tab === 'drive') renderDriveOptions();
}
function renderRail() {
  const ul = $('#runlist'); ul.replaceChildren();
  for (const r of S.runs) {
    const best = r.best_lap != null ? `최고 ${r.best_lap.toFixed(2)}초` : '완주 기록 없음';
    const prog = r.last_progress != null ? `평가 ${Math.min(100, r.last_progress).toFixed(0)}%` : '';
    const cb = h('input', { type: 'checkbox', 'aria-label': r.name + ' 비교에 포함' });
    cb.checked = S.checked.has(r.name);
    cb.addEventListener('click', e => e.stopPropagation());
    cb.addEventListener('change', () => { cb.checked ? S.checked.add(r.name) : S.checked.delete(r.name); updateCompareLabel(); if (S.tab === 'compare') refreshCompare(); });
    const li = h('li', { class: 'run' + (r.name === S.active ? ' active' : ''), tabindex: '0' }, cb,
      h('div', { class: 'top' }, h('span', { class: 'name' }, r.name), h('span', { class: 'mark ' + r.state }, STATE_KO[r.state] || r.state)),
      h('div', { class: 'sub' }, h('span', {}, r.track || ''), prog ? h('span', {}, prog) : null, r.iterations_done ? h('span', {}, best) : null));
    li.addEventListener('click', () => setActive(r.name));
    li.addEventListener('keydown', e => { if (e.key === 'Enter') setActive(r.name); });
    ul.append(li);
  }
  $('#railnote').textContent = S.runs.length ? '체크한 실험은 비교 화면에서 겹쳐 볼 수 있습니다.' : '아직 기록이 없습니다. 새 실험에서 학습을 시작하세요.';
  updateCompareLabel();
}
function updateCompareLabel() {
  $('.menu button[data-tab="compare"]').textContent = S.checked.size ? `비교 (${S.checked.size})` : '비교';
}
function renderLive() {
  const box = $('#live'); box.replaceChildren();
  const r = S.runs.find(x => x.state === 'running');
  if (r) box.append(h('span', { class: 'mark running' }, 'MiniRacer 학습 중'), h('b', {}, r.name), h('span', {}, `iteration ${r.iteration}${r.max_iterations ? ' / ' + r.max_iterations : ''}`), h('span', {}, dur(r.elapsed_s)));
  const n = box.childNodes.length;
  if (typeof DRUI !== 'undefined') DRUI.appendLive(box);
  if (!box.childNodes.length) box.textContent = '실행 중인 학습 없음';
  const pill = $('#mrpill'); if (pill) { pill.textContent = r ? '학습 중' : ''; pill.className = 'pill' + (r ? ' on' : ''); }
}

function setActive(name) {
  S.active = name; S.data = null; S.selIt = null; S.usage = null; S.replayKey = ''; S.usageAt = S.heatAt = 0; S.pending = null;
  renderRail();
  if (!['result', 'drive', 'compare'].includes(S.tab)) setTab('result');
  else if (S.tab === 'result') renderResult();
  loadRun(true);
}

/* ---------------------------------------------------------------- 결과 */
async function loadRun(first) {
  const name = S.active; if (!name) return;
  const row = S.runs.find(r => r.name === name);
  if (row && row.pending) { S.data = null; S.pending = { name, alive: row.state === 'starting', output: '' }; renderResult(); watchJob(name); return; }
  let d;
  try { d = await api('/api/run?name=' + encodeURIComponent(name)); }
  catch (e) {
    if (S.active !== name || S.pending) return;
    S.data = null;
    try { const j = await api('/api/job?name=' + encodeURIComponent(name)); if (j.known) S.pending = { name, alive: j.alive, output: j.output }; } catch (e2) { /* 무시 */ }
    renderResult(e.message); return;
  }
  if (S.active !== name) return;
  const prev = S.data, count = (d.metrics.iteration || []).length, prevCount = prev ? (prev.metrics.iteration || []).length : 0;
  S.data = d; S.pending = null;
  if (first) {
    S.follow = d.state.state === 'running';
    S.selIt = S.follow ? count || null : d.best_iteration || count || null;
  } else if (S.follow && count) S.selIt = count;
  if (first || count !== prevCount || prev.state.state !== d.state.state) renderResult();
  else renderHeader();
}

function renderHeader() {
  const d = S.data; if (!d) return;
  const st = d.state, kind = st.state;
  $('#r-name').textContent = d.name;
  const sEl = $('#r-state'); sEl.textContent = STATE_KO[kind] || kind; sEl.className = 'state ' + kind;
  const bits = [d.info.track];
  if (kind === 'running') bits.push(`iteration ${st.iteration}${st.max_iterations ? ' / ' + st.max_iterations : ''}`, `${st.episodes} 에피소드`, dur(st.elapsed_s) + ' 경과');
  else { if (st.stop_reason) bits.push(STOP_KO[st.stop_reason] || st.stop_reason); bits.push(`${st.iteration} iteration`, dur(st.elapsed_s)); }
  if (d.info.noise) bits.push(`외란 ${d.info.noise}`);
  $('#r-meta').replaceChildren(...bits.map(b => h('span', {}, b)));
  $('#r-stop').hidden = kind !== 'running' && kind !== 'starting';
  $('#logbox').open = kind === 'aborted' || kind === 'failed' || $('#logbox').open;
  $('#r-log').textContent = d.log || '';
}

function renderResult(errText) {
  const empty = $('#res-empty'), body = $('#res-body');
  const d = S.data;
  if (!d) {
    body.hidden = true; empty.hidden = false; empty.replaceChildren();
    if (S.pending) {
      empty.append(h('strong', {}, S.pending.alive ? `${S.pending.name} 학습을 시작하는 중입니다` : `${S.pending.name} 학습을 시작하지 못했습니다`));
      if (S.pending.output) empty.append(h('pre', {}, S.pending.output));
      if (!S.pending.alive) empty.append(h('p', {}, '위 메시지의 원인을 고친 뒤 새 실험에서 다시 시작하세요. 보상함수 오류는 코드의 몇 번째 줄인지 함께 표시됩니다.'));
    } else if (S.active) empty.append(h('strong', {}, '이 실험을 불러오지 못했습니다'), errText || '');
    else empty.append(h('strong', {}, '선택한 실험이 없습니다'), '왼쪽 기록에서 실험을 고르거나, 새 실험에서 학습을 시작하세요.');
    return;
  }
  empty.hidden = true; body.hidden = false;
  renderHeader();
  const m = d.metrics, it = m.iteration || [], last = it.length ? it[it.length - 1] : null;
  const run = d.state.state === 'running';
  const xMax = run && d.state.max_iterations ? Math.max(d.state.max_iterations, last || 1) : last || 2;
  HUB.best = d.best_iteration;
  const X = it;
  const fmtP = v => num(v, 1), fmtL = v => v.toFixed(2) + '초';
  CH.reward.set([{ label: '학습', color: C.gray, x: X, y: m.train_reward || [] }, { label: '평가', color: C.line, w: 2.4, x: X, y: m.eval_reward || [] }], xMax, last);
  CH.prog.set([{ label: '학습', color: C.gray, x: X, y: m.train_progress || [] }, { label: '평가', color: C.line, w: 2.4, x: X, y: m.eval_progress || [] }], xMax, last);
  CH.lap.set([{ label: '평균', color: C.ink, w: 2, dots: true, x: X, y: m.eval_lap_time_mean || [] }, { label: '최고', color: C.good, dash: [4, 3], x: X, y: m.eval_lap_time_best || [] }], xMax, last);
  CH.ent.set([{ label: '엔트로피', color: C.good, x: X, y: m.entropy || [] }], xMax, last);
  CH.kl.set([{ label: 'KL', color: C.bad, x: X, y: m.kl || [] }], xMax, last);
  CH.ploss.set([{ label: '정책 손실', color: C.ink, x: X, y: m.policy_loss || [] }], xMax, last);
  CH.vloss.set([{ label: '가치 손실', color: C.sky, x: X, y: m.value_loss || [] }], xMax, last);
  CH.lap.opt.fmt = fmtL; CH.prog.opt.fmt = fmtP;
  const sc = $('#r-scrub'); sc.max = Math.max(1, last || 1); sc.min = 1;
  if (S.selIt != null) { S.selIt = Math.min(S.selIt, last || 1); sc.value = S.selIt; $('#r-scrub-out').textContent = S.selIt; HUB.setSel(S.selIt); } else HUB.setSel(null);
  $('#r-follow').checked = S.follow;
  renderLedger();
  $('#r-hp').textContent = JSON.stringify(d.hp, null, 2);
  $('#r-code').textContent = d.reward_code;
  $('#r-log').textContent = d.log || '';
  loadMapData();
  maybeUsage(last);
}

async function loadMapData() {
  const d = S.data; if (!d) return;
  try {
    const g = await trackGeom(d.info.track);
    MAP.setTrack(g);
    if (S.layer === 'replay' && S.selIt != null) {
      const key = `${S.active}:${S.selIt}`;
      if (S.replayKey !== key) {
        S.replayKey = key;
        const rp = await api(`/api/replay?name=${encodeURIComponent(S.active)}&iteration=${S.selIt}`);
        if (S.replayKey !== key) return;
        MAP.setReplay(rp.episodes); renderEpButtons(rp);
      }
    } else if (S.layer === 'heat' && Date.now() - S.heatAt > 12000) {
      S.heatAt = Date.now();
      const hd = await api('/api/heat?name=' + encodeURIComponent(S.active)); MAP.setHeat(hd);
      $('#m-info').textContent = hd.x.length ? `학습 후반(iteration ${hd.from_iteration} 이후) 주행 기록에서 뽑은 점입니다. 주황에 가까울수록 그 자리에서 받은 보상이 큽니다. 어느 구간에서 보상을 많이 주는지 확인해 보세요.` : '아직 기록된 학습 주행이 없습니다. 학습 주행 기록은 3 iteration마다 저장됩니다.';
    }
  } catch (e) { $('#m-info').textContent = e.message; }
}

function renderEpButtons(rp) {
  const box = $('#m-eps'); box.replaceChildren();
  rp.episodes.forEach((ep, i) => {
    const b = h('button', { class: i === MAP.idx ? 'on' : '', title: `${END_KO[ep.status] || ep.status}, 진행률 ${ep.progress.toFixed(0)}%` }, `주행 ${i + 1}`);
    b.addEventListener('click', () => { MAP.pick(i); });
    box.append(b);
  });
  S.replayMeta = rp; updateMapInfo(MAP.idx);
}
function updateMapInfo(i) {
  $$('#m-eps button').forEach((b, k) => b.classList.toggle('on', k === i));
  const rp = S.replayMeta; if (S.layer !== 'replay') return;
  const box = $('#m-info'); box.replaceChildren();
  if (!rp || !rp.episodes.length) { box.textContent = '이 iteration 의 평가 주행 기록이 아직 없습니다.'; return; }
  const ep = rp.episodes[i]; if (!ep) return;
  box.append(h('span', {}, h('b', {}, `iteration ${rp.iteration}`), ' 평가 주행'), h('span', {}, h('b', {}, END_KO[ep.status] || ep.status)),
    h('span', {}, `진행률 ${ep.progress.toFixed(0)}%`), ep.lap_time != null ? h('span', {}, `랩타임 ${ep.lap_time.toFixed(2)}초`) : null,
    h('span', {}, `${ep.steps} 스텝`), h('span', {}, '궤적 색: 느림(하늘)에서 빠름(주황), 0에서 최고 속도 기준'));
}

async function maybeUsage(last) {
  if (!S.active || !last) { CH.uspeed.set([], 2, null); CH.usteer.set([], 2, null); return; }
  const running = S.data && S.data.state.state === 'running';
  if (S.usage && S.usage.name === S.active && S.usage.last === last) { applyUsage(); return; }
  if (running && Date.now() - S.usageAt < 12000 && S.usage) { applyUsage(); return; }
  S.usageAt = Date.now();
  const name = S.active;
  try { const u = await api('/api/usage?name=' + encodeURIComponent(name)); if (S.active !== name) return; S.usage = { name, last, u }; applyUsage(); } catch (e) { /* 무시 */ }
}
function applyUsage() {
  const d = S.data, u = S.usage && S.usage.u; if (!d || !u) return;
  const m = d.metrics, last = lastOf(m.iteration);
  const run = d.state.state === 'running', xMax = run && d.state.max_iterations ? Math.max(d.state.max_iterations, last || 1) : last || 2;
  const mk = o => Object.entries(o).map(([k, y], i) => ({ label: k, color: LEVEL_COLORS[i % LEVEL_COLORS.length], x: u.iterations, y, dots: true }));
  CH.uspeed.set(mk(u.speed), xMax, last); CH.usteer.set(mk(u.steer), xMax, last);
}

function renderLedger() {
  const d = S.data, m = d.metrics, it = m.iteration || [], t = $('#ledger'); t.replaceChildren();
  t.append(h('thead', {}, h('tr', {}, ...['iteration', '학습 진행', '학습 완주', '평가 진행', '평가 랩타임', '엔트로피', 'KL'].map(x => h('th', {}, x)))));
  const tb = h('tbody');
  for (let i = it.length - 1; i >= 0; i--) {
    const n = it[i], lap = m.eval_lap_time_mean[i];
    const tr = h('tr', { class: n === S.selIt ? 'sel' : '', 'data-it': n },
      h('td', {}, String(n), n === d.best_iteration ? h('span', { class: 'tag' }, 'best') : null),
      h('td', {}, num(m.train_progress[i]) + '%'), h('td', {}, num(m.train_lap_rate[i], 0) + '%'), h('td', {}, num(Math.min(100, m.eval_progress[i])) + '%'),
      h('td', {}, secs(lap)), h('td', {}, num(m.entropy[i], 2)), h('td', {}, num(m.kl[i], 4)));
    tr.addEventListener('click', () => selectIteration(n, true));
    tb.append(tr);
  }
  t.append(tb);
}

function selectIteration(it, byUser) {
  if (!S.data) return;
  if (byUser) { S.follow = false; $('#r-follow').checked = false; }
  S.selIt = it; $('#r-scrub').value = it; $('#r-scrub-out').textContent = it; HUB.setSel(it);
  $$('#ledger tbody tr').forEach(tr => tr.classList.toggle('sel', +tr.dataset.it === it));
  if (S.layer !== 'replay') setLayer('replay');
  loadMapData();
}
function setLayer(l) {
  S.layer = l; MAP.setLayer(l);
  $$('#layerseg button').forEach(b => b.classList.toggle('on', b.dataset.layer === l));
  $('.mapctl').hidden = l !== 'replay';
  if (l === 'replay') { S.replayKey = ''; updateMapInfo(MAP.idx); }
  loadMapData();
}

/* ---------------------------------------------------------------- 새 실험 */
const HP_FIELDS = [
  ['num_episodes_between_training', '정책 갱신 간격', '정책을 한 번 갱신하기 전에 모으는 에피소드 수. 이 묶음이 1 iteration 입니다.', 1],
  ['term_cond_max_episodes', '총 에피소드', '이만큼 학습하면 끝납니다. 기본 1000이면 약 2분입니다.', 1],
  ['lr', '학습률', '크면 빨리 배우지만 행동이 불안정해질 수 있습니다.', 0.0001],
  ['batch_size', '배치 크기', '한 번에 묶어서 학습하는 샘플 수.', 1],
  ['num_epochs', '반복 횟수', '모은 데이터를 몇 번 되풀이해서 학습할지.', 1],
  ['beta_entropy', '엔트로피 보너스', '클수록 여러 행동을 계속 시도합니다. 너무 작으면 일찍 굳습니다.', 0.001],
  ['discount_factor', '할인율', '1에 가까울수록 먼 미래의 보상까지 중요하게 봅니다.', 0.001],
  ['min_eval_trials', '평가 주행 수', 'iteration마다 평가로 달리는 횟수.', 1],
];
const HP_ADV = [
  ['clip_epsilon', 'PPO 클립 범위', '한 번 갱신에서 정책이 바뀔 수 있는 폭의 한계.', 0.01],
  ['gae_lambda', 'GAE 람다', '보상이 늦게 나타나는 문제를 다루는 정도.', 0.01],
  ['round_robin_advance_dist', '출발 위치 이동', '에피소드마다 출발점을 트랙의 이 비율만큼 옮깁니다.', 0.01],
];
function buildHpGrid(host, fields, hp) {
  for (const [k, ko, hint, step] of fields) {
    host.append(h('label', { class: 'k', for: 'hp-' + k }, ko, h('small', {}, k)), h('input', { id: 'hp-' + k, type: 'number', step: String(step), value: hp[k] }), h('span', { class: 'h' }, hint));
  }
  if (host.id === 'hpgrid') {
    host.append(h('label', { class: 'k', for: 'hp-loss_type' }, '가치 손실 종류', h('small', {}, 'loss_type')),
      h('select', { id: 'hp-loss_type' }, h('option', { value: 'huber' }, 'huber'), h('option', { value: 'mean squared error' }, 'mean squared error')),
      h('span', { class: 'h' }, 'huber 는 가끔 나오는 큰 오차에 덜 민감합니다.'));
    $('#hp-loss_type').value = hp.loss_type;
  }
}
function parseList(s) { return s.split(/[,\s]+/).filter(Boolean).map(Number); }
function updateActCount() {
  const a = parseList($('#f-steer').value), b = parseList($('#f-speed').value);
  const bad = [...a, ...b].some(v => !Number.isFinite(v));
  $('#actcount').textContent = bad ? '숫자만 쉼표로 구분해서 입력하세요.' : `조향각 ${a.length}개 x 속도 ${b.length}개 = 행동 ${a.length * b.length}개`;
}
async function drawPreview() {
  const name = $('#f-track').value, cv = $('#trackPreview'); if (!name) return;
  try {
    const g = await trackGeom(name);
    cv.style.aspectRatio = `${g.bounds[1] - g.bounds[0]} / ${g.bounds[3] - g.bounds[2]}`;
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, hh = cv.clientHeight; if (!w) return;
    cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr);
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    paintTrack(c, g, transformFor(g.bounds, w, hh), w, hh);
    cv.title = `길이 ${g.length.toFixed(1)}m, 폭 ${g.width.toFixed(2)}m`;
  } catch (e) { /* 무시 */ }
}
async function submitNew(ev) {
  ev.preventDefault();
  const msg = $('#newmsg'), btn = $('#f-submit'); msg.className = 'msg'; msg.textContent = '';
  const hp = {};
  for (const [k] of [...HP_FIELDS, ...HP_ADV]) hp[k] = Number($('#hp-' + k).value);
  hp.loss_type = $('#hp-loss_type').value;
  const body = { name: $('#f-name').value.trim(), track: $('#f-track').value, reward_code: $('#f-code').value, hp,
    steer: parseList($('#f-steer').value), speed: parseList($('#f-speed').value), seed: 0 };
  btn.disabled = true;
  try {
    const res = await api('/api/train', body);
    S.pending = { name: res.name, alive: true, output: '' };
    S.active = res.name; S.data = null; S.selIt = null; S.usage = null; S.replayKey = '';
    msg.className = 'msg ok'; msg.textContent = '학습을 시작했습니다.';
    setTab('result'); await refreshRuns(); watchJob(res.name);
  } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
  btn.disabled = false;
}
let jobTimer = null;
function watchJob(name) {
  clearInterval(jobTimer);
  jobTimer = setInterval(async () => {
    if (S.active !== name || !S.pending) { clearInterval(jobTimer); return; }
    try {
      const j = await api('/api/job?name=' + encodeURIComponent(name));
      S.pending = { name, alive: j.alive, output: j.output };
      await refreshRuns();
      if (S.runs.some(r => r.name === name && !r.pending)) { clearInterval(jobTimer); loadRun(true); return; }
      if (!j.alive) clearInterval(jobTimer);
      if (S.tab === 'result' && !S.data) renderResult();
    } catch (e) { clearInterval(jobTimer); }
  }, 1000);
}

/* ---------------------------------------------------------------- 주행 시험 */
function renderDriveOptions() {
  const runSel = $('#d-run'), keep = runSel.value || S.active;
  runSel.replaceChildren(...S.runs.filter(r => r.iterations_done > 0).map(r => h('option', { value: r.name }, `${r.name} (${r.track})`)));
  if ([...runSel.options].some(o => o.value === keep)) runSel.value = keep;
  const tSel = $('#d-track'), tk = tSel.value;
  tSel.replaceChildren(h('option', { value: '' }, '학습한 트랙'), ...(S.info ? S.info.tracks : []).map(t => h('option', { value: t }, t)));
  tSel.value = tk;
  if (!runSel.options.length) $('#d-msg').textContent = '주행해 볼 모델이 아직 없습니다. 첫 iteration 이 끝난 실험이 있어야 합니다.';
}
async function submitDrive(ev) {
  ev.preventDefault();
  const msg = $('#d-msg'); msg.className = 'msg'; msg.textContent = '주행 중입니다…';
  const body = { run: $('#d-run').value, model: $('#d-model').value, track: $('#d-track').value, noise: Number($('#d-noise').value), laps: Number($('#d-laps').value) };
  try {
    const res = await api('/api/play', body);
    DMAP.setTrack(await trackGeom(res.track));
    DMAP.setReplay(res.episodes);
    S.drive = res;
    const ok = res.episodes.filter(e => e.status === 'lap_complete'), laps = ok.map(e => e.lap_time);
    msg.className = 'msg ok';
    msg.textContent = `${res.track} 트랙, 외란 ${res.noise}: ${res.episodes.length}번 중 ${ok.length}번 완주` + (laps.length ? `, 평균 랩타임 ${(laps.reduce((a, b) => a + b, 0) / laps.length).toFixed(2)}초` : '') + (res.track !== res.trained_track ? ` (학습한 트랙: ${res.trained_track})` : '');
    renderDriveTable();
  } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
}
function renderDriveTable() {
  const t = $('#d-table'); t.replaceChildren(); const res = S.drive; if (!res) return;
  t.append(h('thead', {}, h('tr', {}, ...['시도', '결과', '진행률', '랩타임', '총 보상'].map(x => h('th', {}, x)))));
  const tb = h('tbody');
  res.episodes.forEach((e, i) => {
    const tr = h('tr', { class: i === DMAP.idx ? 'sel' : '' }, h('td', {}, String(i + 1)), h('td', {}, END_KO[e.status] || e.status), h('td', {}, e.progress.toFixed(0) + '%'), h('td', {}, secs(e.lap_time)), h('td', {}, e.total_reward.toFixed(0)));
    tr.addEventListener('click', () => { DMAP.pick(i); renderDriveTable(); });
    tb.append(tr);
  });
  t.append(tb);
}

/* ---------------------------------------------------------------- 비교 */
async function refreshCompare() {
  const names = [...S.checked], empty = $('#cmp-empty'), body = $('#cmp-body');
  if (names.length < 1) {
    body.hidden = true; empty.hidden = false;
    empty.replaceChildren(h('strong', {}, '비교할 실험을 고르세요'), '왼쪽 실험 기록에서 2개 이상 체크하면 진행률, 랩타임, 완주율을 겹쳐서 볼 수 있습니다.'); return;
  }
  for (const n of names) {
    const run = S.runs.find(r => r.name === n), c = S.cmp[n];
    if (!c || (run && run.state === 'running') || (c.iters !== (run ? run.iterations_done : 0))) {
      try { S.cmp[n] = { d: await api('/api/run?name=' + encodeURIComponent(n)), iters: run ? run.iterations_done : 0 }; } catch (e) { /* 건너뜀 */ }
    }
  }
  const items = names.filter(n => S.cmp[n]).map((n, i) => ({ n, d: S.cmp[n].d, color: RUN_COLORS[i % RUN_COLORS.length] }));
  empty.hidden = true; body.hidden = false;
  if (!CMPCH.length) {
    const host = $('#cmp-charts');
    CMPCH = [new Chart(host, { title: '평가 진행률', height: 150, xLabels: true, yMin: 0, yMax: 100, fmt: v => num(v, 1), empty: '데이터 없음' }),
      new Chart(host, { title: '평가 랩타임', height: 150, xLabels: true, fmt: v => v.toFixed(2), empty: '완주한 평가 주행이 없습니다' }),
      new Chart(host, { title: '학습 완주율', height: 150, xLabels: true, yMin: 0, yMax: 100, fmt: v => num(v, 0), empty: '데이터 없음' })];
    CMPCH.forEach(c => CMP.add(c));
  }
  const xMax = Math.max(2, ...items.map(o => lastOf(o.d.metrics.iteration) || 0));
  const mk = key => items.map(o => ({ label: o.n, color: o.color, w: 2.2, x: o.d.metrics.iteration || [], y: (o.d.metrics[key] || []).map(v => (key === 'eval_progress' ? Math.min(100, v) : v)) }));
  CMPCH[0].set(mk('eval_progress'), xMax, xMax); CMPCH[1].set(mk('eval_lap_time_mean'), xMax, xMax); CMPCH[2].set(mk('train_lap_rate'), xMax, xMax);
  $('#cmp-legend').replaceChildren(...items.map(o => h('span', {}, h('i', { style: `background:${o.color}` }), o.n)));
  const t = $('#cmp-table'); t.replaceChildren();
  t.append(h('thead', {}, h('tr', {}, ...['실험', '트랙', 'iteration', '최종 평가 진행률', '최고 랩타임', '첫 완주', '최종 엔트로피'].map(x => h('th', {}, x)))));
  const tb = h('tbody');
  for (const o of items) {
    const m = o.d.metrics, laps = (m.eval_lap_time_best || []).filter(Number.isFinite), firstIdx = (m.eval_lap_rate || []).findIndex(v => v > 0);
    tb.append(h('tr', { style: 'cursor:default' }, h('td', {}, o.n), h('td', {}, o.d.info.track), h('td', {}, String(lastOf(m.iteration) || 0)),
      h('td', {}, num(Math.min(100, lastOf(m.eval_progress))) + '%'), h('td', {}, laps.length ? secs(Math.min(...laps)) : '–'),
      h('td', {}, firstIdx >= 0 ? `iteration ${m.iteration[firstIdx]}` : '–'), h('td', {}, num(lastOf(m.entropy), 2))));
  }
  t.append(tb);
}

/* ---------------------------------------------------------------- 시작 */
function buildCharts() {
  const tele = $('#tele'), small = $('#smalls');
  const mk = (host, o) => HUB.add(new Chart(host, o));
  CH.reward = mk(tele, { title: '에피소드 평균 보상', height: 100, fmt: v => num(v, 1) });
  CH.prog = mk(tele, { title: '진행률 (%)', height: 100, yMin: 0, yMax: 100 });
  CH.lap = mk(tele, { title: '평가 랩타임', height: 100, empty: '아직 완주한 평가 주행이 없습니다' });
  CH.ent = mk(tele, { title: '엔트로피 (높으면 탐험, 낮아지면 굳는 중)', height: 100, fmt: v => num(v, 2) });
  CH.kl = mk(tele, { title: 'KL (한 번 갱신에 정책이 바뀐 폭)', height: 100, xLabels: true, fmt: v => num(v, 4) });
  CH.uspeed = mk(small, { title: '속도 선택 (%)', height: 110, yMin: 0, yMax: 100, fmt: v => num(v, 0), empty: '학습 주행 기록이 아직 없습니다' });
  CH.usteer = mk(small, { title: '조향 선택 (%)', height: 110, yMin: 0, yMax: 100, fmt: v => num(v, 0), empty: '학습 주행 기록이 아직 없습니다' });
  CH.ploss = mk(small, { title: '정책 손실', height: 110, xLabels: true, fmt: v => num(v, 4) });
  CH.vloss = mk(small, { title: '가치 손실', height: 110, xLabels: true, fmt: v => num(v, 4) });
  HUB.onPick = it => selectIteration(it, true);
}

async function init() {
  MAP = new MapView($('#map')); DMAP = new MapView($('#dmap'));
  MAP.onEpisode = i => updateMapInfo(i);
  DMAP.onEpisode = () => { if (S.drive) renderDriveTable(); };
  buildCharts();
  $$('.tabs button').forEach(b => b.addEventListener('click', () => gotoProduct(b.dataset.product)));
  $$('.menu button').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
  $$('.grouphead').forEach(b => b.addEventListener('click', () => { const p = b.closest('.group').dataset.group; expandGroup(p, b.getAttribute('aria-expanded') !== 'true'); }));
  { const st = groupState(); expandGroup('mr', st.mr, false); expandGroup('dr', st.dr, false); }
  $$('#layerseg button').forEach(b => b.addEventListener('click', () => setLayer(b.dataset.layer)));
  $('#m-play').addEventListener('click', () => { MAP.playing = !MAP.playing; $('#m-play').textContent = MAP.playing ? '일시정지' : '재생'; });
  $('#d-play').addEventListener('click', () => { DMAP.playing = !DMAP.playing; $('#d-play').textContent = DMAP.playing ? '일시정지' : '재생'; });
  $('#m-speed').addEventListener('change', e => { MAP.mul = Number(e.target.value); DMAP.mul = Number(e.target.value); });
  if (REDUCE) { $('#m-play').textContent = '재생'; $('#d-play').textContent = '재생'; }
  $('#r-scrub').addEventListener('input', e => { const it = Number(e.target.value); S.follow = false; $('#r-follow').checked = false; $('#r-scrub-out').textContent = it; clearTimeout(init.t); init.t = setTimeout(() => selectIteration(it, true), 80); });
  $('#r-follow').addEventListener('change', e => { S.follow = e.target.checked; const last = S.data ? lastOf(S.data.metrics.iteration) : null; if (S.follow && last) selectIteration(last, false); });
  $('#r-stop').addEventListener('click', async () => {
    try { const r = await api('/api/stop', { name: S.active }); $('#r-meta').textContent = r.how; } catch (e) { $('#r-meta').textContent = e.message; }
  });
  $('#newform').addEventListener('submit', submitNew);
  $('#driveform').addEventListener('submit', submitDrive);
  $('#f-steer').addEventListener('input', updateActCount); $('#f-speed').addEventListener('input', updateActCount);
  $('#f-track').addEventListener('change', drawPreview);
  $('#f-code').addEventListener('keydown', e => {
    if (e.key === 'Tab') { e.preventDefault(); const t = e.target, s = t.selectionStart; t.setRangeText('    ', s, t.selectionEnd, 'end'); }
  });
  $('#f-example').addEventListener('change', e => { const ex = S.info.examples.find(x => x.name === e.target.value); if (ex) $('#f-code').value = ex.code; });
  window.addEventListener('resize', () => { HUB.draw(); CMP.draw(); drawPreview(); });

  try {
    S.info = await api('/api/info');
  } catch (e) { $('#newmsg').className = 'msg err'; $('#newmsg').textContent = e.message; setTab('new'); return; }
  const info = S.info;
  DRUI.init();
  initClean();
  $('#f-track').replaceChildren(...info.tracks.map(t => h('option', { value: t }, t)));
  $('#f-steer').value = info.steer.join(', '); $('#f-speed').value = info.speed.join(', '); updateActCount();
  $('#f-code').value = info.reward_code;
  $('#f-example').replaceChildren(h('option', { value: '' }, '선택'), ...info.examples.map(x => h('option', { value: x.name }, x.name)));
  buildHpGrid($('#hpgrid'), HP_FIELDS, info.hp); buildHpGrid($('#hpadv'), HP_ADV, info.hp);

  await refreshRuns();
  const running = S.runs.find(r => r.state === 'running');
  if (S.runs.length) { setTab('result'); setActive((running || S.runs[0]).name); } else setTab('new');

  setInterval(() => { if (!document.hidden) refreshRuns(); }, 3000);
  setInterval(() => {
    if (document.hidden) return;
    if (S.tab === 'result' && S.active && !S.pending && (!S.data || ['running', 'starting'].includes(S.data.state.state))) loadRun(false);
    if (S.tab === 'compare' && S.runs.some(r => S.checked.has(r.name) && r.state === 'running')) refreshCompare();
  }, 2000);
}
init();
