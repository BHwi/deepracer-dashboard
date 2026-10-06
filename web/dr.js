'use strict';
/* DeepRacer 화면 (DRfC 연동). app.js 의 공통 도구($, h, api, Chart, Hub, parseList ...)를 쓴다. */

const DR = {
  ov: null, doctor: null, tpl: null, exps: [], active: null, detail: null, met: null, inited: false,
  hub: new Hub(), charts: null, logRole: 'train', logTimer: null, algo: 'ppo', atype: 'discrete', hpvals: {}, source: '', arch: null,
};
const EP_KO = { 'Lap complete': '완주', 'Off track': '트랙 이탈', 'Crashed': '충돌', 'Reversed': '역주행', 'Immobilized': '정지', 'In progress': '진행 중', 'Incomplete': '미완료 (기록 없음)' };
const bytes = n => (n > 1e6 ? (n / 1e6).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB');
const when = ts => new Date(ts * 1000).toLocaleString('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
const q = s => encodeURIComponent(s);

/* ---------------------------------------------------------------- 작업(task) 출력 */
function showTask(box, t) {
  box.hidden = false; box.replaceChildren();
  const head = h('div', { class: 'th' }, h('span', {}, t.label), h('span', { class: 'state ' + t.state }, { running: '진행 중', ok: '완료', failed: '실패', cancelled: '취소됨' }[t.state] || t.state));
  if (t.state === 'running') {
    const b = h('button', { class: 'quiet', type: 'button' }, '취소');
    b.addEventListener('click', () => api('/api/dr/task/cancel', { id: t.id }).catch(() => {}));
    head.append(b);
  }
  const pre = h('pre', {}, t.output || '(출력 없음)');
  box.append(head, pre);
  if (t.hints && t.hints.length) box.append(h('ul', { class: 'hints' }, ...t.hints.map(x => h('li', {}, x))));
  pre.scrollTop = pre.scrollHeight;
}
function trackTask(box, t, done) {
  clearInterval(box._timer);
  showTask(box, t);
  if (t.state !== 'running') { if (done) done(t); return; }
  box._timer = setInterval(async () => {
    try {
      const v = await api('/api/dr/task?id=' + t.id); showTask(box, v);
      if (v.state !== 'running') { clearInterval(box._timer); if (done) done(v); }
    } catch (e) { clearInterval(box._timer); }
  }, 1000);
}
async function startTask(box, call, done) {
  try { const t = await call(); trackTask(box, t, done); return t; }
  catch (e) {
    box.hidden = false; box.replaceChildren(h('div', { class: 'th' }, h('span', { class: 'state failed' }, '시작하지 못했습니다')), h('ul', { class: 'hints' }, h('li', {}, e.message)));
    return null;
  }
}
function copyLine(cmd) {
  const b = h('button', { type: 'button' }, '복사');
  b.addEventListener('click', async () => { try { await navigator.clipboard.writeText(cmd); b.textContent = '복사했습니다'; } catch (e) { b.textContent = '직접 선택해서 복사하세요'; } setTimeout(() => { b.textContent = '복사'; }, 1800); });
  return h('div', { class: 'copyline' }, h('code', {}, cmd), b);
}

/* ---------------------------------------------------------------- 개요(상태) 갱신 */
function expState(e) {
  const a = DR.ov && DR.ov.active;
  if (e.running) return { cls: 'running', text: e.kind === 'evaluation' ? '평가 중' : '학습 중' };
  if (a && a.starting && a.experiment === e.name) return { cls: 'running', text: '시작 중' };
  if (e.has_model) return { cls: 'idle', text: '모델 있음' };
  return { cls: 'idle', text: '학습 전' };
}
async function refreshOv() {
  try { DR.ov = await api('/api/dr/overview'); } catch (e) { return; }
  const ov = DR.ov, a = ov.active, running = !!(a && (a.running || a.starting));
  const pill = $('#drpill');
  pill.textContent = running ? (a.kind === 'evaluation' ? '평가 중' : '학습 중') : (ov.arch ? ov.arch.toUpperCase() : '');
  pill.className = 'pill' + (running ? ' on' : '');
  $('#dot-run').className = 'dot' + (running ? ' live' : '');
  let envDot = '';
  if (!ov.valid || !ov.initialized) envDot = 'fail'; else if (DR.doctor) { const st = DR.doctor.checks.map(c => c.status); envDot = st.includes('fail') ? 'fail' : st.includes('warn') ? 'warn' : 'ok'; }
  $('#dot-env').className = 'dot ' + envDot;
  renderLive(); renderBanner();
  if (S.tab === 'dr-run') renderRunHeader();
  if (S.tab === 'dr-eval') renderEvalHeader();
  if (S.tab === 'dr-env') { $('#dr-mock').hidden = !ov.mock; }
}
async function refreshExps() {
  try { DR.exps = await api('/api/dr/experiments'); } catch (e) { DR.exps = []; }
  renderDrRail();
  if (DR.active && !DR.exps.some(e => e.name === DR.active)) { DR.active = null; DR.detail = null; DR.met = null; }
}
function renderDrRail() {
  const ul = $('#drlist'); ul.replaceChildren();
  for (const e of DR.exps) {
    const st = expState(e);
    const li = h('li', { class: 'run' + (e.name === DR.active ? ' active' : ''), tabindex: '0', style: 'grid-template-columns:4px 1fr' }, h('span'),
      h('div', { class: 'top' }, h('span', { class: 'name' }, e.name), h('span', { class: 'mark ' + st.cls }, st.text)),
      h('div', { class: 'sub' }, h('span', {}, e.world || ''), e.best_progress != null ? h('span', {}, `평가 ${Math.min(100, e.best_progress).toFixed(0)}%`) : null, e.algo ? h('span', {}, e.algo === 'sac' ? 'SAC' : 'PPO') : null));
    li.querySelector('.sub').style.gridColumn = '2'; li.querySelector('.top').style.gridColumn = '2';
    li.addEventListener('click', () => setActiveExp(e.name, true));
    li.addEventListener('keydown', ev => { if (ev.key === 'Enter') setActiveExp(e.name, true); });
    ul.append(li);
  }
  $('#drnote').textContent = !DR.ov || !DR.ov.valid ? '환경 점검에서 DRfC 폴더를 지정하세요.' : DR.exps.length ? '' : '아직 실험이 없습니다. 새 실험에서 만드세요.';
}
async function setActiveExp(name, go) {
  DR.active = name; DR.detail = null; DR.met = null; renderDrRail();
  if (go && !['dr-eval', 'dr-export'].includes(S.tab)) setTab('dr-run');
  await loadActive(); renderForTab();
}
async function loadActive() {
  const name = DR.active; if (!name) return;
  try {
    const [d, m] = await Promise.all([api('/api/dr/experiment?name=' + q(name)), api('/api/dr/metrics?name=' + q(name))]);
    if (DR.active !== name) return;
    DR.detail = d; DR.met = m; DR.err = null;
  } catch (e) { DR.detail = null; DR.met = null; DR.err = e.message; }
}
function renderForTab() {
  if (S.tab === 'dr-run') renderRun(); else if (S.tab === 'dr-eval') renderEval(); else if (S.tab === 'dr-export') renderExport();
}

/* ---------------------------------------------------------------- 환경 점검 */
const ST_KO = { ok: '정상', warn: '주의', fail: '필요', info: '참고' };
async function renderEnv() {
  if (DR.ov) { $('#dr-dir').value = DR.ov.drfc_dir || ''; $('#dr-mock').hidden = !DR.ov.mock; $('#dr-dir').readOnly = !!DR.ov.in_docker; $('#dr-dirform button').disabled = !!DR.ov.in_docker; }
  const list = $('#dr-checks'); list.replaceChildren(h('li', {}, h('span'), h('span', { class: 'dt' }, '점검 중…')));
  try { DR.doctor = await api('/api/dr/doctor'); } catch (e) { list.replaceChildren(h('li', {}, h('span'), h('span', { class: 'dt' }, e.message))); return; }
  list.replaceChildren(...DR.doctor.checks.map(c => {
    const li = h('li', {}, h('span', { class: 'st ' + c.status }, ST_KO[c.status] || c.status), h('span', { class: 'lb' }, c.label), h('span', { class: 'dt' }, c.detail || ''));
    if (c.fix || c.action) {
      const fx = h('div', { class: 'fx' });
      if (c.fix) fx.append(/^(sudo|docker|apt|git|cd|aws|source|\.\/)/.test(c.fix) ? copyLine(c.fix) : h('span', { class: 'dt' }, c.fix));
      if (c.action === 'create_swarm') {
        const b = h('button', { class: 'primary small', type: 'button' }, 'swarm 만들기');
        b.addEventListener('click', () => startTask($('#dr-envtask'), () => api('/api/dr/swarm', {}), () => renderEnv()));
        fx.append(b);
      }
      if (c.action === 'build_minio') {
        const b = h('button', { class: 'primary small', type: 'button' }, 'minio 이미지 만들기');
        b.addEventListener('click', () => startTask($('#dr-envtask'), () => api('/api/dr/minio-image', {}), () => renderEnv()));
        fx.append(b);
      }
      li.append(fx);
    }
    return li;
  }));
  renderArch(); renderInstall(); refreshOv();
}
function renderArch() {
  const ov = DR.ov || {}, d = DR.doctor || { gpu_names: [] }, box = $('#dr-arch');
  const cur = ov.arch, sel = DR.arch || cur || (d.gpu_names.length ? 'gpu' : 'cpu');
  DR.arch = sel;
  box.replaceChildren();
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'GPU 또는 CPU' }, ...['gpu', 'cpu'].map(v => {
    const b = h('button', { type: 'button', class: v === sel ? 'on' : '' }, v.toUpperCase() + (cur === v ? ' (현재)' : ''));
    b.addEventListener('click', () => { DR.arch = v; renderArch(); }); return b;
  }));
  const cuda = h('input', { id: 'ar-cuda', placeholder: '자동', value: ov.cuda_train || '' }), workers = h('input', { id: 'ar-workers', type: 'number', min: '1', max: '16', value: ov.workers || '1' });
  const apply = h('button', { class: 'primary small', type: 'button' }, '이 모드로 전환');
  const test = h('button', { class: 'quiet', type: 'button' }, 'GPU 컨테이너 시험');
  const msg = h('p', { class: 'note', id: 'ar-note' });
  apply.addEventListener('click', () => {
    if (sel === 'gpu' && !(d.gpu_names || []).length && !confirm('이 컴퓨터에서 NVIDIA GPU 를 찾지 못했습니다. 그래도 GPU 모드로 바꿀까요? (학습이 시작되지 않을 수 있습니다)')) return;
    startTask($('#dr-envtask'), () => api('/api/dr/arch', { arch: sel, cuda_devices: cuda.value.trim(), workers: Number(workers.value) }), () => { refreshOv().then(renderArch); });
  });
  test.addEventListener('click', () => startTask($('#dr-envtask'), () => api('/api/dr/gpu-test', {})));
  box.append(
    h('p', { class: 'note', style: 'margin:0' }, d.in_docker ? 'Docker 로 실행 중이라 이 화면에서 GPU 를 직접 볼 수 없습니다. GPU 컨테이너 시험으로 호스트 Docker 의 GPU 연결을 확인하세요.' : d.gpu_names && d.gpu_names.length ? `감지된 GPU: ${d.gpu_names.join(' / ')}` : '이 컴퓨터에서 NVIDIA GPU 를 찾지 못했습니다. CPU 모드로 쓸 수 있습니다.'),
    h('div', { class: 'archrow' }, seg, h('label', {}, 'GPU 번호', cuda), h('label', {}, '시뮬레이터 워커 수', workers), apply, test),
    h('p', { class: 'note', style: 'margin:0' }, 'GPU 모드는 학습 신경망 계산을 GPU 로 돌리고, CPU 모드는 GPU 없이 돌립니다. 전환은 시뮬레이터 이미지(-gpu / -cpu)를 바꿔 내려받는 것이며 system.env 는 백업 후 수정합니다. GPU 가 여러 개면 번호를 지정하세요(예: 0 또는 0,1). 빈칸이면 자동입니다. 워커 수는 시뮬레이터를 몇 개 동시에 돌릴지입니다.'),
    h('p', { class: 'note', style: 'margin:0' }, 'DRfC 문서의 로컬 권장 사양: NVIDIA GPU 메모리 8GB 이상(시뮬레이터 워커마다 약 1GB 추가), CPU 4코어(8스레드) 이상, 시스템 RAM + GPU 메모리 합 32GB 이상.'),
    msg);
}
function renderInstall() {
  const ins = DR.doctor && DR.doctor.install, body = $('#dr-installbody'); if (!ins) return;
  const mode = DR.arch || 'gpu';
  if (ins.docker) {
    body.replaceChildren(h('p', { class: 'note', style: 'margin:0' }, ins.note), h('p', { class: 'note' }, 'DRfC 초기화가 실패했거나 다시 하고 싶으면 호스트 터미널에서:'), copyLine('./drtrainer init'), h('p', { class: 'note' }, '진행 상황과 오류 로그:'), copyLine('./drtrainer logs -f'));
    $('#dr-install').open = !(DR.ov && DR.ov.initialized); return;
  }
  body.replaceChildren(
    h('p', { class: 'note', style: 'margin:0' }, '터미널에서 아래 명령 한 줄이면 Docker, GPU 도구, DRfC 내려받기와 초기화까지 한 번에 설치합니다. GPU 가 있으면 GPU 모드, 없으면 CPU 모드로 자동 설정합니다.'),
    h('ol', {},
      h('li', {}, '먼저 미리보기 (아무것도 바꾸지 않고 무엇을 할지만 보여 줍니다)', copyLine(ins.preview)),
      h('li', {}, '설치 (관리자 비밀번호를 한 번 입력합니다)', copyLine(mode === 'cpu' ? ins.auto_cpu : ins.auto)),
      h('li', {}, '끝나면 이 화면에서 점검 다시 실행을 누르세요.')),
    h('p', { class: 'note' }, ins.note),
    h('details', {}, h('summary', {}, '더 보기: GPU 드라이버 설치, 수동 설치'),
      h('div', { style: 'display:grid;gap:10px;padding-top:10px' },
        h('p', { class: 'note', style: 'margin:0' }, 'NVIDIA GPU 가 있는데 드라이버가 없으면 스크립트가 멈추고 안내합니다. 드라이버까지 설치하려면 아래 명령을 쓰세요. 설치 후 재부팅이 필요하고, 재부팅 뒤 같은 명령을 다시 실행하면 이어서 진행합니다.'),
        copyLine(ins.driver),
        h('p', { class: 'note', style: 'margin:0' }, '직접 설치하려면: DRfC 내려받기, 그리고 초기화(sudo 사용)'),
        copyLine(ins.clone), copyLine(mode === 'gpu' ? ins.init_gpu : ins.init_cpu))));
  $('#dr-install').open = !(DR.ov && DR.ov.initialized);
}

/* ---------------------------------------------------------------- 새 실험 */
const HP_HELP = {
  batch_size: ['배치 크기', '한 번에 묶어서 학습하는 샘플 수. 클수록 안정적이지만 느립니다.', 1],
  lr: ['학습률', '크면 빨리 배우지만 불안정해질 수 있습니다.', 0.0001],
  discount_factor: ['할인율', '1에 가까울수록 먼 미래의 보상까지 중요하게 봅니다.', 0.001],
  beta_entropy: ['엔트로피 보너스', '클수록 여러 행동을 오래 시도합니다. 너무 작으면 일찍 굳습니다.', 0.001],
  num_epochs: ['반복 횟수', '모은 데이터를 몇 번 되풀이해서 학습할지.', 1],
  sac_alpha: ['SAC 알파', 'SAC 의 탐험 정도. 클수록 탐험을 더 합니다.', 0.01],
  loss_type: ['손실 종류', 'huber 는 가끔 나오는 큰 오차에 덜 민감합니다.', 'select'],
  num_episodes_between_training: ['정책 갱신 간격', '정책을 한 번 갱신하기 전에 모으는 에피소드 수 (1 iteration).', 1],
  term_cond_max_episodes: ['총 에피소드', '이만큼 학습하면 끝납니다.', 1],
  term_cond_avg_score: ['목표 평균 점수', '평균 점수가 이 값에 닿으면 끝납니다. 보상함수의 점수 규모에 맞게 바꾸세요.', 1],
};
const HP_ORDER = { ppo: ['batch_size', 'lr', 'discount_factor', 'beta_entropy', 'num_epochs', 'loss_type', 'num_episodes_between_training', 'term_cond_max_episodes', 'term_cond_avg_score'],
  sac: ['batch_size', 'lr', 'discount_factor', 'sac_alpha', 'loss_type', 'num_episodes_between_training', 'term_cond_max_episodes', 'term_cond_avg_score'] };
const ENV_FIELDS = [
  ['change_start', 'bool', true, '출발 위치 바꾸기', '에피소드마다 출발 위치를 옮깁니다. 초기 학습에 권장됩니다.'],
  ['round_robin', 'num', 0.05, '출발 이동 간격', '에피소드마다 옮길 거리 (0.05 = 트랙의 5%).'],
  ['min_eval_trials', 'num', 5, '최소 평가 횟수', 'iteration 사이에 하는 최소 평가 횟수.'],
  ['best_metric', 'select:progress,reward', 'progress', 'best 모델 기준', 'progress 는 평가 완주율, reward 는 평가 보상이 가장 높은 모델을 best 로 고릅니다.'],
  ['domain_randomization', 'bool', false, '도메인 랜덤화', '에피소드마다 환경 색과 조명을 바꿔서 시뮬레이터에 덜 맞춰지게 합니다.'],
  ['reverse', 'bool', false, '반대 방향 학습', '트랙을 반대 방향으로 달립니다.'],
  ['alternate', 'bool', false, '방향 번갈아', '에피소드마다 달리는 방향을 번갈아 바꿉니다.'],
  ['eval_trials', 'num', 3, '평가 주행 수 (기본)', '나중에 평가 화면에서 바꿀 수 있습니다.'],
  ['eval_checkpoint', 'select:last,best', 'last', '평가 체크포인트 (기본)', '나중에 평가 화면에서 바꿀 수 있습니다.'],
];
function buildHp() {
  const host = $('#dn-hp'), tpl = (DR.tpl && DR.tpl.hp) || {};
  for (const el of $$('[id^="dnh-"]', host)) DR.hpvals[el.id.slice(4)] = el.value;     // 알고리즘을 바꿔도 입력값 유지
  host.replaceChildren();
  for (const k of HP_ORDER[DR.algo]) {
    const [ko, hint, step] = HP_HELP[k], val = DR.hpvals[k] != null ? DR.hpvals[k] : tpl[k];
    let inp;
    if (step === 'select') { inp = h('select', { id: 'dnh-' + k }, h('option', { value: 'huber' }, 'huber'), h('option', { value: 'mean squared error' }, 'mean squared error')); inp.value = val || 'huber'; }
    else inp = h('input', { id: 'dnh-' + k, type: 'number', step: String(step), value: val != null ? val : '' });
    host.append(h('label', { class: 'k', for: 'dnh-' + k }, ko, h('small', {}, k)), inp, h('span', { class: 'h' }, hint));
  }
}
function buildEnvFields() {
  const host = $('#dn-env'); host.replaceChildren();
  for (const [k, type, def, ko, hint] of ENV_FIELDS) {
    let inp;
    if (type === 'bool') { inp = h('input', { id: 'dnv-' + k, type: 'checkbox' }); inp.checked = def; }
    else if (type.startsWith('select:')) { inp = h('select', { id: 'dnv-' + k }, ...type.slice(7).split(',').map(o => h('option', { value: o }, o))); inp.value = def; }
    else inp = h('input', { id: 'dnv-' + k, type: 'number', step: 'any', value: def });
    host.append(h('label', { class: 'k', for: 'dnv-' + k }, ko), inp, h('span', { class: 'h' }, hint));
  }
}
function setSeg(id, v) { $$('#' + id + ' button').forEach(b => b.classList.toggle('on', b.dataset.v === v)); }
function applyAlgo() {
  setSeg('dn-algo', DR.algo);
  if (DR.algo === 'sac') { DR.atype = 'continuous'; }
  setSeg('dn-atype', DR.atype);
  $('#dn-typerow').hidden = DR.algo === 'sac';
  $('#dn-discrete').hidden = DR.atype !== 'discrete';
  $('#dn-continuous').hidden = DR.atype !== 'continuous';
  buildHp();
}
function updateDnCount() {
  const a = parseList($('#dn-steer').value), b = parseList($('#dn-speed').value), bad = [...a, ...b].some(v => !Number.isFinite(v));
  $('#dn-actcount').textContent = bad ? '숫자만 쉼표로 구분해서 입력하세요.' : `조향각 ${a.length}개 x 속도 ${b.length}개 = 행동 ${a.length * b.length}개`;
}
async function initNew() {
  if (!DR.tpl) {
    try { DR.tpl = await api('/api/dr/templates'); } catch (e) { $('#dn-msg').className = 'msg err'; $('#dn-msg').textContent = e.message; return; }
    const t = DR.tpl;
    $('#dn-worlds').replaceChildren(...[...new Set([...t.minirace_tracks, ...t.worlds])].map(w => h('option', { value: w })));
    $('#dn-color').replaceChildren(...t.colors.map(c => h('option', { value: c }, c))); $('#dn-color').value = 'Red';
    $('#dn-example').replaceChildren(h('option', { value: '' }, '선택'), ...t.examples.map(x => h('option', { value: x.name }, x.name)));
    const info = S.info; $('#dn-code').value = info.reward_code; $('#dn-steer').value = info.steer.join(', '); $('#dn-speed').value = info.speed.join(', ');
    $('#dn-world').value = 'reinvent_base'; updateDnCount(); buildEnvFields(); applyAlgo();
  }
  renderSourceOptions();
}
async function renderSourceOptions() {
  try { DR.tpl = Object.assign(DR.tpl || {}, await api('/api/dr/templates')); } catch (e) { /* 이전 값 유지 */ }
  const t = DR.tpl, sel = $('#dn-source'), cur = sel.value;
  sel.replaceChildren(h('option', { value: '' }, '처음부터 만들기'));
  if (t.minirace_runs.length) sel.append(h('optgroup', { label: 'MiniRacer 실험의 보상함수와 행동 공간' }, ...t.minirace_runs.map(r => h('option', { value: 'mr:' + r.name }, `${r.name} (${r.track})`))));
  if (DR.exps.length) sel.append(h('optgroup', { label: 'DeepRacer 실험을 복제' }, ...DR.exps.map(e => h('option', { value: 'dr:' + e.name }, e.name))));
  if (t.models.length) sel.append(h('optgroup', { label: '학습한 모델에서 이어서 학습' }, ...t.models.map(m => h('option', { value: 'cont:' + m }, m))));
  sel.value = [...sel.options].some(o => o.value === cur) ? cur : '';
}
function fillFromDetail(d) {
  $('#dn-code').value = d.reward_code || '';
  const mm = d.metadata || {}, algo = mm.training_algorithm === 'sac' ? 'sac' : 'ppo';
  DR.algo = algo; DR.atype = mm.action_space_type === 'continuous' ? 'continuous' : 'discrete';
  if (DR.atype === 'discrete' && Array.isArray(mm.action_space)) {
    $('#dn-steer').value = [...new Set(mm.action_space.map(a => a.steering_angle))].sort((a, b) => a - b).join(', ');
    $('#dn-speed').value = [...new Set(mm.action_space.map(a => a.speed))].sort((a, b) => a - b).join(', '); updateDnCount();
  } else if (mm.action_space && mm.action_space.speed) {
    $('#dn-slo').value = mm.action_space.steering_angle.low; $('#dn-shi').value = mm.action_space.steering_angle.high;
    $('#dn-vlo').value = mm.action_space.speed.low; $('#dn-vhi').value = mm.action_space.speed.high;
  }
  DR.hpvals = Object.assign({}, d.hp || {}); applyAlgo();
  const e = d.env || {};
  const setv = (k, v) => { const el = $('#dnv-' + k); if (!el || v == null) return; if (el.type === 'checkbox') el.checked = String(v).toLowerCase() === 'true'; else el.value = v; };
  setv('change_start', e.DR_TRAIN_CHANGE_START_POSITION); setv('round_robin', e.DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST); setv('min_eval_trials', e.DR_TRAIN_MIN_EVAL_TRIALS);
  setv('best_metric', e.DR_TRAIN_BEST_MODEL_METRIC); setv('domain_randomization', e.DR_ENABLE_DOMAIN_RANDOMIZATION); setv('reverse', e.DR_TRAIN_REVERSE_DIRECTION);
  setv('alternate', e.DR_TRAIN_ALTERNATE_DRIVING_DIRECTION); setv('eval_trials', e.DR_EVAL_NUMBER_OF_TRIALS); setv('eval_checkpoint', e.DR_EVAL_CHECKPOINT);
  if (e.DR_WORLD_NAME) $('#dn-world').value = e.DR_WORLD_NAME;
  if (e.DR_CAR_COLOR) $('#dn-color').value = e.DR_CAR_COLOR;
}
async function onSourceChange() {
  const v = $('#dn-source').value, note = $('#dn-sourcenote'); DR.source = v; note.textContent = ''; $('#dn-prerow').hidden = true;
  if (!v) return;
  const [kind, name] = [v.split(':')[0], v.slice(v.indexOf(':') + 1)];
  try {
    if (kind === 'mr') {
      const r = await api('/api/dr/import-minirace?run=' + q(name));
      $('#dn-code').value = r.reward_code; $('#dn-steer').value = r.steer.join(', '); $('#dn-speed').value = r.speed.join(', '); updateDnCount();
      DR.algo = 'ppo'; DR.atype = 'discrete'; applyAlgo();
      const tracks = new Set([...(DR.tpl.worlds || []), ...(DR.tpl.minirace_tracks || [])]);
      if (r.track && tracks.has(r.track)) $('#dn-world').value = r.track;
      if (!$('#dn-name').value) $('#dn-name').value = (name + '_dr').slice(0, 40).replace(/[^A-Za-z0-9_-]/g, '_');
      note.textContent = `MiniRacer 실험 '${name}' 의 보상함수와 행동 공간을 가져왔습니다. 하이퍼파라미터는 DRfC 기본값을 씁니다. ` + (r.warnings || []).join(' ');
    } else if (kind === 'dr') {
      const d = await api('/api/dr/experiment?name=' + q(name)); fillFromDetail(d);
      if (!$('#dn-name').value) $('#dn-name').value = (name + '_v2').slice(0, 40);
      note.textContent = `'${name}' 의 설정을 복사했습니다. 새 이름으로 저장됩니다.`;
    } else if (kind === 'cont') {
      const own = DR.exps.find(e => e.prefix === name);
      if (own) { const d = await api('/api/dr/experiment?name=' + q(own.name)); fillFromDetail(d); }
      $('#dn-prerow').hidden = false; $('#dn-prefix').value = name;
      if (!$('#dn-name').value) $('#dn-name').value = (name + '_2').slice(0, 40);
      note.textContent = `모델 '${name}' 에서 이어서 학습합니다. 보상함수나 설정을 바꿔도 되고, 기존 모델 폴더는 건드리지 않습니다.`;
    }
  } catch (e) { note.textContent = e.message; }
}
function collectSpec() {
  const hp = {};
  for (const k of HP_ORDER[DR.algo]) { const el = $('#dnh-' + k); hp[k] = el.tagName === 'SELECT' ? el.value : Number(el.value); }
  const bv = k => $('#dnv-' + k).checked, nv = k => Number($('#dnv-' + k).value), sv = k => $('#dnv-' + k).value;
  const spec = { name: $('#dn-name').value.trim(), world: $('#dn-world').value.trim(), car_color: $('#dn-color').value, race_type: 'TIME_TRIAL', algo: DR.algo,
    action_type: DR.algo === 'sac' ? 'continuous' : DR.atype, hp, reward_code: $('#dn-code').value, source: DR.source || 'form',
    train: { change_start: bv('change_start'), round_robin: nv('round_robin'), min_eval_trials: nv('min_eval_trials'), best_metric: sv('best_metric'),
      domain_randomization: bv('domain_randomization'), reverse: bv('reverse'), alternate: bv('alternate') },
    eval: { trials: nv('eval_trials'), checkpoint: sv('eval_checkpoint') } };
  if (spec.action_type === 'discrete') { spec.steer = parseList($('#dn-steer').value); spec.speed = parseList($('#dn-speed').value); }
  else Object.assign(spec, { steer_low: Number($('#dn-slo').value), steer_high: Number($('#dn-shi').value), speed_low: Number($('#dn-vlo').value), speed_high: Number($('#dn-vhi').value) });
  if (!$('#dn-prerow').hidden) spec.pretrained = { prefix: $('#dn-prefix').value, checkpoint: $('#dn-prechk').value };
  return spec;
}
async function lintNow() {
  const m = $('#dn-lintmsg'); m.className = 'msg'; m.textContent = '점검 중…';
  try { const r = await api('/api/dr/lint', { code: $('#dn-code').value }); m.className = 'msg ' + (r.ok ? 'ok' : 'err'); m.textContent = r.message; return r.ok; }
  catch (e) { m.className = 'msg err'; m.textContent = e.message; return false; }
}
async function submitNew(go) {
  const msg = $('#dn-msg'); msg.className = 'msg'; msg.textContent = '';
  const spec = collectSpec();
  for (const b of [$('#dn-make'), $('#dn-makego')]) b.disabled = true;
  try {
    const r = await api('/api/dr/experiment/create', { spec });
    msg.className = 'msg ok'; msg.textContent = `'${r.name}' 실험을 만들었습니다.`;
    await refreshExps(); DR.active = r.name; renderDrRail(); setTab('dr-run'); await loadActive(); renderRun();
    if (go) startTrain();
  } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
  for (const b of [$('#dn-make'), $('#dn-makego')]) b.disabled = false;
}

/* ---------------------------------------------------------------- 학습 */
function buildCharts() {
  if (DR.charts) return;
  const host = $('#drr-charts');
  const mk = o => DR.hub.add(new Chart(host, Object.assign({ height: 120, xLabels: true }, o)));
  DR.charts = {
    reward: mk({ title: '평균 보상', fmt: v => num(v, 1) }),
    prog: mk({ title: '진행률 (%)', yMin: 0, yMax: 100, fmt: v => num(v, 1) }),
    lap: mk({ title: '완주율 (%)', yMin: 0, yMax: 100, fmt: v => num(v, 0) }),
    time: mk({ title: '평가 랩타임', fmt: v => v.toFixed(2) + '초', empty: '아직 완주한 평가 주행이 없습니다' }),
  };
}
function runStateOf(d) {
  const e = DR.exps.find(x => x.name === d.name) || {};
  return expState({ name: d.name, running: e.running, kind: e.kind, has_model: d.has_model });
}
function renderRunHeader() {
  const d = DR.detail; if (!d || S.tab !== 'dr-run') return;
  const st = runStateOf(d), ov = DR.ov || {}, a = ov.active, running = !!(a && a.running && a.experiment === d.name);
  const sEl = $('#drr-state'); sEl.textContent = st.text; sEl.className = 'state ' + (st.cls === 'running' ? 'running' : '');
  const meta = [d.env.DR_WORLD_NAME, d.metadata.training_algorithm === 'sac' ? 'SAC' : 'PPO', ov.arch ? ov.arch.toUpperCase() + ' 모드' : null, ov.mock ? '시험용 가짜 DRfC' : null].filter(Boolean);
  $('#drr-meta').replaceChildren(...meta.map(m => h('span', {}, m)));
  const busy = !!(a && (a.running || a.starting));
  $('#drr-start').disabled = busy; $('#drr-stop').hidden = !(running || (a && a.starting && a.experiment === d.name));
  $('#drr-wipewrap').hidden = !d.has_model || running;
  const names = { train: '학습 (Sagemaker)', sim: '시뮬레이터 (RoboMaker)', coach: '코치', storage: '저장소 (minio)', other: '기타' };
  $('#drr-chips').replaceChildren(...(ov.containers || []).map(c => h('span', { class: 'chip' }, h('b', {}, names[c.role] || c.role), c.status)));
  if (!(ov.containers || []).length) $('#drr-chips').replaceChildren(h('span', { class: 'chip off' }, '실행 중인 컨테이너가 없습니다'));
}
function renderRun() {
  const empty = $('#drr-empty'), body = $('#drr-body'), d = DR.detail;
  if (!DR.ov || !DR.ov.valid || !DR.ov.initialized) {
    body.hidden = true; empty.hidden = false;
    empty.replaceChildren(h('strong', {}, 'DRfC 환경이 준비되지 않았습니다'), '환경 점검에서 DRfC 폴더와 설치 상태를 확인하세요.'); return;
  }
  if (!d) {
    body.hidden = true; empty.hidden = false;
    empty.replaceChildren(h('strong', {}, DR.active ? '이 실험을 불러오지 못했습니다' : '선택한 실험이 없습니다'), DR.err || '왼쪽 실험 목록에서 고르거나, 새 실험에서 만드세요.'); return;
  }
  empty.hidden = true; body.hidden = false; buildCharts();
  $('#drr-name').textContent = d.name; renderRunHeader();
  const per = d.per_iter || 20; $('#drr-per').textContent = per;
  const m = DR.met && DR.met.metrics, it = m ? m.iteration : [], last = it.length ? it[it.length - 1] : null;
  const planned = Math.ceil((d.hp.term_cond_max_episodes || 1000) / per), running = d.name && DR.exps.find(e => e.name === d.name && e.running);
  const xMax = running ? Math.max(planned, last || 1) : last || 2;
  DR.hub.best = DR.met ? DR.met.best_iteration : null;
  const S2 = (label, color, y, o) => Object.assign({ label, color, x: it, y: m ? y : [] }, o || {});
  DR.charts.reward.set([S2('학습', C.gray, m && m.train_reward), S2('평가', C.line, m && m.eval_reward, { w: 2.4 })], xMax, last);
  DR.charts.prog.set([S2('학습', C.gray, m && m.train_progress), S2('평가', C.line, m && m.eval_progress, { w: 2.4 })], xMax, last);
  DR.charts.lap.set([S2('학습', C.gray, m && m.train_lap_rate), S2('평가', C.line, m && m.eval_lap_rate, { w: 2.4 })], xMax, last);
  DR.charts.time.set([S2('평균', C.ink, m && m.eval_lap_time_mean, { w: 2, dots: true }), S2('최고', C.good, m && m.eval_lap_time_best, { dash: [4, 3] })], xMax, last);
  const t = $('#drr-ledger'); t.replaceChildren();
  t.append(h('thead', {}, h('tr', {}, ...['iteration', '학습 진행', '학습 완주', '평가 진행', '평가 랩타임', '평가 횟수'].map(x => h('th', {}, x)))));
  const tb = h('tbody');
  for (let i = it.length - 1; i >= 0; i--) {
    tb.append(h('tr', { style: 'cursor:default' }, h('td', {}, String(it[i]), it[i] === (DR.met && DR.met.best_iteration) ? h('span', { class: 'tag' }, 'best') : null),
      h('td', {}, num(m.train_progress[i]) + '%'), h('td', {}, num(m.train_lap_rate[i], 0) + '%'), h('td', {}, m.eval_progress[i] == null ? '–' : num(Math.min(100, m.eval_progress[i])) + '%'),
      h('td', {}, secs(m.eval_lap_time_mean[i])), h('td', {}, String(m.n_eval[i]))));
  }
  t.append(tb);
  renderConfig(d); loadLog(false);
}
function renderConfig(d) {
  const e = d.env, rows = [['트랙', e.DR_WORLD_NAME], ['모델 이름(접두사)', d.prefix], ['이어서 학습', String(e.DR_LOCAL_S3_PRETRAINED).toLowerCase() === 'true' ? `${e.DR_LOCAL_S3_PRETRAINED_PREFIX} (${e.DR_LOCAL_S3_PRETRAINED_CHECKPOINT})` : '아니오'],
    ['출발 위치 바꾸기', e.DR_TRAIN_CHANGE_START_POSITION], ['best 기준', e.DR_TRAIN_BEST_MODEL_METRIC], ['도메인 랜덤화', e.DR_ENABLE_DOMAIN_RANDOMIZATION],
    ['알고리즘', d.metadata.training_algorithm], ['행동 공간', d.metadata.action_space_type === 'discrete' ? `이산 ${(d.metadata.action_space || []).length}개` : '연속']];
  const dl = h('dl', {}, ...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v == null ? '–' : String(v))]));
  const clone = h('button', { class: 'quiet', type: 'button', style: 'justify-self:start' }, '이 설정으로 새 실험 만들기');
  clone.addEventListener('click', async () => { setTab('dr-new'); await initNew(); $('#dn-source').value = 'dr:' + d.name; onSourceChange(); });
  $('#drr-config').replaceChildren(dl, h('p', { class: 'note', style: 'margin:0' }, '하이퍼파라미터'), h('pre', {}, JSON.stringify(d.hp, null, 2)), h('p', { class: 'note', style: 'margin:0' }, '보상함수'), h('pre', {}, d.reward_code), clone);
}
async function loadLog(manual) {
  if (S.tab !== 'dr-run') return;
  try {
    const r = await api(`/api/dr/logs?role=${DR.logRole}&tail=300`), pre = $('#drr-log');
    $('#drr-logname').textContent = r.container || r.message || '';
    const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
    pre.replaceChildren(...r.text.split('\n').map(l => { const s = h('span', {}, l + '\n'); if (/error|traceback|exception|failed/i.test(l)) s.className = 'bad'; return s; }));
    if (!r.text) pre.textContent = r.message || '(로그 없음)';
    if (stick || manual) pre.scrollTop = pre.scrollHeight;
  } catch (e) { $('#drr-log').textContent = e.message; }
}
function startTrain() {
  const d = DR.detail; if (!d) return;
  const wipe = !$('#drr-wipewrap').hidden && $('#drr-wipe').checked;
  if (wipe && !confirm(`모델 '${d.prefix}' 을(를) 지우고 처음부터 학습합니다. 되돌릴 수 없습니다. 계속할까요?`)) return;
  startTask($('#drr-task'), () => api('/api/dr/train/start', { name: d.name, wipe }), async () => { await refreshOv(); await refreshExps(); await loadActive(); renderRun(); });
}

/* ---------------------------------------------------------------- 평가 */
/* 평가할 모델 고르기: 모델은 실험 하나에 하나이므로 실험을 고르는 것과 같다. 열려 있는 목록이 닫히지 않게 내용이 바뀔 때만 다시 만든다. */
function renderEvalModels() {
  const sel = $('#dre-model'), exps = (DR.exps || []).slice().sort((a, b) => Number(b.has_model) - Number(a.has_model));     // 모델이 있는 실험을 위로 (같으면 기존 순서)
  const sig = exps.map(e => [e.name, e.prefix, e.has_model, e.iterations, e.best_progress].join(':')).join('|');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    if (!exps.length) sel.replaceChildren(h('option', { value: '' }, '(실험이 없습니다)'));
    else sel.replaceChildren(...exps.map(e => {
      const info = e.has_model ? `학습 ${e.iterations || 0} iteration` + (e.best_progress != null ? ` · 평가 ${Math.min(100, e.best_progress).toFixed(0)}%` : '') : '모델 없음 (먼저 학습)';
      const o = h('option', { value: e.name }, `${e.name}${e.prefix && e.prefix !== e.name ? ` (모델 ${e.prefix})` : ''} — ${info}`);
      if (!e.has_model) o.disabled = true;
      return o;
    }));
  }
  if (DR.active && sel.value !== DR.active) sel.value = DR.active;
}
function renderEvalHeader() {
  const d = DR.detail; if (!d || S.tab !== 'dr-eval') return;
  renderEvalModels();
  const st = runStateOf(d), a = DR.ov && DR.ov.active, busy = !!(a && (a.running || a.starting));
  $('#dre-state').textContent = st.text; $('#dre-state').className = 'state ' + (st.cls === 'running' ? 'running' : '');
  $('#dre-start').disabled = busy || !d.has_model; $('#dre-stop').hidden = !(a && a.running && a.experiment === d.name && a.kind === 'evaluation');
}
async function renderEval() {
  const empty = $('#dre-empty'), body = $('#dre-body'), d = DR.detail;
  if (!d) { body.hidden = true; empty.hidden = false; empty.replaceChildren(h('strong', {}, '선택한 실험이 없습니다'), DR.err || '왼쪽 실험 목록에서 평가할 실험을 고르세요.'); return; }
  empty.hidden = true; body.hidden = false; $('#dre-name').textContent = d.name;
  const e = d.env;
  if (DR.evalFor !== d.name) {
    $('#dre-trials').value = e.DR_EVAL_NUMBER_OF_TRIALS || 3; $('#dre-chk').value = e.DR_EVAL_CHECKPOINT || 'last';
    $('#dre-cont').checked = String(e.DR_EVAL_IS_CONTINUOUS).toLowerCase() !== 'false'; $('#dre-rev').checked = String(e.DR_EVAL_REVERSE_DIRECTION).toLowerCase() === 'true';
    DR.evalFor = d.name;
  }
  renderEvalHeader();
  try {
    const r = await api('/api/dr/evals?name=' + q(d.name)), box = $('#dre-results');
    DR.evalData = r;
    if (!r.runs.length) {
      DR.evalSig = '';
      box.replaceChildren(h('p', { class: 'note' }, d.has_model ? '아직 평가 기록이 없습니다.' : '평가할 모델이 아직 없습니다. 먼저 학습하세요.'));
      return;
    }
    // 목록(이름)을 눌러 그 평가의 결과와 영상을 본다. 4초마다 다시 그리면 재생 중인 영상이 끊기므로 내용이 바뀐 때만 다시 그린다.
    if (!r.runs.some(x => x.file === DR.evalSel)) DR.evalSel = r.runs[0].file;
    const sig = JSON.stringify([d.name, DR.evalSel, r.runs.map(x => [x.file, x.label, x.n, x.laps, x.inferred, x.videos.map(v => v.key + v.size), x.trials.map(t => t.status + t.progress)]), r.unmatched_videos.map(v => v.key)]);
    if (sig === DR.evalSig && box.childElementCount) return;
    DR.evalSig = sig;
    renderEvalList(box, d, r);
  } catch (err) { $('#dre-results').textContent = err.message; }
}

const evTitle = run => {
  if (run.label) return run.label;
  const date = (run.file.match(/evaluation-(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})/) || []);
  return date.length ? `${date[2]}/${date[3]} ${date[4]}:${date[5]}` : run.file;
};
function videoKind(name) {
  const n = name.toLowerCase();
  if (/(main|follow|chase)/.test(n)) return '차를 따라가는 시점';
  if (/(sub|top|overhead|bird)/.test(n)) return '트랙 위에서 본 시점';
  if (/(cam|car|pip|zed|front|racecar)/.test(n)) return '차 안의 카메라';
  return '';
}
const videoUrl = (exp, v, dl) => `/api/dr/video?name=${q(exp)}&key=${q(v.key)}${dl ? '&download=1' : ''}`;

/* 영상 재생기: 고른 영상을 <video> 로 틀고, 재생이 안 되는 형식이면 이유와 다운로드 방법을 알려 준다 */
function mountPlayer(exp, videos) {
  const player = h('video', { controls: '', preload: 'metadata', playsinline: '' }), msg = h('p', { class: 'note' });
  const links = h('div', { class: 'vlinks' });
  const btns = videos.map(v => {
    const kind = videoKind(v.name);
    const b = h('button', { type: 'button', class: 'vbtn', title: `${v.name} (${fmtBytes(v.size)})` }, kind ? `${kind}` : v.name);
    b.addEventListener('click', () => show(v, b)); return b;
  });
  function show(v, b) {
    btns.forEach(x => x.classList.toggle('on', x === b));
    msg.textContent = `${v.name} · ${fmtBytes(v.size)}`;
    player.onerror = () => { msg.textContent = `이 브라우저가 이 영상을 재생하지 못합니다 (${v.name}). MP4 안의 영상 형식이 브라우저가 지원하는 H.264 가 아닐 수 있습니다. 아래 '다운로드'로 받아서 영상 플레이어(VLC 등)로 여세요.`; };
    player.src = videoUrl(exp, v, false);
    links.replaceChildren(h('a', { href: videoUrl(exp, v, true) }, '다운로드'), h('a', { href: videoUrl(exp, v, false), target: '_blank', rel: 'noopener' }, '새 창에서 열기'));
  }
  const wrap = h('div', { class: 'vplayer' }, h('div', { class: 'vtabs' }, ...btns), player, msg, links);
  if (videos.length) show(videos[0], btns[0]);
  return wrap;
}

function renderEvalList(box, d, r) {
  const list = h('div', { class: 'evlist' }, ...r.runs.map(run => {
    const b = h('button', { type: 'button', class: 'evitem' + (run.file === DR.evalSel ? ' on' : '') },
      h('strong', {}, evTitle(run)), h('span', {}, `${run.laps}/${run.n} 완주 · 영상 ${run.videos.length}개`));
    if (run.label) b.append(h('span', { class: 'sub' }, evTitle({ file: run.file })));
    b.addEventListener('click', () => { DR.evalSel = run.file; DR.evalSig = ''; renderEval(); });
    return b;
  }));
  const run = r.runs.find(x => x.file === DR.evalSel), detail = h('div', { class: 'evdetail' });
  // 이름 붙이기 / 바꾸기
  const inp = h('input', { type: 'text', maxlength: '60', value: run.label || '', placeholder: '평가 이름', autocomplete: 'off', 'aria-label': '평가 이름' });
  const save = h('button', { type: 'button' }, '이름 저장'), msg = h('span', { class: 'msg' });
  save.addEventListener('click', async () => {
    try { await api('/api/dr/eval/label', { name: d.name, file: run.file, label: inp.value }); msg.className = 'msg ok'; msg.textContent = '저장했습니다'; DR.evalSig = ''; renderEval(); }
    catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
  });
  const tbl = h('table', {}, h('thead', {}, h('tr', {}, ...['시도', '결과', '진행률', '랩타임'].map(x => h('th', {}, x)))),
    h('tbody', {}, ...run.trials.map((t, i) => h('tr', { style: 'cursor:default' }, h('td', {}, String(t.trial != null ? t.trial : i + 1)),
      h('td', t.inferred ? { title: '최종 상태가 기록되지 않았지만 진행률이 100% 라서 완주로 봤습니다' } : {}, (EP_KO[t.status] || t.status || '–') + (t.inferred ? ' *' : '')),
      h('td', {}, t.progress != null ? num(Math.min(100, t.progress), 0) + '%' : '–'), h('td', {}, secs(t.time))))));
  detail.append(h('h4', {}, evTitle(run)), h('div', { class: 'evname' }, inp, save, msg),
    h('div', { class: 'meta' }, h('span', {}, `${run.n}번 중 ${run.laps}번 완주`), h('span', {}, `평균 진행률 ${num(run.progress, 0)}%`), h('span', {}, `최고 ${secs(run.best)}`), h('span', {}, `평균 ${secs(run.mean)}`)), tbl);
  if (run.inferred) detail.append(h('p', { class: 'note' }, `* ${run.inferred}번은 시뮬레이터가 최종 상태를 기록하지 않았지만(계속 'In progress') 진행률이 100% 라서 완주로 봤습니다. 이 랩타임은 마지막으로 기록된 시점의 값이라 실제와 조금 다를 수 있습니다.`));
  detail.append(h('h4', { style: 'margin-top:18px' }, '주행 영상'));
  if (run.videos.length) {
    detail.append(mountPlayer(d.name, run.videos), h('p', { class: 'note' }, '영상은 파일 이름이 아니라 만들어진 시각으로 이 평가와 짝지었습니다.'));
  } else {
    const why = run.save_mp4 === false ? "이 평가는 '주행 영상 저장'을 켜지 않고 실행해서 영상이 없습니다."
      : run.save_mp4 === true ? '영상을 저장하도록 실행했지만 아직 영상이 없습니다. 영상은 평가가 끝난 뒤에 만들어집니다.'
      : "이 평가에 짝지어진 영상이 없습니다. '주행 영상 저장'을 켜지 않고 실행한 평가에는 영상이 만들어지지 않습니다 (DRfC 의 기본값은 저장 안 함입니다).";
    detail.append(h('p', { class: 'note' }, why + " 같은 모델을 '주행 영상 저장'을 켜고 다시 평가하면 영상이 저장됩니다."));
  }
  if (r.unmatched_videos.length) {
    detail.append(h('details', { style: 'margin-top:14px' }, h('summary', {}, `이 모델에 저장된 영상 중 어느 평가와도 짝지어지지 않은 것 (${r.unmatched_videos.length}개)`), mountPlayer(d.name, r.unmatched_videos)));
  }
  box.replaceChildren(list, detail);
}

function startEval(ev) {
  ev.preventDefault();
  const d = DR.detail; if (!d) return;
  const body = { name: d.name, trials: Number($('#dre-trials').value), checkpoint: $('#dre-chk').value, continuous: $('#dre-cont').checked, reverse: $('#dre-rev').checked, clone: $('#dre-clone').checked,
    label: $('#dre-label').value.trim(), save_mp4: $('#dre-mp4').checked };
  startTask($('#dre-task'), () => api('/api/dr/eval/start', body), async () => { $('#dre-label').value = ''; DR.evalSig = ''; await refreshOv(); await refreshExps(); renderEval(); });
}

/* ---------------------------------------------------------------- 차량용 내보내기 */
function renderExport() {
  const empty = $('#drx-empty'), body = $('#drx-body'), d = DR.detail;
  if (!d) { body.hidden = true; empty.hidden = false; empty.replaceChildren(h('strong', {}, '선택한 실험이 없습니다'), DR.err || '왼쪽 실험 목록에서 내보낼 실험을 고르세요.'); return; }
  empty.hidden = true; body.hidden = false; $('#drx-name').textContent = d.name;
  const st = runStateOf(d); $('#drx-state').textContent = st.text; $('#drx-state').className = 'state ' + (st.cls === 'running' ? 'running' : '');
  $('#drx-make').disabled = !d.has_model;
  const box = $('#drx-files');
  if (!d.zips.length) { box.replaceChildren(h('p', { class: 'note' }, d.has_model ? '아직 만든 파일이 없습니다.' : '내보낼 모델이 아직 없습니다. 먼저 학습하세요.')); return; }
  box.replaceChildren(...d.zips.map(z => {
    const need = ['agent/model.pb', 'model_metadata.json'];
    return h('div', { class: 'filerow' }, h('div', {}, h('div', {}, h('b', {}, z.file), `   ${bytes(z.size)}   ${when(z.mtime)}`),
      h('div', { class: 'meta' }, ...need.map(n => h('span', { class: z.members.includes(n) ? 'ok' : 'no' }, (z.members.includes(n) ? '있음 ' : '없음 ') + n)))),
      h('a', { href: '/api/dr/download?file=' + q(z.file), download: z.file }, '내려받기'));
  }));
}
function makeCar(ev) {
  ev.preventDefault();
  const d = DR.detail; if (!d) return;
  startTask($('#drx-task'), () => api('/api/dr/export', { name: d.name, best: $('#drx-best').value === '1' }), async () => { await loadActive(); renderExport(); });
}

/* ---------------------------------------------------------------- 실시간 영상 */
/* 시뮬레이터(RoboMaker)가 ROS 영상 스트림(MJPEG)을 내보낸다: 학습 8080+실행번호, 평가 8180+실행번호.
   기본은 대시보드가 그 스트림을 브라우저로 중계하는 방식이라 브라우저는 대시보드 포트 하나만 접속하면 된다 (원격 접속, 포트 충돌, 방화벽이 있어도 보인다).
   '직접 접속'은 브라우저가 시뮬레이터의 포트에 바로 접속한다. */
const STREAM = { panels: {}, info: null };
const STREAM_SIZES = [['480x360', '작게 (480x360)'], ['640x480', '보통 (640x480)'], ['960x720', '크게 (960x720)']];
const STREAM_IDLE = '학습이나 평가가 실행되면 여기에 시뮬레이터 영상이 나옵니다. (시뮬레이터가 시작되는 데 1분쯤 걸립니다.)';

function buildStream(hostId, tab) {
  const host = $('#' + hostId);
  const st = { tab, host, topic: '/racecar/deepracer/kvs_stream', userTopic: false, size: '640x480', mode: 'proxy', port: null, playing: null, retry: null, fails: 0, force: false, sig: '' };
  st.title = h('h3', {}, '실시간 영상');
  st.topicSel = h('select', { 'aria-label': '영상 종류' });
  st.sizeSel = h('select', { 'aria-label': '영상 크기' }, ...STREAM_SIZES.map(([v, l]) => h('option', { value: v }, l)));
  st.sizeSel.value = st.size;
  st.modeSel = h('select', { 'aria-label': '연결 방식', title: '대시보드를 통해: 브라우저는 대시보드 포트만 접속합니다. 직접 접속: 브라우저가 시뮬레이터의 영상 포트에 바로 접속합니다.' },
    h('option', { value: 'proxy' }, '연결: 대시보드를 통해'), h('option', { value: 'direct' }, '연결: 직접 접속'));
  st.portSel = h('select', { 'aria-label': '워커 선택' }); st.portSel.hidden = true;
  st.reload = h('button', { type: 'button' }, '다시 연결');
  st.force_ = h('button', { type: 'button', title: '대시보드가 실행 중으로 인식하지 못해도 영상 서버에 직접 연결을 시도합니다' }, '그래도 연결 시도');
  st.open = h('a', { target: '_blank', rel: 'noopener' }, '새 창에서 열기');
  st.diag = h('button', { type: 'button', title: '영상이 안 보일 때 어디가 막혔는지 단계별로 확인합니다' }, '영상 진단');
  st.viewer = h('button', { type: 'button', title: 'DRfC 가 제공하는 뷰어: 여러 워커의 영상을 한 화면에 보여 줍니다' }, 'DRfC 뷰어 (워커 전체)');
  st.viewerLink = h('a', { target: '_blank', rel: 'noopener' }, '뷰어 열기'); st.viewerLink.hidden = true;
  st.img = h('img', { alt: '시뮬레이터 실시간 영상' }); st.img.hidden = true;
  st.ph = h('div', { class: 'ph' }, STREAM_IDLE);
  st.frame = h('div', { class: 'streamframe' }, st.img, st.ph); st.frame.hidden = true;     // 영상이 나올 때만 큰 화면을 보인다
  st.status = h('p', { class: 'streamnote' }, STREAM_IDLE);
  st.note = h('p', { class: 'streamnote' });
  st.diagBox = h('div', { class: 'streamdiag' }); st.diagBox.hidden = true;
  host.replaceChildren(
    h('div', { class: 'streamhead' }, st.title, st.topicSel, st.sizeSel, st.modeSel, st.portSel, st.reload, st.force_, st.open, st.diag, st.viewer, st.viewerLink),
    st.frame, st.status, st.note, st.diagBox);
  st.topicSel.addEventListener('change', () => { st.topic = st.topicSel.value; st.userTopic = true; st.playing = null; syncStreams(); });
  st.sizeSel.addEventListener('change', () => { st.size = st.sizeSel.value; st.playing = null; syncStreams(); });
  st.modeSel.addEventListener('change', () => { st.mode = st.modeSel.value; st.playing = null; syncStreams(); });
  st.portSel.addEventListener('change', () => { st.port = Number(st.portSel.value); st.playing = null; syncStreams(); });
  st.reload.addEventListener('click', () => { st.fails = 0; st.playing = null; syncStreams(); });
  st.force_.addEventListener('click', () => { st.force = true; st.fails = 0; st.playing = null; syncStreams(); });
  st.diag.addEventListener('click', () => runStreamDiag(st));
  st.viewer.addEventListener('click', () => {
    const box = $(tab === 'dr-eval' ? '#dre-task' : '#drr-task'), info = STREAM.info;
    startTask(box, () => api('/api/dr/viewer/start', {}), () => { st.viewerLink.href = `${location.protocol}//${location.hostname}:${info.viewer_port}`; st.viewerLink.hidden = false; });
  });
  STREAM.panels[tab] = st;
}
function streamUrl(st) {
  const [w, hh] = st.size.split('x');
  if (st.mode === 'direct') return `${location.protocol}//${location.hostname}:${st.port}/stream?topic=${encodeURIComponent(st.topic)}&width=${w}&height=${hh}&quality=75`;
  return `/api/dr/stream/proxy?port=${st.port}&topic=${encodeURIComponent(st.topic)}&width=${w}&height=${hh}&quality=75`;
}
const streamWhere = st => (st.mode === 'direct' ? `직접 접속 (포트 ${st.port})` : `대시보드 경유 (시뮬레이터 포트 ${st.port})`);
function stopStream(st, text) {
  clearTimeout(st.retry); st.playing = null; st.img.onload = st.img.onerror = null; st.img.removeAttribute('src');
  st.img.hidden = true; st.ph.hidden = false; st.ph.textContent = text;
  st.frame.hidden = true; st.status.textContent = text;                       // 영상이 없을 때는 큰 빈 화면 대신 한 줄 안내만
}
function startStream(st, url) {
  clearTimeout(st.retry); st.playing = url;
  st.frame.hidden = false; st.ph.hidden = false; st.ph.textContent = '영상에 연결하는 중입니다... (시뮬레이터가 시작되는 데 1분쯤 걸릴 수 있습니다)'; st.img.hidden = true;
  st.status.textContent = `연결 중: ${streamWhere(st)}${st.fails ? ` / 지금까지 ${st.fails}번 실패` : ''}`;
  st.img.onload = () => { if (st.playing === url) { st.fails = 0; st.ph.hidden = true; st.img.hidden = false; st.status.textContent = `영상 수신 중: ${streamWhere(st)}`; } };
  st.img.onerror = () => {
    if (st.playing !== url) return;
    st.fails += 1; st.img.hidden = true; st.ph.hidden = false;
    st.ph.textContent = '영상에 연결할 수 없습니다. 시뮬레이터가 아직 시작 중일 수 있어 3초 뒤에 다시 시도합니다.' + (st.fails >= 3 ? ' 계속 안 되면 위의 "영상 진단"을 눌러 원인을 확인하세요.' : '');
    st.status.textContent = `연결 실패 ${st.fails}번: ${streamWhere(st)}`;
    st.retry = setTimeout(() => { if (st.playing === url) { st.playing = null; syncStreams(); } }, 3000);
  };
  st.open.href = url; st.img.src = url + '&_=' + Date.now();
}
function updateStreamControls(st, info) {
  st.title.textContent = info.running ? (info.kind === 'evaluation' ? '실시간 영상 (평가)' : '실시간 영상 (학습)') : '실시간 영상';
  // 영상 종류: 시뮬레이터가 지금 실제로 내보내는 목록을 반영한다. 사용자가 고르지 않았다면 나오고 있는 영상 중 첫 번째를 쓴다.
  const sig = info.topics.map(t => t.id + t.enabled + t.available).join();
  if (!st.userTopic) {
    const cur = info.topics.find(t => t.id === st.topic), ok = info.topics.find(t => t.enabled && t.available !== false);
    if ((!cur || cur.available === false || !cur.enabled) && ok) st.topic = ok.id;
  }
  if (st.sig !== sig + st.topic) {
    st.sig = sig + st.topic;
    st.topicSel.replaceChildren(...info.topics.map(t => {
      const o = h('option', { value: t.id }, t.label + (t.enabled ? '' : ' (사용 안 함)') + (t.enabled && t.available === false ? ' (지금은 나오지 않음)' : ''));
      if (!t.enabled) o.disabled = true; return o; }));
    st.topicSel.value = st.topic;
  }
  const multi = info.style === 'compose' && info.ports.length > 1;
  st.portSel.hidden = !multi;
  if (multi && st.portSel.options.length !== info.ports.length) st.portSel.replaceChildren(...info.ports.map((p, i) => h('option', { value: p }, `워커 ${i + 1} (포트 ${p})`)));
  if (!info.ports.includes(st.port)) { st.port = info.ports[0] || (st.force ? 8080 : null); if (multi && st.port) st.portSel.value = String(st.port); }
  st.force_.hidden = info.running;
  st.viewer.hidden = !(info.running && info.workers > 1);
  const t = info.topics.find(x => x.id === st.topic);
  st.note.textContent = (t && !t.enabled ? `이 영상은 ${t.need}. ` : '')
    + (info.live && info.live.length === 0 ? '시뮬레이터의 영상 서버는 떠 있지만 아직 내보내는 영상이 없습니다 (시작 중일 수 있음). ' : '')
    + (info.probe_error && info.running ? `영상 서버에 연결하지 못했습니다 (${info.probe_error}). ` : '')
    + (info.style === 'swarm' && info.workers > 1 ? '워커가 여러 개라서 한 주소를 번갈아 보여 줍니다. 다른 워커를 보려면 다시 연결을 누르세요. ' : '');
}
function syncStreams() {
  const info = STREAM.info; if (!info) return;
  for (const st of Object.values(STREAM.panels)) {
    updateStreamControls(st, info);
    const want = S.tab === st.tab && (info.running || st.force) && st.port;
    if (!want) { if (st.playing || st.img.getAttribute('src')) stopStream(st, info.running ? '영상을 보려면 이 화면을 여세요.' : STREAM_IDLE); else if (!info.running) st.status.textContent = STREAM_IDLE; continue; }
    const t = info.topics.find(x => x.id === st.topic);
    if (t && !t.enabled) { stopStream(st, `이 영상은 ${t.need}. 다른 영상을 고르세요.`); continue; }
    const url = streamUrl(st);
    if (st.playing !== url) startStream(st, url);
  }
}
async function refreshStream() {
  try { STREAM.info = await api('/api/dr/stream'); } catch (e) { return; }
  syncStreams();
}

/* 브라우저에서 직접 해 보는 점검 한 가지: 주소에 접속해서 JPEG 프레임이 오는지 (서버 쪽 점검과 합쳐 막힌 구간을 가린다) */
async function probeBrowser(url, label, noCors) {
  const t0 = performance.now(), ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 7000), ms = () => Math.round(performance.now() - t0);
  try {
    const r = await fetch(url, { signal: ctl.signal, mode: noCors ? 'no-cors' : 'same-origin', cache: 'no-store' });
    if (noCors) { ctl.abort(); return `[ 정상 ] ${label}: 브라우저가 이 주소에 접속할 수 있음 (${ms()}ms)`; }
    if (!r.ok) return `[ 문제 ] ${label}: HTTP ${r.status} ${(await r.text().catch(() => '')).slice(0, 200)}`;
    const reader = r.body.getReader(); let got = 0, jpeg = false;
    while (performance.now() - t0 < 7000 && got < 60000) {
      const { value, done } = await reader.read(); if (done) break;
      got += value.length; for (let i = 0; i < value.length - 1; i++) if (value[i] === 0xFF && value[i + 1] === 0xD8) { jpeg = true; break; }
      if (jpeg && got > 1500) break;
    }
    ctl.abort();
    return `[ ${jpeg ? '정상' : '문제'} ] ${label}: ${got} bytes, ${jpeg ? 'JPEG 프레임을 받음' : 'JPEG 프레임을 받지 못함'} (${ms()}ms, ${r.headers.get('content-type') || '형식 없음'})`;
  } catch (e) {
    if (e.name === 'AbortError' && !noCors) return `[ 문제 ] ${label}: 7초 안에 응답이 없음`;
    return `[ 문제 ] ${label}: 접속하지 못함 (${e.message})${noCors ? ' - 이 컴퓨터가 아닌 곳에서 접속 중이거나, 포트를 다른 프로그램이 쓰거나, 방화벽이 막은 경우입니다. 이 경우 "연결: 대시보드를 통해"를 쓰세요.' : ''}`;
  }
}
async function runStreamDiag(st) {
  const box = st.diagBox; box.hidden = false;
  const pre = h('pre', { class: 'diag' }, '진단 중입니다 (최대 30초)...');
  const copy = h('button', { type: 'button' }, '진단 결과 복사'); copy.hidden = true;
  box.replaceChildren(pre, copy);
  let rep;
  try { rep = await api('/api/dr/stream/debug'); } catch (e) { pre.textContent = '진단 요청에 실패했습니다: ' + e.message; return; }
  pre.textContent = rep.text + '\n\n== 브라우저 점검 ==\n(진행 중...)';
  const info = (await api('/api/dr/stream').catch(() => null)) || STREAM.info || { ports: [], topics: [] };
  const port = (info.ports && info.ports[0]) || st.port || 8080;
  const live = (info.live && info.live.length) ? info.live : info.topics.filter(t => t.enabled).map(t => t.id);
  const topic = live.includes(st.topic) ? st.topic : (live[0] || st.topic);
  const [w, hh] = st.size.split('x');
  const lines = [];
  lines.push(`브라우저 주소: ${location.origin}  (이 컴퓨터에서 접속 중이면 localhost/127.0.0.1)`);
  lines.push(await probeBrowser(`/api/dr/stream/proxy?port=${port}&topic=${encodeURIComponent(topic)}&width=${w}&height=${hh}&quality=75`, `브라우저 -> 대시보드 -> 시뮬레이터 (중계, 포트 ${port})`, false));
  lines.push(await probeBrowser(`${location.protocol}//${location.hostname}:${port}/stream?topic=${encodeURIComponent(topic)}&width=${w}&height=${hh}&quality=50`, `브라우저 -> 시뮬레이터 직접 (포트 ${port})`, true));
  lines.push(`현재 화면의 상태: ${st.status.textContent || '(없음)'}`);
  const text = rep.text + '\n\n== 브라우저 점검 ==\n' + lines.join('\n');
  pre.textContent = text; copy.hidden = false;
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); copy.textContent = '복사했습니다'; }
    catch (e) { const r = document.createRange(); r.selectNodeContents(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); copy.textContent = '선택했습니다. Ctrl+C 로 복사하세요'; }
  });
}

/* ---------------------------------------------------------------- 실험 정리 (DeepRacer) */
const DRC = { sel: () => [], upd: () => {} };
async function renderDrClean() {
  const msg = $('#dc-msg'); msg.className = 'msg'; msg.textContent = '';
  let rows;
  try { rows = await api('/api/dr/usage'); } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; return; }
  $('#dc-empty').textContent = '삭제할 DeepRacer 실험이 없습니다.';
  $('#dc-empty').hidden = rows.length > 0; $('#dc-body').hidden = rows.length === 0;
  if (!rows.length) return;
  DRC.upd = () => {
    const n = DRC.sel().length, any = $('#dc-config').checked || $('#dc-model').checked || $('#dc-files').checked;
    $('#dc-go').disabled = !(n > 0 && any && $('#dc-confirm').value.trim() === '삭제');
    $('#dc-go').textContent = n ? `선택한 ${n}개 삭제` : '선택한 실험 삭제';
  };
  DRC.sel = selTable($('#dc-table'), [
    { label: '이름', get: r => r.name }, { label: '트랙', get: r => r.world || '–' },
    { label: '상태', get: r => (r.running ? '실행 중' : r.has_model ? '모델 있음' : '모델 없음') },
    { label: '설정', get: r => fmtBytes(r.config_bytes) }, { label: '모델', get: r => fmtBytes(r.model_bytes) }, { label: '파일', get: r => fmtBytes(r.files_bytes) },
    { label: '비고', get: r => (r.used_by.length ? `${r.used_by.join(', ')} 이(가) 이 모델에서 이어서 학습` : '') },
  ], rows, r => (r.running ? '실행 중이라 삭제할 수 없습니다' : ''), () => DRC.upd());
  $('#dc-confirm').value = ''; DRC.upd();
}
function initDrClean() {
  $('#dc-confirm').addEventListener('input', () => DRC.upd());
  ['dc-config', 'dc-model', 'dc-files'].forEach(id => $('#' + id).addEventListener('change', () => DRC.upd()));
  $('#dc-go').addEventListener('click', () => {
    const names = DRC.sel(); if (!names.length) return;
    $('#dc-go').disabled = true;
    startTask($('#dc-task'), () => api('/api/dr/experiments/delete', { names, confirm: true, config: $('#dc-config').checked, model: $('#dc-model').checked, files: $('#dc-files').checked }),
      async () => {
        await refreshExps();
        if (DR.active && !DR.exps.some(e => e.name === DR.active)) { DR.active = DR.exps.length ? DR.exps[0].name : null; DR.met = null; }
        renderDrRail(); await renderDrClean(); await refreshOv();
      });
    DRC.upd();
  });
}

/* ---------------------------------------------------------------- 관리 (dr-* 명령 버튼) */
const LEVEL_KO = { read: '읽기', change: '변경', danger: '주의' };
async function renderAdmin() {
  const box = $('#dra-groups'), sel = $('#dra-exp');
  let info;
  try { info = await api('/api/dr/admin'); } catch (e) { box.replaceChildren(h('p', { class: 'note' }, e.message)); return; }
  const keep = sel.value || DR.active || '';
  sel.replaceChildren(h('option', { value: '' }, '(실험 없이: DRfC 기본 설정)'), ...info.experiments.map(n => h('option', { value: n }, n)));
  sel.value = info.experiments.includes(keep) ? keep : '';
  const after = async () => { DR.bannerSig = ''; await refreshOv(); await refreshExps(); };
  const run = c => {
    if (c.level === 'danger' && !confirm(`${c.label}\n\n${c.desc}\n\n실행할까요?`)) return;
    startTask($('#dra-task'), () => api('/api/dr/admin/run', { id: c.id, experiment: sel.value, confirm: c.level === 'danger' }), after);
  };
  box.replaceChildren(...info.groups.map(g => h('div', { class: 'admgroup' }, h('h3', {}, g.name),
    h('table', { class: 'admtable' }, h('tbody', {}, ...g.commands.map(c => {
      const b = h('button', { type: 'button', class: c.level === 'danger' ? 'danger small' : 'small' }, '실행');
      b.addEventListener('click', () => run(c));
      return h('tr', {}, h('td', { class: 'admname' }, h('code', {}, c.label)), h('td', { class: 'admlvl' }, h('span', { class: 'lvl ' + c.level }, LEVEL_KO[c.level])),
        h('td', { class: 'admdesc' }, c.desc), h('td', {}, b));
    }))))));
  $('#dra-allowed').textContent = info.custom_allowed.join(', ');
}
function runCustomCommand(ev) {
  ev.preventDefault();
  const cmd = $('#dra-cmd').value.trim(); if (!cmd) return;
  if (!confirm(`다음 명령을 실행합니다.\n\n${cmd}\n\n선택한 실험: ${$('#dra-exp').value || '(없음)'}`)) return;
  startTask($('#dra-task'), () => api('/api/dr/admin/custom', { command: cmd, experiment: $('#dra-exp').value, confirm: true }), async () => { DR.bannerSig = ''; await refreshOv(); await refreshExps(); });
}

/* ---------------------------------------------------------------- 실행 중 배너 (어디서든 중지) */
/* 실행 중인 학습/평가가 있으면 DeepRacer 의 모든 화면 위쪽에 나온다. 어떤 실험인지 몰라도(터미널 등 대시보드 밖에서 시작) 중지할 수 있다. */
function renderBanner() {
  const box = $('#dr-banner'), a = DR.ov && DR.ov.active, onDr = !!(S.tab && S.tab.startsWith('dr-'));
  if (!onDr || !(a && (a.running || a.starting))) { if (!box.hidden) { box.hidden = true; box.replaceChildren(); } DR.bannerSig = ''; return; }
  const sig = [S.tab, a.running, a.starting, a.kind, a.experiment, a.external, DR.active].join('|');
  if (sig === DR.bannerSig && box.childElementCount) return;          // 4초마다 다시 그리면 진행 중인 중지 작업 표시가 사라지므로 바뀐 때만
  DR.bannerSig = sig; box.hidden = false;
  const kindKo = a.kind === 'evaluation' ? '평가' : a.kind === 'training' ? '학습' : '학습/평가', ext = !a.experiment;
  const stop = h('button', { class: 'primary small', type: 'button' }, '중지');
  const force = h('button', { class: 'quiet small', type: 'button', title: "DRfC 의 중지 명령으로 안 멈출 때 씁니다. deepracer-* 스택과 학습/시뮬레이터 컨테이너를 직접 지웁니다 (저장된 모델과 minio 는 그대로)." }, '강제 중지');
  const task = h('div', { class: 'taskbox', hidden: '' });
  const done = async () => { DR.bannerSig = ''; await refreshOv(); await refreshExps(); await loadActive(); renderForTab(); };
  stop.addEventListener('click', () => startTask(task, () => api('/api/dr/stop', {}), done));
  force.addEventListener('click', () => {
    if (!confirm('강제 중지는 DRfC 의 중지 명령 대신 deepracer-* 스택과 학습/시뮬레이터 컨테이너를 직접 지웁니다. 저장된 모델은 지워지지 않습니다. 계속할까요?')) return;
    startTask(task, () => api('/api/dr/stop', { force: true }), done);
  });
  const parts = [h('span', { class: 'mark running' }, `${kindKo} ${a.running ? '실행 중' : '시작 중'}`), h('b', {}, a.experiment || '(외부에서 시작됨)')];
  if (a.experiment && a.experiment !== DR.active) {
    const go = h('button', { class: 'quiet small', type: 'button' }, '그 실험 보기'); go.addEventListener('click', () => setActiveExp(a.experiment, true)); parts.push(go);
  }
  parts.push(stop, force);
  const note = ext ? h('p', { class: 'note', style: 'margin:6px 0 0;flex-basis:100%' }, '대시보드에서 시작한 것이 아니거나, 어떤 실험인지 알 수 없는 실행입니다 (터미널에서 시작했거나 시작 기록이 없는 경우). 중지하면 학습과 평가를 모두 멈춥니다.') : null;
  box.replaceChildren(...parts, ...(note ? [note] : []), h('div', { style: 'flex-basis:100%' }, task));
}

/* ---------------------------------------------------------------- 공개 인터페이스 */
const DRUI = {
  defaultTab() { return !DR.ov || !DR.ov.valid || !DR.ov.initialized ? 'dr-env' : (DR.exps.length ? 'dr-run' : 'dr-new'); },
  appendLive(box) {
    const ov = DR.ov, a = ov && ov.active;
    if (a && (a.running || a.starting)) {
      const kindKo = a.kind === 'evaluation' ? '평가' : a.kind === 'training' ? '학습' : '학습/평가';
      const go = h('button', { class: 'livebtn', type: 'button', title: '눌러서 중지 버튼이 있는 DeepRacer 화면으로 이동' }, h('span', { class: 'mark running' }, `DeepRacer ${kindKo} ${a.running ? '중' : '시작 중'}`), h('b', {}, a.experiment || '(외부에서 시작됨)'), h('span', { class: 'gotostop' }, '중지'));
      go.addEventListener('click', () => setTab(S.tab && S.tab.startsWith('dr-') ? S.tab : 'dr-run'));
      box.append(go);
    }
  },
  leave(tab) { if (!String(tab).startsWith('dr-')) { const b = $('#dr-banner'); b.hidden = true; b.replaceChildren(); DR.bannerSig = ''; }
    if (tab !== 'dr-run' && tab !== 'dr-eval') for (const st of Object.values(STREAM.panels)) if (st.playing || st.img.getAttribute('src')) stopStream(st, STREAM_IDLE); },
  async enter(tab) {
    DR.bannerSig = ''; renderBanner();
    if (tab === 'dr-env') renderEnv();
    else if (tab === 'dr-clean') renderDrClean();
    else if (tab === 'dr-admin') renderAdmin();
    else if (tab === 'dr-new') initNew();
    else { await refreshExps(); if (!DR.active && DR.exps.length) DR.active = (DR.exps.find(e => e.running) || DR.exps[0]).name; renderDrRail(); await loadActive(); renderForTab(); if (tab === 'dr-run' || tab === 'dr-eval') refreshStream(); }
  },
  init() {
    if (DR.inited) return; DR.inited = true;
    $('#dr-recheck').addEventListener('click', renderEnv);
    $('#dr-dirform').addEventListener('submit', async ev => {
      ev.preventDefault(); const m = $('#dr-dirmsg'); m.className = 'msg'; m.textContent = '';
      try { await api('/api/dr/settings', { drfc_dir: $('#dr-dir').value }); m.className = 'msg ok'; m.textContent = '지정했습니다.'; await refreshOv(); await refreshExps(); renderEnv(); }
      catch (e) { m.className = 'msg err'; m.textContent = e.message; }
    });
    for (const [id, key] of [['dn-algo', 'algo'], ['dn-atype', 'atype']]) $$('#' + id + ' button').forEach(b => b.addEventListener('click', () => { DR[key] = b.dataset.v; applyAlgo(); }));
    $('#dn-steer').addEventListener('input', updateDnCount); $('#dn-speed').addEventListener('input', updateDnCount);
    $('#dn-source').addEventListener('change', onSourceChange);
    $('#dn-example').addEventListener('change', e => { const ex = DR.tpl.examples.find(x => x.name === e.target.value); if (ex) $('#dn-code').value = ex.code; });
    $('#dn-code').addEventListener('keydown', e => { if (e.key === 'Tab') { e.preventDefault(); const t = e.target; t.setRangeText('    ', t.selectionStart, t.selectionEnd, 'end'); } });
    $('#dn-lint').addEventListener('click', lintNow);
    $('#dr-newform').addEventListener('submit', ev => { ev.preventDefault(); submitNew(false); });
    $('#dn-makego').addEventListener('click', () => { if ($('#dr-newform').reportValidity()) submitNew(true); });
    $('#drr-start').addEventListener('click', startTrain);
    $('#drr-stop').addEventListener('click', () => startTask($('#drr-task'), () => api('/api/dr/stop', {}), async () => { await refreshOv(); await refreshExps(); await loadActive(); renderRun(); }));
    $('#dra-custom').addEventListener('submit', runCustomCommand);
    $('#dre-model').addEventListener('change', ev => { if (ev.target.value) setActiveExp(ev.target.value, false); });
    $('#dre-stop').addEventListener('click', () => startTask($('#dre-task'), () => api('/api/dr/stop', {}), async () => { await refreshOv(); await refreshExps(); renderEval(); }));
    $('#dre-form').addEventListener('submit', startEval);
    $('#drx-form').addEventListener('submit', makeCar);
    $$('#drr-logrole button').forEach(b => b.addEventListener('click', () => { DR.logRole = b.dataset.v; setSeg('drr-logrole', DR.logRole); loadLog(true); }));
    $('#drr-logbtn').addEventListener('click', () => loadLog(true));
    buildStream('drr-stream', 'dr-run'); buildStream('dre-stream', 'dr-eval'); initDrClean();
    refreshOv().then(refreshExps);
    setInterval(() => { if (!document.hidden) { refreshOv(); if (S.tab === 'dr-run' || S.tab === 'dr-eval') refreshStream(); } }, 4000);
    setInterval(async () => {
      if (document.hidden) return;
      const running = DR.ov && DR.ov.active && DR.ov.active.running;
      if (S.tab === 'dr-run' && DR.active && (running || !DR.met)) { await refreshExps(); await loadActive(); renderRun(); if ($('#drr-logauto').checked && running) loadLog(false); }
      else if (S.tab === 'dr-eval' && running) { await loadActive(); renderEval(); }
      else if (S.product === 'dr' && S.tab !== 'dr-new' && S.tab !== 'dr-env') refreshExps();
    }, 3000);
  },
};
