"""DeepRacer Trainer 대시보드 - 내 컴퓨터에서만 열리는 로컬 웹 서버 (MiniRacer 시뮬레이터 + DeepRacer(DRfC) 학습 관리).

  python dashboard.py                 # 브라우저가 자동으로 열립니다 (http://127.0.0.1:8765)
  python dashboard.py --port 9000
  python dashboard.py --no-browser

표준 라이브러리와 numpy 만 사용합니다. 학습은 별도 프로세스(train.py)로 실행되고,
대시보드는 runs/ 폴더의 파일(metrics.csv, trace_*.csv, status.json)을 읽어서 보여 줍니다.
터미널에서 직접 실행한 학습(python train.py ...)도 같은 폴더에 저장되므로 대시보드에 그대로 나타납니다.

보안: 기본값은 이 컴퓨터(127.0.0.1)에서만 접속됩니다. 이 서버는 사용자가 입력한 보상함수 코드를
실행하므로, --host 로 외부 접속을 열면 같은 네트워크의 누구나 이 컴퓨터에서 코드를 실행할 수 있습니다.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import csv
import http.client
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, parse_qs, urlparse

import numpy as np

from miniracer.playback import play_trajectories
from miniracer.ppo import DEFAULT_HP, load_hyperparameters
from miniracer.track import Track, list_tracks
from trainer import drfc

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
TRACKS = os.path.join(HERE, "tracks")
WEB = os.path.join(HERE, "web")
CUSTOM = os.path.join(HERE, "custom_files")
EXAMPLES = os.path.join(HERE, "examples", "rewards")

NAME_RE = re.compile(r"^[0-9A-Za-z_\-\uAC00-\uD7A3]{1,40}$")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/dr.js": ("dr.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}

jobs = {}                # 대시보드에서 시작한 학습 프로세스: 이름 -> {proc, out_path, started, track}
_track_cache, _csv_cache = {}, {}
_lock = threading.Lock()


# ----------------------------------------------------------------------------- 유틸
def clean(o):
    """JSON 으로 보낼 수 있게 변환 (NaN/inf -> null, numpy -> 기본 타입)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def read_csv(path):
    """CSV -> {열: numpy 배열}. 수정 시각이 같으면 캐시를 쓰고, 학습 중이라 덜 써진 마지막 줄은 버린다."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _csv_cache.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    head = rows[0]
    body = [r for r in rows[1:] if len(r) == len(head)]
    cols = {}
    for j, h in enumerate(head):
        raw = [r[j] for r in body]
        try:
            cols[h] = np.array([float(v) if v != "" else np.nan for v in raw])
        except ValueError:
            cols[h] = np.array(raw, dtype=object)
    _csv_cache[path] = (mt, cols)
    return cols


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_text(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def get_track(name):
    if not NAME_RE.match(name or ""):
        raise ValueError("트랙 이름이 올바르지 않습니다.")
    path = os.path.join(TRACKS, name + ".npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"트랙 '{name}' 이(가) tracks/ 에 없습니다. python fetch_tracks.py 로 내려받을 수 있습니다.")
    if name not in _track_cache:
        _track_cache[name] = Track(path)
    return _track_cache[name]


def run_dir_of(name):
    if not NAME_RE.match(name or ""):
        raise ValueError("실험 이름이 올바르지 않습니다.")
    d = os.path.join(RUNS, name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"실험 '{name}' 을(를) 찾을 수 없습니다.")
    return d


def tail(text, n=40):
    return "\n".join(text.splitlines()[-n:])


# ----------------------------------------------------------------------------- 실험 상태
def run_state(name):
    d = os.path.join(RUNS, name)
    st = read_json(os.path.join(d, "status.json"), {}) or {}
    job = jobs.get(name)
    alive = bool(job and job["proc"].poll() is None)
    state = st.get("state")
    if state == "running" and not alive and time.time() - st.get("updated", 0) > 90:
        state = "aborted"       # 프로세스가 사라졌는데 완료 기록이 없음 (비정상 종료)
    if state is None:
        state = "starting" if alive else ("aborted" if job else "unknown")
    return {"state": state, "iteration": st.get("iteration", 0), "max_iterations": st.get("max_iterations"),
            "episodes": st.get("episodes", 0), "elapsed_s": st.get("elapsed_s", 0), "stop_reason": st.get("stop_reason")}


def best_iteration(m):
    """train.py 의 best 모델 선택 기준과 같은 기준."""
    best, best_key = None, None
    for i in range(len(m.get("iteration", []))):
        lap = m["eval_lap_time_mean"][i]
        key = (round(min(100.0, float(m["eval_progress"][i])), 1), -(1e9 if np.isnan(lap) else float(lap)))
        if best_key is None or key >= best_key:
            best, best_key = int(m["iteration"][i]), key
    return best


def summarize(name):
    d = os.path.join(RUNS, name)
    info = read_json(os.path.join(d, "run_info.json"), {}) or {}
    m = read_csv(os.path.join(d, "metrics.csv"))
    s = run_state(name)
    out = {"name": name, "track": info.get("track"), "started": info.get("started"), **s,
           "best_progress": None, "best_lap": None, "last_progress": None, "iterations_done": 0}
    if m and len(m.get("iteration", [])):
        laps = m["eval_lap_time_best"]
        out["best_progress"] = float(np.nanmax(np.minimum(m["eval_progress"], 100.0)))
        out["best_lap"] = float(np.nanmin(laps)) if np.isfinite(laps).any() else None
        out["last_progress"] = float(m["eval_progress"][-1])
        out["iterations_done"] = int(m["iteration"][-1])
    return out


def list_runs():
    rows = []
    if os.path.isdir(RUNS):
        for n in os.listdir(RUNS):
            if n.startswith(("_", ".")) or not os.path.isfile(os.path.join(RUNS, n, "run_info.json")):
                continue
            rows.append(summarize(n))
    for n, job in jobs.items():          # 아직 폴더가 만들어지지 않은 학습
        if not any(r["name"] == n for r in rows):
            alive = job["proc"].poll() is None
            rows.append({"name": n, "track": job["track"], "started": job["started"], "state": "starting" if alive else "failed",
                         "iteration": 0, "max_iterations": None, "iterations_done": 0, "best_progress": None,
                         "best_lap": None, "last_progress": None, "pending": True})
    rows.sort(key=lambda r: r.get("started") or "", reverse=True)
    return rows


# ----------------------------------------------------------------------------- API 본문
def api_info():
    hp = load_hyperparameters(os.path.join(CUSTOM, "hyperparameters.json"))
    meta = read_json(os.path.join(CUSTOM, "model_metadata.json"), {}) or {}
    acts = meta.get("action_space", [])
    examples = []
    if os.path.isdir(EXAMPLES):
        for f in sorted(os.listdir(EXAMPLES)):
            if f.endswith(".py"):
                examples.append({"name": f[:-3], "code": read_text(os.path.join(EXAMPLES, f))})
    return {"tracks": list_tracks(TRACKS), "examples": examples, "hp": hp,
            "reward_code": read_text(os.path.join(CUSTOM, "reward_function.py")),
            "steer": sorted({a["steering_angle"] for a in acts}), "speed": sorted({a["speed"] for a in acts})}


def api_track(name):
    t = get_track(name)
    r = lambda a: [[round(float(x), 3), round(float(y), 3)] for x, y in a]
    return {"name": t.name, "outer": r(t.outer), "inner": r(t.inner), "center": r(t.center), "bounds": list(map(float, t.bounds)),
            "length": t.length, "width": t.width}


def api_run(name):
    d = run_dir_of(name)
    m = read_csv(os.path.join(d, "metrics.csv"))
    metrics = {k: v for k, v in m.items()} if m else {}
    job = jobs.get(name)
    return {"name": name, "info": read_json(os.path.join(d, "run_info.json"), {}), "state": run_state(name),
            "hp": read_json(os.path.join(d, "hyperparameters.json"), {}), "metadata": read_json(os.path.join(d, "model_metadata.json"), {}),
            "reward_code": read_text(os.path.join(d, "reward_function.py")), "metrics": metrics,
            "best_iteration": best_iteration(m) if m else None,
            "log": tail(read_text(os.path.join(d, "train.log")) or (read_text(job["out_path"]) if job else ""))}


def api_replay(name, iteration):
    d = run_dir_of(name)
    tr = read_csv(os.path.join(d, "trace_eval.csv"))
    if not tr or not len(tr.get("iteration", [])):
        return {"iteration": None, "episodes": []}
    its = tr["iteration"]
    iteration = int(its.max()) if iteration is None else int(iteration)
    mask = its == iteration
    eps = []
    for e in np.unique(tr["episode"][mask]):
        sel = mask & (tr["episode"] == e)
        status = str(tr["episode_status"][sel][-1])
        steps = int(sel.sum())
        eps.append({"status": status, "progress": min(100.0, float(tr["progress"][sel].max())), "steps": steps,
                    "lap_time": round(steps / 15.0, 2) if status == "lap_complete" else None,
                    "x": np.round(tr["x"][sel], 3), "y": np.round(tr["y"][sel], 3), "speed": np.round(tr["speed"][sel], 2),
                    "steer": np.round(tr["steering_angle"][sel], 1), "progress_t": np.round(np.minimum(tr["progress"][sel], 100), 1)})
    return {"iteration": iteration, "episodes": eps}


def api_heat(name):
    d = run_dir_of(name)
    tt = read_csv(os.path.join(d, "trace_train.csv"))
    if not tt or not len(tt.get("iteration", [])):
        return {"x": [], "y": [], "r": [], "from_iteration": None}
    its = np.unique(tt["iteration"])
    first = its[int(len(its) * 0.6)] if len(its) > 2 else its[0]
    idx = np.flatnonzero(tt["iteration"] >= first)
    if len(idx) > 6000:
        idx = np.random.default_rng(0).choice(idx, 6000, replace=False)
    return {"x": np.round(tt["x"][idx], 3), "y": np.round(tt["y"][idx], 3), "r": np.round(tt["reward"][idx], 3),
            "from_iteration": int(first)}


def api_usage(name):
    d = run_dir_of(name)
    tt = read_csv(os.path.join(d, "trace_train.csv"))
    meta = read_json(os.path.join(d, "model_metadata.json"), {}) or {}
    acts = sorted(meta.get("action_space", []), key=lambda a: a.get("index", 0))
    if not tt or not len(tt.get("iteration", [])) or not acts:
        return {"iterations": [], "speed": {}, "steer": {}}
    a = tt["action"].astype(int)
    speeds = np.array([acts[i]["speed"] for i in a])
    steers = np.array([acts[i]["steering_angle"] for i in a])
    its = np.unique(tt["iteration"])
    out = {"iterations": its, "speed": {}, "steer": {}}
    for key, arr in (("speed", speeds), ("steer", steers)):
        for v in sorted(set(arr)):
            out[key][f"{v:g}"] = [100.0 * float(np.mean(arr[tt["iteration"] == i] == v)) for i in its]
    return out


# ----------------------------------------------------------------------------- 학습 시작/중지
def _num(v, typ, lo, hi, label):
    try:
        v = typ(v)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 값이 숫자가 아닙니다: {v!r}")
    if not (lo <= v <= hi):
        raise ValueError(f"{label} 값은 {lo}~{hi} 범위여야 합니다.")
    return v


def api_train(body):
    name = (body.get("name") or "").strip()
    if not NAME_RE.match(name):
        raise ValueError("실험 이름은 한글, 영문, 숫자, _ , - 만 쓸 수 있고 40자 이하여야 합니다.")
    track = body.get("track") or ""
    get_track(track)
    code = body.get("reward_code") or ""
    if "def reward_function" not in code:
        raise ValueError("보상함수 코드에 def reward_function(params): 가 있어야 합니다.")
    if sum(1 for j in jobs.values() if j["proc"].poll() is None) >= 3:
        raise ValueError("동시에 실행할 수 있는 학습은 3개까지입니다. 하나가 끝난 뒤 다시 시도하세요.")

    # 하이퍼파라미터: 허용된 키만, 기본값과 같은 타입으로 변환
    hp_in = body.get("hp") or {}
    hp = {}
    for k, v in hp_in.items():
        if k not in DEFAULT_HP:
            continue
        d = DEFAULT_HP[k]
        if isinstance(d, bool):
            hp[k] = bool(v)
        elif isinstance(d, int):
            hp[k] = int(_num(v, float, -1e9, 1e9, k))
        elif isinstance(d, float):
            hp[k] = _num(v, float, -1e9, 1e9, k)
        else:
            hp[k] = v if k == "max_steps_per_episode" and v == "auto" else (v if isinstance(v, str) else str(v))
    hp["max_steps_per_episode"] = hp.get("max_steps_per_episode", "auto")
    if hp.get("loss_type") not in (None, "huber", "mean squared error"):
        raise ValueError("loss_type 은 huber 또는 mean squared error 여야 합니다.")
    if "num_episodes_between_training" in hp and not 1 <= hp["num_episodes_between_training"] <= 200:
        raise ValueError("num_episodes_between_training 은 1~200 이어야 합니다.")
    if "term_cond_max_episodes" in hp and not 20 <= hp["term_cond_max_episodes"] <= 20000:
        raise ValueError("term_cond_max_episodes 는 20~20000 이어야 합니다.")

    # 행동 공간: 조향각 목록 x 속도 목록
    steers = [_num(s, float, -30, 30, "조향각") for s in (body.get("steer") or [])]
    speeds = [_num(s, float, 0.1, 5, "속도") for s in (body.get("speed") or [])]
    if not steers or not speeds:
        raise ValueError("행동 공간에는 조향각과 속도가 각각 1개 이상 필요합니다.")
    if len(steers) * len(speeds) > 60:
        raise ValueError("행동이 너무 많습니다. (조향각 수 x 속도 수 <= 60)")
    actions = [{"steering_angle": s, "speed": v, "index": i} for i, (s, v) in enumerate((s, v) for s in steers for v in speeds)]

    base, name_final, k = name, name, 2
    while os.path.exists(os.path.join(RUNS, name_final)) or name_final in jobs:
        name_final, k = f"{base}_{k}", k + 1
    name = name_final

    tmp = tempfile.mkdtemp(prefix="miniracer_")
    rp, hpp, mp = (os.path.join(tmp, f) for f in ("reward_function.py", "hyperparameters.json", "model_metadata.json"))
    with open(rp, "w", encoding="utf-8") as f:
        f.write(code)
    with open(hpp, "w", encoding="utf-8") as f:
        json.dump(hp, f)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"action_space": actions, "sensor": ["RAY_SENSOR_120DEG"], "neural_network": "MLP_ON_RAYS",
                   "training_algorithm": "clipped_ppo", "action_space_type": "discrete", "version": "MiniRacer-1"}, f, indent=2)

    out_path = os.path.join(tmp, "out.txt")
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--no-render", "--name", name, "--track", track,
           "--reward", rp, "--hp", hpp, "--metadata", mp, "--seed", str(int(body.get("seed") or 0))]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    out = open(out_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, env=env)
    jobs[name] = {"proc": proc, "out_path": out_path, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "track": track}
    return {"name": name}


def api_job(name):
    job = jobs.get(name)
    if not job:
        return {"known": False}
    rc = job["proc"].poll()
    return {"known": True, "alive": rc is None, "returncode": rc, "output": tail(read_text(job["out_path"]), 30)}


def api_stop(name):
    d = os.path.join(RUNS, name)
    if os.path.isdir(d):
        with open(os.path.join(d, "STOP"), "w") as f:
            f.write("stop")
        return {"ok": True, "how": "다음 iteration 이 끝나면 멈춥니다."}
    job = jobs.get(name)
    if job and job["proc"].poll() is None:
        job["proc"].terminate()
        return {"ok": True, "how": "시작 중인 학습을 종료했습니다."}
    raise FileNotFoundError("중지할 학습이 없습니다.")


def api_play(body):
    name = body.get("run") or ""
    d = run_dir_of(name)
    model = body.get("model") if body.get("model") in ("best", "final") else "best"
    track = body.get("track") or None
    if track:
        get_track(track)
    laps = int(_num(body.get("laps", 3), float, 1, 6, "시도 횟수"))
    noise = _num(body.get("noise", 0), float, 0, 5, "외란")
    seed = int(body.get("seed") or 0)
    return play_trajectories(d, TRACKS, model, track, noise, laps, seed)


# ----------------------------------------------------------------------------- MiniRacer 실험 정리
def runs_usage():
    rows = []
    for r in list_runs():
        if r.get("pending"):
            continue
        d = os.path.join(RUNS, r["name"])
        rows.append({"name": r["name"], "track": r.get("track"), "state": r.get("state"), "started": r.get("started"),
                     "iterations": r.get("iterations_done"), "best_lap": r.get("best_lap"), "bytes": drfc.dir_size(d)})
    return rows


def delete_runs(body):
    """MiniRacer 실험 기록(runs/<이름>/)을 삭제한다 (되돌릴 수 없음). 학습 중인 기록은 지우지 않는다."""
    if body.get("confirm") is not True:
        raise ValueError("삭제를 확인하는 값이 없습니다.")
    names = list(dict.fromkeys(body.get("names") or []))
    if not names:
        raise ValueError("삭제할 실험을 고르세요.")
    for n in names:
        run_dir_of(n)                                                   # 이름 검증과 존재 확인
        if run_state(n)["state"] in ("running", "starting"):
            raise ValueError(f"'{n}' 은(는) 지금 학습 중입니다. 먼저 중지하세요.")
    for n in names:
        shutil.rmtree(os.path.join(RUNS, n), ignore_errors=True)
        jobs.pop(n, None)
    for k in [k for k in _csv_cache if any(k.startswith(os.path.join(RUNS, n) + os.sep) for n in names)]:
        _csv_cache.pop(k, None)
    return {"deleted": names}


# ----------------------------------------------------------------------------- DeepRacer (DRfC)
def dr_delete(body):
    if body.get("confirm") is not True:
        raise ValueError("삭제를 확인하는 값이 없습니다.")
    return drfc.delete_experiments(body.get("names") or [], config=bool(body.get("config", True)), model=bool(body.get("model", True)),
                                   files=bool(body.get("files", True)))


def dr_templates():
    t = drfc.templates()
    t["examples"] = api_info()["examples"]
    t["models"] = drfc.models_in_bucket()
    t["minirace_tracks"] = list_tracks(TRACKS)
    t["minirace_runs"] = [{"name": r["name"], "track": r["track"]} for r in list_runs() if r.get("iterations_done")]
    return t


def dr_import_minirace(name):
    """MiniRacer 실험의 보상함수와 행동 공간을 DeepRacer 새 실험 양식에 채울 값으로 돌려준다."""
    d = run_dir_of(name)
    meta = read_json(os.path.join(d, "model_metadata.json"), {}) or {}
    acts = meta.get("action_space", [])
    steer = sorted({a["steering_angle"] for a in acts})
    speed = sorted({a["speed"] for a in acts})
    full = len(acts) == len(steer) * len(speed)
    info = read_json(os.path.join(d, "run_info.json"), {}) or {}
    warn = []
    if not full:
        warn.append("MiniRacer 의 행동이 '조향각 x 속도' 격자가 아니라서 격자로 바꿔 가져왔습니다. 행동 수가 달라질 수 있습니다.")
    if speed and (max(speed) > 4 or min(speed) < 0.5):
        warn.append("속도 범위가 0.5~4 m/s 밖입니다. 실제 DeepRacer 차량의 속도 한계를 확인하세요.")
    return {"reward_code": read_text(os.path.join(d, "reward_function.py")), "steer": steer, "speed": speed, "track": info.get("track"), "warnings": warn}


def dr_create(body):
    spec = body.get("spec") or {}
    return drfc.create_experiment(spec)


def _send_file_route(handler, file):
    p = drfc.download_path(file)
    with open(p, "rb") as f:
        data = f.read()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/gzip")
    handler.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(p)}"')
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


# ----------------------------------------------------------------------------- HTTP
# 접속을 허용하는 Host 이름. None 이면 검사하지 않는다 (--host 0.0.0.0 처럼 사용자가 일부러 외부에 연 경우).
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniRacer"

    def log_message(self, fmt, *args):      # 2초마다 오는 요청 로그로 터미널이 도배되지 않게
        pass

    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _guard(self, post, api=True):
        """다른 웹사이트가 이 대시보드(127.0.0.1)로 요청을 보내 명령을 실행시키는 것(CSRF, DNS rebinding)을 막는다.
        Host 가 허용된 이름인지, 브라우저가 보낸 Origin 이 같은 출처인지, 다른 사이트에서 온 요청이 아닌지, POST 가 JSON 인지 확인한다."""
        host = self.headers.get("Host") or ""
        low = host.lower()
        hostname = low.split("]")[0] + "]" if low.startswith("[") else low.rsplit(":", 1)[0]
        if ALLOWED_HOSTS is not None and hostname not in ALLOWED_HOSTS:
            self._json(403, {"error": f"허용되지 않은 접속 주소입니다 ({host}). http://127.0.0.1:포트 또는 http://localhost:포트 로 접속하세요."})
            return False
        if api or post:
            origin = self.headers.get("Origin")
            if (origin is not None and urlparse(origin).netloc != host) or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._json(403, {"error": "다른 사이트에서 온 요청은 처리하지 않습니다."})
                return False
        if post and (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            self._json(415, {"error": "요청 형식이 올바르지 않습니다 (Content-Type: application/json 필요)."})
            return False
        return True

    def _json(self, code, obj):
        self._send(code, json.dumps(clean(obj), ensure_ascii=False, allow_nan=False))

    def _handle(self, fn):
        try:
            self._json(200, fn())
        except (ValueError, FileNotFoundError, KeyError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def _serve_video(self, q):
        """저장된 평가 영상(MP4)을 브라우저로 보낸다. 재생 위치를 옮길 수 있도록 Range 요청을 지원한다 (구간만 읽어서 보낸다)."""
        try:
            v = drfc.open_video(q.get("name", ""), q.get("key", ""), self.headers.get("Range"))
        except (drfc.DrfcError, ValueError, FileNotFoundError, KeyError) as e:
            return self._json(400, {"error": str(e)})
        try:
            if v["status"] == 416:
                self.send_response(416)
                for k, val in v["headers"].items():
                    self.send_header(k, val)
                self.end_headers()
                return
            self.send_response(v["status"])
            for k, val in v["headers"].items():
                self.send_header(k, val)
            if q.get("download"):
                fn = os.path.basename(q.get("key", "video.mp4")).replace('"', "")
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(fn))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for chunk in v["chunks"]:
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                     # 재생 위치를 옮기거나 닫으면 브라우저가 연결을 끊는다
        finally:
            v["close"]()
            self.close_connection = True

    def _proxy_stream(self, q):
        """시뮬레이터의 영상 스트림(MJPEG)을 브라우저로 중계한다. 브라우저는 대시보드 포트 하나만 접속하면 되므로, 원격 접속(SSH 터널), 포트 충돌,
        방화벽이 있어도 영상이 나온다. 접속하는 곳은 이 컴퓨터(127.0.0.1)의 DRfC 영상 포트(8080~8089, 8180~8189)로 제한한다."""
        try:
            port, path = drfc.stream_proxy_args(q)
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        conn = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            conn.request("GET", path)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            if conn:
                conn.close()
            return self._json(502, {"error": f"시뮬레이터의 영상 서버(포트 {port})에 연결할 수 없습니다: {type(e).__name__}: {e}"})
        if resp.status != 200:
            body = resp.read(300).decode("utf-8", "replace")
            conn.close()
            return self._json(502, {"error": f"시뮬레이터의 영상 서버가 {resp.status} 로 응답했습니다: {body}"})
        try:
            self.send_response(200)
            self.send_header("Content-Type", resp.getheader("Content-Type") or "multipart/x-mixed-replace")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, http.client.HTTPException):
            pass                                    # 브라우저가 연결을 끊었거나 시뮬레이터가 종료됨
        finally:
            conn.close()
            self.close_connection = True

    def do_GET(self):
        u = urlparse(self.path)
        if not self._guard(False, api=u.path.startswith("/api/")):
            return
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in STATIC:
            fname, ctype = STATIC[u.path]
            try:
                with open(os.path.join(WEB, fname), "rb") as f:
                    return self._send(200, f.read(), ctype)
            except OSError:
                return self._send(404, "not found", "text/plain; charset=utf-8")
        routes = {
            "/api/info": api_info,
            "/api/runs": list_runs,
            "/api/track": lambda: api_track(q.get("name", "")),
            "/api/run": lambda: api_run(q.get("name", "")),
            "/api/replay": lambda: api_replay(q.get("name", ""), q.get("iteration")),
            "/api/heat": lambda: api_heat(q.get("name", "")),
            "/api/usage": lambda: api_usage(q.get("name", "")),
            "/api/job": lambda: api_job(q.get("name", "")),
            "/api/dr/overview": drfc.overview,
            "/api/dr/admin": drfc.admin_info,
            "/api/dr/doctor": drfc.doctor,
            "/api/dr/templates": dr_templates,
            "/api/dr/experiments": drfc.list_experiments,
            "/api/dr/experiment": lambda: drfc.experiment_detail(q.get("name", "")),
            "/api/dr/metrics": lambda: drfc.training_metrics(q.get("name", "")),
            "/api/dr/evals": lambda: drfc.eval_results(q.get("name", "")),
            "/api/dr/logs": lambda: drfc.logs(q.get("role", "train"), q.get("tail", 200)),
            "/api/dr/tasks": drfc.list_tasks,
            "/api/dr/task": lambda: drfc.get_task(q.get("id", "")),
            "/api/dr/import-minirace": lambda: dr_import_minirace(q.get("run", "")),
            "/api/dr/stream": drfc.stream_info,
            "/api/dr/stream/debug": drfc.stream_debug,
            "/api/dr/usage": drfc.experiments_usage,
            "/api/runs/usage": runs_usage,
        }
        if u.path == "/api/dr/stream/proxy":
            return self._proxy_stream(q)
        if u.path == "/api/dr/video":
            return self._serve_video(q)
        if u.path == "/api/dr/download":
            try:
                return _send_file_route(self, q.get("file", ""))
            except (ValueError, OSError) as e:
                return self._json(400, {"error": str(e)})
        if u.path in routes:
            return self._handle(routes[u.path])
        self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        u = urlparse(self.path)
        if not self._guard(True):
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
        except Exception:
            return self._json(400, {"error": "요청 형식이 올바르지 않습니다."})
        routes = {"/api/train": lambda: api_train(body), "/api/stop": lambda: api_stop(body.get("name", "")),
                  "/api/play": lambda: api_play(body),
                  "/api/dr/settings": lambda: {"drfc_dir": drfc.set_dir(body.get("drfc_dir", ""))},
                  "/api/dr/gpu-test": drfc.gpu_test,
                  "/api/dr/minio-image": drfc.build_minio_image,
                  "/api/dr/swarm": drfc.create_swarm,
                  "/api/dr/admin/run": lambda: drfc.admin_run(body.get("id", ""), body.get("experiment"), bool(body.get("confirm"))),
                  "/api/dr/admin/custom": lambda: drfc.admin_custom(body.get("command", ""), body.get("experiment"), bool(body.get("confirm"))),
                  "/api/dr/eval/label": lambda: drfc.set_eval_label(body.get("name", ""), body.get("file", ""), body.get("label", "")),
                  "/api/dr/viewer/start": drfc.start_viewer,
                  "/api/dr/viewer/stop": drfc.stop_viewer,
                  "/api/dr/experiments/delete": lambda: dr_delete(body),
                  "/api/runs/delete": lambda: delete_runs(body),
                  "/api/dr/arch": lambda: drfc.set_arch(body.get("arch"), body.get("cuda_devices"), body.get("workers")),
                  "/api/dr/lint": lambda: drfc.lint_reward(body.get("code", "")),
                  "/api/dr/experiment/create": lambda: dr_create(body),
                  "/api/dr/train/start": lambda: drfc.start_training(body.get("name", ""), bool(body.get("wipe"))),
                  "/api/dr/stop": lambda: drfc.stop_training(bool(body.get("force"))),
                  "/api/dr/eval/start": lambda: drfc.start_evaluation(body.get("name", ""), body),
                  "/api/dr/export": lambda: drfc.export_car(body.get("name", ""), bool(body.get("best"))),
                  "/api/dr/task/cancel": lambda: drfc.cancel_task(body.get("id", ""))}
        if u.path in routes:
            return self._handle(routes[u.path])
        self._send(404, "not found", "text/plain; charset=utf-8")


def main():
    p = argparse.ArgumentParser(description="DeepRacer Trainer 대시보드")
    p.add_argument("--host", default="127.0.0.1", help="기본값은 이 컴퓨터에서만 접속 (바꾸면 위험, 파일 상단 설명 참고)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--strict-port", action="store_true", help="지정한 포트가 사용 중이면 다른 포트를 찾지 않고 종료 (Docker 상태 확인용)")
    args = p.parse_args()
    global ALLOWED_HOSTS
    ALLOWED_HOSTS = None if args.host in ("0.0.0.0", "::", "") else ALLOWED_HOSTS | {args.host.lower()}

    os.makedirs(RUNS, exist_ok=True)
    server = None
    for port in range(args.port, args.port + (1 if args.strict_port else 10)):
        try:
            server = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit(f"포트 {args.port}~{args.port + 9} 가 모두 사용 중입니다. --port 로 다른 번호를 지정하세요.")
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_address[1]}"
    print(f"DeepRacer Trainer 대시보드: {url}   (종료: Ctrl+C)")
    if args.host not in ("127.0.0.1", "localhost"):
        print("[주의] 외부 접속이 열려 있습니다. 같은 네트워크의 누구나 이 컴퓨터에서 보상함수 코드를 실행할 수 있습니다.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n대시보드를 종료합니다. (실행 중인 학습은 계속 진행됩니다.)")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
