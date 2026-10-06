"""DRfC(DeepRacer-for-Cloud) 연동.

원칙
  - DRfC 의 공식 명령(dr-*)만 호출한다. DRfC 내부 파일은 system.env 의 시뮬레이터 이미지 태그 한 줄(GPU/CPU 전환) 외에는 건드리지 않는다.
  - 실험 하나 = DRfC 의 experiments/<이름>/ 폴더 하나 (run.env + custom_files/). 활성화는 `source bin/activate.sh -e <이름>`.
  - 사용자 입력은 모두 검증하고, 실행할 명령은 이 파일에 정해 둔 것만 쓴다 (임의 셸 실행 없음).
  - 설치(init.sh)는 sudo 가 필요해서 대신 실행하지 않고 터미널에 붙여 넣을 명령으로 안내한다.

DRfC 사양은 저장소(2026-08 기준)의 docs/reference.md, bin/*.sh, scripts/*, defaults/* 를 읽고 맞췄다.
"""
import http.client
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
import uuid

from . import metrics as M
from .envfile import fmt_value, parse_env, set_env
from .s3 import S3Client, S3Error

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.environ.get("TRAINER_STATE_DIR") or HERE          # Docker 에서는 호스트에 남는 폴더
SETTINGS_FILE = os.path.join(STATE_DIR, "trainer_settings.json")
STATE_FILE = os.path.join(STATE_DIR, "trainer_state.json")
TASK_DIR = os.path.join(STATE_DIR, "trainer_tasks")
IN_DOCKER = os.environ.get("DRTRAINER_IN_DOCKER") == "1"          # 대시보드가 Docker 컨테이너 안에서 실행 중
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")

# DRfC defaults/hyperparameters.json 과 같은 값 (DRfC 폴더에 파일이 있으면 그 값을 우선 쓴다)
DEFAULT_HP = {"batch_size": 64, "beta_entropy": 0.01, "discount_factor": 0.99, "e_greedy_value": 0.05, "epsilon_steps": 10000,
              "exploration_type": "categorical", "loss_type": "huber", "lr": 0.0003, "num_episodes_between_training": 20,
              "num_epochs": 5, "stack_size": 1, "term_cond_avg_score": 350.0, "term_cond_max_episodes": 1000, "sac_alpha": 0.2}
# 트랙 이름 후보 (DR_WORLD_NAME). 시뮬레이터 이미지에 없는 이름이면 시뮬레이터 로그에 오류가 난다 (직접 입력도 가능)
WORLDS = ["reinvent_base", "reInvent2019_track", "reInvent2019_wide", "reInvent2019_wide_mirrored", "Vegas_track", "Canada_Training",
          "Oval_track", "Tokyo_Training_track", "AWS_track", "Spain_track", "Monaco", "Singapore", "2022_april_open", "2022_june_open"]
CAR_COLORS = ["Black", "Grey", "Blue", "Red", "Orange", "White", "Purple"]
CUDA_IMAGE_FALLBACK = "nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04"


class DrfcError(ValueError):
    """사용자에게 그대로 보여 줄 수 있는 오류."""


# ----------------------------------------------------------------------------- 설정과 경로
def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return default


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def _json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def is_drfc(path):
    return bool(path) and os.path.isfile(os.path.join(path, "bin", "activate.sh")) and os.path.isdir(os.path.join(path, "scripts"))


def default_dir():
    for c in (os.environ.get("DR_DIR"), os.path.expanduser("~/deepracer-for-cloud"), os.path.join(HERE, "deepracer-for-cloud"),
              os.path.join(os.path.dirname(HERE), "deepracer-for-cloud")):
        if c and is_drfc(c):
            return os.path.abspath(c)
    return os.path.expanduser("~/deepracer-for-cloud")


def get_dir():
    if IN_DOCKER and os.environ.get("DRFC_DIR"):                  # 컨테이너에서는 compose 가 정한 경로로 고정 (호스트와 같은 경로여야 함)
        return os.environ["DRFC_DIR"]
    return (_json(SETTINGS_FILE, {}) or {}).get("drfc_dir") or default_dir()


def set_dir(path):
    if IN_DOCKER:
        raise DrfcError("Docker 로 실행 중이라 DRfC 폴더는 여기서 바꿀 수 없습니다. 호스트 터미널에서 './drtrainer up --drfc-dir <경로>' 로 다시 시작하세요.")
    path = os.path.abspath(os.path.expanduser((path or "").strip()))
    if not os.path.isdir(path):
        raise DrfcError(f"폴더가 없습니다: {path}")
    if not is_drfc(path):
        raise DrfcError("DRfC 폴더가 아닙니다 (bin/activate.sh 와 scripts/ 가 있어야 합니다). deepracer-for-cloud 를 git clone 한 폴더를 지정하세요.")
    _write(SETTINGS_FILE, json.dumps({"drfc_dir": path}, ensure_ascii=False))
    return path


def is_mock(d):
    return os.path.isfile(os.path.join(d, ".mock_drfc"))


def _env(d):
    env = dict(os.environ, LANG=os.environ.get("LANG", "C.UTF-8"))
    if d and is_mock(d):                       # UI 시험용 가짜 DRfC: 가짜 docker/nvidia-smi 를 먼저 찾게 한다
        env["PATH"] = os.path.join(d, "shims") + os.pathsep + env.get("PATH", "")
    return env


def sh(argv, timeout=8, cwd=None, d=None):
    """짧은 명령 실행. (returncode, 출력). 없는 명령은 127, 시간 초과는 124."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=_env(d or get_dir()), stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{argv[0]}: 명령을 찾을 수 없습니다"
    except PermissionError:
        return 126, f"{argv[0]}: 실행할 수 없습니다 (PATH 에 접근 권한이 없는 폴더가 있거나 실행 권한이 없음)"
    except OSError as e:
        return 126, f"{argv[0]}: 실행하지 못했습니다 ({e})"
    except subprocess.TimeoutExpired:
        return 124, "시간 초과"


def sysenv(d):
    return parse_env(_read(os.path.join(d, "system.env")))


# init.sh 는 system.env 를 템플릿에서 복사한 뒤 <DOCKER_STYLE> 같은 자리표시자를 단계마다 채운다. 마지막 치환(<DOCKER_STYLE>)은 swarm 생성 뒤라서,
# swarm 생성에서 실패하면 파일은 있지만 값이 비어 있는 '중간에 멈춘' 상태가 된다. 파일이 있다고 초기화된 것이 아니다.
_PLACEHOLDER = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*['\"]?<[A-Z][A-Z0-9_]*>['\"]?\s*$", re.M)


def init_state(d):
    """('missing' | 'incomplete' | 'ok', 채워지지 않은 변수 이름들)"""
    p = os.path.join(d, "system.env")
    if not os.path.isfile(p) or not sysenv(d):
        return "missing", []
    bad = _PLACEHOLDER.findall(_read(p))
    return ("incomplete", bad) if bad else ("ok", [])


def _incomplete_msg(bad):
    return ("DRfC 초기화(init.sh)가 중간에 멈췄습니다: system.env 의 " + ", ".join(bad) + " 값이 채워지지 않았습니다 (<DOCKER_STYLE> 같은 자리표시자가 남아 있음). "
            "보통 swarm 을 만드는 단계에서 실패한 경우입니다.")


# ----------------------------------------------------------------------------- 컨테이너 / 상태
def list_containers(d=None):
    d = d or get_dir()
    rc, out = sh(["docker", "ps", "--format", "{{.Names}}|{{.Image}}|{{.Status}}"], timeout=6, d=d)
    if rc != 0:
        return []
    res = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        name, image, status = parts[0], parts[1], "|".join(parts[2:])
        low = name.lower()
        if "algo-" in low:
            role = "train"
        elif "robomaker" in low:
            role = "sim"
        elif "rl_coach" in low:
            role = "coach"
        elif "minio" in low:
            role = "storage"
        elif "simapp" in image:
            role = "other"
        else:
            continue
        res.append({"name": name, "image": image, "status": status, "role": role})
    return res


def _state():
    return _json(STATE_FILE, None)


def _set_state(experiment, kind):
    _write(STATE_FILE, json.dumps({"experiment": experiment, "kind": kind, "since": time.time()}))


def _clear_state():
    try:
        os.remove(STATE_FILE)
    except OSError:
        pass


_infer_cache = {}


def _container_env(d, name):
    """컨테이너의 환경변수 {이름: 값}. 시뮬레이터에는 MODEL_S3_PREFIX(모델 이름)가 들어 있다 (DRfC 의 docker-compose 파일)."""
    rc, out = sh(["docker", "inspect", "--format", "{{range .Config.Env}}{{println .}}{{end}}", name], timeout=6, d=d)
    env = {}
    if rc == 0:
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    return env


def _experiment_for_prefix(d, prefix):
    """모델 이름(prefix)을 쓰는 실험. 정확히 같은 것을 먼저, 없으면 가장 긴 접두사 일치(복사본 평가 같은 경우)."""
    best, best_len = None, 0
    root = os.path.join(d, "experiments")
    if not (prefix and os.path.isdir(root)):
        return None
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n, "run.env")
        if not os.path.isfile(p):
            continue
        pre = parse_env(_read(p)).get("DR_LOCAL_S3_MODEL_PREFIX") or n
        if pre == prefix:
            return n
        if prefix.startswith(pre) and len(pre) > best_len:
            best, best_len = n, len(pre)
    return best


def _infer_run(d, containers):
    """상태 기록이 없어도(터미널에서 시작했거나 기록을 잃었을 때) 컨테이너에서 학습/평가 종류와 실험을 알아낸다.
    DRfC 의 스택 이름: 학습 deepracer-<번호>, 평가 deepracer-eval-<번호>. 시뮬레이터 컨테이너의 환경변수 MODEL_S3_PREFIX 가 모델 이름이다."""
    names = [c["name"].lower() for c in containers if c["role"] in ("train", "sim", "coach")]
    kind = "evaluation" if any("deepracer-eval" in n for n in names) else ("training" if names else None)
    exp = None
    sim = next((c for c in containers if c["role"] == "sim"), None)
    if sim:
        hit = _infer_cache.get(sim["name"])
        if hit and time.time() - hit[0] < 60:
            exp = hit[1]
        else:
            exp = _experiment_for_prefix(d, _container_env(d, sim["name"]).get("MODEL_S3_PREFIX", ""))
            _infer_cache[sim["name"]] = (time.time(), exp)
    return kind, exp


_NOT_ALIVE = ("shutdown", "failed", "rejected", "complete", "orphaned", "remove")


def _stack_states(d, name):
    """swarm 스택에 남은 작업들의 상태 목록. `docker stack ps` 는 지금 도는 것뿐 아니라 종료·실패한 작업의 기록도 스택을 지우기 전까지 모두 보여 준다."""
    rc, out = sh(["docker", "stack", "ps", name, "--format", "{{.CurrentState}}"], timeout=6, d=d)
    return [l.strip() for l in out.splitlines() if l.strip()] if rc == 0 else []


_live_cache = {}


def _live_stacks(d):
    """작업이 아직 실행 중이거나 준비 중인 deepracer-* 스택 이름들. 컨테이너가 `docker ps` 에 나오기 전(이미지 준비, 자원 부족 등)이나 멈춰 있는 상태도 잡는다.
    종료·실패한 작업의 기록만 남은 스택은 실행 중이 아니므로 포함하지 않는다. 2초 동안 결과를 재사용한다 (화면이 주기적으로 부른다)."""
    hit = _live_cache.get(d)
    if hit and time.time() - hit[0] < 2:
        return hit[1]
    rc, out = sh(["docker", "stack", "ls", "--format", "{{.Name}}"], timeout=6, d=d)
    names = []
    if rc == 0:
        for s in out.splitlines():
            s = s.strip()
            if s.startswith("deepracer-") and any(not st.lower().startswith(_NOT_ALIVE) for st in _stack_states(d, s)):
                names.append(s)
    _live_cache[d] = (time.time(), names)
    return names


# 시작하기 전에 이전 실행이 남긴 '종료된 작업의 기록'을 지운다. DRfC 는 스택에 기록이 하나라도 있으면(종료된 것 포함) 'Processes running in stack' 으로 시작을 거부한다.
# 아직 실행/준비 중인 작업이 있으면 지우지 않고 멈춘다 (중지 버튼으로 먼저 정리해야 한다).
_STALE_STACK = r"""
_st=@@
if docker stack ps "$_st" --format '{{.CurrentState}}' 2>/dev/null | grep -qiE '^(running|starting|preparing|pending|assigned|accepted|ready|new|allocated)'; then
  echo "ERROR: 이전 실행의 작업이 아직 $_st 스택에 남아 있습니다. 화면 위쪽 배너의 '중지' 또는 '강제 중지'로 정리한 뒤 다시 시작하세요."; exit 1
fi
if [ "$(docker stack ps "$_st" 2>/dev/null | wc -l)" -gt 1 ]; then
  echo "이전 실행이 남긴 종료된 작업 기록이 있어서 DRfC 가 시작을 거부합니다. 기록을 지웁니다: docker stack rm $_st"
  docker stack rm "$_st"
  for _i in $(seq 1 60); do [ "$(docker stack ps "$_st" 2>/dev/null | wc -l)" -le 1 ] && break; sleep 1; done
fi
"""


def _stale_stack_script(kind, run_id):
    return _STALE_STACK.replace("@@", ("deepracer-eval-" if kind == "evaluation" else "deepracer-") + str(run_id))


def active(containers, d=None):
    """지금 돌고 있는 학습/평가. DRfC 는 한 번에 하나(DR_RUN_ID 당)만 돌릴 수 있다.
    종류와 실험은 컨테이너에서 알아낸 값을 우선하고(이름과 환경변수가 사실이다), 모르면 대시보드가 시작할 때 적어 둔 기록을 쓴다.
    둘 다 모르면 external=True 로 돌려준다 (터미널 등 대시보드 밖에서 시작한 실행).
    컨테이너가 없어도 swarm 스택에 실행/준비 중인 작업이 남아 있으면 '시작 중'(starting)으로 본다 (멈춰 있는 시작도 중지 버튼으로 정리할 수 있게)."""
    d = d or get_dir()
    st = _state()
    running = any(c["role"] in ("train", "sim", "coach") for c in containers)
    if running:
        kind, exp = _infer_run(d, containers)
        exp = exp or (st or {}).get("experiment")
        kind = kind or (st or {}).get("kind", "unknown")
        return {"experiment": exp, "kind": kind, "running": True, "since": (st or {}).get("since"), "external": exp is None}
    live = _live_stacks(d)
    if live:
        kind = "evaluation" if live[0].startswith("deepracer-eval-") else "training"
        exp = (st or {}).get("experiment") if (st or {}).get("kind") == kind else None
        return {"experiment": exp, "kind": kind, "running": False, "starting": True, "external": exp is None, "since": (st or {}).get("since"), "stack": live[0]}
    if st:
        if time.time() - st.get("since", 0) < 30:      # 명령이 끝난 직후 컨테이너가 뜨기까지의 짧은 유예
            return {**st, "running": False, "starting": True, "external": False}
        _clear_state()
    return None


def _run_id_of(d, experiment):
    try:
        return int(parse_env(_read(os.path.join(exp_path(d, experiment), "run.env"))).get("DR_RUN_ID", "0") or 0) if experiment else 0
    except (ValueError, DrfcError):
        return 0


def overview():
    d = get_dir()
    valid = is_drfc(d)
    info = {"in_docker": IN_DOCKER, "drfc_dir": d, "valid": valid, "mock": bool(valid and is_mock(d)), "platform": platform.system(), "initialized": False,
            "arch": None, "simapp": None, "workers": None, "cloud": None, "containers": [], "active": None}
    if valid:
        se = sysenv(d)
        ver = se.get("DR_SIMAPP_VERSION", "")
        m = re.search(r"-(gpu|cpu)$", ver)
        ist, ibad = init_state(d)
        info.update(initialized=ist == "ok", init_state=ist, init_missing=ibad, arch=m.group(1) if m else None, simapp=f"{se.get('DR_SIMAPP_SOURCE', '')}:{ver}" if ver else None,
                    workers=se.get("DR_WORKERS"), cloud=se.get("DR_CLOUD"), cuda_train=se.get("DR_SAGEMAKER_CUDA_DEVICES"),
                    cuda_sim=se.get("DR_ROBOMAKER_CUDA_DEVICES"))
        info["containers"] = list_containers(d)
        info["active"] = active(info["containers"], d)
    return info


# ----------------------------------------------------------------------------- 환경 점검
def _chk(cid, label, status, detail="", fix="", action=None):
    return {"id": cid, "label": label, "status": status, "detail": detail, "fix": fix, "action": action}


def doctor():
    d = get_dir()
    valid = is_drfc(d)
    checks = []

    sysname = platform.system()
    osr = {}
    for line in _read("/etc/os-release").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            osr[k] = v.strip('"')
    pretty = osr.get("PRETTY_NAME", sysname)
    wsl = "microsoft" in _read("/proc/version").lower()
    if IN_DOCKER:
        checks.append(_chk("os", "실행 환경", "ok", f"Docker 컨테이너 ({pretty}). 호스트 OS 와 무관하게 같은 환경으로 실행 중입니다."))
    elif sysname != "Linux":
        checks.append(_chk("os", "운영체제", "fail", f"{sysname}. DRfC 는 Linux(Ubuntu)용입니다.", "Ubuntu 컴퓨터나 서버에서 이 대시보드를 실행하세요. MiniRacer 는 어디서나 사용할 수 있습니다."))
    elif osr.get("ID") != "ubuntu":
        checks.append(_chk("os", "운영체제", "warn", f"{pretty}. DRfC 문서는 Ubuntu 를 기준으로 하고 다른 배포판은 \"될 가능성이 높다\"고만 적혀 있습니다."))
    else:
        supported = "22.04 24.04 24.10 25.04 25.10 26.04".split()
        m = re.search(r"^SUPPORTED_VERSIONS=\((.*)\)", _read(os.path.join(d, "bin", "prepare.sh")), re.M)
        if m:
            supported = re.findall(r"[0-9]+\.[0-9]+", m.group(1)) or supported
        if osr.get("VERSION_ID") not in supported:
            checks.append(_chk("os", "운영체제", "warn", f"{pretty}. DRfC 설치 스크립트가 지원하는 Ubuntu 버전은 {', '.join(supported)} 입니다."))
        else:
            checks.append(_chk("os", "운영체제", "ok", pretty + (" (WSL2)" if wsl else "")))

    rc, out = sh(["docker", "--version"])
    checks.append(_chk("docker", "Docker 설치", "ok" if rc == 0 else "fail", out if rc == 0 else "docker 명령이 없습니다.",
                       "" if rc == 0 else "https://docs.docker.com/engine/install/ubuntu/ 의 안내대로 설치하세요."))
    docker_ok = rc == 0
    daemon_ok = False
    if docker_ok:
        rc, out = sh(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
        daemon_ok = rc == 0
        if daemon_ok:
            checks.append(_chk("daemon", "Docker 실행", "ok", f"서버 {out}"))
        elif "permission denied" in out.lower():
            checks.append(_chk("daemon", "Docker 실행", "fail", "docker 를 쓸 권한이 없습니다.", "sudo usermod -aG docker $USER  를 실행한 뒤 로그아웃/로그인(또는 재부팅)하세요."))
        else:
            checks.append(_chk("daemon", "Docker 실행", "fail", out[:200] or "docker 서버에 연결할 수 없습니다.", "sudo systemctl start docker"))
        rc, out = sh(["docker", "compose", "version"])
        checks.append(_chk("compose", "Docker Compose", "ok" if rc == 0 else "fail", out if rc == 0 else "docker compose 플러그인이 없습니다.",
                           "" if rc == 0 else "sudo apt install docker-compose-plugin"))

    missing = [t for t in ("jq", "aws", "git") if shutil.which(t, path=_env(d)["PATH"]) is None]
    checks.append(_chk("tools", "필수 도구 (jq, awscli, git)", "ok" if not missing else "fail",
                       "모두 있음" if not missing else "없음: " + ", ".join(missing), "" if not missing else "sudo apt install " + " ".join("awscli" if m == "aws" else m for m in missing)))

    if not valid:
        checks.append(_chk("drfc", "DRfC 폴더", "fail", f"{d} 에서 DRfC 를 찾지 못했습니다.", "아래 '설치 안내'의 1단계로 내려받거나, 이미 있다면 위에서 폴더 경로를 지정하세요."))
    else:
        checks.append(_chk("drfc", "DRfC 폴더", "ok", d + ("  (UI 시험용 가짜 DRfC)" if is_mock(d) else "")))
        se = sysenv(d)
        ist, ibad = init_state(d)
        if ist == "missing":
            checks.append(_chk("init", "DRfC 초기화 (init.sh)", "fail", "system.env 가 없습니다. 아직 init.sh 를 실행하지 않았습니다.",
                               "호스트 터미널에서 ./drtrainer init  (실패 원인은 ./drtrainer logs 에 있습니다)" if IN_DOCKER else "설치 안내의 2단계를 터미널에서 실행하세요."))
        else:
            ver = se.get("DR_SIMAPP_VERSION", "")
            if ist == "incomplete" or "<" in ver or not ver:
                checks.append(_chk("init", "DRfC 초기화 (init.sh)", "fail", _incomplete_msg(ibad or ["DR_SIMAPP_VERSION"]),
                                   "호스트 터미널에서 ./drtrainer init  (비어 있는 설정을 백업하고 처음부터 자동으로 다시 초기화합니다. 원인은 ./drtrainer logs 에 있습니다)" if IN_DOCKER
                                   else "sudo bash scripts/setup_drfc.sh  (비어 있는 설정을 백업하고 init.sh 부터 자동으로 다시 합니다)"))
            else:
                arch = re.search(r"-(gpu|cpu)$", ver)
                checks.append(_chk("init", "DRfC 초기화 (init.sh)", "ok", f"{se.get('DR_CLOUD', '?')} 모드, 이미지 {se.get('DR_SIMAPP_SOURCE', '')}:{ver}"
                                   + (f", 현재 {arch.group(1).upper()} 모드" if arch else "")))
                if IN_DOCKER and os.environ.get("DOCKER_API_VERSION"):
                    checks.append(_chk("docker_api", "Docker 서버와 클라이언트 버전", "info",
                                       f"호스트 Docker 서버가 이 컨테이너의 docker 명령보다 오래돼서 API {os.environ['DOCKER_API_VERSION']} 로 맞춰 쓰고 있습니다. 동작에는 문제가 없지만, 가능하면 호스트의 Docker 를 업데이트하세요."))
                if docker_ok and daemon_ok and not is_mock(d) and (se.get("DR_DOCKER_STYLE") or "swarm").lower() == "swarm":
                    rc, _ = sh(["docker", "node", "ls"], timeout=10, d=d)
                    if rc != 0:
                        checks.append(_chk("swarm", "Docker swarm", "fail",
                                           "DRfC 는 swarm 모드를 쓰는데 이 Docker 는 swarm 이 아닙니다 (swarm 을 만든 적이 없거나 'docker swarm leave' 로 나갔습니다). 이대로는 학습 시작이 'This node is not a swarm manager' 로 실패합니다.",
                                           "아래 버튼으로 swarm 을 만듭니다 (docker swarm init). 이 컴퓨터의 Docker 가 swarm 모드로 바뀝니다. DRfC 가 원래 하는 일입니다.", action="create_swarm"))
                if docker_ok and daemon_ok and se.get("DR_CLOUD", "local") in ("local", "azure") and not is_mock(d):
                    mtag = se.get("DR_MINIO_IMAGE") or "RELEASE.2022-10-24T18-35-07Z"
                    rc, _ = sh(["docker", "image", "inspect", f"minio/minio:{mtag}", "--format", "ok"], timeout=10, d=d)
                    if rc == 0:
                        checks.append(_chk("minio_image", "minio 이미지 (저장소)", "ok", f"minio/minio:{mtag} 가 있음"))
                    else:
                        checks.append(_chk("minio_image", "minio 이미지 (저장소)", "fail",
                                           f"minio/minio:{mtag} 가 이 컴퓨터에 없습니다. MinIO 가 2026-09 에 Docker Hub 의 이미지를 삭제해서 받을 수 없습니다. 이대로는 학습을 시작할 수 없습니다.",
                                           "아래 버튼으로 GitHub 릴리스의 검증된 바이너리에서 이미지를 만듭니다 (1~2분). 학습을 시작할 때도 자동으로 확인하고 만듭니다.", action="build_minio"))
                vpy = os.path.join(d, ".venv", "bin", "python")
                if is_mock(d) or IN_DOCKER:
                    pass
                elif not os.path.isfile(vpy):
                    checks.append(_chk("venv", "DRfC 파이썬 환경 (.venv)", "warn", ".venv 가 없습니다. DRfC 의 activate.sh 가 경고를 냅니다. (init.sh 만 실행하는 로컬 설치에서는 만들어지지 않습니다.)",
                                       f"sudo bash {shlex.quote(os.path.join(HERE, 'scripts', 'setup_drfc.sh'))}  (자동 설치 스크립트가 만들어 줍니다)"))
                else:
                    rc, out = sh([vpy, "-c", "import boto3, yaml, requests"], timeout=15, d=d)
                    checks.append(_chk("venv", "DRfC 파이썬 환경 (.venv)", "ok" if rc == 0 else "warn", ".venv 에 필요한 패키지가 있음" if rc == 0 else "필수 패키지(boto3, pyyaml, requests)가 없습니다.",
                                       "" if rc == 0 else f"{shlex.quote(os.path.join(d, '.venv', 'bin', 'pip'))} install -r {shlex.quote(os.path.join(d, 'requirements.txt'))}"))
                if docker_ok and daemon_ok:
                    rc, _ = sh(["docker", "image", "inspect", f"{se.get('DR_SIMAPP_SOURCE', '')}:{ver}", "--format", "ok"], timeout=10)
                    checks.append(_chk("image", "시뮬레이터 이미지", "ok" if rc == 0 else "warn", "내려받아 있음" if rc == 0 else "아직 내려받지 않았습니다 (용량이 큽니다).",
                                       "" if rc == 0 else f"docker pull {se.get('DR_SIMAPP_SOURCE', '')}:{ver}  (또는 아래 'GPU / CPU' 에서 전환하면서 내려받기)"))

    cred = _read(os.path.expanduser("~/.aws/credentials"))
    checks.append(_chk("minio", "minio 자격 증명", "ok" if "[minio]" in cred else "warn", "~/.aws/credentials 에 [minio] 프로필이 있음" if "[minio]" in cred else
                       "[minio] 프로필이 없습니다.", "" if "[minio]" in cred else "init.sh 가 자동으로 만들어 줍니다. 만들지 못했다면 aws configure --profile minio"))

    # GPU
    rc, out = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=10)
    gpu_names = []
    if rc == 0 and out:
        gpus = [l.strip() for l in out.splitlines() if l.strip()]
        gpu_names = gpus
        mem = []
        for g in gpus:
            m = re.search(r"(\d+)\s*MiB", g)
            if m:
                mem.append(int(m.group(1)) / 1024)
        low = [x for x in mem if x < 8]
        checks.append(_chk("gpu", "NVIDIA GPU", "warn" if low else "ok", " / ".join(gpus) + ("  (GPU 메모리 8GB 미만: DRfC 문서의 최소 권장보다 작습니다)" if low else "")))
    elif IN_DOCKER:
        checks.append(_chk("gpu", "NVIDIA GPU", "info", "컨테이너 안에서는 GPU 를 직접 볼 수 없습니다. 호스트 Docker 의 GPU 연결은 아래 항목과 'GPU 컨테이너 시험'으로 확인하세요."))
    else:
        checks.append(_chk("gpu", "NVIDIA GPU", "info", "nvidia-smi 가 없거나 GPU 를 찾지 못했습니다. GPU 없이 CPU 모드로 쓸 수 있습니다.",
                           "GPU 를 쓰려면 NVIDIA 드라이버를 먼저 설치하세요."))
    if docker_ok and daemon_ok and (gpu_names or IN_DOCKER):
        rc, out = sh(["docker", "info", "--format", "{{json .Runtimes}}"], timeout=10)
        has = rc == 0 and "nvidia" in out.lower()
        if IN_DOCKER and not has:
            checks.append(_chk("gpu_docker", "Docker 의 GPU 연결", "info", "호스트 Docker 에 nvidia 런타임이 보이지 않습니다. GPU 가 없거나 NVIDIA Container Toolkit 이 없는 것입니다. CPU 모드는 그대로 쓸 수 있습니다.",
                               "GPU 를 쓰려면 호스트에 NVIDIA 드라이버와 NVIDIA Container Toolkit 을 설치하세요 (컨테이너 밖의 일입니다)."))
        else:
            checks.append(_chk("gpu_docker", "Docker 의 GPU 연결", "ok" if has else "warn", "nvidia 런타임이 보입니다." if has else
                               "nvidia 런타임이 보이지 않습니다. (이 확인만으로는 확정할 수 없으니 아래 'GPU 컨테이너 시험' 결과를 기준으로 보세요.)",
                               "" if has else "NVIDIA Container Toolkit 설치: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"))
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        checks.append(_chk("cuda_env", "CUDA_VISIBLE_DEVICES", "warn", "이 터미널에 CUDA_VISIBLE_DEVICES 가 설정돼 있습니다. DRfC 는 이 값을 더는 쓰지 않습니다.",
                           "GPU 번호는 아래 'GPU 번호'(DR_SAGEMAKER_CUDA_DEVICES)로 지정하세요."))

    # 자원 (DRfC docs/installation.md 의 Local 권장: 4코어/8 vCPU, RAM+GPU RAM 32GB 이상. 디스크 30~40GB 는 같은 문서의 AWS 항목(OS 디스크 최소 30GB, 권장 40GB)을 참고한 값이고 Local 항목에는 디스크 요건이 없다)
    cores = os.cpu_count() or 0
    ram = 0.0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            ram = int(re.sub(r"\D", "", line)) / 1024 / 1024
    path = d if os.path.isdir(d) else os.path.expanduser("~")
    free = shutil.disk_usage(path).free / 1e9
    notes = []
    if cores and cores < 8:
        notes.append(f"CPU {cores}스레드 (권장 8 이상)")
    if ram and ram < 16:
        notes.append(f"RAM {ram:.0f}GB (RAM+GPU 메모리 합 32GB 이상 권장)")
    if free < 30:
        notes.append(f"여유 디스크 {free:.0f}GB (30~40GB 권장)")
    checks.append(_chk("resources", "컴퓨터 사양", "warn" if notes else "ok", f"CPU {cores}스레드, RAM {ram:.0f}GB, 여유 디스크 {free:.0f}GB" + ("  → " + ", ".join(notes) if notes else "")))

    by = {c["id"]: c["status"] for c in checks}
    base = all(by.get(k) == "ok" for k in ("docker", "daemon", "compose", "tools", "drfc", "init")) and by.get("minio") in ("ok", "warn")
    ov = overview()
    return {"checks": checks, "ready_cpu": bool(base), "ready_gpu": bool(base and (by.get("gpu") in ("ok", "warn") or (IN_DOCKER and by.get("gpu_docker") == "ok")) and ov.get("arch") == "gpu"),
            "in_docker": IN_DOCKER,
            "gpu_names": gpu_names, "arch": ov.get("arch"), "install": install_steps()}


def install_steps(arch="gpu"):
    if IN_DOCKER:
        return {"docker": True, "note": "Docker 로 실행 중이라 설치할 것이 없습니다. DRfC 와 파이썬 도구는 이미지에 들어 있고, 처음 실행할 때 DRfC 가 자동으로 초기화됩니다."}
    d = get_dir()
    script = os.path.join(HERE, "scripts", "setup_drfc.sh")
    q = shlex.quote
    extra = "" if os.path.abspath(d) == os.path.abspath(os.path.expanduser("~/deepracer-for-cloud")) else f" --drfc-dir {q(d)}"
    repo = "https://github.com/aws-deepracer-community/deepracer-for-cloud.git"
    return {"preview": f"bash {q(script)} --dry-run{extra}",
            "auto": f"sudo bash {q(script)}{extra}",
            "auto_cpu": f"sudo bash {q(script)} --arch cpu{extra}",
            "auto_gpu": f"sudo bash {q(script)} --arch gpu{extra}",
            "driver": f"sudo bash {q(script)} --arch gpu --install-driver{extra}",
            "clone": f"git clone {repo} {q(d)}",
            "init_gpu": f"cd {q(d)} && ./bin/init.sh -c local -a gpu",
            "init_cpu": f"cd {q(d)} && ./bin/init.sh -c local -a cpu",
            "note": "스크립트는 여러 번 실행해도 안전합니다(이미 된 단계는 건너뜀). 관리자 권한이 필요해서 비밀번호를 한 번 묻습니다. "
                    "NVIDIA 드라이버는 기본으로 설치하지 않고, 재부팅도 대신 하지 않습니다. 이미 초기화된 DRfC 의 설정(system.env 등)은 덮어쓰지 않습니다."}


# ----------------------------------------------------------------------------- 작업 실행기
_tasks = {}
_tlock = threading.Lock()


def _hints(text):
    h = []
    pairs = [
        (r"Selected path .* exists", "DRfC 가 이 이름의 모델 경로가 이미 있다고 판단했습니다. 같은 이름의 모델이 있거나, 이름이 같은 글자로 시작하는 다른 모델(예: 'exp' 와 'exp_v2')이 있을 때 나옵니다 (DRfC 는 끝에 / 없이 접두사로 검사합니다). 처음부터 다시 학습하려면 학습 화면의 '덮어쓰기'를 체크하세요. 덮어쓰기는 이 이름의 모델만 지웁니다 (이름이 비슷한 다른 모델은 지우지 않는 것을 확인했습니다, aws cli 1.x)."),
        (r"Configuration files were not found", "보상함수·행동 공간·하이퍼파라미터 파일이 minio 에 없습니다. 다시 시작하면 업로드부터 다시 합니다."),
        (r"could not select device driver.*capabilities: \[\[gpu\]\]|could not select device driver \"\"", "Docker 가 GPU 를 쓸 수 있게 설정돼 있지 않습니다. 호스트에 NVIDIA Container Toolkit 이 없거나 설정이 안 된 것입니다. 호스트 터미널에서 'bash scripts/gpu_check.sh' 로 원인을 확인하고, 'sudo bash scripts/setup_drfc.sh --arch gpu' 로 필요한 것만 설치하세요 (Docker 가 다시 시작됩니다)."),
        (r"WSL environment detected but no adapters were found|nvidia-container-cli.*(driver|initialization|libnvidia|nvml)|Failed to initialize NVML", "NVIDIA 드라이버를 컨테이너에서 쓸 수 없습니다. WSL2 라면 Windows 쪽 NVIDIA 드라이버(WSL 지원 버전)를 설치하거나 업데이트한 뒤, Windows 에서 'wsl --shutdown' 하고 다시 여세요. WSL 안에서 'nvidia-smi' 가 보이는지 먼저 확인하세요."),
        (r"unknown or invalid runtime name: nvidia", "daemon.json 에 default-runtime 이 nvidia 로 돼 있는데 nvidia 런타임이 설치돼 있지 않습니다. 'sudo bash scripts/setup_drfc.sh --arch gpu' 로 설치하세요."),
        (r"permission denied", "docker 권한이 없습니다. sudo usermod -aG docker $USER 후 다시 로그인하세요."),
        (r"Cannot connect to the Docker daemon", "docker 서버가 꺼져 있습니다. sudo systemctl start docker"),
        (r"could not select device driver|nvidia", "GPU 를 컨테이너에서 쓸 수 없는 것 같습니다. '환경 점검'에서 GPU 컨테이너 시험을 해 보세요. 안 되면 CPU 모드로 전환하세요."),
        (r"No such image: minio|minio/minio.*(not found|does not exist)|pull access denied for minio", "minio 이미지를 받을 수 없습니다 (MinIO 가 Docker Hub 의 이미지를 삭제했습니다). 환경 점검의 'minio 이미지' 항목에서 만들기 버튼을 누르세요."),
        (r"Processes running in stack", "이전 실행이 남긴 스택(작업 기록)이 있어서 DRfC 가 시작을 거부했습니다. 종료된 기록만 남은 경우는 대시보드가 시작할 때 자동으로 지웁니다. 계속 같은 오류가 나면 화면 위쪽 배너의 '강제 중지' 또는 터미널에서 'docker stack rm deepracer-0' 으로 지우세요."),
        (r"Could not connect to the endpoint URL|minio\(localhost:9000\)에 연결할 수 없습니다", "minio(저장소)에 연결하지 못했습니다. 방금 시작돼 아직 준비 중이거나(처음에는 이미지를 내려받느라 오래 걸림), localhost:9000 이 막혀 있습니다. 잠시 뒤 다시 시도하고, 계속되면 'docker ps' 로 s3_minio 가 떠 있는지 확인하세요. Docker 로 실행 중이면 컨테이너가 호스트 네트워크(network_mode: host)를 쓰는지 확인하세요."),
        (r"DRfC 활성화에 실패", "DRfC 환경을 불러오지 못했습니다. system.env 와 init.sh 실행 여부를 확인하세요."),
        (r"No such file or directory", "필요한 파일을 찾지 못했습니다. 로그 위쪽의 경로를 확인하세요."),
    ]
    for pat, msg in pairs:
        if re.search(pat, text, re.I):
            h.append(msg)
    return h


def _tail(path, n=60):
    return "\n".join(_read(path).splitlines()[-n:])


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


def task_view(t):
    out = _ANSI.sub("", _tail(t["log"])).replace("\r\n", "\n")      # DRfC 가 출력하는 색상 코드(예: ESC[33m WARNING)는 화면에서 글자로 보이므로 지운다
    state = "running" if t["rc"] is None else ("cancelled" if t.get("cancelled") else ("ok" if t["rc"] == 0 else "failed"))
    return {"id": t["id"], "label": t["label"], "state": state, "rc": t["rc"], "started": t["started"], "finished": t.get("finished"),
            "output": out, "hints": _hints(out) if state == "failed" else []}


def busy_task():
    with _tlock:
        for t in _tasks.values():
            if t["rc"] is None and t.get("exclusive"):
                return t
    return None


def run_task(label, argv, cwd=None, env=None, on_done=None, exclusive=True, shown=None):
    if exclusive:
        b = busy_task()
        if b:
            raise DrfcError(f"'{b['label']}' 작업이 진행 중입니다. 끝난 뒤 다시 시도하세요.")
    os.makedirs(TASK_DIR, exist_ok=True)
    tid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    log = os.path.join(TASK_DIR, tid + ".log")
    f = open(log, "w", encoding="utf-8")
    f.write("$ " + (shown or " ".join(shlex.quote(a) for a in argv[:4])) + "\n")
    f.flush()
    proc = subprocess.Popen(argv, cwd=cwd, env=env or _env(get_dir()), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    t = {"id": tid, "label": label, "started": time.time(), "proc": proc, "log": log, "rc": None, "exclusive": exclusive}
    with _tlock:
        _tasks[tid] = t
        for old in sorted(_tasks)[:-30]:
            _tasks.pop(old, None)

    def waiter():
        rc = proc.wait()
        f.close()
        t["finished"] = time.time()
        t["rc"] = rc
        if on_done:
            try:
                on_done(rc, _read(log))
            except Exception:
                pass
    threading.Thread(target=waiter, daemon=True).start()
    return task_view(t)


def get_task(tid):
    t = _tasks.get(tid)
    if not t:
        raise DrfcError("작업을 찾을 수 없습니다.")
    return task_view(t)


def list_tasks():
    with _tlock:
        return [task_view(t) for t in sorted(_tasks.values(), key=lambda x: x["started"], reverse=True)[:12]]


def cancel_task(tid):
    t = _tasks.get(tid)
    if not t or t["rc"] is not None:
        raise DrfcError("취소할 작업이 없습니다.")
    t["cancelled"] = True
    try:
        os.killpg(os.getpgid(t["proc"].pid), signal.SIGTERM)
    except OSError:
        pass
    return {"ok": True}


# DRfC 의 bin/activate.sh 는 minio 를 'docker stack deploy' 로 띄우고 준비를 기다리지 않고 바로 돌아온다. 사람이 직접 칠 때는 다음 명령까지 몇 초가 걸려서
# 문제가 없지만, 이어서 바로 aws s3 를 부르면 minio 가 뜨기 전에 'Could not connect to the endpoint URL' 로 실패한다. S3(minio)를 쓰는 명령 앞에서 기다린다.
# (시험용: DRTRAINER_S3_WAIT_TRIES 번, DRTRAINER_S3_WAIT_SLEEP 초 간격. 기본 60번 x 3초 = 3분, 처음에는 minio 이미지를 내려받느라 오래 걸릴 수 있음)
_WAIT_S3 = r"""
_ok=0
for _i in $(seq 1 "${DRTRAINER_S3_WAIT_TRIES:-60}"); do
  # shellcheck disable=SC2086
  if aws $DR_LOCAL_PROFILE_ENDPOINT_URL s3 ls >/dev/null 2>&1; then _ok=1; break; fi
  [ "$_i" -eq 1 ] && echo 'minio 가 준비되기를 기다립니다 (처음에는 이미지를 내려받느라 1분 넘게 걸릴 수 있습니다)...'
  sleep "${DRTRAINER_S3_WAIT_SLEEP:-3}"
done
if [ "$_ok" -ne 1 ]; then
  echo 'minio(localhost:9000)에 연결할 수 없습니다.'
  docker service ps s3_minio --no-trunc --format '{{.Name}}: {{.CurrentState}} {{.Error}}' 2>/dev/null | head -3 || docker ps --filter name=minio 2>&1 | tail -2
  exit 98
fi
echo 'minio 준비됨'
"""


MINIO_SCRIPT = os.path.join(HERE, "docker", "minio-local-image.sh")


def _bash(d, experiment, inner, s3=False):
    q = shlex.quote
    act = "source bin/activate.sh" + (f" -e {q(experiment)}" if experiment else "")
    # Docker 로 실행 중이면 파이썬 환경은 이미지에 들어 있어서 DRfC 의 '.venv 를 찾을 수 없다' 경고는 해당 없다 (호스트 직접 설치에서는 의미가 있으니 걸러내지 않음)
    filt = " | grep -v 'Python venv not found'" if IN_DOCKER else ""
    ensure = (f"bash {q(MINIO_SCRIPT)} {q(d)} || {{ echo 'minio 이미지를 준비하지 못해 중단합니다.'; exit 96; }}; " if s3 else "")
    pre = (f"cd {q(d)} && " + ensure + "export DR_QUIET_ACTIVATE=True; "
           # activate.sh 는 DR_DOCKER_MAJOR_VERSION 을 341행에서 쓰고 380행에서 정의한다 (DRfC 의 순서 버그). 미리 정의해서 '[: : integer expression expected' 경고를 없앤다.
           "export DR_DOCKER_MAJOR_VERSION=$(docker --version 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+\\.[0-9]+' | head -1 | cut -d. -f1); "
           f"_drlog=$(mktemp); {act} >\"$_drlog\" 2>&1; cat \"$_drlog\"{filt}; rm -f \"$_drlog\"; "
           "if ! type dr-update-env >/dev/null 2>&1; then echo 'DRfC 활성화에 실패했습니다 (bin/activate.sh).'; exit 97; fi; ")
    return ["bash", "-c", pre + (_WAIT_S3 if s3 else "") + inner]


def _need_drfc(ready=True):
    """DRfC 폴더를 돌려준다. ready=True 면 초기화가 끝난 상태여야 한다 (system.env 가 있고 자리표시자가 남아 있지 않음)."""
    d = get_dir()
    if not is_drfc(d):
        raise DrfcError("DRfC 폴더를 찾지 못했습니다. '환경 점검'에서 폴더를 지정하세요.")
    ist, bad = init_state(d)
    if ist == "missing":
        raise DrfcError("DRfC 가 아직 초기화되지 않았습니다 (system.env 없음). '환경 점검'의 설치 안내를 따라 init.sh 를 실행하세요.")
    if ready and ist == "incomplete":
        raise DrfcError(_incomplete_msg(bad) + (" 호스트 터미널에서 ./drtrainer init 을 실행하면 비어 있는 설정을 백업하고 처음부터 다시 초기화합니다." if IN_DOCKER
                                                else " sudo bash scripts/setup_drfc.sh 를 실행하면 비어 있는 설정을 백업하고 다시 초기화합니다."))
    return d


def _check_swarm(d):
    """swarm 모드인데 이 Docker 가 swarm 이 아니면 시작하기 전에 알려 준다 (그대로 두면 활성화 오류가 줄줄이 나고 저장소(minio)를 기다리다 3분 뒤 실패한다)."""
    if is_mock(d) or (sysenv(d).get("DR_DOCKER_STYLE") or "swarm").lower() != "swarm":
        return
    rc, out = sh(["docker", "node", "ls"], timeout=10, d=d)
    if rc != 0 and "swarm manager" in out:
        raise DrfcError("이 Docker 는 swarm 이 아닙니다 (DRfC 는 swarm 모드를 씁니다). '환경 점검'의 'Docker swarm' 항목에서 'swarm 만들기'를 누르세요.")


# ----------------------------------------------------------------------------- GPU / CPU
def create_swarm():
    """docker swarm init (init.sh 가 하는 것과 같은 순서: 기본 방식, 안 되면 기본 IP 를 지정)."""
    d = get_dir()
    if not is_drfc(d):
        raise DrfcError("DRfC 폴더를 찾지 못했습니다. '환경 점검'에서 폴더를 지정하세요.")
    script = r"""
if docker node ls >/dev/null 2>&1; then echo "이미 swarm 입니다."; exit 0; fi
docker swarm init && { echo "swarm 을 만들었습니다."; exit 0; }
echo "기본 방식으로 만들지 못했습니다. 이 컴퓨터의 기본 IP 주소를 지정해서 다시 시도합니다..."
IFACE=$(ip route 2>/dev/null | awk '/^default/ {print $5; exit}')
IP=$(ip -4 addr show "$IFACE" 2>/dev/null | awk '/inet / {print $2}' | cut -d/ -f1 | head -1)
if [ -z "$IP" ]; then echo "기본 IP 주소를 알 수 없습니다. 터미널에서 직접 실행하세요: docker swarm init --advertise-addr <이 컴퓨터의 IP>"; exit 1; fi
echo "IP: $IP"
docker swarm init --advertise-addr "$IP" && echo "swarm 을 만들었습니다."
"""
    return run_task("Docker swarm 만들기", ["bash", "-c", script], cwd=d, env=_env(d), exclusive=True, shown="docker swarm init")


def build_minio_image():
    d = _need_drfc(ready=False)
    return run_task("minio 이미지 확인/만들기", ["bash", MINIO_SCRIPT, d], cwd=d, env=_env(d), exclusive=True, shown=f"bash docker/minio-local-image.sh {d}")


def gpu_test():
    d = _need_drfc()
    init = _read(os.path.join(d, "bin", "init.sh"))
    m = re.search(r"nvcr\.io/nvidia/cuda:[A-Za-z0-9._-]+", init)
    image = m.group(0) if m else CUDA_IMAGE_FALLBACK
    return run_task("GPU 컨테이너 시험", ["docker", "run", "--rm", "--gpus", "all", "--pull=missing", image, "nvidia-smi", "-L"], cwd=d, env=_env(d), exclusive=False)


def set_arch(arch, cuda_devices=None, workers=None):
    """시뮬레이터 이미지 태그의 -gpu/-cpu 를 바꾸고 이미지를 내려받는다 (init.sh 가 하는 일과 같은 설정)."""
    d = _need_drfc()
    if arch not in ("gpu", "cpu"):
        raise DrfcError("arch 는 gpu 또는 cpu 여야 합니다.")
    ov = overview()
    if ov["active"] and ov["active"].get("running"):
        raise DrfcError("학습/평가가 실행 중입니다. 먼저 중지한 뒤 바꾸세요.")
    path = os.path.join(d, "system.env")
    text = _read(path)
    se = parse_env(text)
    ver = se.get("DR_SIMAPP_VERSION", "")
    m = re.match(r"^(.*)-(gpu|cpu)$", ver)
    if not m:
        raise DrfcError("system.env 의 DR_SIMAPP_VERSION 이 '<버전>-gpu' 또는 '<버전>-cpu' 형식이 아닙니다. init.sh 를 먼저 실행하세요.")
    updates = {"DR_SIMAPP_VERSION": f"{m.group(1)}-{arch}"}
    if cuda_devices is not None:
        cd = str(cuda_devices).strip()
        if cd and not re.fullmatch(r"\d+(,\d+)*", cd):
            raise DrfcError("GPU 번호는 0 또는 0,1 처럼 숫자를 쉼표로 구분해서 입력하세요.")
        updates["DR_SAGEMAKER_CUDA_DEVICES"] = cd or None
    if workers is not None:
        w = int(workers)
        if not 1 <= w <= 16:
            raise DrfcError("워커 수는 1~16 이어야 합니다.")
        updates["DR_WORKERS"] = w
    backup = path + time.strftime(".bak-%m%d-%H%M%S")
    shutil.copy(path, backup)
    _write(path, set_env(text, updates))
    image = f"{se.get('DR_SIMAPP_SOURCE', 'awsdeepracercommunity/deepracer-simapp')}:{m.group(1)}-{arch}"
    t = run_task(f"시뮬레이터 이미지 내려받기 ({arch.upper()})", ["docker", "pull", image], cwd=d, env=_env(d), exclusive=False)
    t["backup"] = os.path.basename(backup)
    return t


# ----------------------------------------------------------------------------- 템플릿 / 검증
def templates():
    d = get_dir()
    hp = dict(DEFAULT_HP)
    hp.update(_json(os.path.join(d, "defaults", "hyperparameters.json"), {}) or {})
    return {"hp": hp, "worlds": WORLDS, "colors": CAR_COLORS, "from_drfc": os.path.isfile(os.path.join(d, "defaults", "hyperparameters.json"))}


def lint_reward(code):
    """보상함수 코드 점검: 문법, reward_function 존재, MiniRacer 환경에서 실제로 몇 스텝 실행."""
    code = code or ""
    try:
        compile(code, "reward_function.py", "exec")
    except SyntaxError as e:
        return {"ok": False, "line": e.lineno, "message": f"문법 오류 ({e.lineno}번째 줄): {e.msg}"}
    if not re.search(r"def\s+reward_function\s*\(", code):
        return {"ok": False, "line": None, "message": "def reward_function(params): 가 없습니다."}
    import tempfile

    from miniracer.env import MiniRacerEnv, RewardFunctionError
    from miniracer.track import Track
    tmp = tempfile.mkdtemp(prefix="drlint_")
    p = os.path.join(tmp, "reward_function.py")
    _write(p, code)
    try:
        env = MiniRacerEnv(Track(os.path.join(HERE, "tracks", "reInvent2019_track.npy")), p, os.path.join(HERE, "custom_files", "model_metadata.json"), 60)
        env.reset(0)
        for a in (7, 7, 4, 10, 7):
            env.step(a)
    except RewardFunctionError as e:
        return {"ok": False, "line": None, "message": str(e).split("\n")[0]}
    except Exception as e:
        return {"ok": False, "line": None, "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": True, "line": None, "message": "MiniRacer 환경에서 몇 스텝 실행해 봤고 오류가 없습니다. (실제 DeepRacer 시뮬레이터에서의 동작을 보장하지는 않습니다.)"}


def _num(v, typ, label, lo=None, hi=None):
    try:
        v = typ(v)
    except (TypeError, ValueError):
        raise DrfcError(f"{label} 값이 숫자가 아닙니다: {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise DrfcError(f"{label} 값은 {lo}~{hi} 범위여야 합니다.")
    return v


def build_metadata(spec):
    algo = spec.get("algo", "ppo")
    atype = "continuous" if algo == "sac" else spec.get("action_type", "discrete")
    base = {"sensor": ["FRONT_FACING_CAMERA"], "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"}
    if atype == "discrete":
        steers = [_num(s, float, "조향각", -30, 30) for s in (spec.get("steer") or [])]
        speeds = [_num(s, float, "속도", 0.1, 8) for s in (spec.get("speed") or [])]
        if not steers or not speeds:
            raise DrfcError("행동 공간에는 조향각과 속도가 각각 1개 이상 필요합니다.")
        if len(steers) * len(speeds) > 60:
            raise DrfcError("행동이 너무 많습니다 (조향각 수 x 속도 수 <= 60).")
        acts = [{"steering_angle": int(s) if s == int(s) else s, "speed": v} for s in steers for v in speeds]
        return {"action_space": acts, **base, "training_algorithm": "clipped_ppo", "action_space_type": "discrete", "version": "5"}
    lo_s, hi_s = _num(spec.get("steer_low", -30), float, "조향각 최소", -30, 30), _num(spec.get("steer_high", 30), float, "조향각 최대", -30, 30)
    lo_v, hi_v = _num(spec.get("speed_low", 1), float, "속도 최소", 0.1, 8), _num(spec.get("speed_high", 2), float, "속도 최대", 0.1, 8)
    if lo_s >= hi_s or lo_v >= hi_v:
        raise DrfcError("연속 행동 공간은 최소값이 최대값보다 작아야 합니다.")
    return {"action_space": {"speed": {"high": hi_v, "low": lo_v}, "steering_angle": {"high": hi_s, "low": lo_s}}, **base,
            "training_algorithm": "sac" if algo == "sac" else "clipped_ppo", "action_space_type": "continuous", "version": "4" if algo == "sac" else "5"}


def _hp(spec):
    base = templates()["hp"]
    out = dict(base)
    for k, v in (spec.get("hp") or {}).items():
        if k not in base:
            continue
        d = base[k]
        if isinstance(d, str):
            if k == "loss_type" and v not in ("huber", "mean squared error"):
                raise DrfcError("loss_type 은 huber 또는 mean squared error 여야 합니다.")
            out[k] = v
        elif isinstance(d, int) and not isinstance(d, bool):
            out[k] = _num(v, int, k, 1)
        else:
            out[k] = _num(v, float, k, 0)
    if not 0 < out["lr"] < 1:
        raise DrfcError("lr(학습률)은 0보다 크고 1보다 작아야 합니다.")
    if not 0 < out["discount_factor"] <= 1:
        raise DrfcError("discount_factor 는 0보다 크고 1 이하여야 합니다.")
    return out


def _bool(v):
    return bool(v) if not isinstance(v, str) else v.lower() == "true"


# ----------------------------------------------------------------------------- 모델(minio) / 실험
def bucket_dir(d):
    return os.path.join(d, "data", "minio", sysenv(d).get("DR_LOCAL_S3_BUCKET", "bucket"))


# ---- 저장소(MinIO) 읽기
# MinIO 는 객체를 평범한 파일이 아니라 '<키>/xl.meta' 같은 디렉터리 구조로 저장하므로 내용은 S3 API 로 읽어야 한다.
# 평범한 파일이 있으면(가짜 DRfC, 예전 방식) 파일로 읽고, 없으면 S3 API 로 읽는다. ETag 로 바뀌지 않은 객체는 다시 받지 않는다.
_obj_cache = {}


def _s3(d):
    try:
        return S3Client.from_drfc(sysenv(d))
    except Exception:
        return None


def read_object(d, key):
    """버킷 안의 키(예: '모델/metrics/TrainingMetrics.json')의 내용 bytes. 없거나 읽을 수 없으면 None."""
    se = sysenv(d)
    bucket = se.get("DR_LOCAL_S3_BUCKET", "bucket")
    p = os.path.join(d, "data", "minio", bucket, *key.split("/"))
    if os.path.isfile(p):
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            return None
    c = _s3(d)
    if c is None:
        return None
    ck = (d, bucket, key)
    etag, body = _obj_cache.get(ck, (None, None))
    try:
        status, data, new_etag = c.get(bucket, key, etag=etag)
    except S3Error:
        return body                      # minio 가 잠시 안 닿으면 마지막으로 읽은 값을 보여 준다
    if status == 304:
        return body
    if status == 200:
        _obj_cache[ck] = (new_etag, data)
        return data
    return None


def list_objects(d, prefix):
    """버킷 안에서 접두사로 시작하는 키 목록. S3 API 를 먼저 쓰고, 안 되면 파일 시스템을 본다."""
    se = sysenv(d)
    bucket = se.get("DR_LOCAL_S3_BUCKET", "bucket")
    c = _s3(d)
    if c is not None:
        try:
            keys, _ = c.list(bucket, prefix)
            return keys
        except S3Error:
            pass
    root = os.path.join(d, "data", "minio", bucket)
    out = []
    base = os.path.join(root, *prefix.rstrip("/").split("/")) if prefix else root
    if os.path.isdir(base):
        for dp, _, fs in os.walk(base):
            for f in fs:
                out.append(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
    return out


def list_meta(d, prefix):
    """버킷 안에서 접두사로 시작하는 객체의 [{key, size, modified}]. S3 API 를 먼저 쓰고, 안 되면 파일 시스템을 본다."""
    bucket = sysenv(d).get("DR_LOCAL_S3_BUCKET", "bucket")
    c = _s3(d)
    if c is not None:
        try:
            return c.list_objects(bucket, prefix)
        except S3Error:
            pass
    root = os.path.join(d, "data", "minio", bucket)
    base = os.path.join(root, *prefix.rstrip("/").split("/")) if prefix else root
    out = []
    if os.path.isdir(base):
        for dp, _, fs in os.walk(base):
            for f in fs:
                p = os.path.join(dp, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out.append({"key": os.path.relpath(p, root).replace(os.sep, "/"), "size": st.st_size, "modified": st.st_mtime})
    return out


def models_in_bucket(d=None):
    d = d or get_dir()
    b = bucket_dir(d)
    out = []
    if os.path.isdir(b):
        for n in sorted(os.listdir(b)):
            if os.path.isdir(os.path.join(b, n, "model")):
                out.append(n)
        return out
    c = _s3(d)                            # 데이터 폴더가 이 컴퓨터에 없는 경우(원격 minio)
    if c is not None:
        try:
            _, prefixes = c.list(sysenv(d).get("DR_LOCAL_S3_BUCKET", "bucket"), "", delimiter="/")
            for p in prefixes:
                keys, _ = c.list(sysenv(d).get("DR_LOCAL_S3_BUCKET", "bucket"), p + "model/")
                if keys:
                    out.append(p.rstrip("/"))
        except S3Error:
            pass
    return sorted(out)


def model_exists(d, prefix):
    p = os.path.join(bucket_dir(d), prefix)
    if os.path.isdir(p) and os.listdir(p):
        return True
    return False


def dir_size(path):
    """폴더 안 파일 크기의 합 (bytes). 없으면 0."""
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


_PREFIX_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def exp_path(d, name):
    if not NAME_RE.match(name or ""):
        raise DrfcError("실험 이름은 영문, 숫자, _ , - 만 쓸 수 있고 40자 이하이며 영문/숫자로 시작해야 합니다.")
    return os.path.join(d, "experiments", name)


def create_experiment(spec):
    d = _need_drfc()
    name = (spec.get("name") or "").strip()
    path = exp_path(d, name)
    if os.path.exists(path):
        raise DrfcError(f"'{name}' 실험이 이미 있습니다. 다른 이름을 쓰거나, 설정만 바꾸려면 새 이름으로 복제하세요 (DRfC 도 실험마다 새 모델 접두사를 쓰는 방식입니다).")
    code = spec.get("reward_code") or ""
    lint = lint_reward(code)
    if not lint["ok"]:
        raise DrfcError("보상함수에 문제가 있습니다: " + lint["message"])
    meta = build_metadata(spec)
    hp = _hp(spec)
    race = spec.get("race_type", "TIME_TRIAL")
    if race not in ("TIME_TRIAL", "OBJECT_AVOIDANCE", "HEAD_TO_BOT"):
        raise DrfcError("레이스 유형이 올바르지 않습니다.")
    world = (spec.get("world") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", world):
        raise DrfcError("트랙 이름은 영문, 숫자, _ , - , . 만 쓸 수 있습니다.")
    color = spec.get("car_color", "Red")
    if color not in CAR_COLORS:
        raise DrfcError("차 색상이 올바르지 않습니다.")

    tr = spec.get("train") or {}
    ev = spec.get("eval") or {}
    updates = {
        "DR_WORLD_NAME": world, "DR_RACE_TYPE": race, "DR_CAR_NAME": name[:20], "DR_CAR_COLOR": color,
        "DR_LOCAL_S3_MODEL_PREFIX": name,
        "DR_ENABLE_DOMAIN_RANDOMIZATION": _bool(tr.get("domain_randomization", False)),
        "DR_TRAIN_CHANGE_START_POSITION": _bool(tr.get("change_start", True)),
        "DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST": _num(tr.get("round_robin", 0.05), float, "출발 위치 이동 간격", 0.01, 1),
        "DR_TRAIN_MIN_EVAL_TRIALS": _num(tr.get("min_eval_trials", 5), int, "최소 평가 횟수", 1, 50),
        "DR_TRAIN_BEST_MODEL_METRIC": tr.get("best_metric", "progress") if tr.get("best_metric") in ("progress", "reward") else "progress",
        "DR_TRAIN_REVERSE_DIRECTION": _bool(tr.get("reverse", False)),
        "DR_TRAIN_ALTERNATE_DRIVING_DIRECTION": _bool(tr.get("alternate", False)),
        "DR_EVAL_NUMBER_OF_TRIALS": _num(ev.get("trials", 3), int, "평가 주행 수", 1, 50),
        "DR_EVAL_CHECKPOINT": ev.get("checkpoint", "last") if ev.get("checkpoint") in ("last", "best") else "last",
    }
    pre = spec.get("pretrained") or {}
    if pre.get("prefix"):
        if not NAME_RE.match(pre["prefix"]) or pre["prefix"] not in models_in_bucket(d):
            raise DrfcError(f"이어서 학습할 모델 '{pre['prefix']}' 을(를) minio 저장소에서 찾지 못했습니다.")
        updates.update(DR_LOCAL_S3_PRETRAINED=True, DR_LOCAL_S3_PRETRAINED_PREFIX=pre["prefix"],
                       DR_LOCAL_S3_PRETRAINED_CHECKPOINT=pre.get("checkpoint", "last") if pre.get("checkpoint") in ("last", "best") else "last")
    else:
        updates["DR_LOCAL_S3_PRETRAINED"] = False

    template = _read(os.path.join(d, "defaults", "template-run.env"))
    if not template:
        raise DrfcError("DRfC 의 defaults/template-run.env 를 읽지 못했습니다. DRfC 폴더가 온전한지 확인하세요.")
    os.makedirs(os.path.join(path, "custom_files"))
    _write(os.path.join(path, "run.env"), set_env(template, updates))
    _write(os.path.join(path, "custom_files", "reward_function.py"), code if code.endswith("\n") else code + "\n")
    _write(os.path.join(path, "custom_files", "model_metadata.json"), json.dumps(meta, indent=2))
    _write(os.path.join(path, "custom_files", "hyperparameters.json"), json.dumps(hp, indent=2))
    _write(os.path.join(path, "trainer_meta.json"), json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": spec.get("source", "form"),
                                                               "algo": spec.get("algo", "ppo"), "action_type": meta["action_space_type"]}, ensure_ascii=False))
    return {"name": name, "lint": lint}


_sum_cache = {}


def training_summary(d, prefix, per_iter):
    data = read_object(d, f"{prefix}/metrics/TrainingMetrics.json")
    if data is None:
        return None
    key = (d, prefix, per_iter, hash(data))
    if key not in _sum_cache:
        _sum_cache.clear()
        rows = M.parse_rows(data)
        _sum_cache[key] = M.summarize_training_rows(rows, per_iter) if rows is not None else None
    return _sum_cache[key]


def _exp_info(d, name):
    path = exp_path(d, name)
    env = parse_env(_read(os.path.join(path, "run.env")))
    meta = _json(os.path.join(path, "trainer_meta.json"), {}) or {}
    hp = _json(os.path.join(path, "custom_files", "hyperparameters.json"), {}) or {}
    mm = _json(os.path.join(path, "custom_files", "model_metadata.json"), {}) or {}
    prefix = env.get("DR_LOCAL_S3_MODEL_PREFIX", name)
    return path, env, meta, hp, mm, prefix


def list_experiments():
    d = get_dir()
    root = os.path.join(d, "experiments")
    if not is_drfc(d) or not os.path.isdir(root):
        return []
    ov_active = active(list_containers(d)) if sysenv(d) else None
    rows = []
    for n in sorted(os.listdir(root)):
        if not os.path.isfile(os.path.join(root, n, "run.env")):
            continue
        path, env, meta, hp, mm, prefix = _exp_info(d, n)
        s = training_summary(d, prefix, hp.get("num_episodes_between_training", 20))
        best_p, last_it = None, 0
        if s:
            ps = [p for p in s["metrics"]["eval_progress"] if p is not None]
            best_p = max(ps) if ps else None
            last_it = s["metrics"]["iteration"][-1] if s["metrics"]["iteration"] else 0
        running = bool(ov_active and ov_active.get("running") and ov_active.get("experiment") == n)
        rows.append({"name": n, "prefix": prefix, "world": env.get("DR_WORLD_NAME"), "race_type": env.get("DR_RACE_TYPE"),
                     "algo": mm.get("training_algorithm"), "action_type": mm.get("action_space_type"), "created": meta.get("created"),
                     "has_model": model_exists(d, prefix), "iterations": last_it, "best_progress": best_p, "running": running,
                     "kind": ov_active.get("kind") if running else None, "mtime": os.path.getmtime(os.path.join(root, n, "run.env"))})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


STREAM_TOPICS = [
    ("/racecar/deepracer/kvs_stream", "차를 따라가는 시점 (속도, 진행률 표시 포함)", None),
    ("/racecar/camera/zed/rgb/image_rect_color", "차 안의 카메라 (모델이 실제로 보는 화면)", None),
    ("/racecar/main_camera/zed/rgb/image_rect_color", "차를 따라가는 시점 (표시 없음)", "DR_CAMERA_MAIN_ENABLE"),
    ("/sub_camera/zed/rgb/image_rect_color", "트랙 위에서 내려다본 시점", "DR_CAMERA_SUB_ENABLE"),
]
# 영상 중계가 연결해도 되는 포트: DRfC 가 시뮬레이터 영상(ROS web_video_server)에 쓰는 범위뿐이다 (그 밖의 포트로는 중계하지 않는다)
STREAM_PORTS = frozenset(list(range(8080, 8090)) + list(range(8180, 8190)))
_TOPIC_OK = re.compile(r"^/[A-Za-z0-9_./-]{1,120}$")


def probe_index(port, timeout=2.0):
    """시뮬레이터의 영상 서버(ROS web_video_server) 첫 화면에서 지금 내보내고 있는 영상 토픽 목록을 읽는다. 서버(대시보드)에서 127.0.0.1:포트 로 접속한다."""
    try:
        c = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
        c.request("GET", "/")
        r = c.getresponse()
        body = r.read(80000).decode("utf-8", "replace")
        c.close()
    except (OSError, http.client.HTTPException, ValueError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    topics = sorted({urllib.parse.unquote(t) for t in re.findall(r"(?:stream_viewer|stream|snapshot)\?topic=([^\"'&\s<>]+)", body)})
    title = re.search(r"<title>(.*?)</title>", body, re.S | re.I)
    return {"ok": True, "status": r.status, "server": r.getheader("Server"), "ctype": r.getheader("Content-Type"), "topics": topics,
            "is_video_server": bool(topics) or "Image Topic" in body, "title": (title.group(1).strip() if title else body[:80].strip().replace("\n", " "))}


def probe_stream(port, topic, total=6.0):
    """영상 한 줄기(/stream)에 실제로 접속해서 프레임(JPEG)이 오는지 확인한다."""
    t0 = time.time()
    res = {"topic": topic, "ok": False}
    try:
        c = http.client.HTTPConnection("127.0.0.1", int(port), timeout=4)
        c.request("GET", f"/stream?topic={urllib.parse.quote(topic, safe='/')}&width=320&height=240&quality=50")
        r = c.getresponse()
        res.update(status=r.status, ctype=r.getheader("Content-Type"), first_byte_ms=int((time.time() - t0) * 1000))
        if r.status != 200:
            res["body"] = r.read(200).decode("utf-8", "replace")
            c.close()
            return res
        got, jpeg = 0, False
        while time.time() - t0 < total and got < 60000:
            chunk = r.read1(8192)
            if not chunk:
                break
            got += len(chunk)
            if b"\xff\xd8" in chunk:
                jpeg = True
                if got > 1500:
                    break
        c.close()
        res.update(bytes=got, jpeg=jpeg, ms=int((time.time() - t0) * 1000), ok=jpeg)
    except (OSError, http.client.HTTPException) as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


_index_cache = {}


def _live_topics(port):
    """3초 동안 캐시한 probe_index 결과 (화면이 4초마다 부르므로)."""
    now = time.time()
    hit = _index_cache.get(port)
    if hit and now - hit[0] < 3:
        return hit[1]
    r = probe_index(port, timeout=1.5)
    _index_cache[port] = (now, r)
    return r


def stream_info():
    """실시간 영상 주소를 만드는 데 필요한 정보. 시뮬레이터(RoboMaker)가 ROS 영상 스트림을 내보내고(MJPEG), 대시보드가 그것을 브라우저로 중계한다.
    포트 (DRfC docs/video.md, bin/activate.sh): swarm 은 학습 8080+DR_RUN_ID, 평가 8180+DR_RUN_ID. compose 는 워커마다 8080~8089."""
    d = get_dir()
    out = {"running": False, "kind": None, "experiment": None, "style": "swarm", "workers": 1, "ports": [], "viewer_port": 8100,
           "topics": [], "in_docker": IN_DOCKER, "live": None, "probe_error": None}
    if not (is_drfc(d) and sysenv(d)):
        return out
    se = sysenv(d)
    a = active(list_containers(d)) or {}
    style = (se.get("DR_DOCKER_STYLE") or "swarm").lower()
    if style not in ("swarm", "compose"):
        style = "swarm"
    workers = max(1, min(10, int(se.get("DR_WORKERS")))) if str(se.get("DR_WORKERS") or "1").isdigit() else 1
    run_id = 0
    if a.get("experiment"):
        try:
            run_id = int(parse_env(_read(os.path.join(exp_path(d, a["experiment"]), "run.env"))).get("DR_RUN_ID", "0") or 0)
        except (ValueError, DrfcError):
            run_id = 0
    kind = a.get("kind")
    # swarm: 학습 8080+실행번호, 평가 8180+실행번호. 종류를 모르면(대시보드 밖에서 시작했거나 기록이 지워짐) 둘 다 후보로 두고 실제로 응답하는 쪽을 쓴다.
    if style == "swarm":
        cands = [8180 + run_id] if kind == "evaluation" else [8080 + run_id] if kind == "training" else [8080 + run_id, 8180 + run_id]
    else:
        cands = [8080 + i for i in range(workers)]
    ports = [p for p in cands if p in STREAM_PORTS]
    live = None
    if a.get("running") and ports and not is_mock(d):
        first_err = None
        for p in ([ports[0]] if style != "swarm" else ports):
            idx = _live_topics(p)
            if idx.get("ok") and idx.get("is_video_server"):
                live = idx["topics"]
                if style == "swarm":
                    ports = [p]
                    if kind not in ("training", "evaluation"):
                        kind = "evaluation" if p >= 8180 else "training"
                break
            if not idx.get("ok") and first_err is None:
                first_err = idx.get("error")
        if live is None:
            out["probe_error"] = first_err
    topics = []
    for tid, label, flag in STREAM_TOPICS:
        enabled = True if flag is None else str(se.get(flag, "True" if flag != "DR_CAMERA_SUB_ENABLE" else "False")).lower() == "true"
        topics.append({"id": tid, "label": label, "enabled": enabled, "need": None if enabled else f"system.env 의 {flag}=True 가 필요합니다",
                       "available": (tid in live) if live else None})
    for t in live or []:
        if all(t != k[0] for k in STREAM_TOPICS) and _TOPIC_OK.match(t):
            topics.append({"id": t, "label": t, "enabled": True, "need": None, "available": True, "extra": True})
    out.update(running=bool(a.get("running")), kind=kind if kind in ("training", "evaluation") else a.get("kind"), experiment=a.get("experiment"), style=style, workers=workers, ports=ports, live=live,
               viewer_port=int(se.get("DR_WEBVIEWER_PORT") or 8100) if str(se.get("DR_WEBVIEWER_PORT") or "8100").isdigit() else 8100, topics=topics)
    return out


def stream_proxy_args(q):
    """중계 요청의 값을 검사해서 시뮬레이터에 보낼 (포트, 경로) 를 돌려준다. 허용하지 않는 값이면 ValueError."""
    try:
        port = int(q.get("port", ""))
        width, height, quality = int(q.get("width", 640)), int(q.get("height", 480)), int(q.get("quality", 75))
    except ValueError:
        raise ValueError("영상 요청의 값이 올바르지 않습니다.")
    if port not in STREAM_PORTS:
        raise ValueError(f"허용하지 않는 포트입니다: {port}")
    topic = q.get("topic", "")
    if not _TOPIC_OK.match(topic):
        raise ValueError("영상 종류(topic)가 올바르지 않습니다.")
    if not (64 <= width <= 1920 and 48 <= height <= 1440 and 1 <= quality <= 100):
        raise ValueError("영상 크기나 품질이 범위를 벗어났습니다.")
    return port, f"/stream?topic={urllib.parse.quote(topic, safe='/')}&width={width}&height={height}&quality={quality}"


def stream_debug():
    """실시간 영상이 안 보일 때 어디가 막혔는지 단계별로 확인한다. 각 단계의 결과와 추정 원인을 돌려주고, 그대로 복사해서 보여줄 수 있는 텍스트도 만든다."""
    d = get_dir()
    steps, hints = [], []

    def add(name, status, detail=""):
        steps.append({"name": name, "status": status, "detail": detail})

    add("실행 환경", "info", ("Docker 컨테이너 (호스트 네트워크로 시뮬레이터의 포트에 접속)" if IN_DOCKER else "호스트에서 직접 실행") + f" / 대시보드 주소 기준 DRfC 폴더 {d}")
    if not (is_drfc(d) and sysenv(d)):
        add("DRfC", "fail", "DRfC 폴더나 system.env 를 찾지 못했습니다.")
        return _debug_result(steps, ["DRfC 가 준비되지 않았습니다. 환경 점검을 먼저 확인하세요."])
    se = sysenv(d)
    ist, bad = init_state(d)
    add("DRfC 초기화", "ok" if ist == "ok" else "fail", "정상" if ist == "ok" else _incomplete_msg(bad))
    style = (se.get("DR_DOCKER_STYLE") or "swarm").lower()
    add("영상 관련 설정 (system.env)", "info", f"DR_DOCKER_STYLE={style}, DR_WORKERS={se.get('DR_WORKERS')}, DR_CAMERA_MAIN_ENABLE={se.get('DR_CAMERA_MAIN_ENABLE')}, "
        f"DR_CAMERA_SUB_ENABLE={se.get('DR_CAMERA_SUB_ENABLE')}, DR_CAMERA_KVS_ENABLE={se.get('DR_CAMERA_KVS_ENABLE')}, DR_GUI_ENABLE={se.get('DR_GUI_ENABLE')}")

    # 1) 대시보드가 '실행 중'으로 인식하는가
    conts = list_containers(d)
    act = active(conts) or {}
    robo = [c for c in conts if "robomaker" in (c.get("name", "") + c.get("image", "")).lower()]
    add("실행 중인 컨테이너", "ok" if conts else "warn", "; ".join(f"{c.get('name')} [{c.get('status')}]" for c in conts) or "없음 (docker ps 결과가 비어 있음)")
    add("대시보드가 인식한 실행", "ok" if act.get("running") else "warn",
        f"종류={act.get('kind')}, 실험={act.get('experiment')}, 실행 중={act.get('running')}" if act else "학습/평가가 실행 중으로 인식되지 않음")
    if not robo:
        hints.append("시뮬레이터(robomaker) 컨테이너가 보이지 않습니다. 학습/평가가 시작됐는지 확인하세요. 시작 직후라면 1분쯤 뒤에 다시 진단하세요. 컨테이너 이름이 예상과 다르면 위 '실행 중인 컨테이너' 줄을 알려 주세요.")
    elif not act.get("running"):
        hints.append("시뮬레이터 컨테이너는 있는데 대시보드가 '실행 중'으로 인식하지 못했습니다 (컨테이너 이름 규칙이 다를 수 있음). 영상 패널의 '그래도 연결 시도'로 직접 연결해 볼 수 있습니다.")

    # 2) 포트가 호스트에 공개돼 있는가
    rc, out = sh(["docker", "ps", "--format", "{{.Names}}|{{.Ports}}"], timeout=8, d=d)
    pub = [l for l in out.splitlines() if "robomaker" in l.lower()] if rc == 0 else []
    add("시뮬레이터 컨테이너의 공개 포트 (docker ps)", "ok" if any("8080" in l or "818" in l or "808" in l for l in pub) else "warn", " / ".join(pub) or "robomaker 컨테이너 없음")
    rc, out = sh(["docker", "service", "ls", "--format", "{{.Name}}|{{.Replicas}}|{{.Ports}}"], timeout=8, d=d)
    svc = [l for l in out.splitlines() if "robomaker" in l.lower()] if rc == 0 else []
    if svc:
        add("swarm 서비스의 공개 포트 (docker service ls)", "ok" if any(("808" in l or "818" in l) for l in svc) else "warn", " / ".join(svc))
    rc, out = sh(["ss", "-ltn"], timeout=5, d=d)
    if rc == 0:
        listen = [l.split()[3] for l in out.splitlines()[1:] if l.split() and re.search(r":(80[89]\d|81[89]\d|8100)$", l.split()[3])]
        add("이 컴퓨터에서 열려 있는 영상 포트 (ss)", "ok" if listen else "warn", ", ".join(sorted(set(listen))) or "8080~8089, 8180~8189, 8100 에서 대기 중인 프로그램이 없음")

    # 3) 서버(대시보드) -> 시뮬레이터: 영상 서버가 응답하는가, 어떤 영상이 있는가
    info = stream_info()
    ports = info["ports"] or [8080]
    ok_port, live, other_app = None, None, False
    for p in dict.fromkeys(ports + [8080, 8180]):
        idx = probe_index(p)
        if not idx["ok"]:
            add(f"포트 {p} 접속 (대시보드 -> 시뮬레이터)", "fail" if p in ports else "info", idx["error"])
            continue
        if idx["is_video_server"]:
            add(f"포트 {p} 접속 (대시보드 -> 시뮬레이터)", "ok", f"영상 서버가 응답함. 지금 내보내는 영상 {len(idx['topics'])}개: " + (", ".join(idx["topics"]) or "없음"))
            ok_port, live = p, idx["topics"]
            break
        add(f"포트 {p} 접속 (대시보드 -> 시뮬레이터)", "fail", f"응답은 있지만 시뮬레이터의 영상 서버가 아닙니다 (Server={idx['server']}, 제목='{idx['title']}'). 이 포트를 다른 프로그램이 쓰고 있을 수 있습니다.")
        other_app = True
        hints.append(f"포트 {p} 에 시뮬레이터가 아닌 다른 프로그램이 응답하고 있습니다 ('{idx['title']}'). 그 프로그램을 끄세요. 시뮬레이터의 영상 포트가 이미 사용 중이라 시뮬레이터가 그 포트를 쓰지 못했을 수 있습니다 (DRfC 를 다시 시작해야 합니다).")
    # 4) 실제 프레임
    if ok_port:
        want = [t[0] for t in STREAM_TOPICS if t[0] in (live or [])] or list(live or [])
        if not want:
            add("영상 프레임 수신", "warn", "영상 서버는 떠 있지만 내보내는 영상(토픽)이 없습니다.")
            hints.append("영상 서버는 응답하지만 내보내는 영상이 없습니다. 시뮬레이터가 아직 카메라를 시작하기 전일 수 있으니 1분쯤 뒤 다시 진단하세요. 계속 없으면 system.env 의 DR_CAMERA_*_ENABLE 설정을 확인하세요.")
        else:
            got = probe_stream(ok_port, want[0])
            if got.get("ok"):
                add("영상 프레임 수신", "ok", f"{want[0]}: JPEG 프레임을 받음 ({got['bytes']} bytes, 첫 응답 {got['first_byte_ms']}ms, 형식 {got.get('ctype')})")
            else:
                add("영상 프레임 수신", "fail", f"{want[0]}: " + json.dumps({k: v for k, v in got.items() if k != 'topic'}, ensure_ascii=False))
                hints.append("영상 서버에는 연결되지만 프레임이 오지 않습니다. 시뮬레이터가 아직 시작 중이거나 그 영상 종류를 내보내지 않는 상태일 수 있습니다. 다른 영상 종류로 바꿔 보거나 잠시 뒤 다시 진단하세요.")
        missing = [t[0] for t in STREAM_TOPICS if live is not None and t[0] not in live]
        if missing and want:
            add("지금 내보내지 않는 기본 영상", "info", ", ".join(missing) + " (카메라 설정에 따라 정상일 수 있음)")
    else:
        if other_app:
            pass
        elif robo and not any(("808" in l or "818" in l) for l in pub + svc):
            hints.append("시뮬레이터 컨테이너가 영상 포트를 호스트에 공개하지 않았습니다 (DRfC 의 docker compose 설정이나 swarm 서비스의 포트). DRfC 를 다시 활성화하고 학습을 다시 시작해 보세요.")
        elif robo:
            hints.append("포트는 공개돼 있지만 영상 서버가 응답하지 않습니다. 시뮬레이터(ROS)가 아직 시작 중일 가능성이 큽니다 (시작에 1~2분). 잠시 뒤 다시 진단하세요.")

    # 5) 시뮬레이터 로그에서 영상 관련 줄
    for c in robo[:1]:
        rc, out = sh(["docker", "logs", "--tail", "400", c["name"]], timeout=15, d=d)
        lines = [l for l in out.splitlines() if re.search(r"video|stream|kvs|camera|web_video|kinesis|traceback|exception|error", l, re.I)]
        add(f"시뮬레이터 로그의 영상 관련 줄 ({c['name']})", "info" if lines else "info", "\n".join(l[:200] for l in lines[-12:]) or "관련 줄 없음")

    if not hints:
        if ok_port and any(s["name"] == "영상 프레임 수신" and s["status"] == "ok" for s in steps):
            hints.append("시뮬레이터와 대시보드 쪽은 정상입니다. 화면에 안 보이면 브라우저와 대시보드 사이의 문제입니다: 아래 '브라우저 점검' 결과를 확인하세요 (원격 접속이면 대시보드 포트만 연결돼 있어도 영상은 대시보드를 통해 전달됩니다).")
        else:
            hints.append("뚜렷한 원인을 찾지 못했습니다. 이 진단 결과 전체를 복사해서 보여 주세요.")
    return _debug_result(steps, hints)


def _debug_result(steps, hints):
    mark = {"ok": "[ 정상 ]", "warn": "[ 주의 ]", "fail": "[ 문제 ]", "info": "[ 참고 ]"}
    text = "== 실시간 영상 진단 ==\n" + "\n".join(f"{mark.get(s['status'], '[ ? ]')} {s['name']}\n      " + (s['detail'] or '').replace("\n", "\n      ") for s in steps)
    text += "\n\n== 추정 원인 ==\n" + "\n".join(f"- {h}" for h in hints)
    return {"steps": steps, "hints": hints, "text": text}


def start_viewer():
    """DRfC 자체 뷰어(여러 워커의 영상을 한 화면에, http://localhost:8100). 시뮬레이터가 실행 중일 때만 의미가 있다."""
    d = _need_drfc()
    a = overview()["active"]
    if not (a and a.get("running")):
        raise DrfcError("학습이나 평가가 실행 중일 때만 뷰어를 시작할 수 있습니다.")
    return run_task("DRfC 뷰어 시작", _bash(d, a.get("experiment"), "dr-update-viewer"), cwd=d, env=_env(d), exclusive=False, shown="dr-update-viewer")


def stop_viewer():
    d = _need_drfc()
    a = overview()["active"] or {}
    return run_task("DRfC 뷰어 중지", _bash(d, a.get("experiment"), "dr-stop-viewer"), cwd=d, env=_env(d), exclusive=False, shown="dr-stop-viewer")


def experiments_usage():
    """실험 정리 화면용: 실험마다 설정/모델/차량용 파일이 차지하는 용량."""
    d = get_dir()
    root = os.path.join(d, "experiments")
    rows = []
    if not (is_drfc(d) and os.path.isdir(root)):
        return rows
    active_exp = (active(list_containers(d)) or {}) if sysenv(d) else {}
    for n in sorted(os.listdir(root)):
        if not os.path.isfile(os.path.join(root, n, "run.env")):
            continue
        path, env, meta, hp, mm, prefix = _exp_info(d, n)
        out_dir = os.path.join(d, "data", "output")
        zips = [os.path.join(out_dir, f) for f in (os.listdir(out_dir) if os.path.isdir(out_dir) else []) if f.startswith(prefix + "-") and f.endswith(".tar.gz")]
        dependents = [o for o in _all_experiment_envs(d) if o[0] != n and o[1].get("DR_LOCAL_S3_PRETRAINED", "").lower() == "true" and o[1].get("DR_LOCAL_S3_PRETRAINED_PREFIX") == prefix]
        rows.append({"name": n, "prefix": prefix, "world": env.get("DR_WORLD_NAME"), "created": meta.get("created"), "has_model": model_exists(d, prefix),
                     "config_bytes": dir_size(path), "model_bytes": dir_size(os.path.join(bucket_dir(d), prefix)), "files_bytes": sum(os.path.getsize(z) for z in zips),
                     "running": bool(active_exp.get("running") and active_exp.get("experiment") == n), "used_by": [o[0] for o in dependents]})
    return rows


def _all_experiment_envs(d):
    root = os.path.join(d, "experiments")
    out = []
    if os.path.isdir(root):
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n, "run.env")
            if os.path.isfile(p):
                out.append((n, parse_env(_read(p))))
    return out


def delete_experiments(names, config=True, model=True, files=True):
    """실험을 삭제한다 (되돌릴 수 없음).
      config: experiments/<이름>/ 폴더 (run.env, custom_files)
      model : 저장소(minio)의 모델과 지표. S3 API(aws s3 rm)로 지운다 (minio 의 데이터 폴더를 직접 지우면 minio 가 어긋날 수 있다)
      files : 만든 차량용 파일(data/output)과 시뮬레이터 로그(data/logs/robomaker/<모델>)
    실행 중인 실험, 다른 실험이 이어서 학습하는 데 쓰는 모델은 지우지 않는다."""
    d = _need_drfc(ready=False)
    names = list(dict.fromkeys(names or []))
    if not names:
        raise DrfcError("삭제할 실험을 고르세요.")
    if not (config or model or files):
        raise DrfcError("삭제할 항목(설정, 모델, 파일)을 하나 이상 고르세요.")
    a = overview()["active"]
    busy = bool(a and (a.get("running") or a.get("starting")))
    info = {}
    for n in names:
        path = exp_path(d, n)
        if not os.path.isfile(os.path.join(path, "run.env")):
            raise DrfcError(f"실험 '{n}' 을(를) 찾을 수 없습니다.")
        _, env, _, _, _, prefix = _exp_info(d, n)
        if busy and a.get("experiment") == n:
            raise DrfcError(f"'{n}' 은(는) 지금 실행 중입니다. 먼저 중지하세요.")
        if (model or files) and not _PREFIX_OK.match(prefix or "") or prefix in ("custom_files", "DeepRacer-Metrics"):
            raise DrfcError(f"'{n}' 의 모델 이름('{prefix}')이 안전하지 않아 모델/파일 삭제를 할 수 없습니다. 설정만 삭제하거나 수동으로 정리하세요.")
        info[n] = prefix
    if model and busy and not a.get("experiment"):
        raise DrfcError("외부에서 시작된 학습/평가가 실행 중이라 모델을 삭제하지 않습니다. 먼저 중지하세요.")
    if model:
        sel = set(info.values())
        for other, env in _all_experiment_envs(d):
            if other in info:
                continue
            p = env.get("DR_LOCAL_S3_PRETRAINED_PREFIX")
            if env.get("DR_LOCAL_S3_PRETRAINED", "").lower() == "true" and p in sel:
                raise DrfcError(f"'{other}' 실험이 모델 '{p}' 에서 이어서 학습하도록 설정돼 있어 그 모델을 지울 수 없습니다. '{other}' 도 같이 삭제하거나 '모델 삭제'를 끄세요.")
    q = shlex.quote
    lines = ["_fail=0"]
    need_s3 = False
    for n, prefix in info.items():
        lines.append(f"echo '== 실험 {n} 삭제'")
        if model:
            if model_exists(d, prefix):
                need_s3 = True
                lines.append(f"echo '  모델과 지표 삭제 (저장소): {prefix}'")
                url = '"s3://${DR_LOCAL_S3_BUCKET}/' + prefix + '/"'          # prefix 는 위에서 안전한 문자만 허용하도록 검증함
                lines.append("aws $DR_LOCAL_PROFILE_ENDPOINT_URL s3 rm --recursive --only-show-errors " + url
                             + " || { echo '  모델 삭제에 실패했습니다: " + prefix + "'; _fail=1; }")
            else:
                lines.append("echo '  저장소에 모델이 없습니다 (건너뜀)'")
        if files:
            lines.append(f"rm -f -- {q(os.path.join(d, 'data', 'output', prefix + '-best.tar.gz'))} {q(os.path.join(d, 'data', 'output', prefix + '-last.tar.gz'))}")
            lines.append(f"rm -rf -- {q(os.path.join(d, 'data', 'logs', 'robomaker', prefix))}")
            lines.append("echo '  차량용 파일과 시뮬레이터 로그 삭제'")
        if config:
            lines.append(f"rm -rf -- {q(exp_path(d, n))} && echo '  설정 폴더 삭제'")
    lines.append("exit $_fail")
    script = "\n".join(lines)

    def done(rc, out):
        _sum_cache.clear()
        for k in [k for k in _obj_cache if any(k[2].startswith(p + "/") for p in info.values())]:
            _obj_cache.pop(k, None)
    return run_task(f"실험 삭제 ({len(names)}개)", _bash(d, None, script, s3=need_s3), cwd=d, env=_env(d), on_done=done,
                    shown=f"실험 {', '.join(names)} 삭제 (설정 {'O' if config else 'X'}, 모델 {'O' if model else 'X'}, 파일 {'O' if files else 'X'})")


def experiment_detail(name):
    d = _need_drfc()
    path = exp_path(d, name)
    if not os.path.isfile(os.path.join(path, "run.env")):
        raise DrfcError(f"실험 '{name}' 을(를) 찾을 수 없습니다.")
    _, env, meta, hp, mm, prefix = _exp_info(d, name)
    keys = ["DR_WORLD_NAME", "DR_RACE_TYPE", "DR_CAR_COLOR", "DR_LOCAL_S3_MODEL_PREFIX", "DR_LOCAL_S3_PRETRAINED", "DR_LOCAL_S3_PRETRAINED_PREFIX",
            "DR_LOCAL_S3_PRETRAINED_CHECKPOINT", "DR_TRAIN_CHANGE_START_POSITION", "DR_TRAIN_ROUND_ROBIN_ADVANCE_DIST", "DR_TRAIN_MIN_EVAL_TRIALS",
            "DR_TRAIN_BEST_MODEL_METRIC", "DR_TRAIN_REVERSE_DIRECTION", "DR_TRAIN_ALTERNATE_DRIVING_DIRECTION", "DR_ENABLE_DOMAIN_RANDOMIZATION",
            "DR_EVAL_NUMBER_OF_TRIALS", "DR_EVAL_CHECKPOINT", "DR_EVAL_IS_CONTINUOUS", "DR_EVAL_REVERSE_DIRECTION", "DR_EVAL_SAVE_MP4"]
    out_dir = os.path.join(d, "data", "output")
    zips = []
    if os.path.isdir(out_dir):
        for fn in sorted(os.listdir(out_dir)):
            if fn.startswith(prefix + "-") and fn.endswith(".tar.gz"):
                p = os.path.join(out_dir, fn)
                zips.append({"file": fn, "size": os.path.getsize(p), "mtime": os.path.getmtime(p), "members": _tar_members(p)})
    return {"name": name, "prefix": prefix, "env": {k: env.get(k) for k in keys}, "hp": hp, "metadata": mm, "meta": meta,
            "reward_code": _read(os.path.join(path, "custom_files", "reward_function.py")), "has_model": model_exists(d, prefix), "zips": zips,
            "per_iter": hp.get("num_episodes_between_training", 20)}


def _tar_members(p):
    import tarfile
    try:
        with tarfile.open(p) as t:
            return sorted(m.name for m in t.getmembers() if m.isfile())
    except (OSError, tarfile.TarError):
        return []


def training_metrics(name):
    d = _need_drfc()
    _, env, meta, hp, mm, prefix = _exp_info(d, name)
    s = training_summary(d, prefix, hp.get("num_episodes_between_training", 20))
    return s or {"metrics": None, "episodes": 0, "per_iter": hp.get("num_episodes_between_training", 20), "best_iteration": None}


_EVAL_FILE = re.compile(r"^evaluation-[A-Za-z0-9_-]+\.json$")


def _eval_meta_path(path):
    return os.path.join(path, "evaluations.json")


def _load_eval_meta(path):
    m = _json(_eval_meta_path(path), None) or {}
    m.setdefault("labels", {})
    m.setdefault("pending", [])
    return m


def _save_eval_meta(path, meta):
    _write(_eval_meta_path(path), json.dumps(meta, ensure_ascii=False, indent=1))


def _resolve_pending(path, files):
    """대시보드에서 이름을 붙여 시작한 평가가 만든 새 지표 파일을 찾아 이름을 연결한다.
    평가 파일 이름은 시뮬레이터가 시각으로 정하므로(evaluation-YYYYMMDDHHMMSS.json), 시작할 때 있던 파일 목록을 기억해 두었다가 그 뒤에 생긴 파일을 순서대로 짝짓는다."""
    meta = _load_eval_meta(path)
    now = time.time()
    pend = [p for p in meta["pending"] if now - p.get("started", 0) < 12 * 3600]
    changed = len(pend) != len(meta["pending"])
    for p in sorted(pend, key=lambda p: p.get("started", 0)):
        cands = sorted(fn for fn in files if fn not in p.get("known", []) and fn not in meta["labels"])
        if cands:
            meta["labels"][cands[0]] = {"name": p.get("label", ""), "started": p.get("started"), "save_mp4": p.get("save_mp4"), "options": p.get("options", {})}
            pend.remove(p)
            changed = True
    meta["pending"] = pend
    if changed:
        _save_eval_meta(path, meta)
    return meta


def set_eval_label(name, file, label):
    """평가에 이름을 붙이거나 바꾼다 (이미 끝난 평가에도). 비우면 이름을 지운다."""
    d = _need_drfc(ready=False)
    path, *_ = _exp_info(d, name)
    if not _EVAL_FILE.match(file or ""):
        raise DrfcError("평가 파일 이름이 올바르지 않습니다.")
    label = re.sub(r"[\x00-\x1f]", "", str(label or "")).strip()[:60]
    meta = _load_eval_meta(path)
    meta["labels"].setdefault(file, {})["name"] = label
    _save_eval_meta(path, meta)
    return {"file": file, "label": label}


def eval_results(name):
    d = _need_drfc(ready=False)
    path, env, meta_, hp, mm, prefix = _exp_info(d, name)
    objs = [o for o in list_meta(d, f"{prefix}/metrics/evaluation/") if o["key"].rsplit("/", 1)[-1].startswith("evaluation-") and o["key"].endswith(".json")]
    files, ends = [], {}
    for o in objs:
        fn = o["key"].rsplit("/", 1)[-1]
        data = read_object(d, o["key"])
        files.append((fn, M.parse_rows(data) if data is not None else []))
        ends[fn] = o.get("modified")
    a = active(list_containers(d)) or {}
    running = bool(a.get("running") and a.get("kind") == "evaluation")
    runs = M.summarize_evaluation_runs(files, running=running)
    emeta = _resolve_pending(path, [fn for fn, _ in files])
    videos = sorted([v for v in list_meta(d, f"{prefix}/mp4/") if v["key"].lower().endswith(".mp4")], key=lambda v: v.get("modified") or 0)
    for v in videos:
        v["name"] = v["key"].rsplit("/", 1)[-1]
    by, left = M.match_videos([{"file": r["file"], "end": ends.get(r["file"])} for r in runs], videos)
    for r in runs:
        lab = emeta["labels"].get(r["file"], {})
        r["label"] = lab.get("name", "")
        r["modified"] = ends.get(r["file"])
        r["videos"] = by.get(r["file"], [])
        r["save_mp4"] = lab.get("save_mp4")
    return {"runs": runs, "running": running, "unmatched_videos": left, "mp4_prefix": f"{prefix}/mp4/", "videos_total": len(videos),
            "pending": [p.get("label") or "(이름 없음)" for p in emeta["pending"]]}


def parse_range(header, size):
    """HTTP Range 헤더 한 구간을 (시작, 끝) 으로. 없으면 None, 범위를 벗어났거나 올바르지 않으면 ValueError."""
    if not header:
        return None
    m = re.match(r"^bytes=(\d*)-(\d*)$", header.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        raise ValueError("range")
    if m.group(1) == "":
        n = int(m.group(2))
        start, end = max(0, size - n), size - 1
    else:
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
    if start >= size or start > end:
        raise ValueError("range")
    return start, end


def open_video(name, key, range_header=None):
    """저장된 평가 영상(MP4)을 스트리밍으로 연다 (재생 위치를 옮길 수 있게 Range 지원).
    반환: {status, headers, chunks(iterator), close()}. 그 실험의 <모델>/mp4/ 아래 .mp4 만 연다."""
    d = _need_drfc(ready=False)
    path, env, meta_, hp, mm, prefix = _exp_info(d, name)
    parts = (key or "").split("/")
    if not key.startswith(prefix + "/mp4/") or ".." in parts or "\\" in key or any(ord(c) < 32 for c in key) or not key.lower().endswith(".mp4"):
        raise DrfcError("영상 경로가 올바르지 않습니다.")
    bucket = sysenv(d).get("DR_LOCAL_S3_BUCKET", "bucket")
    p = os.path.join(d, "data", "minio", bucket, *parts)
    if os.path.isfile(p):                                                    # 평범한 파일이면 파일로 (가짜 DRfC, 예전 방식)
        size = os.path.getsize(p)
        try:
            rng = parse_range(range_header, size)
        except ValueError:
            return {"status": 416, "headers": {"Content-Range": f"bytes */{size}"}, "chunks": iter(()), "close": lambda: None}
        start, end = rng if rng else (0, size - 1)
        f = open(p, "rb")
        f.seek(start)

        def chunks():
            left = end - start + 1
            while left > 0:
                b = f.read(min(65536, left))
                if not b:
                    break
                left -= len(b)
                yield b
        h = {"Content-Type": "video/mp4", "Content-Length": str(end - start + 1), "Accept-Ranges": "bytes"}
        if rng:
            h["Content-Range"] = f"bytes {start}-{end}/{size}"
        return {"status": 206 if rng else 200, "headers": h, "chunks": chunks(), "close": f.close}
    c = _s3(d)
    if c is None:
        raise DrfcError("저장소(minio)의 자격 증명(~/.aws/credentials 의 minio 프로필)을 찾지 못했습니다.")
    try:
        st, hdr, resp, conn = c.open_object(bucket, key, range_header)
    except S3Error as e:
        raise DrfcError(str(e))
    if st == 404:
        conn.close()
        raise DrfcError("영상 파일을 찾을 수 없습니다 (이미 삭제됐을 수 있습니다).")
    if st == 416:
        conn.close()
        return {"status": 416, "headers": ({"Content-Range": hdr["content-range"]} if hdr.get("content-range") else {}), "chunks": iter(()), "close": lambda: None}
    if st not in (200, 206):
        body = resp.read(200).decode("utf-8", "replace")
        conn.close()
        raise DrfcError(f"영상을 읽지 못했습니다 ({st}): {body}")
    h = {"Content-Type": "video/mp4", "Accept-Ranges": "bytes"}
    for k_in, k_out in (("content-length", "Content-Length"), ("content-range", "Content-Range")):
        if hdr.get(k_in):
            h[k_out] = hdr[k_in]

    def s3_chunks():
        while True:
            b = resp.read1(65536)
            if not b:
                break
            yield b
    return {"status": st, "headers": h, "chunks": s3_chunks(), "close": conn.close}


def _check_free(d, kind):
    ov = overview()
    a = ov["active"]
    if a and (a.get("running") or a.get("starting")):
        who = f" ('{a['experiment']}')" if a.get("experiment") else ""
        raise DrfcError(f"이미 {'학습' if a.get('kind') == 'training' else '평가' if a.get('kind') == 'evaluation' else '작업'}이 실행 중입니다{who}. 먼저 중지하세요.")


def start_training(name, wipe=False):
    d = _need_drfc()
    _check_swarm(d)
    _check_free(d, "training")
    path, env, meta, hp, mm, prefix = _exp_info(d, name)
    if not os.path.isfile(os.path.join(path, "run.env")):
        raise DrfcError(f"실험 '{name}' 을(를) 찾을 수 없습니다.")
    if model_exists(d, prefix) and not wipe:
        raise DrfcError(f"모델 '{prefix}' 이(가) 이미 있습니다. 이어서 학습하려면 '새 실험'에서 '이어서 학습'으로 새 실험을 만드세요. 처음부터 다시 하려면 '덮어쓰기'를 선택하세요 (기존 모델이 지워집니다).")
    inner = "dr-upload-custom-files && dr-start-training -q" + (" -w" if wipe else "")
    script = _stale_stack_script("training", _run_id_of(d, name)) + inner
    return run_task(f"학습 시작: {name}", _bash(d, name, script, s3=True), cwd=d, env=_env(d), shown=inner,
                    on_done=lambda rc, out: _set_state(name, "training") if rc == 0 else None)


# DRfC 의 중지 스크립트로 멈추지 못했을 때를 위한 정리: deepracer-* 스택과 학습/시뮬레이터 컨테이너를 직접 지운다 (minio 저장소(s3 스택)는 건드리지 않는다)
_FORCE_CLEANUP = r"""
sleep 2
echo '남아 있는 시뮬레이터/학습 컨테이너를 직접 정리합니다...'
for s in $(docker stack ls --format '{{.Name}}' 2>/dev/null | grep '^deepracer-'); do echo "스택 삭제: $s"; docker stack rm "$s"; done
for c in $(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^algo-|deepracer-.*(robomaker|rl_coach|algo)'); do echo "컨테이너 삭제: $c"; docker rm -f "$c" >/dev/null; done
exit 0
"""


def stop_training(force=False):
    """실행 중인 학습/평가를 중지한다. 어떤 실험인지, 학습인지 평가인지 대시보드가 알면 그에 맞는 DRfC 명령만 쓰고, 모르면(터미널 등 밖에서 시작) 둘 다 쓴 뒤
    남은 것을 직접 정리한다. force=True 면 DRfC 명령 뒤에 항상 직접 정리한다 (일반 중지로 안 멈출 때)."""
    d = _need_drfc()
    a = overview()["active"] or {}
    exp = a.get("experiment") or (_state() or {}).get("experiment")
    kind = a.get("kind")
    known = kind in ("training", "evaluation")
    steps = ["dr-stop-evaluation"] if kind == "evaluation" else ["dr-stop-training"] if kind == "training" else ["dr-stop-evaluation", "dr-stop-training"]
    inner = "; ".join(steps)
    if force or not known:
        inner += "; " + _FORCE_CLEANUP
    label = "강제 중지" if force else "중지"
    return run_task(label, _bash(d, exp, inner), cwd=d, env=_env(d), shown=inner.splitlines()[0], on_done=lambda rc, out: _clear_state() if rc == 0 else None)


# ----------------------------------------------------------------------------- 관리 (dr-* 명령 버튼)
# level: read = 읽기만, change = 설정이나 실행 상태를 바꿈, danger = 되돌리기 어렵거나 영향이 큼 (화면에서 확인을 받고, 서버도 confirm 이 없으면 거부)
_STACK_STATES_SCRIPT = r"""
echo "== docker stack ls"; docker stack ls
for s in $(docker stack ls --format '{{.Name}}' 2>/dev/null); do
  echo; echo "== docker stack ps $s"
  docker stack ps "$s" --no-trunc --format 'table {{.Name}}\t{{.CurrentState}}\t{{.Error}}' 2>&1
done
"""
_RM_STALE_SCRIPT = r"""
found=0
for s in $(docker stack ls --format '{{.Name}}' 2>/dev/null | grep '^deepracer-'); do
  found=1
  if docker stack ps "$s" --format '{{.CurrentState}}' 2>/dev/null | grep -qiE '^(running|starting|preparing|pending|assigned|accepted|ready|new|allocated)'; then
    echo "건너뜀 (실행/준비 중인 작업이 있음. 중지 버튼으로 먼저 멈추세요): $s"
  else
    echo "삭제: $s"; docker stack rm "$s"
  fi
done
[ "$found" = 0 ] && echo "남은 deepracer-* 스택이 없습니다."
exit 0
"""

ADMIN_COMMANDS = [
    # (id, 그룹, 이름, 설명, 실행할 셸 명령, level)
    ("stacks", "상태 확인", "스택과 작업 상태", "swarm 스택(deepracer-*, s3 등)과 각 스택에 남은 작업의 상태. 종료된 작업의 기록이 남아 있으면 DRfC 가 시작을 거부합니다.", _STACK_STATES_SCRIPT, "read"),
    ("containers", "상태 확인", "컨테이너 목록", "지금 실행 중인 컨테이너 (docker ps).", "docker ps --format 'table {{.Names}}\\t{{.Image}}\\t{{.Status}}'", "read"),
    ("dr-summary", "상태 확인", "dr-summary", "DRfC 의 환경 요약 (클라우드 설정, 이미지, 실행 중인 서비스와 컨테이너).", "dr-summary", "read"),
    ("dr-find-sagemaker", "상태 확인", "dr-find-sagemaker", "실행 중인 Sagemaker(학습) 컨테이너를 찾습니다.", "dr-find-sagemaker", "read"),
    ("dr-find-robomaker", "상태 확인", "dr-find-robomaker", "실행 중인 RoboMaker(시뮬레이터) 컨테이너를 찾습니다.", "dr-find-robomaker", "read"),
    ("dr-stop-training", "중지와 정리", "dr-stop-training", "학습을 중지하고 스택과 Sagemaker 컨테이너를 정리합니다. 로그 파일을 올립니다.", "dr-stop-training", "change"),
    ("dr-stop-evaluation", "중지와 정리", "dr-stop-evaluation", "평가를 중지하고 스택을 정리합니다.", "dr-stop-evaluation", "change"),
    ("rm-stale-stacks", "중지와 정리", "이전 실행 기록(스택) 지우기", "종료된 작업의 기록만 남은 deepracer-* 스택을 지웁니다 (docker stack rm). 실행/준비 중인 작업이 있는 스택은 건드리지 않습니다. "
        "학습이 끝난 뒤나 실패한 뒤에 'Processes running in stack' 으로 시작이 거부될 때 씁니다.", _RM_STALE_SCRIPT, "change"),
    ("dr-stop-all", "중지와 정리", "dr-stop-all (모든 스택 중지)", "학습, 평가, 뷰어, 로그 분석 등 모든 스택과 minio(s3 스택)까지 내립니다. minio 는 다음 학습·평가를 시작할 때 DRfC 활성화 단계에서 다시 올라옵니다. 저장된 모델은 그대로입니다.", "dr-stop-all", "danger"),
    ("force-stop", "중지와 정리", "강제 중지", "DRfC 의 중지 명령으로 안 멈출 때. deepracer-* 스택과 algo-*, 시뮬레이터, 코치 컨테이너를 직접 지웁니다 (minio 는 그대로).", _FORCE_CLEANUP, "danger"),
    ("dr-update-env", "설정과 파일", "dr-update-env", "system.env 와 선택한 실험의 run.env 를 다시 읽습니다.", "dr-update-env", "change"),
    ("dr-upload-custom-files", "설정과 파일", "dr-upload-custom-files", "선택한 실험의 보상함수, 행동 공간, 하이퍼파라미터 파일을 저장소(minio)의 custom_files 로 올립니다.", "dr-upload-custom-files", "change"),
    ("dr-download-custom-files", "설정과 파일", "dr-download-custom-files", "저장소(minio)의 custom_files 를 로컬 custom_files/ 로 내려받습니다. 같은 이름의 로컬 파일이 바뀔 수 있습니다.", "dr-download-custom-files", "danger"),
    ("dr-start-viewer", "부가 서비스", "dr-start-viewer", "워커 전체의 영상을 한 화면에서 보는 DRfC 뷰어를 시작합니다 (http://localhost:8100).", "dr-start-viewer", "change"),
    ("dr-update-viewer", "부가 서비스", "dr-update-viewer", "뷰어를 껐다가 다시 시작합니다 (워커 구성을 바꾼 뒤).", "dr-update-viewer", "change"),
    ("dr-stop-viewer", "부가 서비스", "dr-stop-viewer", "뷰어를 중지합니다.", "dr-stop-viewer", "change"),
    ("dr-start-loganalysis", "부가 서비스", "dr-start-loganalysis", "Jupyter 로그 분석 컨테이너를 시작합니다 (포트 8888).", "dr-start-loganalysis", "change"),
    ("dr-stop-loganalysis", "부가 서비스", "dr-stop-loganalysis", "로그 분석 컨테이너를 중지합니다.", "dr-stop-loganalysis", "change"),
    ("dr-start-metrics", "부가 서비스", "dr-start-metrics", "DRfC 의 지표 서비스를 시작합니다.", "dr-start-metrics", "change"),
    ("dr-stop-metrics", "부가 서비스", "dr-stop-metrics", "지표 서비스를 중지합니다.", "dr-stop-metrics", "change"),
]
_ADMIN = {c[0]: c for c in ADMIN_COMMANDS}
_STOPS = {"dr-stop-training", "dr-stop-evaluation", "dr-stop-all", "force-stop", "rm-stale-stacks"}

# 직접 입력에서 막는 명령: 끝나지 않고 로그를 따라가는 것, 브라우저를 여는 것, 전용 화면이 있는 것(-q 가 필요한 시작 명령), 실제 AWS 계정에 쓰는 것
_BLOCKED_CUSTOM = {"dr-logs-sagemaker", "dr-logs-robomaker", "dr-logs-loganalysis", "dr-view-stream", "dr-url-loganalysis", "dr-start-training", "dr-start-evaluation",
                   "dr-upload-model", "dr-upload-car-zip", "dr-download-model", "dr-set-upload-model", "dr-increment-upload-model", "dr-list-aws-models", "dr-start-tournament"}
_CUSTOM_FALLBACK = {"dr-update-env", "dr-upload-custom-files", "dr-download-custom-files", "dr-stop-training", "dr-stop-evaluation", "dr-stop-all", "dr-summary", "dr-find-sagemaker",
                    "dr-find-robomaker", "dr-start-viewer", "dr-stop-viewer", "dr-update-viewer", "dr-start-loganalysis", "dr-stop-loganalysis", "dr-start-metrics", "dr-stop-metrics",
                    "dr-create-car-zip", "dr-increment-training"}


def _drfc_commands(d):
    """이 DRfC 가 정의한 dr-* 함수 이름들 (bin/scripts_wrapper.sh). 읽지 못하면 알려진 목록."""
    names = set(re.findall(r"^function (dr-[a-z0-9-]+)", _read(os.path.join(d, "bin", "scripts_wrapper.sh")), flags=re.M))
    return names or set(_CUSTOM_FALLBACK)


def admin_info():
    d = _need_drfc(ready=False)
    groups = []
    for c in ADMIN_COMMANDS:
        if not groups or groups[-1]["name"] != c[1]:
            groups.append({"name": c[1], "commands": []})
        groups[-1]["commands"].append({"id": c[0], "label": c[2], "desc": c[3], "level": c[5]})
    exps = sorted(n for n in os.listdir(os.path.join(d, "experiments")) if os.path.isfile(os.path.join(d, "experiments", n, "run.env"))) if os.path.isdir(os.path.join(d, "experiments")) else []
    return {"groups": groups, "experiments": exps, "custom_allowed": sorted(_drfc_commands(d) - _BLOCKED_CUSTOM), "custom_blocked": sorted(_BLOCKED_CUSTOM & _drfc_commands(d))}


def _admin_exp(d, experiment):
    exp = (experiment or "").strip() or None
    if exp and not (re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$", exp) and os.path.isfile(os.path.join(d, "experiments", exp, "run.env"))):
        raise DrfcError(f"실험 '{exp}' 을(를) 찾을 수 없습니다.")
    return exp


def _admin_run_task(d, label, exp, inner, shown, stop):
    def done(rc, out):
        _live_cache.clear()
        if stop and rc == 0:
            _clear_state()
    return run_task(label, _bash(d, exp, inner), cwd=d, env=_env(d), shown=shown, on_done=done)


def admin_run(cmd_id, experiment=None, confirm=False):
    """관리 화면의 버튼. 등록된 명령만 실행한다 (사용자 입력이 셸에 들어가지 않음)."""
    d = _need_drfc()
    c = _ADMIN.get(cmd_id)
    if not c:
        raise DrfcError("알 수 없는 명령입니다.")
    if c[5] == "danger" and not confirm:
        raise DrfcError("확인이 필요한 명령입니다 (되돌리기 어렵거나 영향이 큽니다).")
    exp = _admin_exp(d, experiment)
    shown = c[4].strip().splitlines()[0] if cmd_id.startswith("dr-") or len(c[4].strip().splitlines()) == 1 else f"({c[2]})"
    return _admin_run_task(d, f"관리: {c[2]}" + (f" [{exp}]" if exp else ""), exp, c[4], shown, cmd_id in _STOPS)


_ARG_OK = re.compile(r"^[A-Za-z0-9_./=:@,+-]{1,120}$")


def admin_custom(command, experiment=None, confirm=False):
    """직접 입력한 dr-* 명령. DRfC 가 정의한 함수 중 허용 목록에 있는 것만, 인자는 안전한 문자만 허용한다 (셸 문법 문자 불가)."""
    d = _need_drfc()
    if re.search(r"[\x00-\x1f]", command or ""):
        raise DrfcError("명령에 줄바꿈이나 제어 문자를 넣을 수 없습니다.")
    try:
        parts = shlex.split(command or "")
    except ValueError:
        raise DrfcError("명령을 해석할 수 없습니다 (따옴표를 확인하세요).")
    if not parts:
        raise DrfcError("명령을 입력하세요. 예: dr-update-env")
    name, args = parts[0], parts[1:]
    allowed = _drfc_commands(d) - _BLOCKED_CUSTOM
    if not re.match(r"^dr-[a-z0-9-]+$", name) or name not in allowed:
        why = " (끝나지 않거나 전용 화면이 있는 명령은 막혀 있습니다)" if name in _BLOCKED_CUSTOM else ""
        raise DrfcError(f"허용되지 않는 명령입니다: {name}{why}. 허용: {', '.join(sorted(allowed))}")
    if len(args) > 8 or any(not _ARG_OK.match(a) for a in args):
        raise DrfcError("인자는 영문, 숫자, 그리고 _ . / = : @ , + - 만 쓸 수 있고, 8개 이하여야 합니다.")
    if not confirm:
        raise DrfcError("직접 입력한 명령은 확인이 필요합니다.")
    exp = _admin_exp(d, experiment)
    inner = " ".join([name] + [shlex.quote(a) for a in args])
    return _admin_run_task(d, f"관리: {inner}" + (f" [{exp}]" if exp else ""), exp, inner, inner, name in {"dr-stop-training", "dr-stop-evaluation", "dr-stop-all"})


def start_evaluation(name, opts):
    d = _need_drfc()
    _check_swarm(d)
    _check_free(d, "evaluation")
    path, env, meta, hp, mm, prefix = _exp_info(d, name)
    if not model_exists(d, prefix):
        raise DrfcError("평가할 모델이 없습니다. 먼저 학습을 해서 모델을 만드세요.")
    up = {"DR_EVAL_NUMBER_OF_TRIALS": _num(opts.get("trials", 3), int, "평가 주행 수", 1, 50),
          "DR_EVAL_CHECKPOINT": opts.get("checkpoint") if opts.get("checkpoint") in ("last", "best") else "last",
          "DR_EVAL_IS_CONTINUOUS": _bool(opts.get("continuous", True)), "DR_EVAL_REVERSE_DIRECTION": _bool(opts.get("reverse", False)),
          "DR_EVAL_SAVE_MP4": _bool(opts.get("save_mp4", False))}
    rp = os.path.join(path, "run.env")
    _write(rp, set_env(_read(rp), up))
    label = re.sub(r"[\x00-\x1f]", "", str(opts.get("label") or "")).strip()[:60]
    known = [o["key"].rsplit("/", 1)[-1] for o in list_meta(d, f"{prefix}/metrics/evaluation/")]     # 시작할 때 이미 있던 평가 파일 (새로 생길 파일과 구별하려고)
    pending = {"label": label, "started": time.time(), "known": known, "save_mp4": bool(up["DR_EVAL_SAVE_MP4"]),
               "options": {"trials": up["DR_EVAL_NUMBER_OF_TRIALS"], "checkpoint": up["DR_EVAL_CHECKPOINT"], "continuous": up["DR_EVAL_IS_CONTINUOUS"], "reverse": up["DR_EVAL_REVERSE_DIRECTION"]}}
    inner = "dr-upload-custom-files && dr-start-evaluation -q" + (" -c" if opts.get("clone") else "")
    script = _stale_stack_script("evaluation", _run_id_of(d, name)) + inner

    def done(rc, out):
        if rc == 0:
            _set_state(name, "evaluation")
            m = _load_eval_meta(path)
            m["pending"].append(pending)
            _save_eval_meta(path, m)
    return run_task(f"평가 시작: {name}" + (f" ({label})" if label else ""), _bash(d, name, script, s3=True), cwd=d, env=_env(d), shown=inner, on_done=done)


def export_car(name, best=False):
    d = _need_drfc()
    path, env, meta, hp, mm, prefix = _exp_info(d, name)
    if not model_exists(d, prefix):
        raise DrfcError("내보낼 모델이 없습니다. 먼저 학습을 해서 모델을 만드세요.")
    out = os.path.join(d, "data", "output", f"{prefix}-{'best' if best else 'last'}.tar.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    inner = f"dr-create-car-zip {'-b ' if best else ''}-p {shlex.quote(prefix)} -o {shlex.quote(out)}"
    return run_task(f"차량용 파일 만들기: {name} ({'best' if best else 'last'})", _bash(d, name, inner, s3=True), cwd=d, env=_env(d), exclusive=False, shown=inner)


def download_path(file):
    d = _need_drfc()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.tar\.gz", file or ""):
        raise DrfcError("파일 이름이 올바르지 않습니다.")
    p = os.path.join(d, "data", "output", file)
    if not os.path.isfile(p):
        raise DrfcError("파일을 찾을 수 없습니다.")
    return p


def logs(role, tail=200):
    d = get_dir()
    tail = max(10, min(int(tail), 2000))
    cs = [c for c in list_containers(d) if c["role"] == role]
    if not cs:
        return {"container": None, "text": "", "message": "해당 컨테이너가 실행 중이 아닙니다."}
    rc, out = sh(["docker", "logs", "--tail", str(tail), cs[0]["name"]], timeout=15, d=d)
    return {"container": cs[0]["name"], "text": out, "message": ""}
