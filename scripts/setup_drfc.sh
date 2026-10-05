#!/usr/bin/env bash
# DeepRacer Trainer: Ubuntu 에서 DRfC(DeepRacer-for-Cloud) 실행 환경을 자동으로 설치한다.
#
#   sudo bash scripts/setup_drfc.sh                    # GPU 가 있으면 GPU, 없으면 CPU 모드 (자동 감지)
#   sudo bash scripts/setup_drfc.sh --arch cpu
#   sudo bash scripts/setup_drfc.sh --arch gpu --install-driver     # NVIDIA 드라이버도 설치 (재부팅 필요)
#   bash scripts/setup_drfc.sh --dry-run               # 아무것도 바꾸지 않고 무엇을 할지만 보여 줌 (sudo 불필요)
#
# 옵션
#   --arch gpu|cpu|auto     (기본 auto)
#   --drfc-dir DIR          DRfC 설치 위치 (기본 ~사용자/deepracer-for-cloud)
#   --user NAME             DRfC 를 쓸 일반 사용자 (기본: sudo 를 실행한 사용자)
#   --install-driver        NVIDIA GPU 가 있는데 드라이버가 없을 때 드라이버를 설치 (기본: 설치하지 않고 멈춤)
#   --dry-run               미리보기
#
# 원칙
#   - 여러 번 실행해도 안전하다. 이미 된 단계는 건너뛰고, 중간에 멈췄으면 이어서 한다.
#   - DRfC 의 bin/prepare.sh 가 하는 일 중 로컬 컴퓨터에 필요한 것만 한다. 시스템 전체 apt upgrade 나 needrestart 제거는 하지 않는다.
#   - 이미 초기화된 DRfC(system.env 가 있음)에는 init.sh 를 다시 실행하지 않는다. init.sh 는 system.env, run.env, custom_files 를 덮어쓴다.
#   - 재부팅은 대신 하지 않는다. 필요하면 안내만 하고 멈춘다.
#
# 종료 코드: 0 완료(또는 미리보기), 1 실패, 2 사용자 조치 필요(드라이버/재부팅/로그인)
# (시험용 환경변수: SETUP_STUB_DIR=가짜 명령 폴더, SETUP_PCI_DIR=PCI 장치 폴더)

set -uo pipefail

REPO_URL="https://github.com/aws-deepracer-community/deepracer-for-cloud.git"
DEFAULT_SUPPORTED="22.04 24.04 24.10 25.04 25.10 26.04"      # DRfC bin/prepare.sh 의 SUPPORTED_VERSIONS (DRfC 폴더가 있으면 거기서 읽음)

ARCH="auto"; DRFC_DIR=""; TARGET_USER="${SUDO_USER:-}"; INSTALL_DRIVER=0; DRY=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch) ARCH="${2:-}"; shift 2 ;;
        --drfc-dir) DRFC_DIR="${2:-}"; shift 2 ;;
        --user) TARGET_USER="${2:-}"; shift 2 ;;
        --install-driver) INSTALL_DRIVER=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "알 수 없는 옵션: $1" >&2; usage 1 ;;
    esac
done
case "$ARCH" in gpu|cpu|auto) ;; *) echo "--arch 는 gpu, cpu, auto 중 하나여야 합니다." >&2; exit 1 ;; esac

export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a      # apt 가 대화형 질문으로 멈추지 않게
[[ -n "${SETUP_STUB_DIR:-}" ]] && PATH="$SETUP_STUB_DIR:$PATH"   # (시험용) 가짜 명령 폴더
export PATH="$PATH:/snap/bin:/usr/sbin:/sbin:/usr/lib/wsl/lib"   # WSL2 의 nvidia-smi 는 /usr/lib/wsl/lib (sudo 환경에서는 PATH 에 없음)
OS_RELEASE_FILE="${SETUP_OS_RELEASE:-/etc/os-release}"; PROC_VERSION_FILE="${SETUP_PROC_VERSION:-/proc/version}"   # (시험용)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ----------------------------------------------------------------------------- 단계 정의
declare -A LABEL=(
    [preflight]="시스템 확인"
    [apt]="기본 도구 설치 (jq, git, python3-venv 등)"
    [awscli]="AWS CLI 설치"
    [driver]="NVIDIA 드라이버 확인"
    [docker]="Docker 설치와 시작"
    [toolkit]="NVIDIA 컨테이너 도구 설정"
    [group]="docker 사용 권한"
    [tmp]="/tmp/sagemaker 폴더 (재부팅 후에도 유지)"
    [clone]="DRfC 내려받기"
    [venv]="DRfC 파이썬 환경"
    [init]="DRfC 초기화 (init.sh)"
    [minio]="minio 이미지 (저장소)"
    [verify]="최종 확인"
)
REBOOT=0; RELOGIN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"


say() { printf '%s\n' "$*"; }
begin() { say ""; say "▶ ${LABEL[$1]}"; }
finish() {   # finish ID 상태 메시지
    case "$2" in
        ok) say "  완료: $3" ;; skipped) say "  건너뜀: $3" ;; plan) say "  할 일: $3" ;;
        warn) say "  주의: $3" ;; blocked) say "  멈춤: $3" ;; failed) say "  실패: $3" ;;
    esac
}
tail_out() { tail -n 14 "$WORK/$1.out" 2>/dev/null || true; }

# 명령을 실행하면서 출력을 단계별 파일에도 남긴다. 실패하면 그 마지막 줄들을 보여 준다.
runv() {   # runv ID 명령...
    local id="$1"; shift
    say "  \$ $*"
    { "$@"; } 2>&1 | tee -a "$WORK/$id.out"
    return "${PIPESTATUS[0]}"
}

fail_step() {   # fail_step ID 메시지  (스크립트 전체를 멈춘다)
    finish "$1" failed "$2"
    tail_out "$1" | sed "s/^/    | /"
    say ""; say "== 실패한 단계의 메시지를 확인하고 원인을 고친 뒤 같은 명령을 다시 실행하세요. 이미 끝난 단계는 건너뜁니다."
    exit 1
}
block_step() {  # 사용자 조치가 필요해서 멈춤
    finish "$1" blocked "$2"
    say ""; say "== ${3:-$2}"
    exit 2
}

# ----------------------------------------------------------------------------- 사용자/경로 도우미
IS_ROOT=0; [[ $EUID -eq 0 ]] && IS_ROOT=1
if [[ -z "$TARGET_USER" ]]; then TARGET_USER="$(id -un)"; fi
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
[[ -n "$DRFC_DIR" ]] || DRFC_DIR="${TARGET_HOME:-$HOME}/deepracer-for-cloud"
DRFC_DIR="${DRFC_DIR/#\~/${TARGET_HOME:-$HOME}}"

as_user() {   # 대상 사용자 권한으로 실행
    if [[ $IS_ROOT -eq 1 && "$(id -un)" != "$TARGET_USER" ]]; then
        runuser -u "$TARGET_USER" -- env HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" PATH="$PATH" "$@"
    else
        env HOME="$TARGET_HOME" USER="$TARGET_USER" LOGNAME="$TARGET_USER" PATH="$PATH" "$@"
    fi
}
as_user_docker() {   # docker 그룹 권한으로 (방금 그룹에 추가해서 아직 로그인에 반영되지 않았어도 동작)
    local cmd; cmd="$(printf '%q ' "$@")"
    as_user sg docker -c "$cmd"
}
have() { command -v "$1" >/dev/null 2>&1; }
pkg_missing() { local p out=(); for p in "$@"; do dpkg -s "$p" >/dev/null 2>&1 || out+=("$p"); done; echo "${out[*]:-}"; }
nvidia_hw() {   # PCI 장치 중 NVIDIA(0x10de) 그래픽 장치 개수. lspci 가 없어도 동작 (SETUP_PCI_DIR 은 시험용)
    local n=0 d v c
    for d in "${SETUP_PCI_DIR:-/sys/bus/pci/devices}"/*; do
        [[ -r "$d/vendor" ]] || continue
        v="$(cat "$d/vendor")"; c="$(cat "$d/class")"
        if [[ "$v" == "0x10de" && "$c" == 0x03* ]]; then n=$((n + 1)); fi
    done
    echo "$n"
}
IS_WSL2=0; if grep -qi microsoft "$PROC_VERSION_FILE" 2>/dev/null && grep -q WSL2 "$PROC_VERSION_FILE" 2>/dev/null; then IS_WSL2=1; fi

# ============================================================================= 1. 시스템 확인
step_preflight() {
    begin preflight
    # shellcheck disable=SC1090,SC1091
    . "$OS_RELEASE_FILE"
    if [[ "${ID:-}" != "ubuntu" ]]; then
        fail_step preflight "Ubuntu 에서만 자동 설치를 지원합니다 (현재: ${PRETTY_NAME:-알 수 없음}). 수동 설치 안내를 따르세요."
    fi
    local supported="$DEFAULT_SUPPORTED" line
    if [[ -f "$DRFC_DIR/bin/prepare.sh" ]]; then
        line="$(grep -m1 '^SUPPORTED_VERSIONS=' "$DRFC_DIR/bin/prepare.sh" | sed -E 's/.*\((.*)\).*/\1/' | tr -d '"')"
        [[ -n "$line" ]] && supported="$line"
    fi
    local os_note=""
    if ! grep -qw -- "$VERSION_ID" <<<"$supported"; then
        # DRfC 의 prepare.sh 지원 목록에는 없지만, DRfC 문서(installation.md, windows.md)는 Ubuntu 20.04 와 WSL2 를 지원한다고 적고 있다. 막지 않고 알린다.
        os_note=" / Ubuntu $VERSION_ID 은(는) DRfC 설치 스크립트의 지원 목록($supported)에는 없지만 DRfC 문서는 20.04 와 WSL2 도 가능하다고 해서 계속 진행합니다"
    fi
    if [[ "$TARGET_USER" == "root" ]] || ! id "$TARGET_USER" >/dev/null 2>&1; then
        fail_step preflight "DRfC 를 쓸 일반 사용자를 찾지 못했습니다 (--user 로 지정하세요). root 에는 설치하지 않습니다."
    fi
    if [[ $DRY -eq 0 && $IS_ROOT -eq 0 ]]; then
        fail_step preflight "관리자 권한이 필요합니다. 앞에 sudo 를 붙여 실행하세요."
    fi
    # GPU 판단
    local gpus; gpus="$(nvidia_hw)"
    if [[ "$gpus" -eq 0 ]] && have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then gpus=1; fi
    local note=""
    if [[ "$ARCH" == "auto" ]]; then
        if [[ "$gpus" -ge 1 ]]; then ARCH="gpu"; note="NVIDIA GPU 를 찾아 GPU 모드로"; else ARCH="cpu"; note="NVIDIA GPU 가 없어 CPU 모드로"; fi
    elif [[ "$ARCH" == "gpu" && "$gpus" -eq 0 && $IS_WSL2 -eq 0 ]]; then
        block_step preflight "NVIDIA GPU 를 찾지 못했습니다. GPU 모드를 고를 수 없습니다. CPU 모드(--arch cpu)로 다시 실행하세요."
    else
        note="${ARCH^^} 모드(직접 선택)로"
    fi
    # 인터넷
    local url bad=""
    for url in https://github.com http://archive.ubuntu.com; do
        curl -fsS -m 10 -o /dev/null -I "$url" 2>/dev/null || bad+=" $url"
    done
    [[ -n "$bad" ]] && fail_step preflight "인터넷에 연결할 수 없습니다:$bad  (프록시가 있으면 환경변수 https_proxy 를 설정하세요)"
    # 디스크 (DRfC 권장: 30~40GB)
    local free_gb; free_gb="$(df -BG --output=avail "${TARGET_HOME:-/}" 2>/dev/null | tail -1 | tr -dc '0-9')"
    local warn=""; [[ -n "$free_gb" && "$free_gb" -lt 30 ]] && warn=" / 여유 디스크 ${free_gb}GB (30~40GB 권장)"
    local wsl=""; [[ $IS_WSL2 -eq 1 ]] && wsl=" (WSL2)"
    local st="ok"; [[ -n "$warn$os_note" ]] && st="warn"
    finish preflight "$st" "Ubuntu ${VERSION_ID}${wsl}, 사용자 ${TARGET_USER}, ${note} 설치${warn}${os_note}"
}

# ============================================================================= 2. 기본 도구
step_apt() {
    begin apt
    local want=(jq python3-boto3 python3-venv screen git curl ca-certificates pciutils openssl) missing
    missing="$(pkg_missing "${want[@]}")"
    if [[ -z "$missing" ]]; then finish apt skipped "이미 설치돼 있음"; return; fi
    if [[ $DRY -eq 1 ]]; then finish apt plan "설치 예정: $missing"; return; fi
    runv apt apt-get update || say "  (apt update 가 일부 실패했습니다. 다른 저장소의 문제일 수 있어 계속합니다.)"
    # shellcheck disable=SC2086
    runv apt apt-get install -y --no-install-recommends $missing || fail_step apt "패키지 설치에 실패했습니다: $missing"
    finish apt ok "설치함: $missing"
}

# ============================================================================= 3. AWS CLI
step_awscli() {
    begin awscli
    if have aws; then finish awscli skipped "이미 설치돼 있음 ($(aws --version 2>&1 | head -1 | cut -c1-40))"; return; fi
    # shellcheck disable=SC1090,SC1091
    . "$OS_RELEASE_FILE"
    local how
    if [[ "$VERSION_ID" == 22.* ]]; then how="apt (awscli)"; elif have snap; then how="snap (aws-cli)"; else how="파이썬 가상환경 (/opt/awscli-venv)"; fi
    if [[ $DRY -eq 1 ]]; then finish awscli plan "설치 예정: $how"; return; fi
    if [[ "$VERSION_ID" == 22.* ]]; then
        runv awscli apt-get install -y awscli || fail_step awscli "awscli 설치에 실패했습니다."
    elif have snap; then
        runv awscli snap install aws-cli --classic || fail_step awscli "snap 으로 aws-cli 를 설치하지 못했습니다."
    else
        # 24.04 이상에는 awscli deb 패키지가 없고 snap 도 없는 서버를 위한 대체 경로
        { runv awscli python3 -m venv /opt/awscli-venv && runv awscli /opt/awscli-venv/bin/pip install --quiet awscli \
            && ln -sf /opt/awscli-venv/bin/aws /usr/local/bin/aws; } || fail_step awscli "awscli 를 설치하지 못했습니다."
    fi
    have aws || fail_step awscli "설치 후에도 aws 명령을 찾을 수 없습니다."
    finish awscli ok "설치함 ($how)"
}

# ============================================================================= 4. NVIDIA 드라이버
step_driver() {
    begin driver
    if [[ "$ARCH" != "gpu" ]]; then finish driver skipped "CPU 모드라 필요 없음"; return; fi
    if [[ $IS_WSL2 -eq 1 ]]; then finish driver skipped "WSL2: Windows 쪽 NVIDIA 드라이버를 사용합니다"; return; fi
    if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
        local ver major
        ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        major="${ver%%.*}"
        if [[ "$major" =~ ^[0-9]+$ && "$major" -lt 560 ]]; then
            finish driver warn "드라이버 $ver 가 동작합니다. 다만 DRfC 의 초기화 시험 이미지(CUDA 12.6)에는 560 이상이 권장됩니다. 시험에 실패하면 CPU 로 설정됩니다."
        else
            finish driver ok "드라이버 ${ver:-확인됨} 가 동작합니다"
        fi
        return
    fi
    if [[ $INSTALL_DRIVER -eq 0 ]]; then
        block_step driver "NVIDIA GPU 는 있지만 드라이버가 동작하지 않습니다. 드라이버를 먼저 설치해야 GPU 모드를 쓸 수 있습니다." \
            "'NVIDIA 드라이버도 설치' 옵션(--install-driver)으로 다시 실행하거나, 직접 설치하고 재부팅한 뒤 다시 실행하세요. GPU 없이 쓰려면 CPU 모드(--arch cpu)로 실행하세요."
    fi
    local best
    best="$(apt-cache search --names-only '^nvidia-driver-[0-9]+$' 2>/dev/null | awk '{print $1}' | grep -oE '[0-9]+$' | awk '$1 >= 560' | sort -nr | head -n1)"
    local pkg=""
    if [[ -n "$best" ]]; then pkg="nvidia-driver-$best"; elif apt-cache show nvidia-driver-560-server >/dev/null 2>&1; then pkg="nvidia-driver-560-server"; fi
    [[ -n "$pkg" ]] || fail_step driver "이 Ubuntu 에서 560 이상 NVIDIA 드라이버 패키지를 찾지 못했습니다. 직접 설치하세요."
    if [[ $DRY -eq 1 ]]; then finish driver plan "설치 예정: $pkg (끝나면 재부팅 필요)"; REBOOT=1; return; fi
    runv driver apt-get install -y "$pkg" --no-install-recommends -o Dpkg::Options::=--force-overwrite || fail_step driver "$pkg 설치에 실패했습니다."
    REBOOT=1
    finish driver ok "$pkg 설치함 (적용하려면 재부팅이 필요합니다)"
}

# ============================================================================= 5. Docker
docker_ok() { have docker && docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; }
restart_docker() {
    if [[ $IS_WSL2 -eq 1 ]] || ! have systemctl; then runv "$1" service docker restart; else runv "$1" systemctl enable docker && runv "$1" systemctl restart docker; fi
}
step_docker() {
    begin docker
    if docker_ok; then finish docker skipped "이미 설치되고 실행 중 ($(docker --version 2>/dev/null | cut -c1-40))"; return; fi
    if [[ $DRY -eq 1 ]]; then finish docker plan "설치 예정: docker.io docker-buildx docker-compose-v2 (Ubuntu 패키지, DRfC 와 같은 방식)"; return; fi
    if ! have docker; then
        runv docker apt-get install -y --no-install-recommends docker.io docker-buildx docker-compose-v2 || fail_step docker "Docker 설치에 실패했습니다."
    fi
    restart_docker docker || fail_step docker "Docker 서비스를 시작하지 못했습니다."
    docker_ok || fail_step docker "Docker 를 설치했지만 docker info / docker compose 가 동작하지 않습니다."
    finish docker ok "설치하고 시작함"
}

# ============================================================================= 6. NVIDIA 컨테이너 도구
step_toolkit() {
    begin toolkit
    if [[ "$ARCH" != "gpu" ]]; then finish toolkit skipped "CPU 모드라 필요 없음"; return; fi
    # WSL2 에서 Docker Desktop 을 쓰면 WSL 안에는 Docker 서버가 없다 (docker 명령이 Windows 쪽 서버에 연결됨). 이 경우 daemon.json 을 고쳐도 소용없다.
    if [[ $IS_WSL2 -eq 1 && ! -x /etc/init.d/docker ]]; then
        finish toolkit skipped "WSL2 안에 Docker 서버가 없습니다 (Docker Desktop 사용 중으로 보임). NVIDIA 설정은 Windows 의 Docker Desktop 이 담당하므로 건드리지 않습니다."
        return
    fi
    local rt=""; have docker && rt="$(docker info --format '{{json .Runtimes}}' 2>/dev/null)"
    local def=""; [[ -f /etc/docker/daemon.json ]] && have jq && def="$(jq -r '."default-runtime" // empty' /etc/docker/daemon.json 2>/dev/null)"
    if [[ "$rt" == *nvidia* && "$def" == "nvidia" ]]; then finish toolkit skipped "nvidia 런타임이 이미 기본으로 설정돼 있음"; return; fi
    if [[ $DRY -eq 1 ]]; then finish toolkit plan "설치 예정: nvidia-docker2, nvidia-container-runtime 와 /etc/docker/daemon.json 의 default-runtime 을 nvidia 로"; return; fi
    local key=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    if [[ "$rt" != *nvidia* ]]; then
        { curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o "$key" \
          && curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
             | sed "s#deb https://#deb [signed-by=$key] https://#g" > /etc/apt/sources.list.d/nvidia-container-toolkit.list; } 2>&1 | tee -a "$WORK/toolkit.out" \
            || fail_step toolkit "NVIDIA 패키지 저장소를 추가하지 못했습니다 (nvidia.github.io 에 접속할 수 있는지 확인하세요)."
        [[ "${PIPESTATUS[0]}" -eq 0 ]] || fail_step toolkit "NVIDIA 패키지 저장소를 추가하지 못했습니다."
        runv toolkit apt-get update || true
        if ! runv toolkit apt-get install -y --no-install-recommends nvidia-docker2 nvidia-container-runtime; then
            runv toolkit apt-get install -y --no-install-recommends nvidia-container-toolkit || fail_step toolkit "NVIDIA 컨테이너 도구 설치에 실패했습니다."
        fi
    fi
    # /etc/docker/daemon.json 에 default-runtime=nvidia (DRfC 의 방식). 바꾸기 전에 백업.
    local f=/etc/docker/daemon.json tmp="$WORK/daemon.json"
    if [[ -f "$f" ]]; then
        cp "$f" "$f.bak-$(date +%m%d-%H%M%S)"
        jq '.runtimes.nvidia //= {"path":"nvidia-container-runtime","runtimeArgs":[]} | ."default-runtime" = "nvidia"' "$f" > "$tmp" \
            || fail_step toolkit "기존 /etc/docker/daemon.json 을 읽지 못했습니다 (JSON 형식 확인)."
    else
        printf '{\n    "runtimes": {\n        "nvidia": {\n            "path": "nvidia-container-runtime",\n            "runtimeArgs": []\n        }\n    },\n    "default-runtime": "nvidia"\n}\n' > "$tmp"
    fi
    jq -e . "$tmp" >/dev/null 2>&1 || fail_step toolkit "생성한 daemon.json 이 올바른 JSON 이 아닙니다."
    mkdir -p /etc/docker && install -m 0644 "$tmp" "$f" || fail_step toolkit "/etc/docker/daemon.json 을 쓰지 못했습니다."
    jq -e '."default-runtime" == "nvidia"' "$f" >/dev/null 2>&1 || fail_step toolkit "daemon.json 에 default-runtime 이 반영되지 않았습니다."
    say "  (Docker 를 다시 시작합니다. 실행 중인 컨테이너가 잠시 멈춥니다. restart 정책이 있는 컨테이너(drtrainer)는 다시 시작됩니다.)"
    restart_docker toolkit || fail_step toolkit "Docker 를 다시 시작하지 못했습니다."
    finish toolkit ok "nvidia 런타임을 설치하고 기본으로 설정함 (이전 daemon.json 은 백업)"
}

# ============================================================================= 7. docker 그룹
step_group() {
    begin group
    if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then finish group skipped "$TARGET_USER 는 이미 docker 그룹에 있음"; return; fi
    if [[ $DRY -eq 1 ]]; then finish group plan "$TARGET_USER 를 docker 그룹에 추가 예정"; return; fi
    runv group usermod -aG docker "$TARGET_USER" || fail_step group "사용자를 docker 그룹에 추가하지 못했습니다."
    RELOGIN=1
    finish group ok "$TARGET_USER 를 docker 그룹에 추가함 (새 터미널/로그인부터 적용되며, 이 설치는 계속 진행됩니다)"
}

# ============================================================================= 8. /tmp/sagemaker
step_tmp() {
    begin tmp
    local conf=/etc/tmpfiles.d/deepracer-trainer.conf
    if [[ -d /tmp/sagemaker ]] && { [[ -f "$conf" ]] || [[ $IS_WSL2 -eq 1 ]]; }; then finish tmp skipped "이미 있음"; return; fi
    if [[ $DRY -eq 1 ]]; then finish tmp plan "/tmp/sagemaker 를 만들고 부팅 때마다 다시 만들도록 설정 예정"; return; fi
    mkdir -p /tmp/sagemaker && chmod -R g+w /tmp/sagemaker || fail_step tmp "/tmp/sagemaker 를 만들지 못했습니다."
    # DRfC 는 이 폴더가 없으면 sudo 로 만드는데, 대시보드에서는 비밀번호를 입력할 수 없어 학습 시작이 멈춘다. 부팅마다 미리 만들어 둔다.
    if [[ $IS_WSL2 -eq 0 && -d /etc/tmpfiles.d ]]; then
        printf '# DeepRacer Trainer: DRfC 가 쓰는 폴더 (재부팅으로 /tmp 가 비워져도 다시 만든다)\nd /tmp/sagemaker 0775 root root -\n' > "$conf"
    fi
    finish tmp ok "만들고 부팅 때마다 다시 만들도록 설정함"
}

# ============================================================================= 9. DRfC 내려받기
is_drfc() { [[ -f "$1/bin/activate.sh" && -d "$1/scripts" ]]; }
step_clone() {
    begin clone
    if is_drfc "$DRFC_DIR"; then finish clone skipped "이미 있음: $DRFC_DIR"; return; fi
    if [[ -e "$DRFC_DIR" && -n "$(ls -A "$DRFC_DIR" 2>/dev/null)" ]]; then fail_step clone "$DRFC_DIR 가 비어 있지 않은데 DRfC 폴더가 아닙니다. 다른 위치(--drfc-dir)를 지정하세요."; fi
    if [[ "$DRFC_DIR" == *" "* ]]; then fail_step clone "DRfC 는 경로에 공백이 있으면 설치할 수 없습니다: $DRFC_DIR"; fi
    if [[ $DRY -eq 1 ]]; then finish clone plan "git clone → $DRFC_DIR"; return; fi
    as_user mkdir -p "$(dirname "$DRFC_DIR")"
    runv clone as_user git clone --depth 1 "$REPO_URL" "$DRFC_DIR" || fail_step clone "DRfC 를 내려받지 못했습니다."
    is_drfc "$DRFC_DIR" || fail_step clone "내려받은 폴더가 DRfC 구조가 아닙니다."
    finish clone ok "내려받음: $DRFC_DIR"
}

# ============================================================================= 10. 파이썬 환경
venv_ok() { [[ -x "$DRFC_DIR/.venv/bin/python" ]] && as_user "$DRFC_DIR/.venv/bin/python" -c 'import boto3, yaml, requests' >/dev/null 2>&1; }
step_venv() {
    begin venv
    if [[ -d "$DRFC_DIR" ]] && venv_ok; then finish venv skipped "이미 만들어져 있음"; return; fi
    if [[ $DRY -eq 1 ]]; then finish venv plan "$DRFC_DIR/.venv 를 만들고 requirements.txt 설치 예정 (DRfC 의 prepare.sh 가 하는 일)"; return; fi
    [[ -f "$DRFC_DIR/requirements.txt" ]] || fail_step venv "$DRFC_DIR/requirements.txt 가 없습니다."
    [[ -x "$DRFC_DIR/.venv/bin/python" ]] || runv venv as_user python3 -m venv --prompt drfc "$DRFC_DIR/.venv" || fail_step venv "가상환경을 만들지 못했습니다 (python3-venv 설치 확인)."
    runv venv as_user "$DRFC_DIR/.venv/bin/pip" install --quiet -r "$DRFC_DIR/requirements.txt" || fail_step venv "requirements.txt 설치에 실패했습니다."
    finish venv ok "가상환경을 만들고 의존성을 설치함"
}

# ============================================================================= 11. init.sh
step_init() {
    begin init
    # init.sh 가 중간에 실패하면(예: swarm 생성 실패) system.env 에 <DOCKER_STYLE> 같은 자리표시자가 남는다. 값이 하나도 채워지지 않은 템플릿이라 잃을 설정이 없으므로
    # 백업하고 처음부터 다시 한다. (파일이 있다고 초기화된 것이 아니다.)
    if [[ -f "$DRFC_DIR/system.env" ]] && grep -Eq "^[A-Za-z0-9_]+=[\"']?<[A-Z][A-Z0-9_]*>" "$DRFC_DIR/system.env"; then
        say "  system.env 에 채워지지 않은 값(<DOCKER_STYLE> 같은 자리표시자)이 있습니다. init.sh 가 중간에 실패한 흔적입니다."
        if [[ $DRY -eq 1 ]]; then finish init plan "system.env 를 백업하고 init.sh 를 처음부터 다시 실행 예정"; return; fi
        local ts; ts="$(date +%m%d-%H%M%S)"
        as_user mv "$DRFC_DIR/system.env" "$DRFC_DIR/system.env.bak-$ts"
        [[ -f "$DRFC_DIR/run.env" ]] && as_user mv "$DRFC_DIR/run.env" "$DRFC_DIR/run.env.bak-$ts"
        say "  이전 파일은 system.env.bak-$ts 로 백업했습니다. 처음부터 다시 초기화합니다."
    fi
    if [[ -f "$DRFC_DIR/system.env" ]]; then
        local cur; cur="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DRFC_DIR/system.env" | cut -d= -f2)"
        finish init skipped "이미 초기화됨 (이미지 ${cur:-?}). init.sh 는 설정을 덮어쓰므로 다시 실행하지 않습니다. GPU/CPU 전환은 대시보드 환경 점검에서 하세요."
        return
    fi
    if [[ $REBOOT -eq 1 ]]; then
        finish init blocked "NVIDIA 드라이버를 새로 설치해서 재부팅이 필요합니다. 재부팅 후 같은 명령을 다시 실행하면 이어서 진행됩니다."
        return
    fi
    if [[ $DRY -eq 1 ]]; then finish init plan "./bin/init.sh -c local -a $ARCH 실행 예정 (도커 이미지 내려받기로 오래 걸립니다)"; return; fi
    if ! as_user_docker docker info >/dev/null 2>&1; then
        block_step init "$TARGET_USER 가 아직 docker 를 쓸 수 없습니다 (그룹 변경이 적용되지 않았거나 Docker 가 꺼져 있음)." \
            "로그아웃 후 다시 로그인(또는 재부팅)한 뒤 같은 명령을 다시 실행하세요."
    fi
    if as_user_docker docker node ls >/dev/null 2>&1; then
        fail_step init "이미 Docker swarm 이 있어서 init.sh 가 중단됩니다. 이 swarm 을 쓰지 않는다면 'docker swarm leave --force' 후 다시 실행하세요."
    fi
    # init.sh 는 내부에서 'sudo mkdir /tmp/sagemaker' 를 부르는데, 이미 위에서 만들었으므로 비밀번호를 묻지 않게 그 두 명령만 건너뛴다.
    local real_sudo shim="$WORK/shim"
    real_sudo="$(command -v sudo || true)"; mkdir -p "$shim"
    cat > "$shim/sudo" <<SHIM
#!/bin/bash
if [[ "\$*" == "mkdir -p /tmp/sagemaker" || "\$*" == "chmod -R g+w /tmp/sagemaker" ]]; then exit 0; fi
if [[ -n "$real_sudo" ]]; then exec "$real_sudo" "\$@"; fi
echo "sudo 를 찾을 수 없습니다" >&2; exit 127
SHIM
    chmod 755 "$shim/sudo"; chmod 755 "$WORK" "$shim"
    say "  (도커 이미지를 내려받는 단계라 몇 분 걸립니다)"
    runv init as_user_docker env PATH="$shim:$PATH" bash -c "cd '$DRFC_DIR' && ./bin/init.sh -c local -a $ARCH" || fail_step init "init.sh 가 실패했습니다."
    [[ -f "$DRFC_DIR/system.env" ]] || fail_step init "init.sh 를 마쳤지만 system.env 가 만들어지지 않았습니다."
    # sg docker 로 실행한 탓에 새 파일의 그룹이 docker 로 남는다. 자격 증명 파일 등이 docker 그룹에 열리지 않게 사용자 기본 그룹으로 되돌린다.
    local grp; grp="$(id -gn "$TARGET_USER")"
    find "$DRFC_DIR" "$TARGET_HOME/.aws" -group docker -not -path '*/.venv/*' -exec chgrp -h "$grp" {} + 2>/dev/null || true
    local tag; tag="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DRFC_DIR/system.env" | sed -E 's/.*-(gpu|cpu)$/\1/')"
    if [[ "$ARCH" == "gpu" && "$tag" != "gpu" && $IS_WSL2 -eq 1 ]]; then
        # DRfC 문서(docs/windows.md): WSL2 에서는 init.sh 가 GPU 를 감지하지 못하니 system.env 의 이미지를 직접 GPU 로 설정하라고 한다.
        local base src
        base="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DRFC_DIR/system.env" | cut -d= -f2 | sed -E 's/-cpu$//')"
        src="$(grep -m1 '^DR_SIMAPP_SOURCE=' "$DRFC_DIR/system.env" | cut -d= -f2)"
        as_user sed -i -E "s/^(DR_SIMAPP_VERSION=.*)-cpu\$/\1-gpu/" "$DRFC_DIR/system.env"
        runv init as_user_docker docker pull "${src:-awsdeepracercommunity/deepracer-simapp}:${base}-gpu" || say "  (GPU 이미지 내려받기에 실패했습니다. 대시보드의 GPU/CPU 전환에서 다시 받을 수 있습니다.)"
        finish init warn "WSL2 에서는 init.sh 가 GPU 를 감지하지 못하는 알려진 문제가 있어(DRfC 문서), 문서대로 GPU 이미지로 설정했습니다. 실제 동작은 대시보드의 'GPU 컨테이너 시험'으로 확인하세요."
    elif [[ "$ARCH" == "gpu" && "$tag" != "gpu" ]]; then
        finish init warn "GPU 모드를 요청했지만 init.sh 의 GPU 컨테이너 시험에 실패해 CPU 모드로 설정됐습니다. 드라이버와 NVIDIA 컨테이너 도구를 확인한 뒤 대시보드에서 GPU 로 전환하세요."
    else
        finish init ok "초기화함 (${tag^^} 모드)"
    fi
}

# ============================================================================= 11-2. minio 이미지
# MinIO 가 2026-09 에 Docker Hub 의 minio/minio 이미지를 삭제했다. DRfC 가 시작할 때 이 이미지가 필요하므로, 없으면 검증된 바이너리로 만든다.
step_minio() {
    begin minio
    local script="$SCRIPT_DIR/../docker/minio-local-image.sh"
    if [[ ! -f "$DRFC_DIR/system.env" ]]; then finish minio skipped "DRfC 초기화 뒤에 확인합니다"; return; fi
    if [[ ! -f "$script" ]]; then finish minio warn "docker/minio-local-image.sh 가 없습니다 (압축 파일 전체를 쓰세요). 대시보드 환경 점검에서도 만들 수 있습니다."; return; fi
    local tag; tag="$(grep -m1 '^DR_MINIO_IMAGE=' "$DRFC_DIR/system.env" | cut -d= -f2-)"; tag="${tag:-RELEASE.2022-10-24T18-35-07Z}"
    if as_user_docker docker image inspect "minio/minio:$tag" >/dev/null 2>&1; then finish minio skipped "minio/minio:$tag 가 있음"; return; fi
    if [[ $DRY -eq 1 ]]; then finish minio plan "minio/minio:$tag 가 없으면(Docker Hub 에서 삭제됨) GitHub 릴리스의 검증된 바이너리로 이미지를 만들 예정"; return; fi
    runv minio as_user_docker bash "$script" "$DRFC_DIR" || fail_step minio "minio 이미지를 만들지 못했습니다."
    finish minio ok "준비함"
}

# ============================================================================= 12. 최종 확인
step_verify() {
    begin verify
    if [[ $DRY -eq 1 ]]; then finish verify plan "설치 후 Docker 와 DRfC 설정을 확인 예정"; return; fi
    if [[ $REBOOT -eq 1 && ! -f "$DRFC_DIR/system.env" ]]; then finish verify skipped "재부팅 후 다시 실행하면 확인합니다"; return; fi
    local problems=""
    as_user_docker docker info >/dev/null 2>&1 || problems+=" docker 사용 권한(로그인 후 적용)"
    [[ -f "$DRFC_DIR/system.env" ]] || problems+=" system.env"
    [[ -d "$DRFC_DIR/.venv" ]] || problems+=" .venv"
    if [[ -n "$problems" ]]; then finish verify warn "확인 필요:$problems"; else finish verify ok "Docker 와 DRfC 설정이 모두 확인됨"; fi
}

# ============================================================================= 실행
say "DeepRacer Trainer 환경 자동 설치$([[ $DRY -eq 1 ]] && echo ' (미리보기: 아무것도 바꾸지 않습니다)')"
step_preflight; step_apt; step_awscli; step_driver; step_docker; step_toolkit; step_group; step_tmp; step_clone; step_venv; step_init; step_minio; step_verify

if [[ $DRY -eq 1 ]]; then
    NEXT="미리보기입니다. 실제로 설치하려면 sudo 로 실행하세요."
elif [[ $REBOOT -eq 1 ]]; then
    NEXT="재부팅하세요. 재부팅 후 같은 명령을 다시 실행하면 남은 단계(DRfC 초기화)를 이어서 진행합니다."
else
    NEXT="끝났습니다. 대시보드(python dashboard.py)의 DeepRacer > 환경 점검에서 '점검 다시 실행'을 누르세요."
    [[ $RELOGIN -eq 1 ]] && NEXT="끝났습니다. 새 터미널을 열거나 로그아웃 후 다시 로그인해야 docker 권한이 적용됩니다. 그다음 대시보드 환경 점검에서 '점검 다시 실행'을 누르세요."
fi
say ""; say "== $NEXT"
[[ $REBOOT -eq 1 && $DRY -eq 0 ]] && exit 2
exit 0
