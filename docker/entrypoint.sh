#!/usr/bin/env bash
# DeepRacer Trainer 컨테이너 진입점 (root 로 시작해서 호스트 사용자 권한으로 내려간다).
#
# 핵심: DRfC 는 호스트의 Docker 로 '형제 컨테이너'(시뮬레이터, minio 등)를 띄운다. 그 컨테이너들의 bind mount 경로는 호스트 기준으로 해석되므로,
#       DRfC 폴더와 ~/.aws 는 호스트와 같은 절대경로로 이 컨테이너에 마운트돼 있어야 한다 (docker-compose.yml 이 그렇게 연결한다).
set -euo pipefail
log() { echo "[drtrainer] $*"; }
die() { echo "[drtrainer] 오류: $*" >&2; exit 1; }

APP_DIR="${APP_DIR:-/app}"; SEED="${DRFC_SEED:-/opt/drfc}"; SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
STATE="${TRAINER_STATE_DIR:-$APP_DIR/state}"
for v in HOST_UID HOST_GID HOST_USER HOST_HOME DRFC_DIR; do [[ -n "${!v:-}" ]] || die "$v 가 비어 있습니다. ./drtrainer up 으로 실행하세요 (.env 를 만들어 줍니다)."; done
[[ -S "$SOCK" ]] || die "Docker 소켓($SOCK)이 연결돼 있지 않습니다. docker-compose.yml 의 volumes 를 확인하세요."
[[ "$DRFC_DIR" == /* ]] || die "DRFC_DIR 는 절대경로여야 합니다: $DRFC_DIR"
[[ "$DRFC_DIR" != *" "* ]] || die "DRfC 는 경로에 공백이 있으면 쓸 수 없습니다: $DRFC_DIR"
export TRAINER_STATE_DIR="$STATE"

# 호스트 Docker 서버가 이 컨테이너의 docker 클라이언트보다 오래됐으면 API 버전을 서버에 맞춘다 (자식 프로세스 전체에 물려준다)
# shellcheck source=docker-api.sh
source "$APP_DIR/docker/docker-api.sh"
pin_docker_api "$SOCK"

# ---------------------------------------------------------------- 호스트와 같은 사용자 (파일 소유권, ~/.aws 심볼릭 링크가 호스트에서도 그대로 유효)
AS=()
RUN_USER=root
if [[ "$HOST_UID" != "0" ]]; then
    if ! getent group "$HOST_GID" >/dev/null; then
        groupadd -g "$HOST_GID" "$HOST_USER" 2>/dev/null || groupadd -g "$HOST_GID" "drt$HOST_GID"
    fi
    GROUP_NAME="$(getent group "$HOST_GID" | cut -d: -f1)"
    if ! getent passwd "$HOST_UID" >/dev/null; then
        useradd -M -u "$HOST_UID" -g "$GROUP_NAME" -d "$HOST_HOME" -s /bin/bash "$HOST_USER" 2>/dev/null \
            || useradd -M -u "$HOST_UID" -g "$GROUP_NAME" -d "$HOST_HOME" -s /bin/bash "drt$HOST_UID"
    fi
    RUN_USER="$(getent passwd "$HOST_UID" | cut -d: -f1)"
    [[ "$(getent passwd "$HOST_UID" | cut -d: -f6)" == "$HOST_HOME" ]] || usermod -d "$HOST_HOME" "$RUN_USER"
    mkdir -p "$HOST_HOME"; chown "$HOST_UID:$HOST_GID" "$HOST_HOME"

    # DRfC 스크립트가 sudo 를 쓴다 (/tmp/sagemaker 만들기, 학습 중지 때 compose 파일 수정과 docker compose stop).
    # Docker 소켓을 쓸 수 있으면 이미 호스트 root 와 같은 권한이므로 컨테이너 안의 sudo 가 늘리는 권한은 없다.
    # (DRfC 의 학습 중지가 'sudo docker compose ...' 를 부르므로, sudo 가 환경변수를 지워도 DOCKER_API_VERSION 은 유지한다)
    printf '%s ALL=(ALL) NOPASSWD:ALL\nDefaults env_keep += "DOCKER_API_VERSION"\n' "$RUN_USER" > /etc/sudoers.d/90-drtrainer; chmod 0440 /etc/sudoers.d/90-drtrainer

    # Docker 소켓 그룹
    SOCK_GID="$(stat -c %g "$SOCK")"
    getent group "$SOCK_GID" >/dev/null || groupadd -g "$SOCK_GID" dockersock
    usermod -aG "$(getent group "$SOCK_GID" | cut -d: -f1)" "$RUN_USER"
    AS=(gosu "$RUN_USER")
else
    log "경고: 호스트 사용자가 root 입니다. 파일이 root 소유로 만들어집니다. 일반 사용자로 실행하는 것을 권장합니다."
fi
RUNENV=(env "HOME=$HOST_HOME" "USER=$RUN_USER" "LOGNAME=$RUN_USER" "TRAINER_STATE_DIR=$STATE")

mkdir -p "$APP_DIR/runs" "$STATE"
chown "$HOST_UID:$HOST_GID" "$APP_DIR/runs" "$STATE" 2>/dev/null || true

# ---------------------------------------------------------------- DRfC 폴더: 처음이면 이미지에 들어 있는 (고정된 버전의) DRfC 를 복사
if [[ ! -f "$DRFC_DIR/bin/activate.sh" ]]; then
    if [[ -d "$DRFC_DIR" && -n "$(ls -A "$DRFC_DIR" 2>/dev/null)" ]]; then die "$DRFC_DIR 가 비어 있지 않은데 DRfC 폴더가 아닙니다. 다른 폴더(DRFC_DIR)를 지정하세요."; fi
    log "처음 실행: DRfC 를 $DRFC_DIR 에 준비합니다 (이미지에 고정된 버전)"
    mkdir -p "$DRFC_DIR"; cp -a "$SEED"/. "$DRFC_DIR"/
    chown -R "$HOST_UID:$HOST_GID" "$DRFC_DIR"
fi
[[ -d /tmp/sagemaker ]] && chmod -R g+w /tmp/sagemaker 2>/dev/null || true

# ---------------------------------------------------------------- DRfC 초기화 (system.env 가 없을 때만 실제로 한다). 실패해도 대시보드는 띄운다 (환경 점검 화면에서 원인을 볼 수 있게).
# 이미 초기화돼 있으면 곧바로 끝나지만, swarm 이 없는 등 초기화가 중간에 멈춘 흔적이 있으면 경고를 낸다.
if [[ ! -f "$DRFC_DIR/system.env" ]]; then
    log "처음 실행: DRfC 를 초기화합니다. 도커 이미지를 내려받느라 몇 분 걸립니다. 끝나면 대시보드가 열립니다."
fi
"${AS[@]}" "${RUNENV[@]}" "DRFC_DIR=$DRFC_DIR" "DRFC_ARCH=${DRFC_ARCH:-auto}" "$APP_DIR/docker/init-drfc.sh" \
    || log "초기화에 실패했습니다. 대시보드는 계속 실행합니다. 원인을 고친 뒤 './drtrainer init' 으로 다시 시도하세요."

# ---------------------------------------------------------------- minio 이미지: MinIO 가 Docker Hub 이미지를 삭제해서(2026-09), 없으면 검증된 바이너리로 직접 만든다
if [[ -f "$DRFC_DIR/system.env" ]]; then
    "${AS[@]}" "${RUNENV[@]}" "$APP_DIR/docker/minio-local-image.sh" "$DRFC_DIR" \
        || log "minio 이미지를 준비하지 못했습니다. 대시보드 환경 점검의 'minio 이미지 만들기' 로 다시 시도할 수 있습니다."
fi

# ---------------------------------------------------------------- 대시보드 (이 컴퓨터에서만 접속: 호스트 네트워크의 127.0.0.1)
cd "$APP_DIR"
log "대시보드 시작: http://127.0.0.1:${DASHBOARD_PORT:-8765}"
exec "${AS[@]}" "${RUNENV[@]}" python3 "$APP_DIR/dashboard.py" --host 127.0.0.1 --port "${DASHBOARD_PORT:-8765}" --strict-port --no-browser
