#!/usr/bin/env bash
# DRfC 초기화 (init.sh). 컨테이너 안에서, 호스트 사용자 권한으로 실행된다.
#   init-drfc.sh [gpu|cpu|auto]
# 이미 system.env 가 있으면 아무것도 하지 않는다. init.sh 는 system.env, run.env, custom_files 를 덮어쓰기 때문이다.
# init.sh 가 중간에 실패하면(예: swarm 생성 실패) 이 실행이 만든 system.env, run.env 를 치워서 다음에 처음부터 다시 할 수 있게 한다.
#   --force : 이미 system.env 가 있어도 백업(system.env.bak-*)하고 다시 초기화 (초기화가 중간에 멈춘 경우)
set -uo pipefail
DRFC_DIR="${DRFC_DIR:?DRFC_DIR 가 필요합니다}"
FORCE=0; ARCH_ARG=""
for a in "$@"; do case "$a" in --force) FORCE=1 ;; *) ARCH_ARG="$a" ;; esac; done
ARCH="${ARCH_ARG:-${DRFC_ARCH:-auto}}"
TS="$(date +%m%d-%H%M%S)"

# init.sh 는 system.env 를 템플릿에서 복사한 뒤 <DOCKER_STYLE> 같은 자리표시자를 단계마다 채운다(마지막 치환은 swarm 생성 뒤). 자리표시자가 남아 있으면
# init.sh 가 중간에 실패한 것이고, 이 파일에는 사용자가 넣은 값이 하나도 없으므로(채워지지 않은 템플릿) 백업하고 처음부터 자동으로 다시 한다.
if [[ -f "$DRFC_DIR/system.env" && $FORCE -eq 0 ]] && grep -Eq "^[A-Za-z0-9_]+=[\"']?<[A-Z][A-Z0-9_]*>" "$DRFC_DIR/system.env"; then
    echo "[init] system.env 에 채워지지 않은 값(<DOCKER_STYLE> 같은 자리표시자)이 있습니다. init.sh 가 중간에 실패한 흔적이라서, 백업하고 처음부터 다시 초기화합니다."
    FORCE=1
fi

if [[ -f "$DRFC_DIR/system.env" ]]; then
    if [[ $FORCE -eq 0 ]]; then
        echo "[init] 이미 초기화돼 있습니다 (system.env 를 그대로 둡니다)."
        style="$(grep -m1 '^DR_DOCKER_STYLE=' "$DRFC_DIR/system.env" | cut -d= -f2 | tr -d '"')"
        if [[ "${style:-swarm}" == "swarm" ]] && ! docker node ls >/dev/null 2>&1; then
            echo "[init] 주의: DRfC 가 swarm 모드를 쓰는데 이 Docker 는 swarm 이 아닙니다. 초기화가 중간에 멈췄을 수 있습니다. 설정을 백업하고 다시 하려면: ./drtrainer init --force" >&2
        fi
        exit 0
    fi
    mv "$DRFC_DIR/system.env" "$DRFC_DIR/system.env.bak-$TS"
    [[ -f "$DRFC_DIR/run.env" ]] && mv "$DRFC_DIR/run.env" "$DRFC_DIR/run.env.bak-$TS"
    echo "[init] 기존 설정을 system.env.bak-$TS 로 백업하고 다시 초기화합니다."
fi

if [[ "$ARCH" == "auto" ]]; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then ARCH=gpu; else ARCH=cpu; fi
    echo "[init] 호스트 Docker 에서 GPU 런타임을 $([[ $ARCH == gpu ]] && echo 찾았습니다 || echo 찾지 못했습니다) -> ${ARCH^^} 모드"
fi
if docker node ls >/dev/null 2>&1; then
    echo "[init] 이미 Docker swarm 이 있어서 init.sh 가 중단됩니다. 이 swarm 을 쓰지 않는다면 호스트에서 'docker swarm leave --force' 후 다시 실행하세요." >&2
    exit 1
fi

cd "$DRFC_DIR" || exit 1
echo "[init] ./bin/init.sh -c local -a $ARCH  (도커 이미지를 내려받는 단계라 몇 분 걸립니다)"
undo() {   # 실패한 실행이 남긴 설정을 치우고, --force 였다면 백업을 되돌린다
    rm -f system.env run.env
    if [[ $FORCE -eq 1 && -f "system.env.bak-$TS" ]]; then mv "system.env.bak-$TS" system.env; [[ -f "run.env.bak-$TS" ]] && mv "run.env.bak-$TS" run.env; echo "[init] 이전 설정을 되돌렸습니다." >&2; fi
}
if ! ./bin/init.sh -c local -a "$ARCH"; then
    echo "[init] init.sh 가 실패했습니다. 반쯤 만들어진 설정(system.env, run.env)을 치웠으니 원인을 고친 뒤 다시 시도하면 처음부터 초기화합니다." >&2
    undo; exit 1
fi
[[ -f system.env ]] || { echo "[init] init.sh 를 마쳤지만 system.env 가 없습니다." >&2; exit 1; }

# DRfC 문서(docs/windows.md)의 알려진 문제: WSL2 에서는 init.sh 가 GPU 를 감지하지 못한다. 문서대로 GPU 이미지로 설정한다.
tag="$(grep -m1 '^DR_SIMAPP_VERSION=' system.env | sed -E 's/.*-(gpu|cpu)$/\1/')"
if [[ "$ARCH" == "gpu" && "$tag" == "cpu" ]] && grep -qi microsoft /proc/version 2>/dev/null && grep -q WSL2 /proc/version 2>/dev/null; then
    base="$(grep -m1 '^DR_SIMAPP_VERSION=' system.env | cut -d= -f2 | sed -E 's/-cpu$//')"
    src="$(grep -m1 '^DR_SIMAPP_SOURCE=' system.env | cut -d= -f2)"
    sed -i -E 's/^(DR_SIMAPP_VERSION=.*)-cpu$/\1-gpu/' system.env
    docker pull "${src:-awsdeepracercommunity/deepracer-simapp}:${base}-gpu" || echo "[init] GPU 이미지 내려받기 실패: 대시보드 GPU/CPU 전환에서 다시 받을 수 있습니다."
    echo "[init] WSL2: init.sh 가 GPU 를 감지하지 못하는 알려진 문제라 GPU 이미지로 설정했습니다. 실제 동작은 대시보드의 'GPU 컨테이너 시험'으로 확인하세요."
elif [[ "$ARCH" == "gpu" && "$tag" == "cpu" ]]; then
    echo "[init] GPU 모드를 요청했지만 init.sh 의 GPU 시험에 실패해 CPU 모드로 설정됐습니다. 호스트의 NVIDIA 드라이버와 NVIDIA Container Toolkit 을 확인한 뒤 대시보드에서 GPU 로 전환하세요."
fi
echo "[init] 완료 (${tag^^} 모드)"
