#!/usr/bin/env bash
# GPU 진단: 컨테이너가 GPU 를 쓸 수 있는지 단계별로 확인하고, 막혔으면 어디가 문제인지와 해야 할 일을 알려 준다.
#
#   bash scripts/gpu_check.sh [DRfC 폴더]        (sudo 필요 없음. 아무것도 바꾸지 않는다)
#
# DeepRacer 학습에서 GPU 를 쓰려면 세 가지가 모두 필요하다. (Docker 가 'GPU 전용'일 필요는 없다.)
#   1) 호스트에 NVIDIA 드라이버 (WSL2 에서는 Windows 쪽 드라이버를 WSL 이 빌려 쓴다)
#   2) Docker 가 GPU 를 컨테이너에 연결할 수 있게 하는 NVIDIA Container Toolkit 과 nvidia 런타임 설정
#   3) DRfC 가 GPU 용 시뮬레이터 이미지(-gpu)를 쓰도록 설정 (대시보드 환경 점검의 GPU/CPU 전환)
# 종료 코드: 0 GPU 사용 가능(모드까지 정상), 1 문제 있음, 2 GPU 는 되지만 DRfC 가 CPU 모드
set -uo pipefail
DIR="${1:-${DRFC_DIR:-$HOME/deepracer-for-cloud}}"
export PATH="$PATH:/usr/lib/wsl/lib"
ok()   { echo "  [ 확인 ] $*"; }
bad()  { echo "  [ 문제 ] $*"; }
info() { echo "  [ 참고 ] $*"; }
VERDICT=""; NEXT=(); RC=1

echo "== GPU 진단"
IS_WSL=0; if grep -qi microsoft "${GPU_CHECK_PROC_VERSION:-/proc/version}" 2>/dev/null; then IS_WSL=1; fi   # (GPU_CHECK_PROC_VERSION 은 시험용)
echo; echo "1) 실행 환경"
if [[ $IS_WSL -eq 1 ]]; then ok "WSL2 (Windows 위의 Linux). GPU 드라이버는 Windows 쪽 것을 씁니다."; else ok "Linux"; fi

echo; echo "2) 호스트의 NVIDIA 드라이버 (nvidia-smi)"
SMI=""; DRIVER_OK=0
if command -v nvidia-smi >/dev/null 2>&1; then SMI="$(command -v nvidia-smi)"; fi
if [[ -n "$SMI" ]] && out="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"; then
    DRIVER_OK=1; ok "$(echo "$out" | head -1)"
    ver="$(echo "$out" | head -1 | awk -F', ' '{print $2}')"; major="${ver%%.*}"
    if [[ "$major" =~ ^[0-9]+$ && "$major" -lt 560 ]]; then info "드라이버 $ver: DRfC 의 GPU 시험 이미지(CUDA 12.6)에는 560 이상이 권장됩니다. 시험에 실패하면 드라이버를 올리세요."; fi
else
    bad "nvidia-smi 가 없거나 실행되지 않습니다. ${out:-}"
    [[ $IS_WSL -eq 1 ]] && info "WSL2 에서는 WSL 안에 드라이버를 설치하지 않습니다. Windows 에 WSL 지원 NVIDIA 드라이버를 설치하세요."
fi

echo; echo "3) Docker"
DOCKER_OK=0; RUNTIMES=""; DEFRT=""
if ! command -v docker >/dev/null 2>&1; then bad "docker 명령이 없습니다."
elif ! docker info >/dev/null 2>&1; then bad "Docker 서버에 연결할 수 없습니다 (꺼져 있거나 권한이 없음: sudo service docker start / sudo usermod -aG docker \$USER)."
else
    DOCKER_OK=1; ok "Docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
    RUNTIMES="$(docker info --format '{{json .Runtimes}}' 2>/dev/null)"; DEFRT="$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null)"
    if [[ "$RUNTIMES" == *nvidia* ]]; then ok "nvidia 런타임이 등록돼 있음 (기본 런타임: ${DEFRT:-?})"; else bad "Docker 에 nvidia 런타임이 등록돼 있지 않음 (등록된 런타임: $(echo "$RUNTIMES" | tr -d '{}"' | sed 's/:[^,]*//g'))"; fi
    if [[ "$RUNTIMES" == *nvidia* && "$DEFRT" != "nvidia" ]]; then info "기본 런타임이 nvidia 가 아닙니다. DRfC(swarm)는 기본 런타임이 nvidia 여야 GPU 를 씁니다 (/etc/docker/daemon.json 의 default-runtime)."; fi
fi

echo; echo "4) NVIDIA Container Toolkit (호스트 패키지)"
PKGS="$(dpkg -l 2>/dev/null | awk '$1 == "ii" && /nvidia-(container-toolkit|docker2|container-runtime)/ {print $2}' | tr '\n' ' ')"
if [[ -n "$PKGS" ]] || command -v nvidia-ctk >/dev/null 2>&1 || command -v nvidia-container-runtime >/dev/null 2>&1; then ok "설치됨: ${PKGS:-nvidia-ctk 또는 nvidia-container-runtime}"; else bad "설치돼 있지 않습니다 (nvidia-container-toolkit, nvidia-docker2, nvidia-container-runtime 없음)."; fi

echo; echo "5) GPU 컨테이너 시험 (docker run --gpus all ... nvidia-smi -L)"
CONTAINER_OK=0; ERR=""
IMG="$(grep -ho 'nvcr.io/nvidia/cuda:[A-Za-z0-9._-]*' "$DIR/bin/init.sh" 2>/dev/null | head -1)"; IMG="${IMG:-nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04}"
if [[ $DOCKER_OK -eq 1 ]]; then
    echo "  (이미지 $IMG 를 처음 받으면 몇 분 걸립니다)"
    if out="$(timeout 600 docker run --rm --gpus all "$IMG" nvidia-smi -L 2>&1)"; then CONTAINER_OK=1; ok "컨테이너 안에서 GPU 가 보입니다: $(echo "$out" | head -1)"
    else ERR="$out"; bad "컨테이너에서 GPU 를 쓸 수 없습니다."; echo "$out" | tail -4 | sed 's/^/        /'; fi
else info "Docker 를 쓸 수 없어서 건너뜁니다."; fi

echo; echo "6) DRfC 가 쓰는 모드 ($DIR)"
TAG=""
if [[ -f "$DIR/system.env" ]]; then
    ver="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DIR/system.env" | cut -d= -f2)"; TAG="${ver##*-}"
    if [[ "$TAG" == "gpu" ]]; then ok "GPU 이미지를 쓰도록 설정됨 ($ver)"; else info "CPU 이미지를 쓰도록 설정됨 ($ver)"; fi
else info "DRfC 가 아직 초기화되지 않았거나 폴더가 다릅니다 (인자로 DRfC 폴더를 주세요)."; fi

# ------------------------------------------------------------------ 판정
echo; echo "== 결론"
if [[ $DRIVER_OK -eq 0 ]]; then
    VERDICT="호스트에서 NVIDIA 드라이버가 보이지 않습니다. 컨테이너 문제가 아니라 그 앞단입니다."
    if [[ $IS_WSL -eq 1 ]]; then
        NEXT=("Windows 에 최신 NVIDIA 드라이버(WSL 지원)를 설치하거나 업데이트하세요. WSL 안에는 드라이버를 설치하지 않습니다." "Windows PowerShell 에서 'wsl --shutdown' 을 실행한 뒤 WSL 을 다시 열고, WSL 안에서 'nvidia-smi' 가 GPU 를 보여주는지 확인하세요." "그래도 안 보이면 'wsl --update' 를 실행하세요.")
    else
        NEXT=("GPU 가 NVIDIA 인지 확인하세요 (lspci | grep -i nvidia)." "드라이버 설치: sudo bash scripts/setup_drfc.sh --arch gpu --install-driver  (설치 후 재부팅이 필요합니다)")
    fi
elif [[ $DOCKER_OK -eq 0 ]]; then
    VERDICT="드라이버는 정상인데 Docker 를 쓸 수 없습니다."
    NEXT=("Docker 서버를 시작하세요: sudo service docker start (또는 sudo systemctl start docker)")
elif [[ $CONTAINER_OK -eq 0 ]]; then
    if grep -qiE 'could not select device driver|unknown or invalid runtime name|no such file or directory.*nvidia' <<<"$ERR" || [[ "$RUNTIMES" != *nvidia* ]]; then
        VERDICT="드라이버는 정상이지만 Docker 가 GPU 를 컨테이너에 연결하지 못합니다. NVIDIA Container Toolkit 이 없거나 Docker 설정에 반영되지 않았습니다. 처음부터 다시 설치할 필요는 없습니다."
        NEXT=("sudo bash scripts/setup_drfc.sh --arch gpu    (이미 된 단계는 건너뛰고 NVIDIA 도구 설치와 Docker 설정만 합니다. Docker 가 다시 시작되어 실행 중인 컨테이너가 잠시 멈춥니다)" "끝나면 이 진단을 다시 실행하세요: bash scripts/gpu_check.sh")
    elif grep -qiE 'adapters were found|nvml|libnvidia-ml|initialization error|driver' <<<"$ERR"; then
        VERDICT="Docker 의 GPU 설정은 있는데 컨테이너가 드라이버를 초기화하지 못합니다."
        if [[ $IS_WSL -eq 1 ]]; then NEXT=("Windows PowerShell 에서 'wsl --shutdown' 후 다시 열고 'nvidia-smi' 를 확인하세요." "Windows NVIDIA 드라이버를 최신(WSL 지원)으로 업데이트하세요."); else NEXT=("드라이버를 재설치하거나 재부팅하세요. 'nvidia-smi' 가 정상인지 먼저 확인하세요."); fi
    elif grep -qiE 'pull access denied|unauthorized|no such host|timeout|TLS|connection refused|dial tcp' <<<"$ERR"; then
        VERDICT="GPU 시험용 이미지($IMG)를 받지 못했습니다. 네트워크 문제로 보입니다."
        NEXT=("인터넷 연결(nvcr.io 접속)을 확인하고 다시 실행하세요.")
    else
        VERDICT="GPU 컨테이너 시험이 실패했습니다. 위의 오류 메시지를 확인하세요."
        NEXT=("오류 메시지를 그대로 알려 주시면 원인을 찾을 수 있습니다.")
    fi
elif [[ -n "$TAG" && "$TAG" != "gpu" ]]; then
    VERDICT="GPU 는 정상입니다. 다만 DRfC 가 CPU 이미지로 설정돼 있어 GPU 를 쓰지 않고 있습니다. (모델과 실험은 그대로 둔 채 이미지만 바꾸면 됩니다.)"
    NEXT=("대시보드 DeepRacer > 환경 점검 > 'GPU / CPU' 에서 GPU 를 고르고 '이 모드로 전환' 을 누르세요. GPU 용 시뮬레이터 이미지를 내려받습니다 (수 GB)." "그다음 'GPU 컨테이너 시험' 을 눌러 확인하세요.")
    RC=2
else
    VERDICT="GPU 를 쓸 준비가 모두 돼 있습니다 (드라이버, Docker 연결, DRfC 모드)."
    NEXT=("학습이 느리게 느껴지면 'nvidia-smi' 로 학습 중 GPU 사용률을 확인하세요 (WSL 안에서 watch -n 1 nvidia-smi).")
    RC=0
fi
echo "  $VERDICT"
if [[ ${#NEXT[@]} -gt 0 ]]; then echo; echo "== 다음 단계"; n=1; for s in "${NEXT[@]}"; do echo "  $n. $s"; n=$((n + 1)); done; fi
exit $RC
