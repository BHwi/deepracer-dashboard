#!/usr/bin/env bash
# GPU 진단: 컨테이너가 GPU 를 쓸 수 있는지 단계별로 확인하고, 막혔으면 어디가 문제인지와 해야 할 일을 알려 준다.
#
#   bash scripts/gpu_check.sh [DRfC 폴더]        (sudo 필요 없음. 아무것도 바꾸지 않는다)
#   bash scripts/gpu_check.sh --container-only [DRfC 폴더]   (GPU 컨테이너 시험과 그 해석만. 대시보드의 'GPU 컨테이너 시험'이 쓴다)
#
# DeepRacer 학습에서 GPU 를 쓰려면 세 가지가 모두 필요하다. (Docker 가 'GPU 전용'일 필요는 없다.)
#   1) 호스트에 NVIDIA 드라이버 (WSL2 에서는 Windows 쪽 드라이버를 WSL 이 빌려 쓴다)
#   2) Docker 가 GPU 를 컨테이너에 연결할 수 있게 하는 NVIDIA Container Toolkit 과 nvidia 런타임 설정
#   3) DRfC 가 GPU 용 시뮬레이터 이미지(-gpu)를 쓰도록 설정 (대시보드 환경 점검의 GPU/CPU 전환)
# 종료 코드: 0 GPU 사용 가능(모드까지 정상), 1 문제 있음, 2 GPU 는 되지만 DRfC 가 CPU 모드,
#            3 GPU 는 컨테이너에 연결되지만 이미지가 요구하는 CUDA 버전을 드라이버가 지원하지 못함
set -uo pipefail
CONTAINER_ONLY=0; ARGS=()
for a in "$@"; do case "$a" in --container-only) CONTAINER_ONLY=1 ;; *) ARGS+=("$a") ;; esac; done
DIR="${ARGS[0]:-${DRFC_DIR:-$HOME/deepracer-for-cloud}}"
export PATH="$PATH:/usr/lib/wsl/lib"
ok()   { echo "  [ 확인 ] $*"; }
bad()  { echo "  [ 문제 ] $*"; }
info() { echo "  [ 참고 ] $*"; }
VERDICT=""; NEXT=(); RC=1
# 버전 비교: ver_ge A B  →  A >= B
ver_ge() { [[ -n "$1" && -n "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]; }
# 드라이버 버전 → 그 드라이버가 지원하는 최대 CUDA (Linux, CUDA Toolkit 릴리스 노트의 최소 드라이버 표 기준). 525 미만은 11.x 이므로 환산하지 않는다.
cuda_for_driver() {
    local m="${1%%.*}"; [[ "$m" =~ ^[0-9]+$ ]] || return 0
    if   [[ $m -ge 590 ]]; then echo "13.1"; elif [[ $m -ge 580 ]]; then echo "13.0"; elif [[ $m -ge 575 ]]; then echo "12.9"; elif [[ $m -ge 570 ]]; then echo "12.8"
    elif [[ $m -ge 565 ]]; then echo "12.7"; elif [[ $m -ge 560 ]]; then echo "12.6"; elif [[ $m -ge 555 ]]; then echo "12.5"; elif [[ $m -ge 550 ]]; then echo "12.4"
    elif [[ $m -ge 545 ]]; then echo "12.3"; elif [[ $m -ge 535 ]]; then echo "12.2"; elif [[ $m -ge 530 ]]; then echo "12.1"; elif [[ $m -ge 525 ]]; then echo "12.0"; fi
}
# nvidia-smi 출력에서 값 뽑기
smi_field() { sed -n "s/.*$1: *\([0-9][0-9.]*\).*/\1/p" | head -1; }

echo "== GPU 진단"
IS_WSL=0; if grep -qi microsoft "${GPU_CHECK_PROC_VERSION:-/proc/version}" 2>/dev/null; then IS_WSL=1; fi   # (GPU_CHECK_PROC_VERSION 은 시험용)
SMI=""; DRIVER_OK=0; HOST_CUDA=""; DOCKER_OK=0; RUNTIMES=""; DEFRT=""
if [[ $CONTAINER_ONLY -eq 1 ]]; then
    if ! command -v docker >/dev/null 2>&1; then bad "docker 명령이 없습니다."
    elif ! docker info >/dev/null 2>&1; then bad "Docker 서버에 연결할 수 없습니다 (꺼져 있거나 권한이 없음)."
    else DOCKER_OK=1; RUNTIMES="$(docker info --format '{{json .Runtimes}}' 2>/dev/null)"; DEFRT="$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null)"; fi
fi
if [[ $CONTAINER_ONLY -eq 0 ]]; then
echo; echo "1) 실행 환경"
if [[ $IS_WSL -eq 1 ]]; then ok "WSL2 (Windows 위의 Linux). GPU 드라이버는 Windows 쪽 것을 씁니다."; else ok "Linux"; fi

echo; echo "2) 호스트의 NVIDIA 드라이버 (nvidia-smi)"
if command -v nvidia-smi >/dev/null 2>&1; then SMI="$(command -v nvidia-smi)"; fi
if [[ -n "$SMI" ]] && out="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"; then
    DRIVER_OK=1; ok "$(echo "$out" | head -1)"
    ccap="$(echo "$out" | head -1 | awk -F', ' 'NF >= 3 {print $3}')"; [[ -n "$ccap" ]] && info "GPU 연산 능력(compute capability): $ccap"
    ver="$(echo "$out" | head -1 | awk -F', ' '{print $2}')"
    HOST_CUDA="$(nvidia-smi 2>/dev/null | smi_field 'CUDA Version')"
    [[ -n "$HOST_CUDA" ]] && info "이 드라이버($ver)가 지원하는 최대 CUDA: $HOST_CUDA"
else
    bad "nvidia-smi 가 없거나 실행되지 않습니다. ${out:-}"
    [[ $IS_WSL -eq 1 ]] && info "WSL2 에서는 WSL 안에 드라이버를 설치하지 않습니다. Windows 에 WSL 지원 NVIDIA 드라이버를 설치하세요."
fi

echo; echo "3) Docker"
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

fi   # (--container-only 가 아닐 때의 1~4단계 끝)

echo; echo "5) GPU 컨테이너 시험 (docker run --gpus all ... nvidia-smi -L)"
CONTAINER_OK=0; ERR=""; REQ=""; BYPASS_OK=0; DRV_VER=""; DRV_CUDA=""; CCAP=""; CONT_CUDA=""; LIBCUDA=""; GPU_LINE=""; CONSUMER=0
IMG="$(grep -ho 'nvcr.io/nvidia/cuda:[A-Za-z0-9._-]*' "$DIR/bin/init.sh" 2>/dev/null | head -1)"; IMG="${IMG:-nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04}"
if [[ $DOCKER_OK -eq 1 ]]; then
    echo "  (이미지 $IMG 를 처음 받으면 몇 분 걸립니다. 이 이미지는 DRfC 의 init.sh 가 GPU 를 감지할 때 쓰는 것과 같습니다)"
    if out="$(timeout 600 docker run --rm --gpus all --pull=missing "$IMG" nvidia-smi -L 2>&1)"; then
        CONTAINER_OK=1; ok "컨테이너 안에서 GPU 가 보입니다: $(echo "$out" | head -1)"
    else
        ERR="$out"; bad "컨테이너에서 GPU 를 쓸 수 없습니다."; echo "$out" | tail -4 | sed 's/^/        /'
        # 이미지가 요구하는 CUDA 버전을 드라이버가 지원하지 못하는 경우: GPU 연결 문제가 아니므로 요구 검사만 끄고 다시 시험해서 구분한다
        REQ="$(grep -io 'unsatisfied condition: cuda>=[0-9][0-9.]*' <<<"$ERR" | head -1 | sed 's/.*>=//')"
        if [[ -n "$REQ" ]]; then
            info "오류에 'cuda>=$REQ' 요구가 있습니다. GPU 연결 문제가 아니라, 이 이미지가 요구하는 CUDA 를 드라이버가 지원하지 못하는 경우입니다."
            info "요구 검사만 끄고(NVIDIA_DISABLE_REQUIRE=1) 다시 시험해서 GPU 가 컨테이너에 연결되는지 확인합니다."
            if out2="$(timeout 600 docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 "$IMG" bash -lc 'nvidia-smi -L; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null; nvidia-smi; echo ---LIBCUDA---; ldconfig -p 2>/dev/null | grep "libcuda.so.1"' 2>&1)"; then
                BYPASS_OK=1; CONT_CUDA="$(smi_field 'CUDA Version' <<<"$out2")"; LIBCUDA="$(sed -n '/---LIBCUDA---/,$p' <<<"$out2" | grep -m1 'libcuda.so.1' | sed 's/.*=> *//')"; GPU_LINE="$(grep -m1 '^GPU ' <<<"$out2")"; CCAP="$(grep -m1 -E ', [0-9]+\.[0-9]+$' <<<"$out2" | awk -F', ' '{print $NF}')"; DRV_VER="$(smi_field 'Driver Version' <<<"$out2")"; DRV_CUDA="$(smi_field 'CUDA Version' <<<"$out2")"
                ok "요구 검사를 끄면 GPU 가 컨테이너에서 보입니다: $(grep -m1 '^GPU ' <<<"$out2")"
                # 드라이버의 실제 한계는 드라이버 버전으로 환산한다. 컨테이너 안의 nvidia-smi 는 이미지의 CUDA 호환(compat) 라이브러리 때문에 더 높게 나올 수 있다.
                DRV_CUDA_MAPPED="$(cuda_for_driver "$DRV_VER")"
                info "이 드라이버: ${DRV_VER:-?}, 지원하는 최대 CUDA: ${DRV_CUDA_MAPPED:-${HOST_CUDA:-?}}${HOST_CUDA:+ (호스트의 nvidia-smi: $HOST_CUDA)}${CCAP:+, GPU 연산 능력(compute capability): $CCAP}"
                DRV_CUDA="${HOST_CUDA:-$DRV_CUDA_MAPPED}"
                if [[ -n "$CONT_CUDA" && -n "$DRV_CUDA" && "$CONT_CUDA" != "$DRV_CUDA" ]]; then
                    info "컨테이너 안의 nvidia-smi 는 CUDA $CONT_CUDA 로 표시하지만 이 드라이버의 실제 한계는 $DRV_CUDA 입니다. 컨테이너가 호스트의 libcuda 가 아니라 이미지의 호환(compat) libcuda 를 쓰기 때문입니다."
                fi
                if [[ "$LIBCUDA" == */compat/* ]]; then
                    info "컨테이너가 이미지의 CUDA 호환(compat) 라이브러리를 씁니다: $LIBCUDA"
                    if grep -qiE 'GeForce|GTX|TITAN' <<<"$GPU_LINE"; then
                        CONSUMER=1; bad "이 GPU(GeForce 계열)는 이 호환 방식(forward compatibility)을 지원하지 않습니다. 이 상태로 CUDA 프로그램을 실행하면 오류 804(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)가 납니다."
                    fi
                elif [[ -n "$LIBCUDA" ]]; then info "컨테이너가 쓰는 libcuda: $LIBCUDA"; fi
            else
                bad "요구 검사를 꺼도 GPU 를 쓸 수 없습니다."; echo "$out2" | tail -4 | sed 's/^/        /'
            fi
        fi
    fi
else info "Docker 를 쓸 수 없어서 건너뜁니다."; fi

# 시뮬레이터 GPU 이미지가 실제로 요구하는 CUDA (이미지가 이 컴퓨터에 있을 때만 읽을 수 있다. 받는 데는 GPU 드라이버가 필요 없다)
SIM_IMG=""; SIM_REQ=""; SIM_BYPASS=0
if [[ $DOCKER_OK -eq 1 && -f "$DIR/system.env" ]]; then
    src="$(grep -m1 '^DR_SIMAPP_SOURCE=' "$DIR/system.env" | cut -d= -f2)"; sver="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DIR/system.env" | cut -d= -f2)"
    if [[ -n "$src" && -n "$sver" ]]; then
        SIM_IMG="$src:${sver%-*}-gpu"
        echo; echo "  시뮬레이터 GPU 이미지($SIM_IMG)가 요구하는 CUDA"
        if envs="$(docker image inspect "$SIM_IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null)"; then
            SIM_REQ="$(grep '^NVIDIA_REQUIRE_CUDA=' <<<"$envs" | grep -o 'cuda>=[0-9][0-9.]*' | head -1 | sed 's/.*>=//')"
            grep -q '^NVIDIA_DISABLE_REQUIRE=' <<<"$envs" && SIM_BYPASS=1
            if [[ $SIM_BYPASS -eq 1 ]]; then ok "이 이미지는 요구 검사가 꺼져 있습니다(NVIDIA_DISABLE_REQUIRE)."
            elif [[ -n "$SIM_REQ" ]]; then ok "이미지의 요구: CUDA $SIM_REQ 이상 (NVIDIA_REQUIRE_CUDA)"
            else ok "이미지에 CUDA 버전 요구가 없습니다."; fi
        else info "이 이미지를 아직 내려받지 않았습니다. 받은 뒤(docker pull $SIM_IMG) 이 진단을 다시 실행하면 이 이미지의 요구를 확인합니다."; fi
    fi
fi

echo; echo "6) DRfC 가 쓰는 모드 ($DIR)"
TAG=""
if [[ -f "$DIR/system.env" ]]; then
    ver="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DIR/system.env" | cut -d= -f2)"; TAG="${ver##*-}"
    if [[ "$TAG" == "gpu" ]]; then ok "GPU 이미지를 쓰도록 설정됨 ($ver)"; else info "CPU 이미지를 쓰도록 설정됨 ($ver)"; fi
else info "DRfC 가 아직 초기화되지 않았거나 폴더가 다릅니다 (인자로 DRfC 폴더를 주세요)."; fi

# ------------------------------------------------------------------ 판정
echo; echo "== 결론"
if [[ $CONTAINER_ONLY -eq 0 && $DRIVER_OK -eq 0 ]]; then
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
    if [[ -n "$REQ" ]]; then
        CUDA_NOW="${HOST_CUDA:-${DRV_CUDA:-}}"; DRV="${DRV_VER:-}"; dmaj="${DRV%%.*}"
        if [[ $BYPASS_OK -eq 1 ]]; then
            RC=3
            VERDICT="Docker 는 GPU 를 컨테이너에 연결할 수 있습니다. 막힌 것은 GPU 연결이 아니라 CUDA 버전입니다. 이 시험 이미지는 CUDA ${REQ} 이상을 요구하는데, 이 드라이버(${DRV:-?})가 지원하는 최대 CUDA 는 ${CUDA_NOW:-?} 입니다. (DRfC 의 init.sh 는 이 시험이 실패하면 GPU 가 없는 것으로 보고 CPU 모드로 설정합니다.)"
            if [[ $SIM_BYPASS -eq 1 ]]; then
                VERDICT+=" 시뮬레이터 GPU 이미지($SIM_IMG)는 요구 검사가 꺼져 있어서 이 검사에 막히지 않습니다. 다만 CUDA 12.6 로 만든 프로그램이 이 드라이버에서 실제로 도는지는 학습을 시작해 봐야 알 수 있습니다."
            elif [[ -n "$SIM_REQ" ]]; then
                if ver_ge "$CUDA_NOW" "$SIM_REQ"; then
                    VERDICT+=" 시뮬레이터 GPU 이미지($SIM_IMG)는 CUDA ${SIM_REQ} 이상만 요구하고 이 드라이버가 충족하므로 GPU 로 시작할 수 있습니다. 지금 DRfC 가 CPU 모드라면 위 시험 때문입니다."
                    NEXT+=("대시보드 DeepRacer > 환경 점검 > 'GPU / CPU' 에서 GPU 를 고르고 '이 모드로 전환' 을 누르세요.")
                else
                    VERDICT+=" 시뮬레이터 GPU 이미지($SIM_IMG)도 CUDA ${SIM_REQ} 이상을 요구하므로 같은 이유로 시작되지 않습니다."
                fi
            else
                VERDICT+=" DRfC 6.x 의 GPU 시뮬레이터 이미지는 CUDA 12.6 기반이라(DRfC 문서) 같은 이유로 시작되지 않을 가능성이 큽니다. 이미지를 받은 뒤 이 진단을 다시 실행하면 그 이미지의 요구를 정확히 알려 줍니다."
            fi
            if [[ $CONSUMER -eq 1 ]]; then
                NEXT+=("중요: 이 GPU(GeForce)는 이미지의 CUDA 호환(compat) 라이브러리를 쓰는 방식을 지원하지 않습니다. 이미지에서 compat 폴더를 없애 호스트의 libcuda 를 쓰게 하고(마이너 버전 호환성) 요구 검사만 끄면 시도해 볼 수 있습니다. 'bash scripts/gpu_norequire.sh' 가 이것을 합니다.")
            fi
            NEXT+=("드라이버나 공용 Docker 를 바꿀 수 없다면 CPU 모드로 쓰세요. 대시보드 DeepRacer > 환경 점검 > 'GPU / CPU' 에서 CPU 를 고릅니다. 호스트 설정은 건드리지 않고 모델과 실험은 그대로이며, 학습은 GPU 보다 느립니다.")
            if [[ "$dmaj" =~ ^[0-9]+$ ]]; then
                if [[ "$dmaj" -ge 525 ]]; then
                    low=""; [[ "$dmaj" -eq 525 ]] && low=" 525 대는 그 하한이라 새 드라이버 API 를 부르는 코드에 걸릴 위험이 더 큽니다."
                    NEXT+=("이 드라이버(${DRV})는 525 이상입니다. NVIDIA 의 CUDA 마이너 버전 호환성(CUDA 12.x 는 드라이버 525 이상)에 따라, CUDA ${REQ} 로 만든 이미지도 시작 검사만 끄면 동작할 수 있습니다.${low} NVIDIA 문서의 제한: PTX 를 쓰는 코드와 새 드라이버 API 를 부르는 코드는 실패할 수 있습니다. 실제로 되는지는 'bash scripts/gpu_norequire.sh' 로 한 번에 시험할 수 있습니다 (README 의 'CUDA 요구 검사를 끄고 시험하기'). DRfC 이미지로는 확인되지 않은 방법입니다.")
                else
                    NEXT+=("이 드라이버(${DRV})는 525 미만이라 CUDA 12 로 만든 이미지는 시작 검사를 꺼도 실행되지 않습니다. CPU 모드로 쓰거나 드라이버를 올려야 합니다.")
                fi
            fi
            NEXT+=("근본 해결은 CUDA ${REQ} 를 지원하는 드라이버(CUDA 12.6 은 560 이상)로 올리는 것입니다. 공용 서버라면 관리자와 일정을 정하세요. 업데이트하는 동안 같은 호스트에서 GPU 를 쓰는 컨테이너가 영향을 받을 수 있습니다.")
        else
            VERDICT="시험 이미지가 CUDA ${REQ} 이상을 요구해서 실패했고, 요구 검사를 꺼도 GPU 를 쓸 수 없었습니다. 위의 두 오류 메시지를 확인하세요."
            NEXT=("오류 메시지를 그대로 알려 주세요. 그때까지는 CPU 모드로 쓸 수 있습니다 (대시보드 환경 점검 > 'GPU / CPU').")
        fi
    elif grep -qiE 'could not select device driver|unknown or invalid runtime name|no such file or directory.*nvidia' <<<"$ERR" || [[ "$RUNTIMES" != *nvidia* ]]; then
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
