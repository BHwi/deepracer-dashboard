#!/usr/bin/env bash
# 드라이버를 올릴 수 없을 때, CUDA 버전 요구 검사를 끈 시뮬레이터 GPU 이미지로 GPU 를 쓸 수 있는지 시험한다.
#
#   bash scripts/gpu_norequire.sh [DRfC 폴더] [--pull] [--apply] [--gpu 학습[,시뮬레이터]] [--rebuild] [--keep-compat]
#
# 배경: nvidia-container-cli: requirement error: unsatisfied condition: cuda>=12.6 은 이미지의 NVIDIA_REQUIRE_CUDA 요구를 드라이버가
#       지원하지 못할 때 컨테이너 시작을 막는 검사다. NVIDIA 는 NVIDIA_DISABLE_REQUIRE=1 로 이 검사를 끌 수 있다고 안내한다.
#       검사를 꺼도 프로그램이 실제로 도는지는 별개이므로(CUDA 12.x 는 드라이버 525 이상에서 마이너 버전 호환성으로 실행될 수 있고,
#       PTX 를 쓰는 코드와 새 드라이버 API 를 부르는 코드는 실패할 수 있다) 마지막에 TensorFlow 로 실제 연산을 해 본다.
#       또 NVIDIA Container Toolkit 의 기본 설정(cuda-compat-mode=ldconfig)은 이미지 안의 CUDA 호환(compat) libcuda 를 호스트 것보다
#       우선시킨다. 이 방식(forward compatibility)은 데이터센터/Quadro GPU 만 지원해서 GeForce 에서는 오류 804
#       (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)가 난다. 그래서 이미지에서 compat 폴더를 없애 호스트의 libcuda 를 쓰게 한다
#       (마이너 버전 호환성 경로). 데이터센터 GPU 에서 compat 를 그대로 쓰려면 --keep-compat.
#
# 하는 일 (원본 이미지와 태그, 호스트와 Docker 설정, 다른 사용자의 것은 바꾸지 않는다)
#   1) system.env 에서 시뮬레이터 이미지 이름과 버전을 읽는다 (-cpu 로 돼 있어도 같은 버전의 -gpu 이미지를 대상으로 한다)
#   2) GPU 이미지가 이 컴퓨터에 없으면 --pull 일 때만 내려받는다 (수 GB. 받는 데는 GPU 드라이버가 필요 없다)
#   3) 원본에 'ENV NVIDIA_DISABLE_REQUIRE=1' 과 compat 폴더 삭제만 얹은 로컬 이미지 drtrainer/deepracer-simapp:<버전>-gpu 를 만든다
#      만든 뒤, 컨테이너가 호스트의 libcuda 를 쓰는지 확인한다
#   4) 그 이미지에서 TensorFlow 로 conv2d 와 matmul 을 GPU 에서 실행해 본다 (TensorFlow 는 이미지의 conda 환경 sagemaker_env 에 있다)
#   5) 통과하면 system.env 에 쓸 줄을 알려 준다. --apply 를 주면 백업 후 직접 고친다.
#
# 여러 GPU 가 있는 (공용) 서버: 시뮬레이터 이미지는 NVIDIA_VISIBLE_DEVICES=all 이고 DRfC 의 DR_SAGEMAKER_CUDA_DEVICES / DR_ROBOMAKER_CUDA_DEVICES 는
#   기본으로 주석 처리돼 있어서(compose 는 CUDA_VISIBLE_DEVICES=${...:-} 로 쓴다) 지정하지 않으면 학습이 모든 GPU 를 본다.
#   --gpu 1 은 학습과 시뮬레이터 모두 1번 GPU, --gpu 1,0 은 학습 1번, 시뮬레이터 0번. 시험도 지정한 GPU 에서만 한다.
#   GPU 가 둘 이상 보이면 --gpu 없이는 --apply 로 설정을 바꾸지 않는다.
# 종료 코드: 0 통과, 1 실패 또는 중단, 2 이미지가 없어서 시험하지 못함, 4 통과했지만 GPU 를 지정해야 해서 적용하지 않음
set -uo pipefail
PULL=0; APPLY=0; REBUILD=0; KEEP_COMPAT=0; GPU_ARG=""; ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull) PULL=1 ;; --apply) APPLY=1 ;; --rebuild) REBUILD=1 ;; --keep-compat) KEEP_COMPAT=1 ;;
        --gpu) GPU_ARG="${2:-}"; shift ;; --gpu=*) GPU_ARG="${1#--gpu=}" ;;
        *) ARGS+=("$1") ;;
    esac; shift
done
DIR="${ARGS[0]:-${DRFC_DIR:-$HOME/deepracer-for-cloud}}"
LOCAL_SRC="drtrainer/deepracer-simapp"
PYBIN="/root/anaconda/envs/sagemaker_env/bin/python"
smi_field() { sed -n "s/.*$1: *\([0-9][0-9.]*\).*/\1/p" | head -1; }
ok()   { echo "  [ 확인 ] $*"; }
bad()  { echo "  [ 문제 ] $*"; }
info() { echo "  [ 참고 ] $*"; }
die()  { echo "  [ 문제 ] $*"; exit 1; }

[[ -z "$GPU_ARG" || "$GPU_ARG" =~ ^[0-9]+(,[0-9]+)?$ ]] || { echo "  [ 문제 ] --gpu 는 번호 하나(1) 또는 학습,시뮬레이터(1,0) 형식이어야 합니다: '$GPU_ARG'"; exit 1; }
SAGE_GPU="${GPU_ARG%%,*}"; ROBO_GPU="${GPU_ARG##*,}"
echo "== CUDA 요구 검사를 끈 시뮬레이터 이미지 시험"
echo; echo "1) 설정과 Docker"
[[ -f "$DIR/system.env" ]] || die "$DIR/system.env 가 없습니다. DRfC 폴더를 인자로 주세요 (DRfC 가 아직 초기화되지 않았을 수 있습니다)."
command -v docker >/dev/null 2>&1 || die "docker 명령이 없습니다."
docker info >/dev/null 2>&1 || die "Docker 서버에 연결할 수 없습니다 (꺼져 있거나 권한이 없음: sudo usermod -aG docker \$USER 후 다시 로그인)."
SRC="$(grep -m1 '^DR_SIMAPP_SOURCE=' "$DIR/system.env" | cut -d= -f2)"; VER="$(grep -m1 '^DR_SIMAPP_VERSION=' "$DIR/system.env" | cut -d= -f2)"
[[ -n "$VER" ]] || die "system.env 에 DR_SIMAPP_VERSION 이 없습니다."
ORIG_SRC="${SRC:-awsdeepracercommunity/deepracer-simapp}"; [[ "$ORIG_SRC" == "$LOCAL_SRC" ]] && ORIG_SRC="awsdeepracercommunity/deepracer-simapp"
BASEVER="${VER%-gpu}"; BASEVER="${BASEVER%-cpu}"; GPUVER="$BASEVER-gpu"
ORIG="$ORIG_SRC:$GPUVER"; NEW="$LOCAL_SRC:$GPUVER"
ok "대상: 원본 $ORIG  →  로컬 $NEW"

NG=0
if command -v nvidia-smi >/dev/null 2>&1 && smi="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)"; then
    info "GPU: $(head -1 <<<"$smi")"
    drv="$(head -1 <<<"$smi" | awk -F', ' '{print $2}')"; dmaj="${drv%%.*}"
    if [[ "$dmaj" =~ ^[0-9]+$ && "$dmaj" -lt 525 ]]; then
        bad "드라이버 $drv 는 525 미만입니다. CUDA 12 로 만든 이미지는 요구 검사를 꺼도 실행되지 않습니다. CPU 모드로 쓰거나 드라이버를 올려야 합니다."
        exit 1
    fi
    [[ "$dmaj" =~ ^[0-9]+$ && "$dmaj" -eq 525 ]] && info "드라이버 525 대는 CUDA 12 마이너 버전 호환성의 하한입니다. 새 드라이버 API 를 부르는 코드에 걸릴 위험이 더 큽니다."
    NG="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)"
    if [[ "${NG:-0}" -gt 1 ]]; then
        info "GPU 가 ${NG}개 보입니다. 사용 현황 (번호, 메모리 사용/전체, 사용률):"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/        GPU /'
        if [[ -z "$GPU_ARG" ]]; then info "GPU 를 지정하지 않으면 학습이 모든 GPU 를 보고 각 GPU 에서 메모리를 잡으려 합니다. 공용 서버라면 비어 있는 GPU 를 --gpu 로 지정하세요."
        else info "지정한 GPU: 학습 $SAGE_GPU, 시뮬레이터 $ROBO_GPU (시험도 학습용 GPU $SAGE_GPU 에서만 합니다)"; fi
    fi
else
    info "호스트의 nvidia-smi 를 쓸 수 없어 드라이버 확인은 건너뜁니다 (아래 시험에서 컨테이너가 판단합니다)."
fi

echo; echo "2) 시뮬레이터 GPU 이미지"
if docker image inspect "$ORIG" >/dev/null 2>&1; then ok "이미 있습니다: $ORIG"
elif [[ $PULL -eq 1 ]]; then
    echo "  내려받는 중입니다 (수 GB 라 시간이 걸립니다. 받는 데는 GPU 드라이버가 필요 없습니다)"
    docker pull "$ORIG" >/dev/null 2>&1 || die "내려받지 못했습니다: $ORIG (인터넷 연결, 또는 버전 이름을 확인하세요)"
    ok "내려받았습니다: $ORIG"
else
    info "$ORIG 가 이 컴퓨터에 없습니다. 내려받으려면 --pull 을 붙여 다시 실행하세요 (수 GB, 공용 서버라면 디스크와 네트워크를 쓰는 것이므로 먼저 확인하세요)."
    exit 2
fi
req="$(docker image inspect "$ORIG" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep '^NVIDIA_REQUIRE_CUDA=' | grep -o 'cuda>=[0-9][0-9.]*' | head -1)"
[[ -n "$req" ]] && info "원본 이미지의 CUDA 요구: $req (NVIDIA_REQUIRE_CUDA). 이 요구 때문에 원본은 낮은 드라이버에서 시작되지 않습니다."

echo; echo "3) 요구 검사를 끈 로컬 이미지"
if docker image inspect "$NEW" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]]; then ok "이미 있습니다: $NEW (다시 만들려면 --rebuild)"
else
    if [[ $KEEP_COMPAT -eq 1 ]]; then DF="$(printf 'FROM %s\nENV NVIDIA_DISABLE_REQUIRE=1\n' "$ORIG")"
    else DF="$(printf 'FROM %s\nENV NVIDIA_DISABLE_REQUIRE=1\nRUN rm -rf /usr/local/cuda*/compat\n' "$ORIG")"; fi
    printf '%s\n' "$DF" | docker build -q -t "$NEW" - >/dev/null 2>&1 || die "이미지를 만들지 못했습니다: $NEW"
    ok "만들었습니다: $NEW (원본 $ORIG 는 그대로입니다)"
    [[ $KEEP_COMPAT -eq 1 ]] && info "--keep-compat: 이미지의 CUDA 호환(compat) 라이브러리를 남겼습니다. GeForce 에서는 오류 804 가 납니다."
fi
# 컨테이너가 어느 libcuda 를 쓰는지 확인한다 (호스트 것이어야 마이너 버전 호환성 경로로 동작한다)
chk="$(timeout 120 docker run --rm --gpus all --entrypoint bash "$NEW" -lc 'ldconfig -p 2>/dev/null | grep "libcuda.so.1"; nvidia-smi 2>/dev/null' 2>&1)"
lib="$(grep -m1 'libcuda.so.1' <<<"$chk" | sed 's/.*=> *//')"; cont="$(smi_field 'CUDA Version' <<<"$chk")"
if [[ -z "$lib" ]]; then info "컨테이너의 libcuda 위치를 확인하지 못했습니다."
elif [[ "$lib" == */compat/* ]]; then bad "컨테이너가 아직 이미지의 호환(compat) libcuda 를 씁니다: $lib (GeForce 에서는 오류 804 가 납니다)"
else ok "컨테이너가 호스트의 libcuda 를 씁니다: $lib${cont:+ (nvidia-smi 의 CUDA Version: $cont)}"; fi

echo; echo "4) TensorFlow 로 GPU 연산 시험 (conv2d, matmul)"
info "GPU 세대가 TensorFlow 에 미리 컴파일돼 있지 않으면 PTX 를 컴파일해서 매우 오래 걸릴 수 있습니다 (최대 15분까지 기다립니다)."
read -r -d '' TF_SCRIPT <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "| GPU:", [g.name for g in gpus])
if not gpus:
    print("RESULT_NO_GPU")
    raise SystemExit(3)
with tf.device("/GPU:0"):
    x = tf.random.normal((1, 64, 64, 3)); w = tf.random.normal((3, 3, 3, 8))
    y = tf.nn.conv2d(x, w, 1, "SAME"); _ = y.numpy(); print("conv2d ok", tuple(y.shape))
    a = tf.random.normal((1024, 1024)); b = tf.matmul(a, a); _ = b.numpy(); print("matmul ok", tuple(b.shape))
print("RESULT_OK")
PY
RUN_ENV=(-e TF_FORCE_GPU_ALLOW_GROWTH=true); [[ -n "$GPU_ARG" ]] && RUN_ENV+=(-e "CUDA_VISIBLE_DEVICES=$SAGE_GPU")
out="$(printf '%s\n' "$TF_SCRIPT" | timeout 900 docker run --rm -i --gpus all "${RUN_ENV[@]}" --entrypoint "$PYBIN" "$NEW" - 2>&1)"; rc=$?
echo "$out" | grep -v '^$' | tail -12 | sed 's/^/        /'

echo; echo "== 결론"
PASS=0
if grep -q 'RESULT_OK' <<<"$out"; then
    PASS=1
    echo "  이 드라이버에서 TensorFlow 가 GPU 로 conv2d 와 matmul 을 실행했습니다."
    if grep -qi 'jit-compiled from PTX' <<<"$out"; then
        echo "  다만 이 GPU 세대는 TensorFlow 에 미리 컴파일돼 있지 않아 PTX 컴파일을 거쳤습니다. 첫 실행이 오래 걸렸을 수 있고, 학습 중에도 새 연산마다 지연이 있을 수 있습니다."
    fi
    echo "  이것은 연산 두 개가 도는 것까지의 확인이며, 실제 학습이 끝까지 안정적으로 도는지는 학습을 돌려 봐야 압니다."
elif grep -q 'RESULT_NO_GPU' <<<"$out"; then
    echo "  TensorFlow 가 GPU 를 찾지 못했습니다. CPU 모드를 쓰세요."
elif grep -qiE 'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE|forward compatibility was attempted|cuInit.*804|error 804' <<<"$out"; then
    echo "  오류 804(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)입니다. 컨테이너가 이미지의 CUDA 호환(compat) libcuda 를 골랐는데, GeForce 는 이 방식(forward compatibility)을 지원하지 않습니다."
    if [[ $KEEP_COMPAT -eq 1 ]]; then echo "  --keep-compat 없이 다시 만드세요 (--rebuild). 이미지에서 compat 폴더를 없애면 호스트의 libcuda 를 씁니다."
    else echo "  이 이미지에서는 compat 폴더를 이미 없앴는데도 났습니다. 위의 3) 단계에서 컨테이너가 쓰는 libcuda 경로를 확인하고, 그 경로를 알려 주세요. (NVIDIA Container Toolkit 의 cuda-compat-mode 설정이 호스트 쪽에서 라이브러리를 넣는 경우일 수 있습니다.)"; fi
elif grep -qiE 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION|cudaErrorCallRequiresNewerDriver|driver version is insufficient|forward compat|CUDA_ERROR_[A-Z_]+|cudaError[A-Za-z]+|CUDNN_STATUS_[A-Z_]+|Failed call to cuInit' <<<"$out"; then
    echo "  CUDA 오류로 실패했습니다. 이 드라이버는 이 이미지가 쓰는 CUDA 기능을 지원하지 못하는 것입니다 (NVIDIA 문서의 마이너 버전 호환성의 제한). 위 오류 이름을 기록해 두고 CPU 모드를 쓰거나 드라이버를 올려야 합니다."
elif [[ $rc -eq 124 ]]; then
    echo "  15분 안에 끝나지 않았습니다. GPU 세대가 PTX 컴파일을 요구하는 경우일 가능성이 큽니다. 실용적이지 않으므로 CPU 모드를 쓰세요."
elif grep -qiE 'no such file or directory|executable file not found|not found in \$PATH' <<<"$out"; then
    echo "  이미지 안에서 TensorFlow 파이썬($PYBIN)을 찾지 못했습니다. 이 이미지 버전의 구성이 다른 것으로 보입니다. docker run --rm -it --entrypoint bash $NEW 로 들어가 which -a python3 python 으로 찾아 보세요."
else
    echo "  원인을 구분하지 못했습니다. 위 출력 전체를 알려 주세요. 그때까지는 CPU 모드를 쓸 수 있습니다."
fi
[[ $PASS -eq 1 ]] || exit 1

echo; echo "== 다음 단계"
# system.env 의 키를 설정한다 (이미 있으면 바꾸고, 주석 처리된 템플릿 줄이 있으면 그 줄을 켜고, 없으면 추가)
set_kv() {
    local k="$1" v="$2" f="$DIR/system.env"
    if grep -q "^$k=" "$f"; then sed -i "s#^$k=.*#$k=$v#" "$f"
    elif grep -q "^# *$k=" "$f"; then sed -i "0,/^# *$k=.*/s##$k=$v#" "$f"
    else echo "$k=$v" >> "$f"; fi
}
if [[ "${NG:-0}" -gt 1 && -z "$GPU_ARG" ]] && ! grep -qE '^DR_SAGEMAKER_CUDA_DEVICES=[0-9]' "$DIR/system.env"; then
    echo "  GPU 가 ${NG}개 보이는데 사용할 GPU 가 지정돼 있지 않습니다. 이대로 학습하면 모든 GPU 를 보고 각 GPU 에서 메모리를 잡으려 해서, 공용 서버에서는 다른 사람의 작업과 부딪힐 수 있습니다."
    echo "  비어 있는 GPU 번호를 위의 사용 현황에서 고르고 다시 실행하세요. 예: bash scripts/gpu_norequire.sh --gpu 1 --apply   (학습과 시뮬레이터 모두 1번)"
    echo "                                                          bash scripts/gpu_norequire.sh --gpu 1,0 --apply (학습 1번, 시뮬레이터 0번)"
    [[ $APPLY -eq 1 ]] && { echo "  --apply 를 요청하셨지만 GPU 지정 없이는 system.env 를 바꾸지 않았습니다."; exit 4; }
    exit 0
fi
if [[ $APPLY -eq 1 ]]; then
    bk="$DIR/system.env.bak-$(date +%m%d-%H%M%S)"; cp "$DIR/system.env" "$bk"
    set_kv DR_SIMAPP_SOURCE "$LOCAL_SRC"; set_kv DR_SIMAPP_VERSION "$GPUVER"
    [[ -n "$GPU_ARG" ]] && { set_kv DR_SAGEMAKER_CUDA_DEVICES "$SAGE_GPU"; set_kv DR_ROBOMAKER_CUDA_DEVICES "$ROBO_GPU"; }
    ok "system.env 를 고쳤습니다 (백업: $(basename "$bk")): DR_SIMAPP_SOURCE=$LOCAL_SRC, DR_SIMAPP_VERSION=$GPUVER${GPU_ARG:+, DR_SAGEMAKER_CUDA_DEVICES=$SAGE_GPU, DR_ROBOMAKER_CUDA_DEVICES=$ROBO_GPU}"
    echo "  1. 이제 평소처럼 학습을 시작하세요. 시뮬레이터, 학습, 코치 컨테이너가 모두 이 이미지를 씁니다."
else
    echo "  1. DRfC 폴더의 system.env 에서 아래 줄로 바꾸세요 (또는 이 스크립트를 --apply 로 다시 실행하면 백업 후 고쳐 줍니다)."
    echo "       DR_SIMAPP_SOURCE=$LOCAL_SRC"
    echo "       DR_SIMAPP_VERSION=$GPUVER"
    [[ -n "$GPU_ARG" ]] && { echo "       DR_SAGEMAKER_CUDA_DEVICES=$SAGE_GPU"; echo "       DR_ROBOMAKER_CUDA_DEVICES=$ROBO_GPU"; }
fi
# 학습이 정말 지정한 GPU 에서 도는지 확인하는 방법과, 시뮬레이터의 GPU 사용 조건
if [[ -n "$GPU_ARG" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    bus="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | awk -F', ' -v g="$SAGE_GPU" '$1 == g {print $2}' | head -1)"; bus="${bus#0000}"
    echo "  확인: 학습을 시작한 뒤 대시보드 DeepRacer > 학습 > 로그 > '학습' 에서 'Created device ... pci bus id: ${bus:-<GPU $SAGE_GPU 의 버스>}' 줄이 보여야 합니다. 다른 버스 번호이거나 GPU 줄이 없으면 학습이 GPU 로 돌지 않는 것입니다."
    echo "        학습 중 호스트에서: nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv   (그 버스에만 프로세스가 있어야 합니다)"
fi
defrt="$(docker info --format '{{.DefaultRuntime}}' 2>/dev/null)"
if [[ -n "$defrt" && "$defrt" != "nvidia" ]]; then
    echo "  참고: Docker 의 기본 런타임이 '$defrt' 입니다. 학습 컨테이너는 이미지 이름에 gpu 가 있으면 SageMaker local_gpu 로 GPU 를 받으므로 영향이 없습니다. 다만 시뮬레이터(swarm 서비스)는 --gpus 를 쓸 수 없어서 기본 런타임이 nvidia 가 아니면 GPU 를 못 쓸 수 있습니다. 공용 호스트의 설정이라 바꾸지 않습니다."
fi
echo "  2. 시작할 때 'image ... could not be accessed on a registry to record its digest' 경고가 나와도 무시해도 됩니다 (로컬 이미지라서 나옵니다)."
echo "  3. DRfC 의 init.sh 는 다시 실행하지 마세요 (system.env 를 덮어씁니다). 대시보드의 'GPU / CPU' 전환은 이 로컬 이미지가 있으면 내려받지 않고, CPU 를 고르면 DR_SIMAPP_SOURCE 도 원래 이름으로 되돌립니다."
echo "  4. 되돌리려면 system.env 의 DR_SIMAPP_SOURCE 를 $ORIG_SRC 로, 백업 파일(system.env.bak-*)에서 복원하거나 직접 고치세요."
exit 0
