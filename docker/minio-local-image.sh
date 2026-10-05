#!/usr/bin/env bash
# DRfC 의 minio 이미지를 확인하고, 없으면 만들어서 쓰게 한다.
#
#   bash docker/minio-local-image.sh [DRfC 폴더]
#
# 배경: MinIO 가 2026-09 에 Docker Hub 의 minio/minio 이미지를 삭제했다. DRfC(로컬 모드)는 시작할 때 이 이미지로 저장소(minio)를 띄우므로,
#       이미지가 없는 컴퓨터에서는 'No such image: minio/minio:latest' 로 학습을 시작할 수 없다.
# 하는 일 (여러 번 실행해도 안전):
#   1) system.env 의 DR_MINIO_IMAGE 로 정해진 이미지가 이미 있으면 아무것도 하지 않는다.
#   2) 없으면 한 번 받아 보고, 안 받아지면 GitHub 릴리스의 마지막 공개 바이너리(SHA-256 검증)로 docker/minio/ 의 Dockerfile 을 빌드한다.
#   3) system.env 의 DR_MINIO_IMAGE 를 만든 이미지의 태그로 바꾼다 (원본은 백업).
#   4) 이전 이미지로 실패 중이던 minio 서비스(s3 스택)가 있으면 지워서, 다음 DRfC 활성화 때 새 이미지로 다시 뜨게 한다.
# 시험용 환경변수: MINIO_LOCAL_TAG (만들 태그 이름)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DIR="${1:-${DR_DIR:-}}"
LOCAL_TAG="${MINIO_LOCAL_TAG:-RELEASE.2025-09-07T16-13-09Z-local}"
say() { echo "[minio] $*"; }

[[ -n "$DIR" && -f "$DIR/system.env" ]] || { say "DRfC 폴더(system.env)를 찾지 못했습니다: ${DIR:-없음}"; exit 1; }
[[ -f "$DIR/.mock_drfc" ]] && exit 0                                    # 시험용 가짜 DRfC
getv() { grep -m1 "^$1=" "$DIR/system.env" | cut -d= -f2- | tr -d '"'"'"; }

cloud="$(getv DR_CLOUD)"
case "$cloud" in local|azure) ;; *) say "DR_CLOUD=${cloud:-?} 라서 로컬 minio 가 필요 없습니다."; exit 0 ;; esac
command -v docker >/dev/null 2>&1 || { say "docker 명령이 없습니다."; exit 1; }
if ! info_out="$(docker info 2>&1 >/dev/null)"; then
    say "docker 명령이 실패했습니다. 실제 오류:"; echo "$info_out" | head -3 | sed 's/^/        /'
    grep -qi "client version.*too new" <<<"$info_out" && say "호스트 Docker 서버가 이 docker 명령보다 오래된 경우입니다. Docker 컨테이너로 실행 중이면 ./drtrainer up --rebuild 로 다시 시작하세요 (서버 버전에 맞춰 자동으로 맞춥니다)."
    exit 1
fi

tag="$(getv DR_MINIO_IMAGE)"
tag="${tag:-RELEASE.2022-10-24T18-35-07Z}"                               # activate.sh 가 DR_MINIO_IMAGE 가 비었을 때 쓰는 값
image="minio/minio:$tag"

if docker image inspect "$image" >/dev/null 2>&1; then
    say "이미지가 있습니다: $image"
    exit 0
fi
say "이미지($image)가 이 컴퓨터에 없습니다. 받아 봅니다..."
if timeout 120 docker pull "$image" >/dev/null 2>&1; then
    say "받았습니다: $image"
    exit 0
fi
say "받을 수 없습니다. (MinIO 가 Docker Hub 의 minio/minio 를 삭제했습니다.)"

new="minio/minio:$LOCAL_TAG"
if docker image inspect "$new" >/dev/null 2>&1; then
    say "이미 만들어 둔 이미지가 있습니다: $new"
else
    say "GitHub 릴리스의 검증된 바이너리로 $new 를 만듭니다 (1~2분)..."
    if ! docker build -t "$new" "$HERE/minio"; then
        say "기본 방식으로 빌드하지 못했습니다. 오래된 Docker 서버를 위해 이전 빌더로 다시 시도합니다..."
        DOCKER_BUILDKIT=0 docker build -t "$new" "$HERE/minio" || { say "이미지를 만들지 못했습니다. 인터넷(github.com) 연결과 위 오류를 확인하세요."; exit 1; }
    fi
fi

# system.env 의 DR_MINIO_IMAGE 를 새 태그로 (백업 후)
cp "$DIR/system.env" "$DIR/system.env.bak-$(date +%m%d-%H%M%S)"
if grep -q "^DR_MINIO_IMAGE=" "$DIR/system.env"; then
    sed -i "s|^DR_MINIO_IMAGE=.*|DR_MINIO_IMAGE=$LOCAL_TAG|" "$DIR/system.env"
else
    printf '\nDR_MINIO_IMAGE=%s\n' "$LOCAL_TAG" >> "$DIR/system.env"
fi
say "system.env 의 DR_MINIO_IMAGE 를 $LOCAL_TAG 로 바꿨습니다 (이전 파일은 system.env.bak-* 로 백업)."

# 예전 이미지로 실패 중인 minio 서비스를 치워서 새 이미지로 다시 뜨게 한다
cur="$(docker service inspect s3_minio --format '{{.Spec.TaskTemplate.ContainerSpec.Image}}' 2>/dev/null || true)"
if [[ -n "$cur" && "$cur" != "$new"* ]]; then
    say "이전 이미지($cur)로 만들어진 minio 서비스를 지웁니다. 다음에 DRfC 를 활성화할 때 새 이미지로 다시 만들어집니다."
    docker stack rm s3 >/dev/null 2>&1 || docker service rm s3_minio >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do docker service inspect s3_minio >/dev/null 2>&1 || break; sleep 1; done
fi
old_c="$(docker ps -a --filter "label=com.docker.compose.project=s3" -q 2>/dev/null || true)"
if [[ -n "$old_c" ]]; then say "compose 로 만든 이전 minio 컨테이너를 지웁니다."; echo "$old_c" | xargs docker rm -f >/dev/null 2>&1 || true; fi
say "준비됐습니다."
