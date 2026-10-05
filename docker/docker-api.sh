#!/usr/bin/env bash
# 호스트의 Docker 서버가 이 컨테이너의 docker 명령(클라이언트)보다 오래된 경우를 처리한다.
#
# 이 이미지의 docker 클라이언트는 Ubuntu 24.04 의 것(API 1.52)이라, 오래된 서버(예: Ubuntu 22.04 의 Docker 24, API 1.43)에서는
#   Error response from daemon: client version 1.52 is too new. Maximum supported API version is 1.43
# 로 모든 docker 명령이 실패할 수 있다. Docker 문서가 안내하는 방법대로, 서버가 지원하는 API 버전을 소켓에서 직접 읽어
# DOCKER_API_VERSION 으로 고정한다 (서버가 더 새것이면 아무것도 하지 않는다).
#
#   source docker/docker-api.sh; pin_docker_api [소켓 경로] [quiet]
pin_docker_api() {
    local sock="${1:-/var/run/docker.sock}" quiet="${2:-}" server client
    if [[ -n "${DOCKER_API_VERSION:-}" ]]; then
        [[ -n "$quiet" ]] || echo "[drtrainer] DOCKER_API_VERSION=$DOCKER_API_VERSION (지정된 값을 씁니다)" >&2
        return 0
    fi
    # 이 함수는 'set -e', 'pipefail' 로 실행되는 스크립트에서도 불린다. 조회가 실패해도 호출한 스크립트가 죽으면 안 되므로 모든 실패를 삼킨다.
    server="$(curl -fsS -m 5 --unix-socket "$sock" http://localhost/version 2>/dev/null | jq -r '.ApiVersion // empty' 2>/dev/null)" || server=""
    [[ "$server" =~ ^1\.[0-9]+$ ]] || return 0
    client="$(docker version --format '{{.Client.APIVersion}}' 2>/dev/null)" || client=""
    [[ "$client" =~ ^1\.[0-9]+$ ]] || client="1.52"             # 클라이언트 버전을 못 읽으면 이 이미지의 값으로 간주
    if [[ "${server#1.}" -lt "${client#1.}" ]]; then
        export DOCKER_API_VERSION="$server"
        [[ -n "$quiet" ]] || echo "[drtrainer] 호스트 Docker 서버(API $server)가 이 컨테이너의 docker 클라이언트(API $client)보다 오래돼서 API $server 로 맞춰 씁니다. 가능하면 호스트의 Docker 를 업데이트하세요." >&2
    fi
}
