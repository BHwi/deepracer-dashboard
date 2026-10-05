#!/usr/bin/env bash
# 컨테이너 안에서 명령을 실행하기 전에 Docker API 버전을 맞춘다 (docker exec 로 들어온 명령은 진입점이 설정한 환경변수를 물려받지 못하므로).
#   run-pinned.sh 명령 [인자...]
# shellcheck source=docker-api.sh
source "$(dirname "${BASH_SOURCE[0]}")/docker-api.sh"
pin_docker_api "${DOCKER_SOCK:-/var/run/docker.sock}" quiet
exec "$@"
