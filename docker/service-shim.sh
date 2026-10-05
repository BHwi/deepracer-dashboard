#!/bin/sh
# 컨테이너 안에서는 Docker 서버를 띄우지 않는다 (호스트의 Docker 를 docker.sock 으로 쓴다).
# DRfC 의 bin/activate.sh 가 'service docker status || sudo service docker start' 를 부르는데,
# 진짜 service 를 부르면 컨테이너 안에서 두 번째 Docker 서버가 뜨려 한다. 호스트 Docker 가 살아 있는지로 대답한다.
if [ "$1" = "docker" ]; then
    case "$2" in
        status|start|restart)
            if docker info >/dev/null 2>&1; then exit 0; fi
            echo "호스트 Docker 서버에 연결할 수 없습니다 (docker.sock 연결 확인)." >&2
            exit 3 ;;
        *) exit 0 ;;
    esac
fi
[ -x /usr/sbin/service ] && exec /usr/sbin/service "$@"
exit 0
