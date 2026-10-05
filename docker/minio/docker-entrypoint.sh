#!/bin/sh
# 공식 minio 이미지의 진입점과 같은 동작: 'server ...' 처럼 minio 를 생략한 인자도 받고,
# MINIO_UID / MINIO_GID 가 있으면 그 사용자로 권한을 낮춰서 실행한다 (DRfC 는 이것으로 data/minio 의 파일 소유자를 호스트 사용자로 맞춘다).
set -e
if [ "${1}" != "minio" ] && [ -n "${1}" ]; then
    set -- minio "$@"
fi
if [ "$(id -u)" = "0" ] && [ -n "${MINIO_UID}" ] && [ -n "${MINIO_GID}" ]; then
    getent group "${MINIO_GID}" >/dev/null 2>&1 || groupadd -g "${MINIO_GID}" "${MINIO_GROUPNAME:-minio}" 2>/dev/null || groupadd -g "${MINIO_GID}" "minio${MINIO_GID}"
    getent passwd "${MINIO_UID}" >/dev/null 2>&1 || useradd -M -N -u "${MINIO_UID}" -g "${MINIO_GID}" -s /usr/sbin/nologin "${MINIO_USERNAME:-minio}" 2>/dev/null || useradd -M -N -u "${MINIO_UID}" -g "${MINIO_GID}" -s /usr/sbin/nologin "minio${MINIO_UID}"
    exec setpriv --reuid="${MINIO_UID}" --regid="${MINIO_GID}" --init-groups "$@"
fi
exec "$@"
