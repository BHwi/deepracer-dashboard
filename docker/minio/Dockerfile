# syntax=docker/dockerfile:1
# MinIO 가 2026-09 에 Docker Hub 의 minio/minio 이미지를 삭제했다. DRfC(로컬 모드)는 이 이미지를 쓰므로, GitHub 릴리스에 남아 있는
# 마지막 공개 바이너리(RELEASE.2025-09-07T16-13-09Z)를 SHA-256 으로 검증해서 같은 모양의 이미지로 직접 만든다.
FROM ubuntu:24.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG MINIO_TAG=RELEASE.2025-09-07T16-13-09Z
# 릴리스 페이지의 .sha256sum 값 (아키텍처별)
ARG SHA256_AMD64=7c5bd8512c6e966455b1d198209358b2d191c77a83ab377c4073281065fb855f
ARG SHA256_ARM64=5c83cd2cf151717ba0243f73e1c7802ff36e272b67144bdd7f1f7d684fd6f03d

# hadolint ignore=DL3008
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends ca-certificates curl; rm -rf /var/lib/apt/lists/*; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in amd64) sha="$SHA256_AMD64" ;; arm64) sha="$SHA256_ARM64" ;; *) echo "지원하지 않는 아키텍처: $arch" >&2; exit 1 ;; esac; \
    curl -fsSL -o /usr/local/bin/minio "https://github.com/minio/minio/releases/download/${MINIO_TAG}/minio.linux-${arch}.${MINIO_TAG}"; \
    echo "${sha}  /usr/local/bin/minio" | sha256sum -c -; \
    chmod +x /usr/local/bin/minio; mkdir -p /data

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
VOLUME /data
EXPOSE 9000 9001
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["minio"]
