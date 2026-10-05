# syntax=docker/dockerfile:1
# DeepRacer Trainer 이미지: 대시보드(MiniRacer + DeepRacer 관리) + DRfC 도구 일체.
# 호스트에는 Docker Engine(GPU 를 쓰면 NVIDIA 드라이버와 NVIDIA Container Toolkit)만 있으면 된다.
FROM ubuntu:24.04

# 이 대시보드를 시험한 DRfC 버전(2026-08-09)으로 고정한다. 다른 버전을 쓰려면: docker compose build --build-arg DRFC_COMMIT=<커밋>
ARG DRFC_REPO=https://github.com/aws-deepracer-community/deepracer-for-cloud.git
ARG DRFC_COMMIT=245bd7f0fe0810043744d988edcae78c03617fee

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DRTRAINER_IN_DOCKER=1 MPLCONFIGDIR=/tmp/matplotlib

# DRfC 스크립트가 쓰는 명령(aws, jq, awk, sed, ip, netstat, perl, bc, pstree, envsubst, screen, openssl, curl, git, sudo ...)을 모두 넣는다.
# docker.io / docker-buildx / docker-compose-v2 는 DRfC 의 bin/prepare.sh 와 같은 Ubuntu 패키지다. 여기서는 docker 명령(CLI)만 쓰고
# Docker 서버(dockerd)는 띄우지 않는다 (호스트의 서버를 docker.sock 으로 쓴다).
# apt 패키지 버전은 고정하지 않는다: 고정하면 Ubuntu 보안 업데이트로 버전이 바뀔 때 빌드가 깨진다.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq screen openssl bc psmisc gettext-base perl gawk sed findutils coreutils tar procps \
        iproute2 net-tools ncurses-bin sudo gosu \
        python3 python3-pip python3-venv \
        docker.io docker-buildx docker-compose-v2 \
    && rm -rf /var/lib/apt/lists/* \
    && (userdel -r ubuntu 2>/dev/null || true) && (groupdel ubuntu 2>/dev/null || true)

# AWS CLI 는 별도 가상환경에 둔다 (awscli 1.x 가 요구하는 botocore 버전이 DRfC 의 boto3 와 충돌하지 않게).
RUN python3 -m venv /opt/awscli && /opt/awscli/bin/pip install --no-cache-dir awscli && ln -s /opt/awscli/bin/aws /usr/local/bin/aws

# DRfC: 고정한 커밋을 /opt/drfc 에 두고, 처음 실행할 때 호스트 폴더로 복사한다 (형제 컨테이너가 호스트 경로를 쓰기 때문).
RUN git init -q /opt/drfc && git -C /opt/drfc remote add origin "$DRFC_REPO" \
    && git -C /opt/drfc fetch -q --depth 1 origin "$DRFC_COMMIT" && git -C /opt/drfc checkout -q FETCH_HEAD

# 파이썬 패키지: 대시보드(numpy, matplotlib)와 DRfC(requirements.txt)를 한 가상환경에. DRfC 스크립트의 python3 도 이것을 쓴다.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir numpy matplotlib \
    && /opt/venv/bin/pip install --no-cache-dir -r /opt/drfc/requirements.txt
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY . /app
COPY docker/service-shim.sh /usr/local/sbin/service
RUN chmod +x /usr/local/sbin/service /app/docker/*.sh && mkdir -p /app/runs /app/state

# 호스트 네트워크로 실행하므로 포트 공개(EXPOSE)는 필요 없다. 대시보드는 127.0.0.1 에만 열린다.
HEALTHCHECK --interval=20s --timeout=5s --start-period=900s --retries=3 \
    CMD ["sh", "-c", "curl -fsS http://127.0.0.1:${DASHBOARD_PORT:-8765}/api/info >/dev/null || exit 1"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
