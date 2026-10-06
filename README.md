<h1 align="center">DeepRacer Trainer</h1>

<p align="center">
  <b>보상함수를 노트북에서 몇 분 안에 시험하고, 실제 DeepRacer 시뮬레이터로 학습·평가해서, 차량용 파일까지 만드는 웹 대시보드</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-%EC%A7%80%EC%9B%90-2496ED?logo=docker&logoColor=white">
  <img alt="Platform" src="https://img.shields.io/badge/Linux%20%C2%B7%20WSL2-DeepRacer-555555">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-orange">
</p>

<p align="center">
  <img src="docs/img/miniracer-result.png" alt="MiniRacer 결과 화면" width="880">
</p>

## 목차

1. [설치 방법](#1-설치-방법)
2. [localhost 접속](#2-localhost-접속)
3. [사용 방법](#3-사용-방법)
   1. [MiniRacer](#31-miniracer)
   2. [DeepRacer simulator](#32-deepracer-simulator)
4. [Trouble shooting](#4-trouble-shooting)
5. [Reference](#5-reference)

---

## 1. 설치 방법

Linux(Ubuntu 22.04 이상 권장)와 Windows(WSL2)에서 설치합니다. 순서는 필요한 것 설치, 실행 권한 부여, 시작입니다.

### 1.1 설치 순서

**(1) 필요한 것 설치**

```bash
git clone https://github.com/BHwi/deepracer-dashboard.git
cd deepracer-dashboard

# Docker Engine 과 Compose 플러그인
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER        # sudo 없이 docker 를 쓰기 위한 설정
```

`usermod` 를 실행했다면 로그아웃했다가 다시 로그인합니다(또는 `newgrp docker`). 그래야 `docker` 그룹이 적용됩니다.

```bash
# 파이썬 패키지 (numpy, matplotlib). MiniRacer 를 Docker 없이 호스트에서 직접 실행할 때만 필요하므로, ./drtrainer up 만 쓴다면 생략해도 됩니다.
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```

**(2) 실행 권한 부여**

```bash
chmod +x drtrainer
```

**(3) 시작**

```bash
./drtrainer up
```

`./drtrainer up` 은 이미지 빌드, DRfC 초기화, 대시보드 시작을 순서대로 합니다. 처음 실행은 이미지 빌드와 시뮬레이터 이미지(수 GB) 다운로드 때문에 수 분에서 십수 분 걸립니다. 진행 상황은 `./drtrainer logs -f` 로 봅니다. GPU 가 없으면 `./drtrainer up --arch cpu` 로 시작합니다(기본은 자동 선택이며, 호스트 Docker 에 nvidia 런타임이 있으면 GPU 모드). 끝나면 [2. localhost 접속](#2-localhost-접속)으로 이동합니다.

**`./drtrainer up` 에 필요한 것**

| 항목 | 용도 | 설치 |
| --- | --- | --- |
| Docker Engine | 대시보드 컨테이너, 시뮬레이터, 저장소(minio) 실행 | `sudo apt install docker.io` |
| Docker Compose 플러그인 (`docker compose`) | `./drtrainer up` 이 대시보드 컨테이너를 만들 때 사용 | `sudo apt install docker-compose-v2` |
| `docker` 그룹 권한 | sudo 없이 `docker` 실행 | `sudo usermod -aG docker $USER` 후 다시 로그인 |
| `bash`, `curl`, `tar`, `awk` 등 | `drtrainer` 스크립트가 사용 | Ubuntu 에 기본 설치되어 있음 |
| 인터넷 | 이미지 빌드(`ubuntu:24.04`, DRfC 내려받기), 시뮬레이터 이미지 다운로드 | |
| NVIDIA 드라이버, NVIDIA Container Toolkit | GPU 모드일 때만. 호스트에 설치해야 함 | [GPU](#gpu) 참고 |
| Windows | WSL2 안에 설치한 Docker Engine. Docker Desktop 에서는 동작하지 않을 수 있음 | 위와 같은 방법으로 WSL2 안에서 설치 |

- 사양은 DRfC 설치 문서의 로컬 기준으로 CPU 4코어(8 vCPU) 이상, GPU 모드면 GPU 메모리 8GB 이상(시뮬레이터 워커마다 약 1GB 추가), 시스템 RAM 과 GPU 메모리의 합 32GB 이상입니다. 디스크는 시뮬레이터 이미지가 수 GB 이므로 여유 공간이 필요합니다(DRfC 문서는 클라우드 VM 기준으로 OS 디스크 최소 30GB, 권장 40GB 를 안내합니다).
- 파이썬, numpy, matplotlib, AWS CLI, jq, DRfC 는 이미지 안에 설치되므로 호스트에 필요하지 않습니다. `requirements.txt` 는 위의 파이썬 패키지 단계에서만 쓰입니다.
- Docker 와 NVIDIA 드라이버는 pip 로 설치할 수 없으므로 `requirements.txt` 에는 들어 있지 않습니다.

### 1.2 Docker 없이 실행 (비권장)

- MiniRacer 만 쓰는 경우: 위의 파이썬 패키지 단계를 마친 뒤 `python3 dashboard.py` 를 실행하면 브라우저가 열립니다.
- DeepRacer simulator 를 Docker 이미지 없이 호스트에 직접 설치하는 경우(Ubuntu): `scripts/setup_drfc.sh` 를 사용합니다. [직접 설치 (Ubuntu)](#직접-설치-ubuntu)를 참고합니다.

### 1.3 저장소 구조

| 경로 | 내용 |
| --- | --- |
| `miniracer/`, `train.py`, `play.py`, `analyze.py`, `compare.py`, `fetch_tracks.py` | MiniRacer (2D 시뮬레이터, NumPy PPO, 명령줄 도구) |
| `custom_files/`, `examples/rewards/`, `tracks/`, `runs/` | 기본 설정, 보상함수 예제, 트랙, MiniRacer 실험 결과 |
| `trainer/` | DeepRacer simulator 연동 (실험 관리, 지표, 영상 중계, S3 클라이언트) |
| `Dockerfile`, `docker-compose.yml`, `docker/`, `drtrainer` | Docker 실행 구성 |
| `scripts/` | Ubuntu 설치 스크립트(`setup_drfc.sh`), GPU 진단(`gpu_check.sh`) |
| `tools/mock_drfc/` | 시험용 가짜 DRfC. Docker 와 GPU 없이 화면만 확인할 때 사용 |
| `dashboard.py`, `web/` | 대시보드 서버와 화면 (외부 라이브러리 없음) |

---

## 2. localhost 접속

`./drtrainer up` 이 끝나면 브라우저에서 아래 주소를 엽니다.

```text
http://localhost:8765
```

`http://127.0.0.1:8765` 도 같은 주소입니다. `python3 dashboard.py` 로 직접 실행한 경우에는 브라우저가 자동으로 열립니다.

**접속 확인**

1. 위쪽에 `MiniRacer`, `DeepRacer` 탭이 있는 대시보드가 열리면 정상입니다.
2. `DeepRacer` 탭의 `환경 점검` 에서 모든 항목이 `정상` 인지 확인합니다.
3. 열리지 않으면 아래 명령으로 상태와 로그를 확인합니다. 원인별 해결은 [4. Trouble shooting](#4-trouble-shooting)에 있습니다.

```bash
./drtrainer status       # 컨테이너 상태, DRfC 폴더, 모드, 주소
./drtrainer logs -f      # 시작 로그
```

**포트 변경**

```bash
./drtrainer up --port 9000        # http://localhost:9000
python3 dashboard.py --port 9000  # Docker 없이 실행할 때
```

**원격 서버에 설치한 경우.** 대시보드는 `Host` 가 `localhost` 또는 `127.0.0.1` 인 요청만 처리하고(다른 주소는 403), 인증이 없습니다. 서버에서 `--host 0.0.0.0` 으로 열지 말고 SSH 터널을 사용합니다.

```bash
ssh -L 8765:127.0.0.1:8765 사용자@서버      # 이 터미널을 열어 둔 채로
# 내 컴퓨터의 브라우저에서 http://localhost:8765
```

---

## 3. 사용 방법

위쪽의 `MiniRacer`, `DeepRacer` 탭으로 전환합니다. 보상함수 전략은 MiniRacer 에서 먼저 시험하고, 정한 전략은 DeepRacer simulator 에서 실제 시뮬레이터로 학습·평가합니다. 두 구성 요소의 보상함수 형식이 같아서, MiniRacer 실험을 DeepRacer `새 실험` 으로 가져올 수 있습니다.

### 3.1 MiniRacer

시작: Docker 로 설치했다면 [2. localhost 접속](#2-localhost-접속)의 주소에서 이미 열려 있습니다. Docker 없이는 `python3 dashboard.py` 를 실행합니다.

#### 개요와 DeepRacer 와의 차이

DeepRacer 와 같은 구조(보상함수, PPO 강화학습, 평가 주행)를 2D 시뮬레이터로 구현한 것입니다. 보상함수 전략을 빠르게 시험하고 비교하는 용도입니다. 기본 설정(1000 에피소드)이 화면 없이 약 1.5분 걸립니다.

DeepRacer 와의 차이는 다음과 같습니다.

| 항목 | DeepRacer | MiniRacer |
| --- | --- | --- |
| 알고리즘 | clipped PPO, 이산 행동 | 같음 (NumPy 로 구현) |
| 설정 파일 | `reward_function.py`, `model_metadata.json`, `hyperparameters.json` | 같은 이름, `custom_files/` 폴더 |
| 보상함수 입력 | `params` 딕셔너리 | 같은 키 이름 (DeepRacer 콘솔에 그대로 붙여 넣을 수 있음) |
| 학습 단위 | iteration 단위로 정책 갱신 후 평가 | 같음 (`num_episodes_between_training`) |
| 하이퍼파라미터 | `lr`, `batch_size`, `num_epochs`, `beta_entropy`, `discount_factor`, `loss_type` | 같은 이름과 기본값 (DRfC 의 `defaults/hyperparameters.json` 과 대조. `term_cond_avg_score` 만 조기 종료를 막으려고 크게 설정) |
| 센서 | 카메라 이미지, CNN | 120도 레이 센서 13개, 작은 MLP |
| 속도·조향 입력 | 받지 않음 | 관측에 포함 (학습 속도를 높이기 위한 단순화) |
| 차량 | 실물 또는 Gazebo | 자전거 운동학 모델, 15 Hz |

센서가 다르므로 학습 속도, 안정성, 랩타임의 절대 수치는 DeepRacer 와 맞지 않습니다. 전략의 상대 비교에만 사용합니다.

#### 대시보드 사용 순서 (MiniRacer)

1. 왼쪽 MiniRacer 그룹의 `새 실험`에서 이름, 트랙, 행동 공간(조향각 × 속도), 학습 설정, 보상함수 코드를 입력합니다. `예제에서 불러오기`로 예제 보상함수를 채울 수 있습니다.
2. `학습 시작`을 누릅니다. 학습 중에도 `결과` 화면이 갱신됩니다.
3. `결과`에서 그래프나 슬라이더로 iteration 을 고르면, 그 시점의 평가 주행이 지도에서 재생됩니다.
4. 필요하면 `주행 시험`(다른 트랙 또는 외란을 주고 주행), `비교`(왼쪽 목록에서 체크한 실험을 겹쳐 비교)를 사용합니다.

| 메뉴 | 내용 |
| --- | --- |
| 새 실험 | 실험 설정 입력, 학습 시작 |
| 결과 | 실시간 그래프, iteration 별 평가 주행 재생, 보상 분포 지도, 속도·조향 선택 비율, iteration 별 기록표, 학습 로그 |
| 주행 시험 | 학습된 모델을 다른 트랙에서 또는 외란을 주고 주행 |
| 비교 | 체크한 실험 겹쳐 비교 |
| 실험 정리 | 실험 기록(`runs/<이름>/`)을 여러 개 선택해 삭제. 확인란에 `삭제` 입력 필요 |

- 학습은 별도 프로세스(`train.py`)로 실행되므로 대시보드를 닫아도 계속됩니다. 터미널에서 실행한 학습도 `runs/` 에 저장되어 대시보드에 나타납니다.
- 보상함수에 오류가 있으면 몇 번째 줄인지 화면에 표시됩니다.
- 학습 중지는 다음 iteration 이 끝난 뒤 멈추고, 그때까지의 모델과 기록을 저장합니다.
- 동시에 3개까지 학습합니다. 코어 수가 적으면 1~2개를 권합니다.

#### 명령줄

```bash
python3 train.py --name my_first                                   # 학습
python3 train.py --reward examples/rewards/01_center_line.py --name center
python3 train.py --track Oval_track --name oval_test               # 트랙 지정
python3 play.py --run my_first                                     # 학습된 모델 주행
python3 play.py --run my_first --track Oval_track                  # 다른 트랙에서 주행
python3 play.py --run my_first --noise 1.0                         # 외란을 주고 주행
python3 analyze.py --run my_first                                  # 분석 리포트(report.png) 재생성
python3 compare.py center progress                                 # 실험 비교
```

| 명령 | 옵션 |
| --- | --- |
| `train.py` | `--name` 실험 이름 · `--track` 트랙 이름 또는 `.npy` 경로 · `--custom-files` 설정 폴더 · `--reward`, `--hp`, `--metadata` 파일 개별 지정 · `--render-every N` N iteration 마다 화면 표시 · `--no-render` · `--no-rays` · `--trace-every N` 스텝별 기록 저장 간격 · `--noise` · `--seed` · `--max-minutes` · `--list-tracks` |
| `play.py` | `--run`(필수) · `--model best\|final\|파일` · `--track` · `--laps` · `--noise` · `--no-render` · `--no-rays` · `--seed` |
| `analyze.py` | `--run`(필수) |
| `compare.py` | `실험이름...` · `--out` |
| `dashboard.py` | `--port` · `--no-browser` · `--host`(외부 접속, 아래 주의 참고) · `--strict-port` |
| `fetch_tracks.py` | `이름...`(없으면 추천 트랙) · `--list` |

`train.py` 의 주행 화면은 `--render-every`(기본 5) iteration 마다 첫 학습 에피소드를 보여 줍니다. 스페이스바로 일시정지합니다. 화면을 켜면 그만큼 느려지므로 서버에서는 `--no-render` 를 사용합니다.

#### 설정 파일

`custom_files/` 의 3개 파일입니다. `--reward`, `--hp`, `--metadata` 로 개별 지정할 수도 있습니다.

| 파일 | 내용 |
| --- | --- |
| `reward_function.py` | 기본 보상함수. 새 실험을 만들 때 처음 채워지는 코드입니다. `params` 의 키 이름이 공식 문서와 같습니다. 사용 가능한 라이브러리: `math`, `random`, `numpy`, `scipy`, `shapely` |
| `model_metadata.json` | 행동 공간 (조향각 × 속도 조합) |
| `hyperparameters.json` | PPO 하이퍼파라미터. `term_cond_max_episodes` 가 학습 길이 |

실험 결과는 `runs/<이름>/` 에 저장되며, 그 실험에 쓴 3개 파일의 사본도 함께 저장됩니다.

#### 보상함수 예제 (`examples/rewards/`)

| 파일 | 내용 |
| --- | --- |
| `01_center_line` | 중앙선만 보상. 가장 느린 속도 쪽으로 쏠려 평가 랩타임이 약 27초 |
| `02_speed_only` | 매 스텝 속도만큼 보상. 총 보상이 이동 거리에 비례하므로 빨리 달릴 이유가 생기지 않음 |
| `03_balanced` | 중앙선, 속도, 조향 억제의 가중합 |
| `04_heading_alignment` | 트랙 진행 방향과 차의 방향 정렬 (`waypoints`, `closest_waypoints`, `heading`) |
| `05_progress_steps` | 진행률과 걸린 스텝으로 빠른 주행을 직접 보상. `TOTAL_NUM_STEPS` 는 트랙마다 조정 필요 |
| `06_speed_squared` | 속도의 제곱에 비례. 같은 거리를 빨리 갈수록 총 보상이 커짐 |

각 파일 맨 위 주석에 측정 결과와 이유가 적혀 있습니다.

#### 트랙

```bash
python3 train.py --list-tracks
python3 fetch_tracks.py              # 추천 트랙 내려받기 (인터넷 필요)
```

기본 포함 트랙은 `reInvent2019_track` 하나입니다. 닫힌 루프가 아닌 트랙은 쓸 수 없고, 약 50m 이상의 긴 트랙은 학습이 오래 걸립니다. 트랙 출처의 라이선스는 [5. Reference](#5-reference)를 참고합니다.

#### 실험 비교

```bash
python3 train.py --reward examples/rewards/01_center_line.py --name center
python3 train.py --reward examples/rewards/05_progress_steps.py --name progress
python3 compare.py center progress
```

보상함수가 다르면 보상 값은 비교할 수 없으므로 진행률, 완주율, 랩타임으로 비교합니다. 결과는 시드 1개의 값입니다. 차이가 작으면 `--seed` 를 바꿔 반복합니다.

#### 결과 파일 (`runs/<이름>/`)

| 파일 | 내용 |
| --- | --- |
| `metrics.csv` | iteration 별 보상, 진행률, 완주율, 랩타임, 엔트로피, KL, 손실 |
| `episodes.csv` | 에피소드별 결과 (종료 사유 포함) |
| `trace_train.csv`, `trace_eval.csv` | 스텝별 기록 (위치, 속도, 조향, 행동, 보상, 진행률). `trace_train.csv` 는 `--trace-every`(기본 3) iteration 마다만 저장 |
| `model_best.npz`, `model_final.npz` | best 는 평가 진행률이 가장 높은(같으면 랩타임이 짧은) iteration |
| `status.json`, `train.log` | 대시보드가 읽는 학습 상태와 로그 |
| `report.png` | 학습 곡선, 엔트로피·KL·손실, 보상 히트맵, 구간별 속도·조향, 행동 선택 비율, 종료 사유 |

#### 참고 사항

- 하이퍼파라미터 `e_greedy_value`, `epsilon_steps`, `stack_size`, `exploration_type` 은 호환용 이름이며 사용되지 않습니다.
- 보상 크기는 내부에서 정규화됩니다. 보상함수의 절대 크기를 바꿔도 학습 속도는 달라지지 않습니다.
- 평가 주행은 확률이 가장 높은 행동만 고르고, 출발 위치를 트랙에 고르게 나눠 `min_eval_trials` 번 달립니다.
- 엔트로피는 보통 2.7(15개 행동이 균등)에서 시작해 내려갑니다. DeepRacer 의 entropy 와 해석은 같지만 수치는 비교할 수 없습니다.

#### 동작 확인

학습을 한 번 실행해서 동작을 확인합니다.

```bash
python3 train.py --name check --no-render
```

1. 약 1.5분 뒤(CPU 에 따라 다름) 마지막 줄에 `종료 사유: term_cond_max_episodes | 50 iterations, 1000 episodes, ...` 가 출력되면 정상 종료입니다.
2. `runs/check/` 에 `metrics.csv`, `episodes.csv`, `model_best.npz`, `model_final.npz`, `report.png`, `status.json`, `train.log`, `trace_*.csv` 가 생겼는지 확인합니다.
3. `python3 dashboard.py` 의 `결과` 화면에서 `check` 실험이 `완료` 로 보이고, 평가 진행률이 100% 로 올라가는지 확인합니다. 개발 환경 측정에서는 아래 두 예제 모두 iteration 4 에서 처음 완주(평가 진행률 100%)했습니다.
4. `python3 play.py --run check` 로 학습된 모델이 트랙을 도는지 확인합니다.

참고값입니다. 개발 환경에서 시드 1, 기본 설정(1000 에피소드)으로 측정한 결과이며 환경에 따라 달라집니다.

| 보상함수 | 평가 진행률 | 최고 랩타임 |
| --- | --- | --- |
| `06_speed_squared` | 100% | 9.27초 |
| `01_center_line` | 100% | 25.27초 |

두 보상함수 모두 완주하지만 랩타임이 크게 다릅니다. 두 실험을 `비교` 화면에서 겹쳐 볼 때 이 차이가 나오면 정상입니다.

![MiniRacer 결과 화면](docs/img/miniracer-result.png)

![MiniRacer 실험 비교](docs/img/miniracer-compare.png)

### 3.2 DeepRacer simulator

시작: Docker 로 설치했다면 [2. localhost 접속](#2-localhost-접속)의 주소에서 `DeepRacer` 탭을 엽니다.

#### 개요와 구성

DRfC(DeepRacer-for-Cloud)의 실제 DeepRacer 시뮬레이터로 모델을 학습·평가하고, 차량에 올릴 파일을 만드는 과정을 대시보드에서 관리합니다. DRfC 의 공식 명령(`dr-upload-custom-files`, `dr-start-training -q`, `dr-stop-training`, `dr-start-evaluation -q`, `dr-create-car-zip` 등)을 호출하는 방식이며, 하나의 실험은 DRfC 의 `experiments/<이름>/` 폴더 하나에 대응합니다.

Docker 로 실행하는 경우 구성은 다음과 같습니다.

```mermaid
flowchart TB
    Browser["브라우저<br/>127.0.0.1:8765"] --> Dash
    subgraph Host["호스트 컴퓨터"]
        subgraph Trainer["drtrainer 컨테이너 (호스트 네트워크)"]
            Dash["대시보드"]
            Scripts["DRfC 스크립트<br/>(고정된 버전)"]
        end
        Engine[("Docker Engine")]
        Sibling["시뮬레이터, 학습, minio<br/>(DRfC 가 띄우는 컨테이너)"]
    end
    Scripts -->|"docker.sock"| Engine
    Engine --> Sibling
    Dash -.->|"S3 API, 영상 중계"| Sibling
```

DRfC 는 호스트의 Docker 로 시뮬레이터, minio 같은 다른 컨테이너를 띄웁니다. 그래서 drtrainer 컨테이너는 다음과 같이 구성합니다.

- `docker.sock` 을 연결해 호스트의 Docker 를 사용합니다.
- DRfC 폴더와 `~/.aws` 를 호스트와 같은 절대경로로 마운트합니다. 형제 컨테이너의 bind mount 는 호스트 기준 경로로 해석되기 때문입니다.
- 호스트 네트워크를 사용합니다. DRfC 가 minio 주소를 `localhost:9000` 으로 고정하고 있습니다. 대시보드는 호스트의 `127.0.0.1` 에만 열립니다.
- 호스트와 같은 uid/gid 로 실행해서 파일 소유권을 유지합니다.

#### 대시보드 사용 순서 (DeepRacer)

위쪽 `DeepRacer` 탭을 선택합니다.

1. `환경 점검`: 모든 항목이 `정상` 인지 확인합니다. 문제 항목에는 원인과 해결 방법이 표시되고, 일부는 버튼(`minio 이미지 만들기`, `swarm 만들기`)으로 해결합니다.
2. `새 실험`: 이름(모델 이름), 트랙, 알고리즘(PPO/SAC), 행동 공간, 하이퍼파라미터, 보상함수를 입력합니다. MiniRacer 실험 가져오기, 기존 실험 복제, 학습한 모델에서 이어서 학습을 선택할 수 있습니다.
3. `학습`: 시작과 중지, 컨테이너 상태, 실시간 영상, iteration 별 그래프, 로그를 확인합니다. 같은 이름의 모델이 이미 있으면 시작을 막습니다. 덮어쓰기(`-w`)는 별도로 체크하고 한 번 더 확인해야 합니다.
4. `평가`: `평가할 모델`(모델이 있는 실험의 목록. 왼쪽 실험 목록에서 고른 것과 같고, 모델은 실험 하나에 하나)을 고르고, 체크포인트(`last` 또는 `best`), 평가 이름, 영상 저장 옵션을 지정해 시작합니다. 평가 기록에서 결과표와 영상을 확인합니다.
5. `차량용 내보내기`: `agent/model.pb` 와 `model_metadata.json` 이 든 `tar.gz` 를 만들어 내려받습니다.
6. `실험 정리`: 설정, 모델, 차량용 파일을 선택해 여러 실험을 한 번에 삭제합니다.
7. `관리`: DRfC 의 `dr-*` 명령(중지, 정리, 설정 다시 읽기 등)을 버튼으로 실행합니다.

#### drtrainer 명령과 데이터 위치

| 명령 | 내용 |
| --- | --- |
| `up [--arch gpu\|cpu\|auto] [--drfc-dir 경로] [--port 번호] [--rebuild]` | 시작. 이미지가 이미 있으면 다시 빌드하지 않음 |
| `down`, `restart`, `status`, `logs [-f]`, `shell` | 관리 |
| `init [--arch gpu\|cpu] [--force]` | DRfC 초기화 재시도. `system.env` 가 있으면 건너뛰고, `--force` 는 백업 후 처음부터 |
| `gpu` | GPU 진단 (아무것도 바꾸지 않음) |
| `export [--with-simapp] [폴더]` | 다른 컴퓨터로 옮길 묶음 생성 (`docker save`). `--with-simapp` 은 시뮬레이터 이미지 포함 |
| `import [폴더]` | 옮겨 온 묶음에서 이미지 로드 |

데이터는 DRfC 폴더(기본 `~/deepracer-for-cloud`: 모델, 설정, 실험)와 `./data`(MiniRacer 기록, 대시보드 상태)에 남습니다. 컨테이너를 지우거나 이미지를 다시 빌드해도 유지됩니다. 다른 컴퓨터로 옮길 때는 두 폴더를 함께 복사하고, 같은 절대경로가 되도록 `--drfc-dir` 을 지정합니다. DRfC 버전은 이미지에 고정되어 있고, `docker compose build --build-arg DRFC_COMMIT=<커밋>` 으로 바꿉니다.

#### 직접 설치 (Ubuntu)

```bash
bash scripts/setup_drfc.sh --dry-run     # 미리보기. 아무것도 변경하지 않음
sudo bash scripts/setup_drfc.sh          # 설치
python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt     # 파이썬 패키지 (Ubuntu 24.04 는 가상환경 필요)
python3 dashboard.py                     # 환경 점검에서 DRfC 폴더 지정 후 '점검 다시 실행'
```

스크립트는 Docker, (GPU 모드면) NVIDIA 컨테이너 도구, AWS CLI, DRfC 내려받기, 파이썬 환경, `init.sh` 초기화를 처리합니다.

| 옵션 | 내용 |
| --- | --- |
| `--arch gpu\|cpu` | 모드 지정 (기본: 자동 감지) |
| `--install-driver` | GPU 가 있는데 드라이버가 없을 때 NVIDIA 드라이버 설치. 설치 후 재부팅 필요 |
| `--drfc-dir 경로` | DRfC 설치 위치 (기본 `~/deepracer-for-cloud`) |
| `--user 이름` | DRfC 를 사용할 일반 사용자 (기본: sudo 를 실행한 사용자) |

- 여러 번 실행해도 됩니다. 완료된 단계는 건너뛰고 중간에 멈춘 곳부터 이어서 합니다.
- 시스템 전체 `apt upgrade`, NVIDIA 드라이버 설치, 재부팅은 기본으로 하지 않습니다.
- Ubuntu 20.04 처럼 DRfC 설치 스크립트의 지원 목록(22.04 이상)에 없는 버전은 경고만 하고 진행합니다. WSL2 에서 Docker Desktop 을 쓰면 WSL 안에 Docker 서버가 없으므로 NVIDIA 설정 단계를 건너뜁니다. WSL2 에서 `init.sh` 가 GPU 를 감지하지 못하는 DRfC 의 알려진 문제는 GPU 이미지로 설정해서 처리합니다.
- `system.env` 가 이미 있으면 `init.sh` 를 다시 실행하지 않습니다. `init.sh` 는 `system.env`, `run.env`, `custom_files/` 를 덮어쓰기 때문입니다. 단, `<DOCKER_STYLE>` 같은 자리표시자가 남은 `system.env` 는 초기화가 중간에 실패한 것으로 보고 백업한 뒤 처음부터 다시 실행합니다.
- `/tmp/sagemaker` 는 재부팅하면 사라지는데 DRfC 가 없으면 `sudo` 로 만들려 하므로, 부팅 때마다 만들어지도록 설정합니다.

#### 실시간 영상

학습이나 평가가 실행 중이면 `학습`, `평가` 화면 위쪽에 시뮬레이터 영상이 표시됩니다. 시뮬레이터가 내보내는 ROS 영상 스트림(MJPEG)을 대시보드가 받아서 브라우저로 중계하므로, 브라우저는 대시보드 포트(8765)만 접속하면 됩니다. 시뮬레이터가 시작되는 데 1분 정도 걸리며, 그동안 3초마다 재연결을 시도합니다.

- 영상 종류는 시뮬레이터가 실제로 내보내고 있는 목록을 읽어서 표시합니다. 차를 따라가는 시점(속도·진행률 표시 포함), 차 안의 카메라, 표시 없는 시점, 트랙 위에서 본 시점이 있으며, 마지막 것은 `system.env` 에 `DR_CAMERA_SUB_ENABLE=True` 가 있어야 나옵니다.
- 포트는 DRfC 의 `docs/video.md` 를 따릅니다. swarm 은 학습 `8080 + 실행번호`, 평가 `8180 + 실행번호`, compose 는 워커마다 `8080~8089` 입니다. 중계는 이 컴퓨터(`127.0.0.1`)의 이 포트 범위로만 연결합니다. `연결: 직접 접속` 으로 바꾸면 브라우저가 시뮬레이터 포트에 직접 접속합니다.
- 워커가 여러 개이면 `DRfC 뷰어` 버튼으로 DRfC 자체 뷰어(`http://localhost:8100`)를 시작할 수 있습니다.
- 영상이 나오지 않으면 `영상 진단` 버튼을 사용합니다. 시뮬레이터 컨테이너와 공개 포트, 대시보드의 실행 인식 여부, 영상 서버 응답, 실제 프레임 수신, 시뮬레이터 로그, 브라우저에서의 접속 가능 여부를 순서대로 점검하고 추정 원인을 표시합니다. 결과는 복사할 수 있습니다. 실행 중으로 인식되지 않을 때는 `그래도 연결 시도` 로 직접 연결할 수 있습니다.

#### 평가 이름과 주행 영상

- 평가 시작 시 이름을 붙이거나, 끝난 평가에 나중에 붙일 수 있습니다. 이름은 `experiments/<실험>/evaluations.json` 에 저장됩니다. DRfC 는 평가 이름을 저장하지 않고 지표 파일을 `evaluation-<시각>.json` 으로만 남기므로, 대시보드가 시작 시점의 파일 목록을 기억해 두었다가 그 뒤에 생긴 파일에 이름을 연결합니다.
- 주행 영상은 `주행 영상 저장 (MP4)` 을 켠 평가에만 생깁니다. DRfC 의 기본값은 저장 안 함(`DR_EVAL_SAVE_MP4=False`)입니다. 켜면 평가 종료 후 영상 파일이 저장소의 `<모델>/mp4/` 에 만들어집니다. 켜지 않고 한 평가의 영상은 복구할 수 없으므로 다시 평가해야 합니다.
- 영상 파일 이름 규칙은 시뮬레이터가 정하며 알려져 있지 않습니다. 그래서 파일 이름이 아니라 생성 시각으로 평가와 짝지어서, 영상이 만들어진 시각과 가장 가까운 시각에 끝난 평가의 영상으로 봅니다. 10분 넘게 떨어져 있으면 짝짓지 않고 별도 목록에 둡니다.
- 영상은 저장소에서 읽어 브라우저로 전달하며 재생 위치를 옮길 수 있습니다. 브라우저는 보통 H.264 만 재생하므로, 다른 형식이면 안내가 표시되고 다운로드해서 별도 플레이어로 엽니다.
- `복사본으로 평가` 를 켜면 결과와 영상이 복사된 모델 폴더에 저장되어 평가 기록에 나타나지 않습니다.

평가 지표 파일에는 진행률이 100% 인데 `episode_status` 가 `In progress` 로 남는 행이 있습니다. 시뮬레이터가 최종 상태를 기록하지 않은 경우이며 시간이 지나도 갱신되지 않습니다. 대시보드는 다음과 같이 표시합니다.

| 지표 파일의 상태 | 화면 표시 |
| --- | --- |
| 진행률 100% 의 `In progress` | 완주로 표시하고 `*` 를 붙임. 마지막으로 기록된 시점의 시간을 랩타임으로 사용하므로 실제와 조금 다를 수 있음 |
| 진행률 100% 미만의 `In progress` | 평가 실행 중이면 `진행 중`, 끝난 뒤에는 `미완료` |
| 같은 시도의 행이 여러 개 | 마지막 행만 사용 |

#### 중지

학습이나 평가가 실행 중이면 DeepRacer 의 모든 화면 위쪽에 `중지`, `강제 중지` 버튼이 있는 배너가 표시됩니다. 오른쪽 위 상태 표시의 `중지` 를 누르면 배너로 이동합니다.

- 어떤 실험인지, 학습인지 평가인지는 컨테이너에서 알아냅니다. DRfC 스택 이름(학습 `deepracer-<번호>`, 평가 `deepracer-eval-<번호>`)과 시뮬레이터 컨테이너의 `MODEL_S3_PREFIX` 환경변수를 읽습니다. 알아내지 못하면 `(외부에서 시작됨)` 으로 표시하며, 이때도 중지할 수 있습니다.
- 종류를 알 수 없으면 `dr-stop-evaluation` 과 `dr-stop-training` 을 모두 실행한 뒤 남은 `deepracer-*` 스택과 학습·시뮬레이터 컨테이너를 직접 정리합니다.
- `강제 중지` 는 DRfC 중지 명령이 실패하거나 멈추지 않을 때 사용합니다. `deepracer-*` 스택과 `algo-*`, 시뮬레이터, 코치 컨테이너를 직접 삭제합니다. 저장된 모델과 minio 는 그대로입니다.
- 시작 직후 컨테이너가 `docker ps` 에 나타나기 전이라도 swarm 스택에 준비 중인 작업이 있으면 최대 15분까지 `시작 중` 으로 유지합니다.

#### 관리 (dr-* 명령 버튼)

`관리` 메뉴는 DRfC 의 `dr-*` 명령과 상태 조회 명령을 버튼으로 실행합니다. 터미널에서 DRfC 를 활성화하지 않아도 되고, 결과는 화면의 작업 상자에 표시됩니다. 명령은 `실행할 실험` 에서 고른 실험의 설정(`run.env`)으로 실행됩니다. 학습과 평가 시작은 각 화면에서 합니다.

| 그룹 | 명령 | 위험도 |
| --- | --- | --- |
| 상태 확인 | 스택과 작업 상태(`docker stack ls`, `docker stack ps`), 컨테이너 목록, `dr-summary`, `dr-find-sagemaker`, `dr-find-robomaker` | 읽기 |
| 중지와 정리 | `dr-stop-training`, `dr-stop-evaluation`, 이전 실행 기록(스택) 지우기 | 변경 |
| 중지와 정리 | `dr-stop-all`, 강제 중지 | 주의 |
| 설정과 파일 | `dr-update-env`, `dr-upload-custom-files` | 변경 |
| 설정과 파일 | `dr-download-custom-files` (같은 이름의 로컬 파일이 바뀜) | 주의 |
| 부가 서비스 | `dr-start-viewer`, `dr-update-viewer`, `dr-stop-viewer`, `dr-start-loganalysis`, `dr-stop-loganalysis`, `dr-start-metrics`, `dr-stop-metrics` | 변경 |

- `주의` 명령은 확인 창을 거치며, 서버도 확인 없이는 실행을 거부합니다.
- `dr-stop-all` 은 학습, 평가, 뷰어, 로그 분석 스택과 minio(`s3` 스택)까지 내립니다. 저장된 모델은 그대로이고, minio 는 다음 학습·평가를 시작할 때 DRfC 의 `bin/activate.sh` 가 `s3` 스택을 다시 배포해서 올라옵니다.
- `이전 실행 기록(스택) 지우기` 는 종료된 작업의 기록만 남은 `deepracer-*` 스택을 `docker stack rm` 으로 지웁니다. 실행 중이거나 준비 중인 작업이 있는 스택은 건드리지 않습니다.
- 맨 아래 `직접 입력` 은 DRfC 가 정의한 `dr-*` 명령 중 허용된 것만 실행합니다. 명령 이름은 `bin/scripts_wrapper.sh` 에 정의된 함수와 대조하고, 인자는 영문, 숫자, `_ . / = : @ , + -` 만 허용하며 8개 이하입니다. `;`, `|`, `$()`, 백틱 같은 셸 문법 문자와 줄바꿈은 거부합니다. 실행 전에 항상 확인 창이 뜹니다.
- 직접 입력에서 막는 명령: 끝나지 않고 로그를 따라가는 명령(`dr-logs-*`), 브라우저를 여는 명령(`dr-view-stream`), 전용 화면이 있는 시작 명령(`dr-start-training`, `dr-start-evaluation`), 실제 AWS 계정에 쓰는 명령(`dr-upload-model`, `dr-upload-car-zip`, `dr-download-model`, `dr-set-upload-model`, `dr-increment-upload-model`, `dr-list-aws-models`). 이들은 터미널에서 실행합니다.

#### GPU

- `환경 점검` 의 GPU / CPU 선택은 시뮬레이터 이미지 태그(`-gpu`, `-cpu`)와 `system.env` 의 `DR_SIMAPP_VERSION` 을 바꾸고 이미지를 내려받습니다. `system.env` 는 백업 후 수정하며 모델과 실험은 유지됩니다.
- `GPU 컨테이너 시험` 은 `docker run --gpus all ... nvidia-smi -L` 로 컨테이너 안에서 GPU 가 보이는지 확인합니다. 이 결과가 가장 확실한 기준입니다.
- GPU 가 여러 개면 `DR_SAGEMAKER_CUDA_DEVICES` 로, 시뮬레이터 워커 수는 `DR_WORKERS` 로 지정합니다.
- DRfC 문서의 로컬 권장 사양은 GPU 메모리 8GB 이상(워커마다 약 1GB 추가), CPU 4코어(8스레드) 이상, RAM 과 GPU 메모리 합 32GB 이상입니다.
- 처음 설치할 때의 모드는 호스트 Docker 에 nvidia 런타임이 있는지로 한 번 정해집니다. 그때 GPU 도구가 없었다면 CPU 모드로 설정되어 있으므로 아래 순서로 확인합니다.

```bash
./drtrainer gpu        # 또는 bash scripts/gpu_check.sh
```

| 막힌 곳 | 진단 결과 | 조치 |
| --- | --- | --- |
| 호스트 NVIDIA 드라이버 | `nvidia-smi` 가 안 보임 | WSL2: Windows 에 최신 NVIDIA 드라이버를 설치하고 PowerShell 에서 `wsl --shutdown` (WSL 안에는 설치하지 않음). Linux: `sudo bash scripts/setup_drfc.sh --arch gpu --install-driver` (재부팅 필요) |
| Docker 의 GPU 연결 | 드라이버는 정상인데 `could not select device driver` | `sudo bash scripts/setup_drfc.sh --arch gpu`. 완료된 단계는 건너뛰고 NVIDIA 도구 설치와 Docker 설정만 수행. Docker 가 재시작되어 실행 중인 컨테이너가 잠시 멈춤 |
| DRfC 모드 | GPU 컨테이너는 되는데 DRfC 가 `-cpu` 이미지 | `환경 점검` 에서 GPU 를 선택하고 `이 모드로 전환` |

#### minio 이미지

MinIO 가 2026-09-11 경 Docker Hub 의 `minio/minio` 이미지를 삭제했습니다. DRfC 는 학습을 시작할 때 이 이미지로 저장소를 띄우므로, 이미지가 컴퓨터에 없으면 `No such image: minio/minio:latest` 로 학습을 시작할 수 없습니다. 이 프로젝트는 GitHub 릴리스에 남아 있는 마지막 공개 바이너리(`RELEASE.2025-09-07T16-13-09Z`)를 SHA-256 으로 검증해서 같은 형태의 이미지(`minio/minio:RELEASE.2025-09-07T16-13-09Z-local`)를 직접 빌드하고(`docker/minio/Dockerfile`), `system.env` 의 `DR_MINIO_IMAGE` 를 그 태그로 바꿉니다. 원본은 `system.env.bak-*` 로 백업합니다.

- 학습·평가 시작 전, `환경 점검` 의 `minio 이미지 만들기` 버튼, 설치 스크립트, Docker 진입점이 이미지 존재를 확인하고 없으면 만듭니다. 수동 실행: `bash docker/minio-local-image.sh <DRfC 폴더>`
- 이 minio 는 더 이상 유지보수되지 않으므로 이후의 보안 취약점은 수정되지 않습니다. DRfC 는 minio 포트(9000, 9001)를 모든 네트워크 인터페이스에 열기 때문에, 신뢰할 수 있는 네트워크에서만 사용하거나 방화벽으로 막습니다.
- 바이너리는 공개된 체크섬과 대조해 손상과 변조를 확인합니다. MinIO 서명(`.minisig`)은 검증하지 않습니다.
- MinIO 는 객체를 `<키>/xl.meta` 같은 디렉터리 구조로 저장하므로 파일로 읽을 수 없습니다. 대시보드는 지표, 평가 결과, 영상을 S3 API 로 읽습니다(`~/.aws/credentials` 의 minio 프로필 사용).

#### 실험 정리

`실험 정리` 에서 여러 실험을 한 번에 삭제합니다. 되돌릴 수 없으며, 확인란에 `삭제` 를 입력해야 버튼이 활성화됩니다.

- 삭제 항목은 설정(`experiments/<이름>/`), 학습한 모델과 지표(저장소), 차량용 파일과 시뮬레이터 로그 중에서 선택합니다. 모델은 minio 데이터 폴더를 직접 지우지 않고 S3 API(`aws s3 rm`)로 삭제합니다.
- 실행 중인 실험은 선택할 수 없고, 다른 실험이 이어서 학습하는 데 쓰는 모델은 삭제되지 않습니다.
- 같은 이름으로 다시 학습하려는데 "모델이 이미 있습니다" 오류가 나면, 여기서 그 실험의 모델을 삭제합니다.

#### 보안 관련 주의

- Docker 방식의 컨테이너는 `docker.sock` 을 사용하므로 호스트 root 와 같은 권한을 가집니다. 컨테이너 안에는 DRfC 가 사용하는 `sudo` 가 비밀번호 없이 동작하도록 되어 있습니다.
- 대시보드는 다른 웹사이트가 이 컴퓨터의 대시보드로 요청을 보내는 것(CSRF, DNS rebinding)을 막습니다. `Host` 가 `127.0.0.1`, `localhost`(또는 `--host` 로 지정한 주소)가 아니거나, `Origin` 이 같은 출처가 아니거나, `Sec-Fetch-Site: cross-site` 이거나, POST 가 `application/json` 이 아니면 거부합니다(403, 415). `--host 0.0.0.0` 이면 `Host` 검사는 하지 않지만 다른 출처의 요청은 계속 막습니다.
- 대시보드는 입력된 보상함수 코드를 실행합니다. `--host` 로 외부 접속을 열면 같은 네트워크의 누구나 그 컴퓨터에서 코드를 실행할 수 있습니다. 인증이 없으므로 신뢰할 수 있는 컴퓨터에서 본인만 사용하는 것을 전제로 합니다.

#### 알아둘 점

- 타임 트라이얼(카메라 한 대) 기준입니다. 물체 회피와 봇 대결은 `experiments/<이름>/run.env` 와 `model_metadata.json` 을 직접 수정해야 합니다.
- 트랙 이름은 시뮬레이터 이미지에 있는 이름이어야 합니다. 목록에 없는 이름도 입력할 수 있지만 틀리면 시뮬레이터 로그에 오류가 납니다.
- 지표 파일(`TrainingMetrics.json`)에는 iteration 번호가 없어서, 학습 에피소드를 `num_episodes_between_training` 개씩 묶어 환산합니다. 추정값입니다.
- DRfC 는 한 번에 하나(학습 또는 평가)만 실행합니다. `DR_RUN_ID` 로 여러 실험을 동시에 돌리는 구성은 지원하지 않습니다.
- DRfC 폴더 구조와 명령은 2026년 8월 기준 저장소를 읽고 맞췄습니다.

#### 동작 확인

##### 설치 후

1. `DeepRacer` 탭의 `환경 점검` 에서 모든 항목이 `정상` 인지 확인합니다. 확인 항목은 운영체제, Docker 설치·실행, Docker Compose, 필수 도구, DRfC 폴더, DRfC 초기화, 시뮬레이터 이미지, minio 이미지, minio 자격 증명, GPU, Docker 의 GPU 연결, 컴퓨터 사양입니다. `주의` 는 동작은 가능하지만 확인이 필요한 항목입니다. `필요` 는 해결해야 하는 항목입니다.
2. GPU 모드라면 `GPU 컨테이너 시험` 을 눌러 컨테이너 안에서 GPU 가 보이는지 확인합니다.
3. 터미널에서 확인할 때:

```bash
./drtrainer status       # 컨테이너 상태, DRfC 폴더, 모드, 주소
./drtrainer gpu          # GPU 진단
docker ps                # drtrainer 컨테이너가 healthy 인지
docker node ls           # swarm 이 활성인지
```

##### 학습

1. `새 실험` 에서 학습 길이를 작게(예: `term_cond_max_episodes` 100) 한 실험을 만들고 `학습` 에서 시작합니다.
2. 컨테이너 상태 표시에 저장소(minio), 학습(Sagemaker), 시뮬레이터(RoboMaker), 코치가 모두 표시되는지 확인합니다.
3. 첫 iteration 이 끝나면 보상, 진행률, 완주율 그래프와 iteration 별 기록표가 채워집니다. iteration 은 `num_episodes_between_training`(기본 20)개 에피소드마다 끝납니다.
4. 영상 패널에 시뮬레이터 영상이 나오는지 확인합니다. 나오지 않으면 `영상 진단` 을 실행합니다.
5. 학습 중에는 어느 DeepRacer 화면에서나 위쪽에 중지 배너가 표시됩니다.

##### 평가와 내보내기

1. `평가` 에서 이름과 `주행 영상 저장 (MP4)` 을 지정해 시작합니다.
2. 평가가 끝나면 평가 기록에 이름이 표시되고, 이름을 누르면 시도별 결과(진행률, 랩타임)가 나옵니다. 영상 저장을 켰다면 주행 영상 탭이 나타납니다.
3. `차량용 내보내기` 로 만든 `tar.gz` 안에 `agent/model.pb` 와 `model_metadata.json` 이 있는지 확인합니다.

##### 화면만 확인 (Docker, GPU 없이)

`tools/mock_drfc/` 는 시험용 가짜 DRfC 입니다. 실제 학습을 하지 않고 가짜 학습 곡선을 만듭니다. `환경 점검` 에서 DRfC 폴더를 이 폴더로 지정하면 `시험용 가짜 DRfC` 표시가 나오며, DeepRacer 화면의 흐름(실험 생성, 학습, 평가, 내보내기, 중지, 실험 정리)을 확인할 수 있습니다. 실제 DRfC 로 바꾸려면 폴더를 다시 지정합니다. 아래 화면은 이 가짜 DRfC 로 찍은 것이며 그래프 값과 GPU 정보는 가짜입니다.

![환경 점검 (시험용 가짜 DRfC)](docs/img/deepracer-env.png)

![학습 (시험용 가짜 DRfC)](docs/img/deepracer-train.png)

![평가 (시험용 가짜 DRfC)](docs/img/deepracer-eval.png)

![관리 (시험용 가짜 DRfC)](docs/img/deepracer-admin.png)

##### 확인 범위

| 영역 | 확인 방법 | 상태 |
| --- | --- | --- |
| MiniRacer 학습, 결과, 비교, 실험 정리 | 실제 학습 실행과 실제 브라우저 | 확인함 (Linux, Python 3.12) |
| DeepRacer 화면 전체 | 시험용 가짜 DRfC 와 실제 브라우저(Chromium) | 확인함 |
| minio 연동 (지표, 평가 결과, 영상 중계·재생 위치 이동, 삭제) | 실제 MinIO 바이너리(2022-10, 2025-09), 실제 MP4 파일 | 확인함 |
| 셸 스크립트, Dockerfile | shellcheck, hadolint, 가짜 docker 로 진입점·초기화·minio 이미지 스크립트 실행 | 확인함 |
| 실제 Docker, DRfC | 개발 중 실제 Ubuntu/WSL2 에서 이미지 실행, minio 이미지 생성, 학습·평가 결과 표시 확인 | 일부 확인. 전 과정을 자동 시험으로 검증하지는 않음 |
| 실시간 영상 | 가짜 영상 서버(ROS web_video_server 형식)로만 시험 | 실제 시뮬레이터 영상은 미확인 |
| 저장된 평가 영상(MP4) | 시각 기반 짝짓기와 재생을 가짜 파일로 시험 | 실제 파일 이름과 코덱은 미확인 |
| GPU 진단·설정 | 가짜 환경으로 여러 상황을 시뮬레이션 | 실제 NVIDIA 환경은 미확인 |
| Windows, macOS 에서 직접 실행 | 시험하지 않음 | 미확인 (Windows 는 WSL2 사용) |

---

## 4. Trouble shooting

### 4.1 `error: externally-managed-environment` (`pip3 install`)

Ubuntu 24.04 등은 시스템 파이썬에 pip 로 직접 설치하는 것을 막습니다(PEP 668). 가상환경을 만들어서 설치합니다.

```bash
sudo apt install -y python3-venv python3-pip     # "ensurepip is not available" 오류가 나올 때
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```

새 터미널을 열 때마다 `source .venv/bin/activate` 가 필요합니다. `pip3 install --break-system-packages -r requirements.txt` 로도 설치되지만 시스템 파이썬이 깨질 수 있어 권하지 않습니다. `./drtrainer up` 만 쓴다면 이 설치는 필요하지 않습니다.

### 4.2 `docker 명령이 없습니다`, `Docker 서버에 연결할 수 없습니다`

`./drtrainer` 가 시작하기 전에 Docker 를 점검하면서 출력하는 메시지입니다.

- `docker 명령이 없습니다`: Docker Engine 이 설치되어 있지 않습니다. `sudo apt install docker.io` 로 설치합니다.
- `Docker 서버에 연결할 수 없습니다`: Docker 서버가 꺼져 있거나(`sudo service docker start`) 권한이 없는 경우입니다. `docker` 를 직접 실행하면 `permission denied while trying to connect to the Docker daemon socket` 가 나옵니다. 아래를 실행하고 로그아웃했다가 다시 로그인합니다(또는 `newgrp docker`).

```bash
sudo usermod -aG docker $USER
```

### 4.3 `docker compose 플러그인이 없습니다`

```bash
sudo apt install -y docker-compose-v2
docker compose version        # 버전이 출력되면 정상
```

패키지를 찾을 수 없으면 Docker 공식 저장소의 `docker-compose-plugin` 패키지를 설치합니다.

### 4.4 `./drtrainer: Permission denied`

압축을 Windows 에서 풀거나 Windows 에서 git 으로 받으면 실행 권한이 사라집니다.

```bash
chmod +x drtrainer          # 또는 bash drtrainer up
```

### 4.5 `client version 1.52 is too new. Maximum supported API version is 1.43`

호스트 Docker 서버(예: Ubuntu 22.04 의 Docker 24, API 1.43)가 이미지 안의 docker 명령(API 1.52)보다 오래된 경우입니다. 컨테이너는 시작할 때 서버의 API 버전을 소켓에서 읽어 `DOCKER_API_VERSION` 으로 맞춰 사용합니다(Docker 문서가 안내하는 방법). 이 오류가 보이면 최신 버전으로 다시 시작합니다.

```bash
./drtrainer up --rebuild
```

적용되면 `환경 점검` 에 맞춰 쓰는 중이라는 알림이 표시됩니다. 호스트 Docker 를 업데이트하는 것이 근본적인 해결입니다.

### 4.6 `No such image: minio/minio:latest`

MinIO 가 Docker Hub 이미지를 삭제했기 때문입니다. `환경 점검` 의 `minio 이미지 만들기` 를 누르거나 `bash docker/minio-local-image.sh <DRfC 폴더>` 를 실행합니다. 학습 시작 시에도 자동으로 확인하고 만듭니다. 자세한 내용은 [minio 이미지](#minio-이미지)를 참고합니다.

### 4.7 `export DR_DOCKER_STYLE=<DOCKER_STYLE>`, `This node is not a swarm manager`

`init.sh` 가 swarm 을 만드는 단계에서 실패하면 `system.env` 는 만들어졌지만 `<DOCKER_STYLE>` 같은 값이 채워지지 않은 상태로 남습니다. 대시보드는 이 상태를 초기화 미완료로 판정하고 학습·평가 시작을 막습니다. 다음 명령이 해당 파일을 백업하고 `init.sh` 를 처음부터 다시 실행합니다.

```bash
./drtrainer init            # 직접 설치는 sudo bash scripts/setup_drfc.sh
```

정상인 파일까지 백업하고 다시 하려면 `./drtrainer init --force` 를 사용합니다.

### 4.8 swarm 만 없는 경우

`docker swarm leave` 를 했거나 Docker 를 새로 설치한 경우입니다. `환경 점검` 의 `Docker swarm` 항목에서 `swarm 만들기` 를 누릅니다(`docker swarm init`, 안 되면 기본 IP 를 지정해 재시도). 이 컴퓨터의 Docker 가 swarm 모드로 바뀌며, DRfC 가 원래 요구하는 설정입니다.

### 4.9 GPU 가 인식되지 않음, `could not select device driver`

Docker 를 다시 설치할 필요는 없습니다. `./drtrainer gpu` 로 막힌 곳을 확인하고 [GPU](#gpu)의 표대로 조치합니다.

### 4.10 실시간 영상이 나오지 않음

영상 패널의 `영상 진단` 을 실행하고 결과 전체를 확인합니다. 대시보드 포트만 접속 가능하면(SSH 터널 포함) 영상은 대시보드를 통해 전달됩니다. 시뮬레이터가 시작되는 데 1분 정도 걸립니다.

### 4.11 평가 결과가 `In progress` 로 남음

시뮬레이터가 최종 상태를 기록하지 않은 경우입니다. 진행률이 100% 면 완주(`*`)로 표시됩니다. 판정 규칙은 [평가 이름과 주행 영상](#평가-이름과-주행-영상)을 참고합니다.

### 4.12 평가 영상이 없거나 재생되지 않음

- 영상이 없음: `주행 영상 저장 (MP4)` 을 켜고 평가한 경우에만 영상이 만들어집니다. 켜지 않고 한 평가는 다시 평가해야 합니다.
- 재생되지 않음: MP4 안의 영상 형식이 브라우저가 지원하는 H.264 가 아닙니다. `다운로드` 해서 별도 플레이어로 엽니다.

### 4.13 중지 버튼이 없음, `(외부에서 시작됨)`

DeepRacer 의 모든 화면 위쪽 배너에 `중지`, `강제 중지` 가 있습니다. `중지` 로 멈추지 않으면 `강제 중지` 를 사용합니다.

### 4.14 `모델 ... 이(가) 이미 있습니다`

같은 이름으로 다시 학습하려는 경우입니다. `실험 정리` 에서 그 실험의 모델을 삭제하거나, `학습` 화면에서 `덮어쓰기` 를 체크합니다.

### 4.15 포트 8765 가 사용 중

```bash
python3 dashboard.py --port 9000     # Docker: ./drtrainer up --port 9000
```

### 4.16 `train.py` 에서 창이 뜨지 않음

화면이 없는 서버 등에서 발생합니다. `--no-render` 로 실행하거나 대시보드를 사용합니다.

### 4.17 `ERROR: Processes running in stack deepracer-0. Stop training with dr-stop-training.`

DRfC 의 `dr-start-training` 은 시작할 때 아래 세 가지를 이 순서로 검사합니다.

| 순서 | 검사 | 실패 메시지 | `-w` 의 영향 |
| --- | --- | --- | --- |
| 1 | 스택에 작업 기록이 있는가 (`docker stack ps <스택> \| wc -l` 이 1 보다 큼) | `Processes running in stack ...` | 없음 (이 검사가 먼저 실행됨) |
| 2 | 설정 파일이 저장소에 있는가 | `Training aborted. Configuration files were not found` | 없음 |
| 3 | 모델 경로가 이미 있는가 | `Selected path ... exists. Delete it, or use -w option` | 여기서만 `-w` 가 모델을 지우고 진행 |

`docker stack ps` 는 지금 도는 작업뿐 아니라 종료되거나 실패한 작업의 기록도 스택을 지우기 전까지 모두 나열합니다. 그래서 학습이 끝난 뒤나 실패한 뒤에 스택(`deepracer-0`)이 남아 있으면, 아무것도 실행 중이 아니어도 1번에서 거부됩니다. 이 메시지는 `-w` 로 해결되지 않고, 스택을 지워야 합니다(`dr-stop-training` 이 `docker stack rm` 을 실행).

대시보드는 다음과 같이 처리합니다.

- 종료된 작업의 기록만 남아 있으면, 학습·평가를 시작할 때 `docker stack rm` 으로 자동으로 지우고 진행합니다.
- 실행 중이거나 준비 중인 작업이 남아 있으면 컨테이너가 보이지 않아도 `시작 중` 으로 표시하고 위쪽 배너에 `중지` 를 표시합니다. 이 상태에서는 새로 시작하지 않습니다.
- 수동으로 정리하려면 `관리` 의 `이전 실행 기록(스택) 지우기`, 또는 터미널에서 `docker stack rm deepracer-0` (평가는 `deepracer-eval-0`)을 실행합니다.

### 4.18 `Selected path s3://... exists. Delete it, or use -w option. Exiting.`

모델 경로가 이미 있다고 DRfC 가 판단한 경우입니다. 원인은 두 가지입니다.

- 같은 이름의 모델이 있는 경우
- 이름이 같은 글자로 시작하는 다른 모델이 있는 경우. DRfC 는 `aws s3 ls s3://버킷/<이름>` 으로, 끝에 `/` 없이 접두사로 검사합니다. 예를 들어 모델 `exp` 를 시작하려는데 `exp_v2` 가 있으면 `exp` 가 없어도 거부됩니다. 실제 MinIO 와 aws CLI 로 확인했습니다.

처음부터 다시 학습하려면 학습 화면에서 `덮어쓰기` 를 체크합니다(기존 모델이 지워지며 되돌릴 수 없습니다). `-w` 는 `aws s3 rm --recursive s3://버킷/<이름>` 을 실행하며, 개발 환경(aws CLI 1.46, MinIO)에서는 `exp` 를 지워도 `exp_v2` 는 남았습니다. 다른 버전의 aws CLI 에서는 확인하지 않았습니다. 기존 모델에서 이어서 학습하려면 `새 실험` 의 `이어서 학습` 을 사용합니다.

### 4.19 학습 시작 때 나오는 경고 (무시해도 되는 것)

- `image minio/minio:...-local could not be accessed on a registry to record its digest`: 레지스트리 없이 로컬에서 빌드한 minio 이미지를 swarm 이 사용하기 때문에 나옵니다. 노드가 하나인 이 구성에서는 영향이 없고, 뒤이어 `minio 준비됨` 이 나오면 정상입니다.
- `WARNING: CPU governor is not set to performance on all cores.`: DRfC 가 CPU 성능 모드를 권하는 메시지입니다(`sudo cpupower frequency-set -g performance`). 학습은 진행되며 속도에만 영향이 있습니다.

문제가 해결되지 않으면 이슈에 `./drtrainer logs` 출력, `환경 점검` 의 실패 항목, 영상 문제라면 `영상 진단` 결과를 함께 올립니다.

---

## 5. Reference

이 프로젝트는 AWS, Amazon 과 관련 없는 비공식 도구입니다. AWS DeepRacer 는 Amazon 의 서비스입니다.

### 5.1 문서

| 문서 | 이 프로젝트에서 참고한 내용 |
| --- | --- |
| [DRfC installation](https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/245bd7f0fe0810043744d988edcae78c03617fee/docs/installation.md) | 설치 절차, 로컬 요구 사양 |
| [DRfC windows](https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/245bd7f0fe0810043744d988edcae78c03617fee/docs/windows.md) | Windows(WSL2) 설치 |
| [DRfC reference](https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/245bd7f0fe0810043744d988edcae78c03617fee/docs/reference.md) | `system.env`, `run.env` 변수, `dr-*` 명령 |
| [DRfC video](https://github.com/aws-deepracer-community/deepracer-for-cloud/blob/245bd7f0fe0810043744d988edcae78c03617fee/docs/video.md) | 영상 스트림 포트, 카메라 종류 |
| [AWS DeepRacer 보상함수 입력 파라미터](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html) | 보상함수 `params` 의 키 이름 |

DRfC 문서 링크는 이 프로젝트의 Docker 이미지가 고정한 커밋(`245bd7f`) 기준입니다.

### 5.2 서드파티

| 구성 요소 | 사용 방식 | 비고 |
| --- | --- | --- |
| [DeepRacer-for-Cloud](https://github.com/aws-deepracer-community/deepracer-for-cloud) | Docker 이미지가 고정한 커밋(`245bd7f`)을 내려받아 사용. `tools/mock_drfc/defaults/` 의 일부 파일은 이 저장소에서 복사 | 라이선스: `tools/mock_drfc/LICENSE-DRfC.txt` |
| [MinIO](https://github.com/minio/minio) | minio 이미지를 빌드할 때 GitHub 릴리스의 바이너리를 내려받음. 저장소에 바이너리는 포함하지 않음 | AGPL-3.0. 더 이상 유지보수되지 않음 |
| 트랙 (`tracks/reInvent2019_track.npy`, `fetch_tracks.py` 가 내려받는 파일) | [deepracer-race-data](https://github.com/aws-deepracer-community/deepracer-race-data) 저장소의 `raw_data/tracks/npy` | 해당 저장소에서 라이선스 파일을 찾지 못했음. 내려받는 트랙은 저장소에 포함하지 않음(`.gitignore`). 포함된 `reInvent2019_track.npy` 도 같은 출처이므로 재배포 전에 확인 필요 |
| numpy, matplotlib | MiniRacer 의 파이썬 의존성 | 각 프로젝트의 라이선스를 따름 |
