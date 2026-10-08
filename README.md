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

<table>
  <tr>
    <td width="50%"><img src="docs/img/miniracer-compare.png" alt="MiniRacer 실험 비교"><br><sub>MiniRacer: 실험 비교</sub></td>
    <td width="50%"><img src="docs/img/deepracer-train.png" alt="DeepRacer 학습"><br><sub>DeepRacer simulator: 학습 (시험용 화면)</sub></td>
  </tr>
</table>

## 1. 실행 방법

Ubuntu 또는 Windows(WSL2)에서 실행합니다. 필요한 것은 Docker 입니다.

```bash
git clone https://github.com/<사용자>/<저장소>.git
cd <저장소>

sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER        # 실행 후 로그아웃했다가 다시 로그인

chmod +x drtrainer
./drtrainer up                       # 처음 실행은 이미지 빌드와 다운로드로 수 분~십수 분
```

브라우저에서 **http://localhost:8765** 를 엽니다.

- 상태 `./drtrainer status`, 로그 `./drtrainer logs -f`, 끄기 `./drtrainer down`
- GPU 가 없으면 `./drtrainer up --arch cpu`, 포트를 바꾸려면 `--port 9000`
- 원격 서버는 SSH 터널로 접속합니다: `ssh -L 8765:127.0.0.1:8765 사용자@서버` 후 내 브라우저에서 `http://localhost:8765`. 대시보드는 인증이 없으므로 외부에 열지 마세요.
- Docker 없이 MiniRacer 만 쓰려면:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt
python3 dashboard.py
```

### GPU 설정

`./drtrainer up` 은 GPU 를 자동으로 감지하고, 감지하지 못하면 CPU 모드로 시작합니다. 진단은 `./drtrainer gpu` 입니다.

**GPU 가 여러 개인 서버는 사용할 GPU 번호를 지정해야 합니다.** 지정하지 않으면 학습이 모든 GPU 를 잡으려 해서 다른 사람의 작업과 부딪힙니다. 비어 있는 GPU 를 먼저 확인하고, DRfC 폴더(기본 `~/deepracer-for-cloud`)의 `system.env` 에 번호를 씁니다.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

```text
DR_SAGEMAKER_CUDA_DEVICES=3      # 학습
DR_ROBOMAKER_CUDA_DEVICES=3      # 시뮬레이터
```

**드라이버가 낮은 경우.** GPU 시뮬레이터 이미지는 CUDA 12.6 기반이라 드라이버 560 이상이 필요합니다. 그보다 낮으면 `unsatisfied condition: cuda>=12.6` 오류가 나고 DRfC 는 CPU 모드로 설정합니다. 드라이버를 올릴 수 없으면(525 이상) CUDA 버전 검사를 끈 로컬 이미지로 시도합니다.

```bash
bash scripts/gpu_check.sh                       # 진단 (선택)
bash scripts/gpu_norequire.sh --pull            # 이미지 받기 + TensorFlow 로 GPU 연산 시험. 설정은 바꾸지 않음
bash scripts/gpu_norequire.sh --gpu 3 --apply   # 통과하면 system.env 를 백업 후 수정
```

- 시험이 통과하면 `RESULT_OK` 가 나옵니다. 실패하면(`CUDA_ERROR_...` 등) `system.env` 는 바뀌지 않으니 CPU 모드를 쓰세요.
- GPU 가 둘 이상이면 `--gpu` 가 필요합니다(`--gpu 3` 은 학습과 시뮬레이터 모두 3번, `--gpu 3,0` 은 학습 3번, 시뮬레이터 0번). 없으면 `--apply` 를 줘도 적용하지 않습니다.
- 적용 후 학습을 시작하면 대시보드 DeepRacer > 학습 > 로그 > `학습` 에 `Created device ... pci bus id: ...` 가 지정한 GPU 의 버스 번호로 나와야 합니다(번호와 버스는 `nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader`). 이어서 `Policy training>` 줄이 나오면 GPU 학습이 도는 것입니다.
- 되돌리기: 대시보드 DeepRacer > 환경 점검 > GPU / CPU 에서 CPU 를 선택하거나, `system.env.bak-*` 에서 복원합니다.
- DRfC 의 `init.sh` 는 다시 실행하지 마세요. `system.env` 를 덮어씁니다.

## 2. 사용 방법

위쪽의 `MiniRacer`, `DeepRacer` 탭으로 전환합니다. 보상함수 전략은 MiniRacer 에서 먼저 시험하고, 정한 전략을 DeepRacer simulator 에서 학습합니다.

### 2.1 MiniRacer

GPU 없이 노트북에서 보상함수를 빠르게 학습하고 비교하는 2D 시뮬레이터입니다. 기본 설정은 약 1.5분 걸립니다. 센서가 실제 DeepRacer 와 달라서 절대 수치가 아니라 전략의 상대 비교에 씁니다.

1. `새 실험`: 이름, 트랙, 행동 공간, 보상함수를 입력합니다(`예제에서 불러오기` 가능). `학습 시작`을 누릅니다.
2. `결과`: 그래프나 슬라이더로 iteration 을 고르면 그때의 평가 주행이 지도에서 재생됩니다.
3. `비교`: 왼쪽 목록에서 체크한 실험을 겹쳐 봅니다. 보상함수가 다르면 보상 값 대신 진행률, 완주율, 랩타임으로 비교합니다.
4. `주행 시험`: 학습된 모델을 다른 트랙이나 외란을 주고 달려 봅니다. `실험 정리`: 실험 기록을 삭제합니다.

보상함수의 `params` 키는 AWS DeepRacer 와 같아서 콘솔에 그대로 붙여 넣을 수 있습니다. 예제는 `examples/rewards/`, 기본 설정은 `custom_files/`, 결과는 `runs/<이름>/` 에 있습니다.

명령줄로도 실행할 수 있습니다(가상환경을 켠 상태).

```bash
python3 train.py --name my_first                                    # 학습
python3 train.py --reward examples/rewards/01_center_line.py --name center
python3 play.py --run my_first                                      # 학습된 모델 주행
python3 compare.py center my_first                                  # 비교
```

### 2.2 DeepRacer simulator

DRfC 의 실제 시뮬레이터로 학습하고 평가해서 차량용 파일까지 만듭니다.

1. `환경 점검`: 모든 항목이 `정상`인지 확인합니다. 막힌 항목은 안내와 버튼으로 해결합니다.
2. `새 실험`: 이름(모델 이름), 트랙, 행동 공간, 하이퍼파라미터, 보상함수를 입력합니다. MiniRacer 실험을 가져올 수 있습니다.
3. `학습`: `학습 시작`. 실시간 영상, iteration 별 그래프, 로그가 나옵니다. 영상이 안 나오면 `영상 진단`을 누릅니다. 같은 이름의 모델이 이미 있으면 `덮어쓰기`를 체크해야 시작합니다(기존 모델이 지워집니다).
4. `평가`: `평가할 모델`을 고르고, 평가 이름과 `주행 영상 저장`을 지정해 시작합니다. 평가 기록에서 이름을 누르면 결과표와 저장된 영상이 나옵니다.
5. `차량용 내보내기`: 차량에 올릴 `tar.gz` 를 만들어 내려받습니다.

그 밖에:

- 학습이나 평가가 실행 중이면 모든 화면 위쪽에 `중지`, `강제 중지` 배너가 나옵니다.
- `관리`: DRfC 의 `dr-*` 명령(`dr-stop-training` 등)을 버튼으로 실행합니다.
- `실험 정리`: 설정과 모델을 삭제합니다. 되돌릴 수 없으므로 확인란에 `삭제` 를 입력해야 합니다.

## 3. 출처

AWS, Amazon 과 관련 없는 비공식 도구입니다.

- [DeepRacer-for-Cloud](https://github.com/aws-deepracer-community/deepracer-for-cloud)(라이선스 파일: `tools/mock_drfc/LICENSE-DRfC.txt`)를 고정한 커밋으로 내려받아 사용합니다.
- DRfC 의 저장소로 쓰는 [MinIO](https://github.com/minio/minio)는 Docker Hub 이미지가 삭제돼, 빌드할 때 GitHub 릴리스의 바이너리(AGPL-3.0, 더 이상 유지보수되지 않음)로 로컬 이미지를 자동으로 만듭니다.
- `tracks/reInvent2019_track.npy` 는 [deepracer-race-data](https://github.com/aws-deepracer-community/deepracer-race-data) 에서 가져왔습니다. 그 저장소에서 라이선스 파일을 찾지 못했으니 재배포하기 전에 확인하세요.
