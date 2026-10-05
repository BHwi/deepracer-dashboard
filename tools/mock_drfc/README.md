# 시험용 가짜 DRfC

**실제 DeepRacer 학습을 하지 않습니다.** Docker/GPU 가 없는 컴퓨터에서 대시보드의 DeepRacer 화면을 미리 보고 시험하기 위한 모형입니다.
실제 DRfC 의 명령(dr-upload-custom-files, dr-start-training -q/-w, dr-stop-training, dr-start-evaluation, dr-create-car-zip)과
실험 폴더(`-e 이름`) 동작, 지표 파일(`data/minio/bucket/<접두사>/metrics/TrainingMetrics.json`) 형식을 흉내 내어 가짜 학습 곡선을 만듭니다.
대시보드 '환경 점검'에서 DRfC 폴더를 이 폴더(tools/mock_drfc)로 지정하면 화면에 '모의 환경' 표시가 나옵니다.
- GPU 없음을 흉내 내려면 `MOCK_NO_GPU=1 python dashboard.py`
- 학습 속도: `MOCK_SPEED=0.05` (에피소드당 초, 기본 0.08)

`defaults/` 안의 파일은 aws-deepracer-community/deepracer-for-cloud 저장소(같은 폴더의 LICENSE-DRfC.txt, MIT 계열)에서 가져온 것입니다.
