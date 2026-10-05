"""[예제 1] 중앙선 보상만.

실험으로 확인된 점: 이 보상으로 학습한 모델은 평가 랩타임이 약 27초(평균 약 0.85 m/s)로,
선택 가능한 가장 느린 속도(0.8 m/s) 쪽으로 쏠렸다. 매 스텝 보상을 받으니 오래 달릴수록 총 보상이 커지기 때문이다."""


def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    if distance_from_center <= 0.1 * track_width:
        return 1.0
    if distance_from_center <= 0.25 * track_width:
        return 0.5
    if distance_from_center <= 0.5 * track_width:
        return 0.1
    return 1e-3
