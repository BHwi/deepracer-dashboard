"""기본 보상함수 - 새 실험을 만들 때 처음 채워지는 보상함수입니다. 필요에 맞게 고쳐서 쓰세요.

DeepRacer on AWS 콘솔의 보상함수 편집기에 그대로 붙여 넣어도 동작하도록,
params 딕셔너리의 키 이름은 공식 문서와 같습니다.
(all_wheels_on_track, distance_from_center, track_width, speed, steering_angle,
 progress, steps, heading, closest_waypoints, waypoints, is_left_of_center ...)
사용 가능한 라이브러리: math, random, numpy, scipy, shapely (DeepRacer 와 동일)
"""


def reward_function(params):
    '''
    예제: 중앙선 따라가기 (AWS 공식 예제 1과 동일)
    '''
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # 트랙 이탈 직전/이탈

    return float(reward)
