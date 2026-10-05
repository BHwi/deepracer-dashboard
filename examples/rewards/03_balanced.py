"""[예제 3] 중앙선 + 속도 + 조향 억제를 가중합으로 섞은 예. W_CENTER, W_SPEED 를 바꿔 가며 비교해 볼 수 있다."""


def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    abs_steer = abs(params['steering_angle'])

    if not params['all_wheels_on_track']:
        return 1e-3

    r_center = max(0.0, 1.0 - (distance_from_center / (0.5 * track_width)) ** 2)   # 0~1
    r_speed = speed / 2.4                                                          # 0~1 (최고 속도 2.4 기준)
    r_smooth = 0.6 if abs_steer > 20 else 1.0                                      # 지그재그 억제

    W_CENTER, W_SPEED = 1.0, 1.0
    return float((W_CENTER * r_center + W_SPEED * r_speed) * r_smooth)
