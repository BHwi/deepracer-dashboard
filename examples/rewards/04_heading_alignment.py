"""[예제 4] 트랙 진행 방향과 차의 방향(heading)이 맞는지 보는 보상. waypoints / closest_waypoints / heading 사용법 (AWS 공식 예제 변형)."""
import math


def reward_function(params):
    if not params['all_wheels_on_track']:
        return 1e-3

    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']

    prev_point = waypoints[closest_waypoints[0]]
    next_point = waypoints[closest_waypoints[1]]
    track_direction = math.degrees(math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]))

    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    reward = 1.0
    if direction_diff > 10.0:
        reward *= 0.5
    if direction_diff > 25.0:
        reward *= 0.5
    return float(reward)
