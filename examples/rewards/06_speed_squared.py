"""[예제 6] 속도의 제곱에 비례하는 보상 (이탈하면 거의 0).

예제 2(속도에 비례)는 총 보상이 '이동 거리'에 비례해서 빠르든 느리든 한 바퀴 총 보상이 같다.
속도를 제곱하면 같은 거리를 더 빨리 갈수록 총 보상이 커진다.
이유: 한 바퀴 총 보상 = (속도^2 의 합) 이고, 일정한 속도 v 로 L 만큼 달리면 스텝 수가 L/(v*dt) 라서 총 보상은 L*v/dt, 즉 v 에 비례한다."""


def reward_function(params):
    if not params['all_wheels_on_track']:
        return 1e-3
    return float((params['speed'] / 2.4) ** 2)
