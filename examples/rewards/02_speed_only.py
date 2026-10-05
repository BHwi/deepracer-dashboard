"""[예제 2] 매 스텝 속도만큼 보상.

실험으로 확인된 점: 한 바퀴를 완주하면 총 보상이 약 15 x 트랙길이(m)로 거의 일정했다.
매 스텝 보상의 합 = 속도의 합 = 이동 거리에 비례하기 때문이다. 그래서 이 보상은 '빨리' 달리게 만드는 힘이 없다.
(트랙 이탈하면 이후 보상을 못 받으므로 '완주'는 학습되지만, 속도에 따른 보상 차이는 없다.)"""


def reward_function(params):
    if not params['all_wheels_on_track']:
        return 1e-3
    return float(params['speed'])
