"""[예제 5] 진행률(progress)과 걸린 스텝(steps)을 이용해 '빨리 가는 것'을 직접 보상. 랩타임 단축용.
TOTAL_NUM_STEPS 는 트랙 길이와 목표 속도에 따라 바꿔야 한다 (15 스텝 = 1초)."""


def reward_function(params):
    if not params['all_wheels_on_track']:
        return 1e-3

    steps = params['steps']
    progress = params['progress']
    TOTAL_NUM_STEPS = 250     # 1바퀴를 이 스텝 안에 돌면 좋다 (reInvent2019 기준 약 17초)

    reward = 1.0
    if steps > 0 and progress > (steps / TOTAL_NUM_STEPS) * 100:
        reward += 2.0
    return float(reward)
