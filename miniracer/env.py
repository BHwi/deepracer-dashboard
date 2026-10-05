"""MiniRacer 환경.

- 차량: 단순 자전거(bicycle) 운동학 모델, 15 Hz (실제 DeepRacer 카메라 fps 와 동일)
- 행동 공간: model_metadata.json (DeepRacer 포맷, discrete)
- 보상: reward_function.py 의 reward_function(params) 를 그대로 호출.
        params 의 키 이름은 DeepRacer 공식 문서와 같다.
- 관측(에이전트가 보는 것): 실제 DeepRacer 는 카메라 이미지(160x120 흑백)를 CNN 에 넣지만,
  MiniRacer 는 같은 120도 시야각을 13개의 레이(ray)로 단순화해서 "경계선까지의 거리"를 본다.
  + 현재 속도, 현재 조향각 (실제 DeepRacer 는 이 둘을 직접 받지 않는다. 학습을 빠르게 하려는 단순화)
  보상함수가 쓰는 params(위치, 중앙선 거리 등)는 에이전트에게 직접 주어지지 않고 '채점'에만 쓰인다.
"""
import importlib.util
import json
import math
import os
import traceback

import numpy as np

FPS = 15
DT = 1.0 / FPS
WHEELBASE = 0.165        # 축간거리 (m), DeepRacer 실차 근사
CAR_LEN, CAR_WID = 0.20, 0.12
MAX_STEER_RATE = 300.0   # 조향각 변화 한계 (deg/s)
MAX_ACCEL = 4.0          # 가·감속 한계 (m/s^2)

RAY_ANGLES = np.linspace(-60, 60, 13)   # 120도 시야각 (실제 DeepRacer 카메라와 동일)
RAY_RANGE = 2.5                          # m
RAY_STEP = 0.05                          # m


class RewardFunctionError(RuntimeError):
    pass


def load_reward_function(path):
    spec = importlib.util.spec_from_file_location("user_reward_function", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                      # 문법 오류 등
        raise RewardFunctionError(f"{path} 를 불러오는 중 오류: {type(e).__name__}: {e}") from e
    if not hasattr(mod, "reward_function"):
        raise RewardFunctionError(f"{path} 안에 reward_function(params) 가 없습니다.")
    return mod.reward_function


def load_action_space(path):
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    acts = sorted(meta["action_space"], key=lambda a: a.get("index", 0))
    return [(float(a["steering_angle"]), float(a["speed"])) for a in acts]


def auto_max_steps(track, actions, factor=4.0):
    """트랙을 최고 속도로 도는 데 걸리는 스텝 수의 factor 배."""
    vmax = max(v for _, v in actions)
    return int(factor * track.length * FPS / vmax)


class MiniRacerEnv:
    def __init__(self, track, reward_path, metadata_path, max_steps=None, noise=0.0, seed=None):
        self.track = track
        self.reward_path = reward_path
        self.reward_fn = load_reward_function(reward_path)
        self.actions = load_action_space(metadata_path)
        self.n_actions = len(self.actions)
        self.v_max = max(v for _, v in self.actions)
        self.steer_max = max(30.0, max(abs(s) for s, _ in self.actions))
        self.max_steps = max_steps or auto_max_steps(track, self.actions)
        self.noise = noise                      # Sim-to-Real 실험용 외란 세기 (0 = 없음)
        self.rng = np.random.default_rng(seed)
        self.obs_dim = len(RAY_ANGLES) + 2
        self.reset(0)
        self._validate_reward()

    # ------------------------------------------------------------------ 에피소드 제어
    def reset(self, start_idx=0):
        tr = self.track
        start_idx = int(start_idx) % tr.n
        self.start_idx = start_idx
        self.x, self.y = tr.center[start_idx]
        self.heading = tr.heading_at(start_idx)   # deg
        self.speed = 0.0
        self.steer = 0.0
        self.steps = 0
        self.progress = 0.0
        self._s_prev = tr.locate(self.x, self.y)["s"]
        self.done = False
        self.status = "running"
        self.last_reward = 0.0
        self.params = self._make_params(True, tr.locate(self.x, self.y))
        return self._observe()

    def _validate_reward(self):
        """학습 시작 전에 보상함수가 숫자를 돌려주는지 한 번 확인."""
        self._call_reward(dict(self.params))
        self.reset(0)

    def _call_reward(self, params):
        try:
            r = self.reward_fn(params)
        except Exception as e:
            line = next((f.lineno for f in reversed(traceback.extract_tb(e.__traceback__)) if f.filename == self.reward_path), None)
            where = f"{os.path.basename(self.reward_path)} {line}번째 줄" if line else "reward_function"
            raise RewardFunctionError(
                f"{where}에서 오류: {type(e).__name__}: {e}\n"
                f"(그때 상태: steps={params['steps']}, progress={params['progress']:.1f})") from e
        try:
            r = float(r)
        except (TypeError, ValueError):
            raise RewardFunctionError(f"reward_function 은 숫자(float)를 돌려줘야 합니다. 받은 값: {r!r}")
        if not math.isfinite(r):
            raise RewardFunctionError(f"reward_function 이 유한하지 않은 값({r})을 돌려줬습니다.")
        return r

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("에피소드가 끝났습니다. reset() 을 호출하세요.")
        steer_cmd, speed_cmd = self.actions[int(action_idx)]

        # 조향·속도는 명령값으로 즉시 바뀌지 않고 한계 속도로 따라감 (실차 느낌)
        self.steer += float(np.clip(steer_cmd - self.steer, -MAX_STEER_RATE * DT, MAX_STEER_RATE * DT))
        self.speed += float(np.clip(speed_cmd - self.speed, -MAX_ACCEL * DT, MAX_ACCEL * DT))

        if self.noise > 0:   # 외란: 조향 오차·속도 오차 (도메인 갭 흉내)
            steer_eff = self.steer + self.rng.normal(0, 5.0 * self.noise)
            speed_eff = self.speed * (1 + self.rng.normal(0, 0.1 * self.noise))
        else:
            steer_eff, speed_eff = self.steer, self.speed

        h = math.radians(self.heading)
        self.x += speed_eff * math.cos(h) * DT
        self.y += speed_eff * math.sin(h) * DT
        self.heading += math.degrees(speed_eff / WHEELBASE * math.tan(math.radians(steer_eff)) * DT)
        self.heading = (self.heading + 180) % 360 - 180
        self.steps += 1

        loc = self.track.locate(self.x, self.y)
        L = self.track.length
        ds = (loc["s"] - self._s_prev + L / 2) % L - L / 2
        self._s_prev = loc["s"]
        self.progress = max(0.0, self.progress + ds / L * 100.0)

        on_track = self._all_wheels_on_track()
        self.params = self._make_params(on_track, loc)
        reward = self._call_reward(dict(self.params))
        self.last_reward = reward

        if not on_track:
            self.done, self.status = True, "offtrack"
        elif self.progress >= 100.0:
            self.done, self.status = True, "lap_complete"
        elif self.steps >= self.max_steps:
            self.done, self.status = True, "timeout"
        return self._observe(), reward, self.done, {"status": self.status}

    # ------------------------------------------------------------------ 내부
    def _corners(self):
        h = math.radians(self.heading)
        c, s = math.cos(h), math.sin(h)
        d = np.array([(CAR_LEN / 2, CAR_WID / 2), (CAR_LEN / 2, -CAR_WID / 2),
                      (-CAR_LEN / 2, -CAR_WID / 2), (-CAR_LEN / 2, CAR_WID / 2)])
        xs = self.x + d[:, 0] * c - d[:, 1] * s
        ys = self.y + d[:, 0] * s + d[:, 1] * c
        return xs, ys

    def _all_wheels_on_track(self):
        xs, ys = self._corners()
        return bool(self.track.on_track(xs, ys).all())

    def car_polygon(self):
        xs, ys = self._corners()
        return list(zip(xs, ys))

    def _make_params(self, on_track, loc):
        tr = self.track
        return {
            "all_wheels_on_track": bool(on_track),
            "x": float(self.x),
            "y": float(self.y),
            "closest_objects": [0, 0],
            "closest_waypoints": loc["closest_waypoints"],
            "distance_from_center": loc["distance_from_center"],
            "is_crashed": False,
            "is_left_of_center": loc["is_left_of_center"],
            "is_offtrack": not bool(on_track),
            "is_reversed": False,
            "heading": float(self.heading),
            "objects_distance": [],
            "objects_heading": [],
            "objects_left_of_center": [],
            "objects_location": [],
            "objects_speed": [],
            "progress": float(min(self.progress, 100.0)),
            "speed": float(self.speed),
            "steering_angle": float(self.steer),
            "steps": int(self.steps),
            "track_length": float(tr.length),
            "track_width": float(tr.width),
            "waypoints": tr.waypoints_list(),
        }

    def _observe(self):
        dist, ex, ey = self.track.cast_rays(self.x, self.y, self.heading, RAY_ANGLES, RAY_RANGE, RAY_STEP)
        self.ray_ends = (ex, ey)
        d = dist / RAY_RANGE
        if self.noise > 0:
            d = np.clip(d + self.rng.normal(0, 0.03 * self.noise, d.shape), 0.0, 1.0)
        return np.concatenate([d, [self.speed / self.v_max, self.steer / self.steer_max]]).astype(np.float64)
